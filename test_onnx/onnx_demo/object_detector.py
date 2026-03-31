"""
ONNX-based LocalFeatureObjectDetector

Replaces the PyTorch-based LocalFeatureObjectDetector with one that uses
ONNX Runtime for both SuperPoint (feature extraction) and SuperGlue
(2D-2D matching).
"""

from __future__ import annotations

import cv2
import numpy as np
import os.path as osp
from pathlib import Path
from typing import Tuple

from onnx_demo.onnx_models import SuperPointOnnx, SuperGlueOnnx


def _pack_extract_data(img_path: str) -> np.ndarray:
    """Load a grayscale image and return [1, H, W] float32 array."""
    path = Path(img_path)
    if not path.is_file():
        # COLMAP image names are often stored as relative paths (e.g. ../../data/demo/...).
        # Re-resolve them when benchmark cwd is different.
        repo_root = Path(__file__).resolve().parents[2]
        alt_from_repo = (repo_root / img_path).resolve()
        if alt_from_repo.is_file():
            path = alt_from_repo
        elif "data/demo/" in img_path:
            rel = img_path.split("data/demo/", 1)[1]
            alt_from_demo = (repo_root / "data/demo" / rel).resolve()
            if alt_from_demo.is_file():
                path = alt_from_demo
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read reference image: {img_path} (resolved: {path})")
    return (image[np.newaxis] / 255.0).astype(np.float32)


def _pack_match_data(db_det: dict, q_det: dict,
                     db_size: np.ndarray, q_size: np.ndarray) -> dict:
    """Build the data dict expected by SuperGlueOnnx.__call__."""
    import torch

    def _t(arr):
        return torch.from_numpy(arr)[None].float()

    data = {}
    for k, v in db_det.items():
        if k != 'size':
            data[k + '0'] = _t(v)
    for k, v in q_det.items():
        if k != 'size':
            data[k + '1'] = _t(v)

    data['image0'] = np.empty((1, 1) + tuple(db_size)[::-1])
    data['image1'] = np.empty((1, 1) + tuple(q_size)[::-1])
    return data


class LocalFeatureObjectDetectorOnnx:
    """
    Object detector based on 2D local feature matching (ONNX version).

    Identical interface to the original LocalFeatureObjectDetector but
    uses SuperPointOnnx + SuperGlueOnnx instead of PyTorch models.
    """

    def __init__(
        self,
        superpoint_onnx_path: str,
        superglue_onnx_path: str,
        sfm_ws_dir: str,
        n_ref_view: int = 15,
        output_results: bool = False,
        detect_save_dir: str | None = None,
        K_crop_save_dir: str | None = None,
        sp_config: dict | None = None,
    ):
        self.extractor = SuperPointOnnx(superpoint_onnx_path, config=sp_config)
        self.matcher   = SuperGlueOnnx(superglue_onnx_path)
        self.output_results  = output_results
        self.detect_save_dir = detect_save_dir
        self.K_crop_save_dir = K_crop_save_dir
        self.db_dict = self._extract_ref_view_features(sfm_ws_dir, n_ref_view)

    # ── reference-view feature extraction ────────────────────────────────────

    def _extract_ref_view_features(self, sfm_ws_dir: str, n_ref_views: int) -> dict:
        from onnx_demo.src.utils.colmap.read_write_model import read_model

        assert osp.exists(sfm_ws_dir), f"SfM workspace not found: {sfm_ws_dir}"
        cameras, images, points3D = read_model(sfm_ws_dir)

        sample_gap = max(len(images) // n_ref_views, 1)
        db_dict = {}
        for idx in range(1, len(images), sample_gap):
            db_img_path = images[idx].name
            db_img = _pack_extract_data(db_img_path)          # [1, H, W]
            db_inp = db_img[np.newaxis]                        # [1, 1, H, W]
            det = self.extractor(db_inp)
            det['size'] = np.array(db_img.shape[-2:])
            db_dict[idx] = det
        return db_dict

    # ── matching ──────────────────────────────────────────────────────────────

    def _match_worker(self, query: dict) -> dict:
        results = {}
        for idx, db in self.db_dict.items():
            match_data = _pack_match_data(db, query, db['size'], query['size'])
            match_pred = self.matcher(match_data)

            matches = match_pred['matches0'][0].numpy()
            confs   = match_pred['matching_scores0'][0].numpy()
            valid   = matches > -1

            mkpts0 = db['keypoints'][valid]
            mkpts1 = query['keypoints'][matches[valid]]

            if mkpts0.shape[0] < 6:
                results[idx] = {
                    'inliers': np.empty((0,)),
                    'bbox': np.array([0, 0, query['size'][0], query['size'][1]]),
                }
                continue

            affine, inliers = cv2.estimateAffinePartial2D(
                mkpts0, mkpts1, ransacReprojThreshold=6
            )
            if affine is None:
                results[idx] = {
                    'inliers': np.empty((0,)),
                    'bbox': np.array([0, 0, query['size'][0], query['size'][1]]),
                }
                continue

            db_shape = db['size']
            four_corner = np.array([
                [0,           0,           1],
                [db_shape[1], 0,           1],
                [0,           db_shape[0], 1],
                [db_shape[1], db_shape[0], 1],
            ]).T
            bbox = (affine @ four_corner).T.astype(np.int32)
            lt = np.min(bbox, axis=0)
            rb = np.max(bbox, axis=0)
            results[idx] = {
                'inliers': inliers,
                'bbox': np.array([lt[0], lt[1], rb[0], rb[1]]),
            }
        return results

    def _detect_by_matching(self, query: dict) -> np.ndarray:
        results = self._match_worker(query)
        best_idx = max(results, key=lambda k: results[k]['inliers'].shape[0])
        return results[best_idx]['bbox']

    # ── image cropping ────────────────────────────────────────────────────────

    def crop_img_by_bbox(
        self,
        query_img_path: str,
        bbox: np.ndarray,
        K: np.ndarray | None = None,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        from src.utils.data_utils import get_K_crop_resize, get_image_crop_resize

        x0, y0, x1, y1 = bbox
        origin_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)

        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop = None
        if K is not None:
            K_crop, _ = get_K_crop_resize(bbox, K, resize_shape)
        image_crop, _ = get_image_crop_resize(origin_img, bbox, resize_shape)

        bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape2 = np.array([crop_size, crop_size])
        if K is not None:
            K_crop, _ = get_K_crop_resize(bbox_new, K_crop, resize_shape2)
        image_crop, _ = get_image_crop_resize(image_crop, bbox_new, resize_shape2)

        return image_crop, K_crop

    def _save_detection(self, crop_img: np.ndarray, query_img_path: str):
        if self.output_results and self.detect_save_dir:
            cv2.imwrite(
                osp.join(self.detect_save_dir, osp.basename(query_img_path)),
                crop_img,
            )

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        query_img: np.ndarray,
        query_img_path: str,
        K: np.ndarray,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect object by 2D local feature matching and crop image.

        Parameters
        ----------
        query_img      : [1, 1, H, W] float32 (0-1)
        query_img_path : path to the original image file
        K              : [3, 3] camera intrinsic matrix
        crop_size      : output crop size in pixels

        Returns
        -------
        bbox        : [x0, y0, x1, y1]
        image_crop  : [1, 1, crop_size, crop_size] float32 (0-1)
        K_crop      : [3, 3]
        """
        # Extract query features
        q_inp = query_img if query_img.ndim == 4 else query_img[np.newaxis]
        q_det = self.extractor(q_inp)
        q_det['size'] = np.array(query_img.shape[-2:])

        bbox = self._detect_by_matching(q_det)
        image_crop, K_crop = self.crop_img_by_bbox(
            query_img_path, bbox, K, crop_size=crop_size
        )
        self._save_detection(image_crop, query_img_path)

        image_crop_f = image_crop.astype(np.float32) / 255.0
        image_crop_t = image_crop_f[np.newaxis, np.newaxis]   # [1,1,H,W]
        return bbox, image_crop_t, K_crop

    def previous_pose_detect(
        self,
        query_img_path: str,
        K: np.ndarray,
        pre_pose: np.ndarray,
        bbox3D_corner: np.ndarray,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect object by projecting 3D bbox with the previous frame's pose.
        """
        from src.utils.vis_utils import reproj

        proj_2d = reproj(K, pre_pose, bbox3D_corner)
        x0, y0 = np.min(proj_2d, axis=0)
        x1, y1 = np.max(proj_2d, axis=0)
        bbox = np.array([x0, y0, x1, y1]).astype(np.int32)

        image_crop, K_crop = self.crop_img_by_bbox(
            query_img_path, bbox, K, crop_size=crop_size
        )
        self._save_detection(image_crop, query_img_path)

        image_crop_f = image_crop.astype(np.float32) / 255.0
        image_crop_t = image_crop_f[np.newaxis, np.newaxis]
        return bbox, image_crop_t, K_crop
