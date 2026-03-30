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


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point: np.ndarray, rot_rad: float) -> np.ndarray:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array(
        [
            src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs,
        ],
        dtype=np.float32,
    )


def _get_affine_transform(center: np.ndarray, scale: np.ndarray, output_size: np.ndarray) -> np.ndarray:
    src_w = float(scale[0])
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])
    src_dir = _get_dir(np.array([0.0, src_w * -0.5], dtype=np.float32), 0.0)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(src, dst)


def _get_image_crop_resize(image: np.ndarray, box: np.ndarray, resize_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
    scale = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32)
    resize_h, resize_w = int(resize_shape[0]), int(resize_shape[1])
    trans_crop = _get_affine_transform(center, scale, np.array([resize_w, resize_h], dtype=np.float32))
    image_crop = cv2.warpAffine(image, trans_crop, (resize_w, resize_h), flags=cv2.INTER_LINEAR)
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]], dtype=np.float32)], axis=0)
    return image_crop, trans_crop_homo


def _get_K_crop_resize(box: np.ndarray, K_orig: np.ndarray, resize_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)
    scale = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32)
    resize_h, resize_w = int(resize_shape[0]), int(resize_shape[1])
    trans_crop = _get_affine_transform(center, scale, np.array([resize_w, resize_h], dtype=np.float32))
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]], dtype=np.float32)], axis=0)
    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1), dtype=K_orig.dtype)], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    K_crop_homo = trans_crop_homo @ K_orig_homo
    return K_crop_homo[:3, :3], K_crop_homo


def _pack_extract_data(img_path: str) -> np.ndarray:
    """Load a grayscale image and return [1, H, W] float32 array."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return (image[np.newaxis] / 255.0).astype(np.float32)


def _resolve_colmap_image_path(raw_name: str, seq_dir: str | None) -> str:
    """
    COLMAP often stores paths like ../../data/demo/... relative to the project
    ``data/demo`` directory. Resolve using ``seq_dir`` (…/mark_cup/mark_cup-annotate).
    """
    if osp.isabs(raw_name) and osp.isfile(raw_name):
        return raw_name
    if osp.isfile(raw_name):
        return raw_name
    if seq_dir:
        seq_path = Path(seq_dir).resolve()
        basename = Path(raw_name).name
        candidates = []

        # Most common case: COLMAP stores old absolute path, but filename stays same.
        candidates.append(seq_path / "color" / basename)
        candidates.append(seq_path / "color_full" / basename)

        # Try to reconstruct trailing "<seq-name>/color/<img>.png" fragment.
        parts = Path(raw_name).parts
        if "color" in parts:
            idx = parts.index("color")
            if idx >= 1:
                candidates.append(seq_path.parent / Path(*parts[idx - 1 :]))
        if "color_full" in parts:
            idx = parts.index("color_full")
            if idx >= 1:
                candidates.append(seq_path.parent / Path(*parts[idx - 1 :]))

        # Legacy relative form used by some datasets.
        candidates.append((seq_path.parent.parent / raw_name).resolve())

        for cand in candidates:
            if cand.is_file():
                return str(cand)
    return raw_name


def _pack_match_data(db_det: dict, q_det: dict,
                     db_size: np.ndarray, q_size: np.ndarray) -> dict:
    """Build the data dict expected by SuperGlueOnnx.__call__."""
    def _t(arr: np.ndarray) -> np.ndarray:
        return arr[np.newaxis].astype(np.float32, copy=False)

    data = {}
    for k, v in db_det.items():
        if k != 'size':
            data[k + '0'] = _t(v)
    for k, v in q_det.items():
        if k != 'size':
            data[k + '1'] = _t(v)

    data['image0'] = np.empty((1, 1) + tuple(db_size)[::-1], dtype=np.float32)
    data['image1'] = np.empty((1, 1) + tuple(q_size)[::-1], dtype=np.float32)
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
        seq_dir: str | None = None,
    ):
        self.extractor = SuperPointOnnx(superpoint_onnx_path, config=sp_config)
        self.matcher   = SuperGlueOnnx(superglue_onnx_path)
        self.output_results  = output_results
        self.detect_save_dir = detect_save_dir
        self.K_crop_save_dir = K_crop_save_dir
        self._seq_dir = seq_dir
        self.db_dict = self._extract_ref_view_features(sfm_ws_dir, n_ref_view)

    # ── reference-view feature extraction ────────────────────────────────────

    def _extract_ref_view_features(self, sfm_ws_dir: str, n_ref_views: int) -> dict:
        from onnx_demo.src.utils.colmap.read_write_model import read_model

        assert osp.exists(sfm_ws_dir), f"SfM workspace not found: {sfm_ws_dir}"
        cameras, images, points3D = read_model(sfm_ws_dir)

        sample_gap = max(len(images) // n_ref_views, 1)
        db_dict = {}
        for idx in range(1, len(images), sample_gap):
            db_img_path = _resolve_colmap_image_path(
                images[idx].name, self._seq_dir
            )

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
        x0, y0, x1, y1 = bbox
        origin_img = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE)

        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop = None
        if K is not None:
            K_crop, _ = _get_K_crop_resize(bbox, K, resize_shape)
        image_crop, _ = _get_image_crop_resize(origin_img, bbox, resize_shape)

        bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape2 = np.array([crop_size, crop_size])
        if K is not None:
            K_crop, _ = _get_K_crop_resize(bbox_new, K_crop, resize_shape2)
        image_crop, _ = _get_image_crop_resize(image_crop, bbox_new, resize_shape2)

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
        try:
            from src.utils.vis_utils import reproj
        except ModuleNotFoundError:
            from onnx_demo.src.utils.vis_utils import reproj

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
