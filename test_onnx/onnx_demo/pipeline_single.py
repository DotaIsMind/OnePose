"""
Single-file ONNX OnePose pipeline.

Consolidates ``pipeline.py``, ``onnx_models.py``, and ``object_detector.py`` for
easier review and deployment. External deps remain: ``src.utils.*`` (COLMAP IO,
PnP, vis, intrinsics), OpenCV, NumPy, SciPy, ONNX Runtime, PyTorch (detector
match tensors), natsort, tqdm.

Run (from anywhere):
    python /path/to/onnx_demo/pipeline_single.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import natsort
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# ONNX model helpers & wrappers (from onnx_models.py)
# ─────────────────────────────────────────────────────────────────────────────


def _get_session(onnx_path: str):
    """Create an ONNX Runtime InferenceSession (CPU only)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return sess


def _simple_nms(scores: np.ndarray, nms_radius: int) -> np.ndarray:
    """Non-maximum suppression on a 2-D score map (NumPy)."""
    from scipy.ndimage import maximum_filter

    max_scores = maximum_filter(scores, size=2 * nms_radius + 1, mode="constant", cval=0)
    mask = scores == max_scores
    return scores * mask


def _sample_descriptors(
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    s: int = 8,
) -> np.ndarray:
    """Bilinear interpolation of dense descriptors at keypoint locations."""
    C, H, W = descriptors.shape
    kpts = keypoints.copy().astype(np.float32)
    kpts[:, 0] = (kpts[:, 0] - s / 2 + 0.5) / (W * s - s / 2 - 0.5) * 2 - 1
    kpts[:, 1] = (kpts[:, 1] - s / 2 + 0.5) / (H * s - s / 2 - 0.5) * 2 - 1
    map_x = ((kpts[:, 0] + 1) / 2 * (W - 1)).astype(np.float32)
    map_y = ((kpts[:, 1] + 1) / 2 * (H - 1)).astype(np.float32)
    desc_sampled = np.zeros((C, len(kpts)), dtype=np.float32)
    mx = map_x.reshape(1, -1)
    my = map_y.reshape(1, -1)
    for c in range(C):
        desc_sampled[c] = cv2.remap(
            descriptors[c],
            mx,
            my,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).flatten()
    norms = np.linalg.norm(desc_sampled, axis=0, keepdims=True) + 1e-8
    return desc_sampled / norms


class SuperPointOnnx:
    """Drop-in replacement for the PyTorch SuperPoint model."""

    def __init__(self, onnx_path: str, config: dict | None = None):
        self.sess = _get_session(onnx_path)
        default_cfg = {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
            "remove_borders": 4,
        }
        self.cfg = {**default_cfg, **(config or {})}

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        if image.ndim == 3:
            image = image[np.newaxis]

        scores_dense, desc_dense = self.sess.run(None, {"image": image.astype(np.float32)})
        scores = scores_dense[0]
        desc = desc_dense[0]
        H, W = scores.shape

        scores = _simple_nms(scores, self.cfg["nms_radius"])
        ys, xs = np.where(scores > self.cfg["keypoint_threshold"])
        kpt_scores = scores[ys, xs]

        b = self.cfg["remove_borders"]
        mask = (ys >= b) & (ys < H - b) & (xs >= b) & (xs < W - b)
        ys, xs, kpt_scores = ys[mask], xs[mask], kpt_scores[mask]

        max_kp = self.cfg["max_keypoints"]
        if max_kp >= 0 and len(kpt_scores) > max_kp:
            idx = np.argsort(kpt_scores)[::-1][:max_kp]
            ys, xs, kpt_scores = ys[idx], xs[idx], kpt_scores[idx]

        keypoints = np.stack([xs, ys], axis=1).astype(np.float32)

        if len(keypoints) > 0:
            descriptors = _sample_descriptors(keypoints, desc, s=8)
        else:
            descriptors = np.zeros((256, 0), dtype=np.float32)

        return {
            "keypoints": keypoints,
            "scores": kpt_scores,
            "descriptors": descriptors,
        }


class SuperGlueOnnx:
    """Drop-in replacement for the PyTorch SuperGlue model."""

    def __init__(self, onnx_path: str):
        self.sess = _get_session(onnx_path)

    @staticmethod
    def _normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([[w, h]], dtype=np.float32)
        center = size / 2.0
        scaling = size.max() * 0.7
        return (kpts - center) / scaling

    def __call__(self, data: dict) -> dict:
        kpts0 = (
            data["keypoints0"][0].numpy()
            if hasattr(data["keypoints0"], "numpy")
            else np.array(data["keypoints0"][0])
        )
        kpts1 = (
            data["keypoints1"][0].numpy()
            if hasattr(data["keypoints1"], "numpy")
            else np.array(data["keypoints1"][0])
        )
        desc0 = (
            data["descriptors0"][0].numpy()
            if hasattr(data["descriptors0"], "numpy")
            else np.array(data["descriptors0"][0])
        )
        desc1 = (
            data["descriptors1"][0].numpy()
            if hasattr(data["descriptors1"], "numpy")
            else np.array(data["descriptors1"][0])
        )
        sc0 = data["scores0"][0].numpy() if hasattr(data["scores0"], "numpy") else np.array(data["scores0"][0])
        sc1 = data["scores1"][0].numpy() if hasattr(data["scores1"], "numpy") else np.array(data["scores1"][0])

        h0, w0 = int(data["image0"].shape[2]), int(data["image0"].shape[3])
        h1, w1 = int(data["image1"].shape[2]), int(data["image1"].shape[3])

        kpts0_norm = self._normalize_keypoints(kpts0, h0, w0)[np.newaxis]
        kpts1_norm = self._normalize_keypoints(kpts1, h1, w1)[np.newaxis]

        feeds = {
            "kpts0": kpts0_norm.astype(np.float32),
            "kpts1": kpts1_norm.astype(np.float32),
            "desc0": desc0[np.newaxis].astype(np.float32),
            "desc1": desc1[np.newaxis].astype(np.float32),
            "scores0": sc0[np.newaxis].astype(np.float32),
            "scores1": sc1[np.newaxis].astype(np.float32),
        }

        m0, m1, ms0, ms1 = self.sess.run(None, feeds)
        m0 = m0.astype(np.int32)
        m1 = m1.astype(np.int32)
        return _MatchResult(m0, m1, ms0, ms1)


class _MatchResult:
    def __init__(self, m0, m1, ms0, ms1):
        self._m0 = m0
        self._m1 = m1
        self._ms0 = ms0
        self._ms1 = ms1

    def __getitem__(self, key):
        mapping = {
            "matches0": self._m0,
            "matches1": self._m1,
            "matching_scores0": self._ms0,
            "matching_scores1": self._ms1,
        }
        arr = mapping[key]
        return _NumpyTensor(arr)


class _NumpyTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        return _NumpyTensor(self._arr[idx])

    @property
    def shape(self):
        return self._arr.shape


class GATsSPGOnnx:
    """Drop-in replacement for the PyTorch GATsSuperGlue matcher."""

    def __init__(self, onnx_path: str):
        self.sess = _get_session(onnx_path)

    def __call__(self, data: dict) -> Tuple[dict, None]:
        def _to_np(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.array(x, dtype=np.float32)

        feeds = {}
        for inp in self.sess.get_inputs():
            feeds[inp.name] = _to_np(data[inp.name])

        m0, m1, ms0, ms1 = self.sess.run(None, feeds)
        m0 = m0.astype(np.int32)
        m1 = m1.astype(np.int32)

        pred = {
            "matches0": _NumpyTensor(m0),
            "matches1": _NumpyTensor(m1),
            "matching_scores0": _NumpyTensor(ms0),
            "matching_scores1": _NumpyTensor(ms1),
        }
        return pred, None


# ─────────────────────────────────────────────────────────────────────────────
# Object detector (from object_detector.py)
# ─────────────────────────────────────────────────────────────────────────────


def _pack_extract_data(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return (image[np.newaxis] / 255.0).astype(np.float32)


def _pack_match_data(db_det: dict, q_det: dict, db_size: np.ndarray, q_size: np.ndarray) -> dict:
    import torch

    def _t(arr):
        return torch.from_numpy(arr)[None].float()

    data = {}
    for k, v in db_det.items():
        if k != "size":
            data[k + "0"] = _t(v)
    for k, v in q_det.items():
        if k != "size":
            data[k + "1"] = _t(v)

    data["image0"] = np.empty((1, 1) + tuple(db_size)[::-1])
    data["image1"] = np.empty((1, 1) + tuple(q_size)[::-1])
    return data


class LocalFeatureObjectDetectorOnnx:
    """Object detector based on 2D local feature matching (ONNX version)."""

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
        self.matcher = SuperGlueOnnx(superglue_onnx_path)
        self.output_results = output_results
        self.detect_save_dir = detect_save_dir
        self.K_crop_save_dir = K_crop_save_dir
        self.db_dict = self._extract_ref_view_features(sfm_ws_dir, n_ref_view)

    def _extract_ref_view_features(self, sfm_ws_dir: str, n_ref_views: int) -> dict:
        from src.utils.colmap.read_write_model import read_model

        assert osp.exists(sfm_ws_dir), f"SfM workspace not found: {sfm_ws_dir}"
        cameras, images, points3D = read_model(sfm_ws_dir)

        sample_gap = max(len(images) // n_ref_views, 1)
        db_dict = {}
        for idx in range(1, len(images), sample_gap):
            db_img_path = images[idx].name
            db_img = _pack_extract_data(db_img_path)
            db_inp = db_img[np.newaxis]
            det = self.extractor(db_inp)
            det["size"] = np.array(db_img.shape[-2:])
            db_dict[idx] = det
        return db_dict

    def _match_worker(self, query: dict) -> dict:
        results = {}
        for idx, db in self.db_dict.items():
            match_data = _pack_match_data(db, query, db["size"], query["size"])
            match_pred = self.matcher(match_data)

            matches = match_pred["matches0"][0].numpy()
            confs = match_pred["matching_scores0"][0].numpy()
            valid = matches > -1

            mkpts0 = db["keypoints"][valid]
            mkpts1 = query["keypoints"][matches[valid]]

            if mkpts0.shape[0] < 6:
                results[idx] = {
                    "inliers": np.empty((0,)),
                    "bbox": np.array([0, 0, query["size"][0], query["size"][1]]),
                }
                continue

            affine, inliers = cv2.estimateAffinePartial2D(mkpts0, mkpts1, ransacReprojThreshold=6)
            if affine is None:
                results[idx] = {
                    "inliers": np.empty((0,)),
                    "bbox": np.array([0, 0, query["size"][0], query["size"][1]]),
                }
                continue

            db_shape = db["size"]
            four_corner = np.array(
                [
                    [0, 0, 1],
                    [db_shape[1], 0, 1],
                    [0, db_shape[0], 1],
                    [db_shape[1], db_shape[0], 1],
                ]
            ).T
            bbox = (affine @ four_corner).T.astype(np.int32)
            lt = np.min(bbox, axis=0)
            rb = np.max(bbox, axis=0)
            results[idx] = {
                "inliers": inliers,
                "bbox": np.array([lt[0], lt[1], rb[0], rb[1]]),
            }
        return results

    def _detect_by_matching(self, query: dict) -> np.ndarray:
        results = self._match_worker(query)
        best_idx = max(results, key=lambda k: results[k]["inliers"].shape[0])
        return results[best_idx]["bbox"]

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

    def detect(
        self,
        query_img: np.ndarray,
        query_img_path: str,
        K: np.ndarray,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_inp = query_img if query_img.ndim == 4 else query_img[np.newaxis]
        q_det = self.extractor(q_inp)
        q_det["size"] = np.array(query_img.shape[-2:])

        bbox = self._detect_by_matching(q_det)
        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self._save_detection(image_crop, query_img_path)

        image_crop_f = image_crop.astype(np.float32) / 255.0
        image_crop_t = image_crop_f[np.newaxis, np.newaxis]
        return bbox, image_crop_t, K_crop

    def previous_pose_detect(
        self,
        query_img_path: str,
        K: np.ndarray,
        pre_pose: np.ndarray,
        bbox3D_corner: np.ndarray,
        crop_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from src.utils.vis_utils import reproj

        proj_2d = reproj(K, pre_pose, bbox3D_corner)
        x0, y0 = np.min(proj_2d, axis=0)
        x1, y1 = np.max(proj_2d, axis=0)
        bbox = np.array([x0, y0, x1, y1]).astype(np.int32)

        image_crop, K_crop = self.crop_img_by_bbox(query_img_path, bbox, K, crop_size=crop_size)
        self._save_detection(image_crop, query_img_path)

        image_crop_f = image_crop.astype(np.float32) / 255.0
        image_crop_t = image_crop_f[np.newaxis, np.newaxis]
        return bbox, image_crop_t, K_crop


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline data helpers & OnnxOnePosePipeline (from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────


def _get_paths(data_root: str, data_dir: str, sfm_model_dir: str) -> Tuple[List[str], dict]:
    anno_dir = osp.join(sfm_model_dir, "outputs_superpoint_superglue", "anno")
    sfm_ws_dir = osp.join(sfm_model_dir, "outputs_superpoint_superglue", "sfm_ws", "model")

    img_lists = natsort.natsorted(glob.glob(osp.join(data_dir, "color_full", "*.png")))

    vis_detector_dir = osp.join(data_dir, "detector_vis")
    vis_box_dir = osp.join(data_dir, "pred_vis_onnx")
    os.makedirs(vis_detector_dir, exist_ok=True)
    if osp.exists(vis_box_dir):
        import shutil

        shutil.rmtree(vis_box_dir)
    os.makedirs(vis_box_dir, exist_ok=True)

    paths = {
        "data_root": data_root,
        "data_dir": data_dir,
        "sfm_model_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "avg_anno_3d_path": osp.join(anno_dir, "anno_3d_average.npz"),
        "clt_anno_3d_path": osp.join(anno_dir, "anno_3d_collect.npz"),
        "idxs_path": osp.join(anno_dir, "idxs.npy"),
        "intrin_full_path": osp.join(data_dir, "intrinsics.txt"),
        "vis_box_dir": vis_box_dir,
        "vis_detector_dir": vis_detector_dir,
        "demo_video_path": osp.join(data_dir, "demo_video_onnx.mp4"),
    }
    return img_lists, paths


def _pad_features3d(
    descriptors: np.ndarray, scores: np.ndarray, n_target: int
) -> Tuple[np.ndarray, np.ndarray]:
    dim = descriptors.shape[0]
    n = descriptors.shape[1]
    n_pad = n_target - n
    if n_pad < 0:
        return descriptors[:, :n_target], scores[:n_target]
    if n_pad > 0:
        descriptors = np.concatenate([descriptors, np.ones((dim, n_pad), dtype=np.float32)], axis=1)
        scores = np.concatenate([scores, np.zeros((n_pad, 1), dtype=np.float32)], axis=0)
    return descriptors, scores


def _build_features3d_leaves(
    descriptors: np.ndarray,
    scores: np.ndarray,
    idxs: np.ndarray,
    n_target: int,
    num_leaf: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = descriptors.shape[0]
    orig_num = idxs.shape[0]
    n_pad = n_target - orig_num

    desc_db = np.concatenate([descriptors, np.ones((dim, 1), dtype=np.float32)], axis=1)
    sc_db = np.concatenate([scores, np.zeros((1, 1), dtype=np.float32)], axis=0)
    dustbin = desc_db.shape[1] - 1

    upper = np.cumsum(idxs)
    lower = np.insert(upper[:-1], 0, 0)
    aff_idxs = []
    for s, e in zip(lower, upper):
        if num_leaf > e - s:
            idx_list = list(range(s, e)) + [dustbin] * (num_leaf - (e - s))
            aff_idxs.append(np.random.permutation(idx_list))
        else:
            aff_idxs.append(np.random.permutation(np.arange(s, e))[:num_leaf])
    aff_idxs = np.concatenate(aff_idxs)

    desc_out = desc_db[:, aff_idxs]
    sc_out = sc_db[aff_idxs]

    if n_pad < 0:
        desc_out = desc_out[:, : num_leaf * n_target]
        sc_out = sc_out[: num_leaf * n_target]
    elif n_pad > 0:
        desc_out = np.concatenate(
            [desc_out, np.ones((dim, n_pad * num_leaf), dtype=np.float32)], axis=1
        )
        sc_out = np.concatenate(
            [sc_out, np.zeros((n_pad * num_leaf, 1), dtype=np.float32)], axis=0
        )
    return desc_out, sc_out


class OnnxOnePosePipeline:
    """
    Full OnePose inference pipeline using ONNX Runtime.

    Parameters
    ----------
    superpoint_onnx : path to superpoint.onnx
    superglue_onnx  : path to superglue.onnx
    gatsspg_onnx    : path to gatsspg.onnx
    num_leaf        : number of 2-D leaves per 3-D point (default 8)
    max_num_kp3d    : max number of 3-D keypoints (default 2500)
    """

    def __init__(
        self,
        superpoint_onnx: str,
        superglue_onnx: str,
        gatsspg_onnx: str,
        num_leaf: int = 8,
        max_num_kp3d: int = 2500,
    ):
        sp_cfg = {
            "nms_radius": 3,
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
            "remove_borders": 4,
        }
        self.extractor = SuperPointOnnx(superpoint_onnx, config=sp_cfg)
        self.matcher_3d = GATsSPGOnnx(gatsspg_onnx)
        self.sg_onnx_path = superglue_onnx
        self.sp_onnx_path = superpoint_onnx
        self.num_leaf = num_leaf
        self.max_num_kp3d = max_num_kp3d

    def run_sequence(
        self,
        data_root: str,
        seq_dir: str,
        sfm_model_dir: str,
    ) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[str, List[float]]]:
        from src.utils.data_utils import get_K
        from src.utils.eval_utils import ransac_PnP
        from src.utils.path_utils import get_3d_box_path
        from src.utils.vis_utils import make_video, save_demo_image

        img_lists, paths = _get_paths(data_root, seq_dir, sfm_model_dir)

        im_ids = [int(osp.basename(p).replace(".png", "")) for p in img_lists]
        im_ids.sort()
        img_lists = [osp.join(osp.dirname(img_lists[0]), f"{i}.png") for i in im_ids]

        K, _ = get_K(paths["intrin_full_path"])
        box3d_path = get_3d_box_path(data_root)
        bbox3d = np.loadtxt(box3d_path)

        detector = LocalFeatureObjectDetectorOnnx(
            superpoint_onnx_path=self.sp_onnx_path,
            superglue_onnx_path=self.sg_onnx_path,
            sfm_ws_dir=paths["sfm_ws_dir"],
            output_results=False,
            detect_save_dir=paths["vis_detector_dir"],
        )

        avg_data = np.load(paths["avg_anno_3d_path"])
        clt_data = np.load(paths["clt_anno_3d_path"])
        idxs = np.load(paths["idxs_path"])

        keypoints3d = clt_data["keypoints3d"].astype(np.float32)
        num_3d = keypoints3d.shape[0]

        avg_desc3d, _ = _pad_features3d(
            avg_data["descriptors3d"].astype(np.float32),
            avg_data["scores3d"].astype(np.float32),
            num_3d,
        )
        clt_desc, _ = _build_features3d_leaves(
            clt_data["descriptors3d"].astype(np.float32),
            clt_data["scores3d"].astype(np.float32),
            idxs,
            num_3d,
            self.num_leaf,
        )

        avg_desc3d_b = avg_desc3d[np.newaxis]
        clt_desc_b = clt_desc[np.newaxis]
        kpts3d_b = keypoints3d[np.newaxis]

        pred_poses: Dict[int, Tuple] = {}
        timing: Dict[str, List[float]] = {"detect": [], "extract": [], "match3d": [], "pnp": []}

        for frame_id, img_path in enumerate(tqdm(img_lists, desc="ONNX inference")):
            t0 = time.perf_counter()
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            inp = (img_gray[np.newaxis, np.newaxis] / 255.0).astype(np.float32)

            if frame_id == 0:
                bbox, inp_crop, K_crop = detector.detect(inp, img_path, K)
            else:
                prev_pose, prev_inliers = pred_poses[frame_id - 1]
                if len(prev_inliers) < 8:
                    bbox, inp_crop, K_crop = detector.detect(inp, img_path, K)
                else:
                    bbox, inp_crop, K_crop = detector.previous_pose_detect(
                        img_path, K, prev_pose, bbox3d
                    )
            timing["detect"].append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            pred_det = self.extractor(inp_crop)
            timing["extract"].append(time.perf_counter() - t1)

            kpts2d = pred_det["keypoints"]
            desc2d = pred_det["descriptors"]

            t2 = time.perf_counter()
            inp_data = {
                "keypoints2d": kpts2d[np.newaxis].astype(np.float32),
                "keypoints3d": kpts3d_b,
                "descriptors2d_query": desc2d[np.newaxis].astype(np.float32),
                "descriptors3d_db": avg_desc3d_b,
                "descriptors2d_db": clt_desc_b,
            }
            pred, _ = self.matcher_3d(inp_data)
            timing["match3d"].append(time.perf_counter() - t2)

            matches = pred["matches0"].numpy().flatten().astype(np.int32)
            mscores = pred["matching_scores0"].numpy().flatten()
            valid = matches > -1
            mkpts2d = kpts2d[valid]
            mkpts3d = keypoints3d[matches[valid]]
            mconf = mscores[valid]

            t3 = time.perf_counter()
            pose_pred, pose_pred_homo, inliers = ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
            timing["pnp"].append(time.perf_counter() - t3)

            pred_poses[frame_id] = (pose_pred, inliers)

            save_demo_image(
                pose_pred_homo,
                K,
                image_path=img_path,
                box3d_path=box3d_path,
                draw_box=len(inliers) > 6,
                save_path=osp.join(paths["vis_box_dir"], f"{frame_id}.jpg"),
                pose_homo=pose_pred_homo,
                draw_axes=True,
            )

        make_video(paths["vis_box_dir"], paths["demo_video_path"])
        print(f"\n[ONNX] Demo video saved to: {paths['demo_video_path']}")

        for k, v in timing.items():
            print(f"  avg {k:10s}: {np.mean(v)*1000:.1f} ms/frame")

        return pred_poses, timing


def _ensure_import_path():
    """Ensure ``onnx_demo`` directory (this file's folder) is on ``sys.path`` for ``src``."""
    here = Path(__file__).resolve().parent
    s = str(here)
    if s not in sys.path:
        sys.path.insert(0, s)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OnePose ONNX single-file pipeline (demo).")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Only process the first N frames (debug / quick validation).",
    )
    args = parser.parse_args()

    _ensure_import_path()

    _demo_root = Path("/raid/tengf/6d-pose-resource/OnePose/data/demo/test_coffee")
    data_root = str(_demo_root)
    seq_dir = str(_demo_root / "test_coffee-test")
    sfm_model_dir = str(_demo_root / "sfm_model")

    onnx_dir = Path(__file__).resolve().parent / "models"
    sp = str(onnx_dir / "superpoint.onnx")
    sg = str(onnx_dir / "superglue.onnx")
    gat = str(onnx_dir / "gatsspg.onnx")

    for p, label in [(sp, "superpoint"), (sg, "superglue"), (gat, "gatsspg")]:
        if not Path(p).exists():
            print(f"[ERROR] Missing ONNX model ({label}): {p}")
            sys.exit(1)

    pipeline = OnnxOnePosePipeline(
        superpoint_onnx=sp,
        superglue_onnx=sg,
        gatsspg_onnx=gat,
        num_leaf=8,
        max_num_kp3d=2500,
    )

    if args.max_frames is not None:
        _mod = sys.modules[__name__]
        _orig_get_paths = _mod._get_paths
        mf = args.max_frames

        def _limited_get_paths(dr: str, dd: str, sfm: str):
            lists, paths = _orig_get_paths(dr, dd, sfm)
            return lists[:mf], paths

        _mod._get_paths = _limited_get_paths
        try:
            pred_poses, timing = pipeline.run_sequence(data_root, seq_dir, sfm_model_dir)
        finally:
            _mod._get_paths = _orig_get_paths
    else:
        pred_poses, timing = pipeline.run_sequence(data_root, seq_dir, sfm_model_dir)

    print(f"Processed {len(pred_poses)} frames.")


if __name__ == "__main__":
    main()
