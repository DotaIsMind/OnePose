"""
在线视频流 OnePose ONNX 推理流水线（设计见仓库根目录 ``camera_pipeline.md`` 第 5–11 节）。

检测器 ``LocalFeatureObjectDetectorOnnx.crop_img_by_bbox`` 仍通过 ``cv2.imread(query_img_path)``
读图，故本阶段采用**临时 PNG** 与 ROS 节点相同的打通方式；每帧将灰度图写入固定路径再调用
``detect`` / ``previous_pose_detect``。

**配置**：数据路径、ONNX 路径及模型/流水线参数由 YAML 提供（默认 ``pipeline_online.yml``，
与本脚本同目录）。相对路径均相对于该 YAML 文件所在目录解析。

``runtime.vis_dir`` 启用时（如 ``./outputs``，相对 YAML 目录），会在该目录下写入
``detect/*.jpg``（全图检测框）、``match/*.jpg``（裁剪图 2D–3D 匹配与重投影）、
``pose/*.jpg``（全图位姿与 3D 框可视化）。

运行示例::

    cd test_onnx/onnx_demo
    python pipeline_online.py
    python pipeline_online.py --config /path/to/custom.yml
"""

from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import OmegaConf

# 与 pipeline_single 相同：保证可导入 onnx_demo 下的 ``src.utils``
_ONNX_DEMO = Path(__file__).resolve().parent
if str(_ONNX_DEMO) not in sys.path:
    sys.path.insert(0, str(_ONNX_DEMO))

from pipeline_single import (  # noqa: E402
    GATsSPGOnnx,
    LocalFeatureObjectDetectorOnnx,
    SuperPointOnnx,
    _build_features3d_leaves,
    _pad_features3d,
)

_TMP_GRAY = "/tmp/onepose_online_query_gray.png"
_TMP_BGR = "/tmp/onepose_online_query_vis.png"

_logger = logging.getLogger(__name__)


def _pose_mat_summary(pose_3x4: np.ndarray) -> str:
    """Short English summary of a 3x4 pose matrix for logs (translation + rotation trace)."""
    R = pose_3x4[:, :3].astype(np.float64)
    t = pose_3x4[:, 3].astype(np.float64)
    return f"t=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f}) trace(R)={np.trace(R):.4f}"


def _scale_K_to_resolution(
    K: np.ndarray,
    ref_wh: Tuple[int, int],
    new_wh: Tuple[int, int],
) -> np.ndarray:
    """将内参从参考分辨率缩放到新分辨率 (width, height)。"""
    rw, rh = ref_wh
    nw, nh = new_wh
    if (rw, rh) == (nw, nh):
        return K.astype(np.float64)
    sx, sy = nw / rw, nh / rh
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def _bbox_area_ratio_xyxy(bbox: np.ndarray, W: int, H: int) -> float:
    x0, y0, x1, y1 = bbox.astype(float)
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return area / float(W * H) if W * H > 0 else 1.0


def _clamp_bbox_xyxy(bbox: np.ndarray, W: int, H: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox.astype(float)
    x0i = int(np.clip(np.floor(x0), 0, max(0, W - 1)))
    y0i = int(np.clip(np.floor(y0), 0, max(0, H - 1)))
    x1i = int(np.clip(np.ceil(x1), x0i + 4, W))
    y1i = int(np.clip(np.ceil(y1), y0i + 4, H))
    return np.array([x0i, y0i, x1i, y1i], dtype=np.int32)


def _center_fallback_bbox(W: int, H: int, scale: float) -> np.ndarray:
    side = max(int(min(W, H) * scale), 96)
    cx, cy = W // 2, H // 2
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(W, x0 + side)
    y1 = min(H, y0 + side)
    if x1 <= x0 + 8:
        x1 = min(W, x0 + 32)
    if y1 <= y0 + 8:
        y1 = min(H, y0 + 32)
    return np.array([x0, y0, x1, y1], dtype=np.int32)


def _crop_resize_from_gray(
    gray: np.ndarray,
    bbox: np.ndarray,
    K: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与 ``LocalFeatureObjectDetectorOnnx.crop_img_by_bbox`` 等价，但整图来自 ``gray`` ndarray。"""
    from src.utils.data_utils import get_image_crop_resize, get_K_crop_resize

    H, W = gray.shape[:2]
    bbox = _clamp_bbox_xyxy(bbox, W, H)
    x0, y0, x1, y1 = bbox.astype(np.float32)
    resize_shape = np.array([y1 - y0, x1 - x0], dtype=np.float32)
    K_crop, _ = get_K_crop_resize(bbox, K, resize_shape)
    image_crop, _ = get_image_crop_resize(gray, bbox, resize_shape)
    bbox_new = np.array([0.0, 0.0, x1 - x0, y1 - y0], dtype=np.float32)
    resize_shape2 = np.array([crop_size, crop_size], dtype=np.float32)
    K_crop, _ = get_K_crop_resize(bbox_new, K_crop, resize_shape2)
    image_crop, _ = get_image_crop_resize(image_crop, bbox_new, resize_shape2)
    inp_crop = (image_crop.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]
    return bbox, inp_crop, K_crop


def _save_detect_visualization(gray: np.ndarray, bbox: np.ndarray, out_path: str) -> None:
    """全图灰度上绘制检测框并保存。"""
    h, w = gray.shape[:2]
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = bbox.astype(float)
    x0 = int(np.clip(np.floor(x0), 0, w - 1))
    y0 = int(np.clip(np.floor(y0), 0, h - 1))
    x1 = int(np.clip(np.ceil(x1), 0, w - 1))
    y1 = int(np.clip(np.ceil(y1), 0, h - 1))
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        bgr,
        "detect",
        (x0, max(y0 - 6, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, bgr)


def _save_match_visualization(
    crop_gray: np.ndarray,
    kpts2d: np.ndarray,
    matches: np.ndarray,
    mscores: np.ndarray,
    keypoints3d: np.ndarray,
    pose_3x4: np.ndarray,
    K_crop: np.ndarray,
    out_path: str,
    num_inliers: int,
) -> None:
    """裁剪图上：全部 2D 关键点 + GAT 匹配点，连线至 PnP 位姿下 3D 点重投影（按匹配置信度着色）。"""
    from src.utils.vis_utils import reproj

    bgr = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
    for xy in kpts2d.astype(np.float32):
        xi, yi = int(round(float(xy[0]))), int(round(float(xy[1])))
        if 0 <= xi < bgr.shape[1] and 0 <= yi < bgr.shape[0]:
            cv2.circle(bgr, (xi, yi), 1, (70, 70, 70), -1, cv2.LINE_AA)

    valid = matches > -1
    if not np.any(valid):
        cv2.putText(
            bgr,
            "matches=0",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, bgr)
        return

    mk2 = kpts2d[valid]
    mk3 = keypoints3d[matches[valid]]
    mc = mscores[valid].astype(np.float64)
    mn, mx = float(mc.min()), float(mc.max())
    pr = reproj(K_crop, pose_3x4, mk3)

    Hc, Wc = bgr.shape[0], bgr.shape[1]
    for i in range(mk2.shape[0]):
        t = int(np.clip(255.0 * (mc[i] - mn) / (mx - mn + 1e-6), 0, 255))
        jet = cv2.applyColorMap(np.array([[t]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        color = (int(jet[0]), int(jet[1]), int(jet[2]))
        p2x, p2y = float(mk2[i, 0]), float(mk2[i, 1])
        p3x, p3y = float(pr[i, 0]), float(pr[i, 1])
        if not (np.isfinite(p2x) and np.isfinite(p2y)):
            continue
        p2 = (int(round(p2x)), int(round(p2y)))
        if not (np.isfinite(p3x) and np.isfinite(p3y)):
            if 0 <= p2[0] < Wc and 0 <= p2[1] < Hc:
                cv2.circle(bgr, p2, 3, color, -1, cv2.LINE_AA)
            continue
        p3 = (int(round(p3x)), int(round(p3y)))
        cv2.line(bgr, p2, p3, color, 1, cv2.LINE_AA)
        cv2.circle(bgr, p2, 3, color, -1, cv2.LINE_AA)

    cv2.putText(
        bgr,
        f"gat_matches={mk2.shape[0]} pnp_inliers={num_inliers}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, bgr)


@dataclass
class CameraPipelineConfig:
    superpoint_onnx: str
    superglue_onnx: str
    gatsspg_onnx: str
    sfm_model_dir: str
    data_root: str
    intrinsics_path: str
    num_leaf: int = 8
    max_num_kp3d: int = 2500
    crop_size: int = 512
    sp_config: Optional[Dict[str, Any]] = None
    tmp_gray_path: str = _TMP_GRAY
    tmp_bgr_path: str = _TMP_BGR
    min_inliers_for_prev_pose: int = 8
    min_inliers_ok: int = 6
    detector_n_ref_view: int = 15
    detector_bbox_max_area_ratio: Optional[float] = 0.45
    detector_fallback_center_scale: float = 0.55


@dataclass
class OnlineRuntimeConfig:
    """从 YAML ``runtime`` + ``input`` 解析出的运行期选项（非 CameraPipelineConfig）。"""

    video: str
    max_frames: Optional[int] = 30
    vis_dir: Optional[str] = None
    frame_stride: int = 1
    ref_width: Optional[int] = None
    ref_height: Optional[int] = None


@dataclass
class OnlineFrameVis:
    """单帧三类可视化输出路径（由 ``run_camera_loop`` 根据 ``vis_dir`` 生成）。"""

    detect_path: Optional[str] = None
    match_path: Optional[str] = None
    pose_path: Optional[str] = None


@dataclass
class FrameInput:
    """One grayscale frame with intrinsics; optional ``frame_id`` for logging (video index)."""

    gray: np.ndarray
    K: np.ndarray
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None


@dataclass
class PoseEstimationResult:
    """Per-frame pose output: PnP pose, inlier count, crop intrinsics, bbox, timings, quality flag."""

    pose_mat: np.ndarray
    pose_homo: np.ndarray
    inliers: np.ndarray
    num_inliers: int
    K_crop: np.ndarray
    bbox: np.ndarray
    timing_ms: Dict[str, float]
    ok: bool


class CameraPosePipeline:
    """
    Single-frame API mirroring ``OnnxOnePosePipeline.run_sequence`` (detection via temp image paths).

    Logs (via ``logging``): frame index, detection branch and bbox, 2D–3D match counts, PnP / pose summary.
    """

    def __init__(self, config: CameraPipelineConfig) -> None:
        from src.utils.data_utils import get_K
        from src.utils.path_utils import get_3d_box_path

        self.config = config
        self.crop_size = config.crop_size
        self._min_prev = config.min_inliers_for_prev_pose
        self._min_ok = config.min_inliers_ok

        anno_dir = osp.join(config.sfm_model_dir, "outputs_superpoint_superglue", "anno")
        sfm_ws_dir = osp.join(
            config.sfm_model_dir, "outputs_superpoint_superglue", "sfm_ws", "model"
        )
        self._paths = {
            "sfm_ws_dir": sfm_ws_dir,
            "avg_anno_3d_path": osp.join(anno_dir, "anno_3d_average.npz"),
            "clt_anno_3d_path": osp.join(anno_dir, "anno_3d_collect.npz"),
            "idxs_path": osp.join(anno_dir, "idxs.npy"),
        }
        for k, p in self._paths.items():
            if not osp.exists(p):
                raise FileNotFoundError(f"CameraPosePipeline 缺少必要文件 [{k}]: {p}")

        sp_cfg = config.sp_config or {
            "nms_radius": 3,
            "keypoint_threshold": 0.005,
            "max_keypoints": 4096,
            "remove_borders": 4,
        }
        self.extractor = SuperPointOnnx(config.superpoint_onnx, config=sp_cfg)
        self.matcher_3d = GATsSPGOnnx(config.gatsspg_onnx)
        self.sg_onnx_path = config.superglue_onnx
        self.sp_onnx_path = config.superpoint_onnx
        self.num_leaf = config.num_leaf
        self.max_num_kp3d = config.max_num_kp3d

        self.detector = LocalFeatureObjectDetectorOnnx(
            superpoint_onnx_path=self.sp_onnx_path,
            superglue_onnx_path=self.sg_onnx_path,
            sfm_ws_dir=sfm_ws_dir,
            n_ref_view=config.detector_n_ref_view,
            output_results=False,
            sp_config=sp_cfg,
            bbox_max_area_ratio=config.detector_bbox_max_area_ratio,
        )

        self.K_full, _ = get_K(config.intrinsics_path)
        box3d_path = get_3d_box_path(config.data_root)
        if not osp.isfile(box3d_path):
            raise FileNotFoundError(f"3D 框文件不存在: {box3d_path}")
        self.bbox3d = np.loadtxt(box3d_path)
        self.box3d_path = box3d_path

        avg_data = np.load(self._paths["avg_anno_3d_path"])
        clt_data = np.load(self._paths["clt_anno_3d_path"])
        idxs = np.load(self._paths["idxs_path"])

        self.keypoints3d = clt_data["keypoints3d"].astype(np.float32)
        num_3d = self.keypoints3d.shape[0]

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

        self._avg_desc3d_b = avg_desc3d[np.newaxis]
        self._clt_desc_b = clt_desc[np.newaxis]
        self._kpts3d_b = self.keypoints3d[np.newaxis]

        self._frame_count = 0
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_inliers: Any = []
        self._bbox_fallback_log_left = 3

    def reset(self) -> None:
        self._frame_count = 0
        self._prev_pose = None
        self._prev_inliers = []
        self._bbox_fallback_log_left = 3

    def process_frame(
        self,
        frame: FrameInput,
        *,
        save_vis: Optional[OnlineFrameVis] = None,
    ) -> PoseEstimationResult:
        """
        Run detect → SuperPoint crop → GAT 2D–3D match → RANSAC PnP for one frame.

        Emits INFO logs: ``[frame]``, ``[detect]``, ``[match]``, ``[pose]`` (pose = PnP / 6-DoF estimate).
        """
        from src.utils.eval_utils import ransac_PnP
        from src.utils.vis_utils import save_demo_image

        gray = np.ascontiguousarray(frame.gray)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if gray.ndim != 2:
            raise ValueError(f"FrameInput.gray 应为 (H,W) uint8，当前 shape={gray.shape}")

        K = frame.K.astype(np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K 应为 3×3，当前 shape={K.shape}")

        cv2.imwrite(self.config.tmp_gray_path, gray)
        tmp_gray = self.config.tmp_gray_path
        if save_vis is not None and save_vis.pose_path is not None:
            cv2.imwrite(self.config.tmp_bgr_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        inp = (gray[np.newaxis, np.newaxis] / 255.0).astype(np.float32)
        timing_ms: Dict[str, float] = {}

        frame_label = frame.frame_id if frame.frame_id is not None else self._frame_count
        _logger.info("[frame] index=%s pipeline_internal_count=%s", frame_label, self._frame_count)

        t0 = time.perf_counter()
        used_full_detect = self._frame_count == 0 or len(self._prev_inliers) < self._min_prev
        if used_full_detect:
            bbox, inp_crop, K_crop = self.detector.detect(inp, tmp_gray, K, crop_size=self.crop_size)
            detect_branch = "full_match"
        else:
            bbox, inp_crop, K_crop = self.detector.previous_pose_detect(
                tmp_gray,
                K,
                self._prev_pose,
                self.bbox3d,
                crop_size=self.crop_size,
            )
            detect_branch = "previous_pose_reproj"
        timing_ms["detect"] = (time.perf_counter() - t0) * 1000.0

        Hg, Wg = gray.shape[0], gray.shape[1]
        bbox = _clamp_bbox_xyxy(bbox, Wg, Hg)
        thr_area = self.config.detector_bbox_max_area_ratio
        bbox_fallback = False
        if thr_area is not None and thr_area > 0 and _bbox_area_ratio_xyxy(bbox, Wg, Hg) > thr_area:
            fb = _center_fallback_bbox(Wg, Hg, self.config.detector_fallback_center_scale)
            bbox, inp_crop, K_crop = _crop_resize_from_gray(gray, fb, K, self.crop_size)
            bbox_fallback = True
            if self._bbox_fallback_log_left > 0:
                print(
                    f"[pipeline_online] 检测框面积占比 > {thr_area}，改用中心 ROI 裁剪 "
                    f"(scale={self.config.detector_fallback_center_scale})"
                )
                self._bbox_fallback_log_left -= 1

        _logger.info(
            "[detect] success=True branch=%s bbox_xyxy=%s area_ratio=%.4f fallback_center=%s time_ms=%.2f",
            detect_branch,
            np.array2string(bbox, precision=1, separator=","),
            float(_bbox_area_ratio_xyxy(bbox, Wg, Hg)),
            bbox_fallback,
            timing_ms["detect"],
        )

        if save_vis is not None and save_vis.detect_path is not None:
            _save_detect_visualization(gray, bbox, save_vis.detect_path)

        t1 = time.perf_counter()
        pred_det = self.extractor(inp_crop)
        timing_ms["extract"] = (time.perf_counter() - t1) * 1000.0

        kpts2d = pred_det["keypoints"]
        desc2d = pred_det["descriptors"]
        n_kpts_crop = int(kpts2d.shape[0])

        t2 = time.perf_counter()
        inp_data = {
            "keypoints2d": kpts2d[np.newaxis].astype(np.float32),
            "keypoints3d": self._kpts3d_b,
            "descriptors2d_query": desc2d[np.newaxis].astype(np.float32),
            "descriptors3d_db": self._avg_desc3d_b,
            "descriptors2d_db": self._clt_desc_b,
        }
        pred, _ = self.matcher_3d(inp_data)
        timing_ms["match3d"] = (time.perf_counter() - t2) * 1000.0

        matches = pred["matches0"].numpy().flatten().astype(np.int32)
        mscores = pred["matching_scores0"].numpy().flatten()
        valid = matches > -1
        n_match = int(np.sum(valid))
        match_ok = n_match > 0
        mscore_mean = float(np.mean(mscores[valid])) if match_ok else float("nan")
        _logger.info(
            "[match] success=%s num_2d3d=%s mean_mscore=%.4f time_ms=%.2f (crop_keypoints=%s)",
            match_ok,
            n_match,
            mscore_mean,
            timing_ms["match3d"],
            n_kpts_crop,
        )

        mkpts2d = kpts2d[valid]
        mkpts3d = self.keypoints3d[matches[valid]]

        t3 = time.perf_counter()
        pose_pred, pose_pred_homo, inliers = ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)
        timing_ms["pnp"] = (time.perf_counter() - t3) * 1000.0

        self._prev_pose = pose_pred
        self._prev_inliers = inliers
        self._frame_count += 1

        n_inl = len(inliers) if inliers is not None else 0
        ok = n_inl >= self._min_ok
        pose_finite = bool(np.isfinite(pose_pred).all() and np.isfinite(pose_pred_homo).all())
        _logger.info(
            "[pose] estimation ok=%s num_inliers=%s min_ok=%s pose_finite=%s %s time_pnp_ms=%.2f",
            ok,
            n_inl,
            self._min_ok,
            pose_finite,
            _pose_mat_summary(pose_pred),
            timing_ms["pnp"],
        )

        crop_gray = (inp_crop[0, 0] * 255.0).clip(0, 255).astype(np.uint8)
        if save_vis is not None and save_vis.match_path is not None:
            _save_match_visualization(
                crop_gray,
                kpts2d,
                matches,
                mscores,
                self.keypoints3d,
                pose_pred,
                K_crop,
                save_vis.match_path,
                num_inliers=n_inl,
            )

        if save_vis is not None and save_vis.pose_path is not None:
            # Degenerate poses yield NaN in reprojection; guard draw_pose_axes / draw_3d_box.
            save_demo_image(
                pose_pred,
                K,
                image_path=self.config.tmp_bgr_path,
                box3d_path=self.box3d_path,
                draw_box=pose_finite and n_inl > self._min_ok,
                save_path=save_vis.pose_path,
                pose_homo=pose_pred_homo,
                draw_axes=pose_finite and ok,
            )

        return PoseEstimationResult(
            pose_mat=pose_pred,
            pose_homo=pose_pred_homo,
            inliers=np.asarray(inliers).reshape(-1) if len(inliers) else np.array([], dtype=np.int32),
            num_inliers=n_inl,
            K_crop=K_crop,
            bbox=bbox,
            timing_ms=timing_ms,
            ok=ok,
        )


class VideoFileSource:
    """离线视频文件作为帧源（模拟相机）。"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return False, None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return True, gray

    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        self._cap.release()


def run_camera_loop(
    pipeline: CameraPosePipeline,
    source: VideoFileSource,
    K: np.ndarray,
    *,
    max_frames: Optional[int] = None,
    vis_dir: Optional[str] = None,
    frame_stride: int = 1,
) -> List[PoseEstimationResult]:
    """Read frames until EOF or ``max_frames``; pass monotonic ``frame_id`` into ``process_frame`` for logs.

    If ``vis_dir`` is set, writes ``detect/``, ``match/``, ``pose/`` sequences as ``%06d.jpg``.
    """
    results: List[PoseEstimationResult] = []
    video_frame_idx = 0
    vis_root: Optional[Path] = None
    if vis_dir is not None:
        vis_root = Path(vis_dir).resolve()
        for sub in ("detect", "match", "pose"):
            (vis_root / sub).mkdir(parents=True, exist_ok=True)

    while True:
        ok, gray = source.read()
        if not ok or gray is None:
            break
        if frame_stride > 1 and video_frame_idx % frame_stride != 0:
            video_frame_idx += 1
            continue
        video_frame_idx += 1

        save_vis: Optional[OnlineFrameVis] = None
        if vis_root is not None:
            idx = len(results)
            save_vis = OnlineFrameVis(
                detect_path=str(vis_root / "detect" / f"{idx:06d}.jpg"),
                match_path=str(vis_root / "match" / f"{idx:06d}.jpg"),
                pose_path=str(vis_root / "pose" / f"{idx:06d}.jpg"),
            )

        r = pipeline.process_frame(
            FrameInput(gray=gray, K=K, frame_id=len(results)),
            save_vis=save_vis,
        )
        results.append(r)

        if max_frames is not None and len(results) >= max_frames:
            break
    return results


def _resolve_path(base_dir: Path, p: Optional[str]) -> Optional[str]:
    if p is None or p == "":
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp.resolve())
    return str((base_dir / pp).resolve())


def load_online_yaml(config_path: Union[str, Path]) -> Tuple[CameraPipelineConfig, OnlineRuntimeConfig]:
    """从 YAML 构建 ``CameraPipelineConfig`` 与 ``OnlineRuntimeConfig``；相对路径相对 YAML 所在目录。"""
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    raw = OmegaConf.load(str(config_path))
    tree: Dict[str, Any] = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    base = config_path.parent

    def R(p: Optional[str]) -> Optional[str]:
        return _resolve_path(base, p) if p else None

    inp = tree.get("input") or {}
    models = tree.get("models") or {}
    data = tree.get("data") or {}
    runtime = tree.get("runtime") or {}
    pipe = tree.get("pipeline") or {}
    det = tree.get("detector") or {}
    sp = tree.get("superpoint") or {}

    video = R(inp.get("video"))
    if not video:
        raise ValueError("YAML 缺少 input.video")

    onnx_dir_s = models.get("onnx_dir")
    onnx_dir = R(onnx_dir_s) if onnx_dir_s else None

    def onnx_explicit(key: str) -> Optional[str]:
        v = models.get(key)
        return R(v) if v else None

    sp_path = onnx_explicit("superpoint_onnx")
    sg_path = onnx_explicit("superglue_onnx")
    gat_path = onnx_explicit("gatsspg_onnx")

    if onnx_dir:
        od = Path(onnx_dir)
        if not sp_path:
            sp_path = str(od / "superpoint.onnx")
        if not sg_path:
            sg_path = str(od / "superglue.onnx")
        if not gat_path:
            gat_path = str(od / "gatsspg.onnx")
    elif not (sp_path and sg_path and gat_path):
        raise ValueError(
            "请在 models 中设置 onnx_dir，或同时设置 superpoint_onnx / superglue_onnx / gatsspg_onnx"
        )

    dr = R(data.get("data_root"))
    sd = R(data.get("seq_dir"))
    sfm = R(data.get("sfm_model_dir"))
    intr_opt = data.get("intrinsics_file")
    intr_path = R(intr_opt) if intr_opt else (osp.join(sd, "intrinsics.txt") if sd else None)

    if not dr or not sd or not sfm or not intr_path:
        raise ValueError(
            "YAML data 需提供 data_root、seq_dir、sfm_model_dir；intrinsics_file 可省略（默认 seq_dir/intrinsics.txt）"
        )

    sp_cfg: Optional[Dict[str, Any]] = dict(sp) if sp else None

    if "bbox_max_area_ratio" in det:
        _bma = det["bbox_max_area_ratio"]
        bbox_m: Optional[float] = None if _bma is None else float(_bma)
    else:
        bbox_m = 0.45

    cfg = CameraPipelineConfig(
        superpoint_onnx=sp_path,
        superglue_onnx=sg_path,
        gatsspg_onnx=gat_path,
        sfm_model_dir=sfm,
        data_root=dr,
        intrinsics_path=intr_path,
        num_leaf=int(pipe.get("num_leaf", 8)),
        max_num_kp3d=int(pipe.get("max_num_kp3d", 2500)),
        crop_size=int(pipe.get("crop_size", 512)),
        sp_config=sp_cfg,
        tmp_gray_path=str(pipe.get("tmp_gray_path", _TMP_GRAY)),
        tmp_bgr_path=str(pipe.get("tmp_bgr_path", _TMP_BGR)),
        min_inliers_for_prev_pose=int(pipe.get("min_inliers_for_prev_pose", 8)),
        min_inliers_ok=int(pipe.get("min_inliers_ok", 6)),
        detector_n_ref_view=int(det.get("n_ref_view", 15)),
        detector_bbox_max_area_ratio=bbox_m,
        detector_fallback_center_scale=float(det.get("fallback_center_scale", 0.55)),
    )

    max_f = runtime.get("max_frames")
    if max_f is not None:
        max_f = int(max_f)

    vis_raw = runtime.get("vis_dir")
    vis_r = R(vis_raw) if vis_raw else None

    rw, rh = runtime.get("ref_width"), runtime.get("ref_height")

    rt = OnlineRuntimeConfig(
        video=video,
        max_frames=max_f,
        vis_dir=vis_r,
        frame_stride=int(runtime.get("frame_stride", 1)),
        ref_width=int(rw) if rw is not None else None,
        ref_height=int(rh) if rh is not None else None,
    )
    return cfg, rt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OnePose 视频流在线 ONNX 推理：路径与参数由 YAML 配置（默认 pipeline_online.yml）。"
    )
    default_yml = _ONNX_DEMO / "pipeline_online.yml"
    p.add_argument(
        "--config",
        "-c",
        default=str(default_yml),
        help=f"YAML 配置文件路径（默认: {default_yml}）",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    args = _parse_args()
    try:
        cfg, run_cfg = load_online_yaml(args.config)
    except Exception as e:
        print(f"[ERROR] 配置解析失败: {e}")
        sys.exit(1)

    for path, name in [
        (cfg.superpoint_onnx, "superpoint"),
        (cfg.superglue_onnx, "superglue"),
        (cfg.gatsspg_onnx, "gatsspg"),
    ]:
        if not Path(path).is_file():
            print(f"[ERROR] 缺少 ONNX: {name} -> {path}")
            sys.exit(1)

    if not Path(run_cfg.video).is_file():
        print(f"[ERROR] 视频不存在: {run_cfg.video}")
        sys.exit(1)

    if not osp.isfile(cfg.intrinsics_path):
        print(f"[ERROR] 内参文件不存在: {cfg.intrinsics_path}")
        sys.exit(1)

    print(f"[config] {Path(args.config).resolve()}")
    print("[CameraPosePipeline] 加载模型与标注…")
    pipeline = CameraPosePipeline(cfg)

    source = VideoFileSource(run_cfg.video)
    try:
        K = pipeline.K_full.copy()
        w, h = source.width(), source.height()
        if run_cfg.ref_width is not None and run_cfg.ref_height is not None:
            K = _scale_K_to_resolution(K, (run_cfg.ref_width, run_cfg.ref_height), (w, h))
            print(f"[intrinsics] 已按参考 {run_cfg.ref_width}x{run_cfg.ref_height} -> 视频 {w}x{h} 缩放 K")

        print(
            f"[video] {run_cfg.video} 分辨率 {w}x{h}，"
            f"max_frames={run_cfg.max_frames} stride={run_cfg.frame_stride}"
        )
        t_wall0 = time.perf_counter()
        results = run_camera_loop(
            pipeline,
            source,
            K,
            max_frames=run_cfg.max_frames,
            vis_dir=run_cfg.vis_dir,
            frame_stride=run_cfg.frame_stride,
        )
        wall_s = time.perf_counter() - t_wall0
    finally:
        source.release()

    n_ok = sum(1 for r in results if r.ok)
    print(f"\n[summary] 共 {len(results)} 帧, ok={n_ok} ({100.0 * n_ok / max(len(results), 1):.1f}%)")
    print(f"[summary] 墙钟 {wall_s:.2f} s, 平均每帧 {wall_s / max(len(results), 1) * 1000:.1f} ms")

    if results:
        for stage in ["detect", "extract", "match3d", "pnp"]:
            arr = [r.timing_ms[stage] for r in results]
            print(f"  avg {stage:10s}: {np.mean(arr):.1f} ms/frame  (PF-01)")

    if run_cfg.vis_dir:
        print(
            f"[vis] 输出根目录: {run_cfg.vis_dir} "
            f"（子目录 detect/、match/、pose/，与处理帧序号对齐）"
        )


if __name__ == "__main__":
    main()
