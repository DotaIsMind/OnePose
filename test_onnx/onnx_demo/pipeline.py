"""
ONNX-based OnePose inference pipeline.

This module replicates the logic of inference_demo.py but uses ONNX Runtime
models instead of PyTorch, making it suitable for CPU-only deployment.
"""

from __future__ import annotations

import os
import glob
import time
import cv2
import numpy as np
# import natsort
import os.path as osp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from onnx_demo.onnx_models import SuperPointOnnx, GATsSPGOnnx
from onnx_demo.object_detector import LocalFeatureObjectDetectorOnnx


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (mirrors inference_demo.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_paths(data_root: str, data_dir: str, sfm_model_dir: str) -> Tuple[List[str], dict]:
    anno_dir = osp.join(
        sfm_model_dir, "outputs_superpoint_superglue", "anno"
    )
    sfm_ws_dir = osp.join(
        sfm_model_dir, "outputs_superpoint_superglue", "sfm_ws", "model"
    )

    img_lists = sorted(
        glob.glob(osp.join(data_dir, "color_full", "*.png"))
    )

    vis_detector_dir = osp.join(data_dir, "detector_vis")
    vis_box_dir      = osp.join(data_dir, "pred_vis_onnx")
    os.makedirs(vis_detector_dir, exist_ok=True)
    if osp.exists(vis_box_dir):
        import shutil; shutil.rmtree(vis_box_dir)
    os.makedirs(vis_box_dir, exist_ok=True)

    paths = {
        "data_root":          data_root,
        "data_dir":           data_dir,
        "sfm_model_dir":      sfm_model_dir,
        "sfm_ws_dir":         sfm_ws_dir,
        "avg_anno_3d_path":   osp.join(anno_dir, "anno_3d_average.npz"),
        "clt_anno_3d_path":   osp.join(anno_dir, "anno_3d_collect.npz"),
        "idxs_path":          osp.join(anno_dir, "idxs.npy"),
        "intrin_full_path":   osp.join(data_dir, "intrinsics.txt"),
        "vis_box_dir":        vis_box_dir,
        "vis_detector_dir":   vis_detector_dir,
        "demo_video_path":    osp.join(data_dir, "demo_video_onnx.mp4"),
    }
    return img_lists, paths


def _pad_features3d(descriptors: np.ndarray, scores: np.ndarray,
                    n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad / truncate 3-D features to a fixed size."""
    dim = descriptors.shape[0]
    n   = descriptors.shape[1]
    n_pad = n_target - n
    if n_pad < 0:
        return descriptors[:, :n_target], scores[:n_target]
    if n_pad > 0:
        descriptors = np.concatenate(
            [descriptors, np.ones((dim, n_pad), dtype=np.float32)], axis=1
        )
        scores = np.concatenate(
            [scores, np.zeros((n_pad, 1), dtype=np.float32)], axis=0
        )
    return descriptors, scores


def _build_features3d_leaves(descriptors: np.ndarray, scores: np.ndarray,
                              idxs: np.ndarray, n_target: int,
                              num_leaf: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build leaf-level 3-D features (mirrors data_utils.build_features3d_leaves)."""
    dim      = descriptors.shape[0]
    orig_num = idxs.shape[0]
    n_pad    = n_target - orig_num

    desc_db  = np.concatenate(
        [descriptors, np.ones((dim, 1), dtype=np.float32)], axis=1
    )
    sc_db    = np.concatenate(
        [scores, np.zeros((1, 1), dtype=np.float32)], axis=0
    )
    dustbin  = desc_db.shape[1] - 1

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
    sc_out   = sc_db[aff_idxs]

    if n_pad < 0:
        desc_out = desc_out[:, :num_leaf * n_target]
        sc_out   = sc_out[:num_leaf * n_target]
    elif n_pad > 0:
        desc_out = np.concatenate(
            [desc_out, np.ones((dim, n_pad * num_leaf), dtype=np.float32)], axis=1
        )
        sc_out = np.concatenate(
            [sc_out, np.zeros((n_pad * num_leaf, 1), dtype=np.float32)], axis=0
        )
    return desc_out, sc_out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ─────────────────────────────────────────────────────────────────────────────

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
            'nms_radius':         3,
            'keypoint_threshold': 0.005,
            'max_keypoints':      4096,
            'remove_borders':     4,
        }
        self.extractor  = SuperPointOnnx(superpoint_onnx, config=sp_cfg)
        self.matcher_3d = GATsSPGOnnx(gatsspg_onnx)
        self.sg_onnx_path = superglue_onnx
        self.sp_onnx_path = superpoint_onnx
        self.num_leaf     = num_leaf
        self.max_num_kp3d = max_num_kp3d

    # ── inference on one sequence ─────────────────────────────────────────────

    def run_sequence(
        self,
        data_root: str,
        seq_dir: str,
        sfm_model_dir: str,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Run ONNX inference on a single test sequence.

        Returns
        -------
        pred_poses : {frame_id: (pose_3x4, inliers)}
        timing     : dict with per-stage timing info
        """
        from src.utils.data_utils import get_K, pad_features3d_random
        from src.utils.path_utils import get_3d_box_path
        from src.utils.eval_utils import ransac_PnP
        from src.utils.vis_utils import save_demo_image, make_video

        img_lists, paths = _get_paths(data_root, seq_dir, sfm_model_dir)

        # Sort by frame id
        im_ids = [int(osp.basename(p).replace('.png', '')) for p in img_lists]
        im_ids.sort()
        img_lists = [
            osp.join(osp.dirname(img_lists[0]), f'{i}.png') for i in im_ids
        ]

        K, _ = get_K(paths['intrin_full_path'])
        box3d_path = get_3d_box_path(data_root)
        bbox3d = np.loadtxt(box3d_path)

        # Build object detector
        detector = LocalFeatureObjectDetectorOnnx(
            superpoint_onnx_path=self.sp_onnx_path,
            superglue_onnx_path=self.sg_onnx_path,
            sfm_ws_dir=paths['sfm_ws_dir'],
            output_results=False,
            detect_save_dir=paths['vis_detector_dir'],
        )

        # Load 3-D annotations
        avg_data = np.load(paths['avg_anno_3d_path'])
        clt_data = np.load(paths['clt_anno_3d_path'])
        idxs     = np.load(paths['idxs_path'])

        keypoints3d = clt_data['keypoints3d'].astype(np.float32)  # [M, 3]
        num_3d      = keypoints3d.shape[0]

        avg_desc3d, _ = _pad_features3d(
            avg_data['descriptors3d'].astype(np.float32),
            avg_data['scores3d'].astype(np.float32),
            num_3d,
        )
        clt_desc, _ = _build_features3d_leaves(
            clt_data['descriptors3d'].astype(np.float32),
            clt_data['scores3d'].astype(np.float32),
            idxs, num_3d, self.num_leaf,
        )

        # [1, 256, M]  and  [1, 256, M*num_leaf]
        avg_desc3d_b = avg_desc3d[np.newaxis]
        clt_desc_b   = clt_desc[np.newaxis]
        kpts3d_b     = keypoints3d[np.newaxis]   # [1, M, 3]

        pred_poses: Dict[int, Tuple] = {}
        timing: Dict[str, List[float]] = {
            'detect': [], 'extract': [], 'match3d': [], 'pnp': []
        }

        for frame_id, img_path in enumerate(tqdm(img_lists, desc="ONNX inference")):
            # ── 1. Object detection ──────────────────────────────────────────
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
            timing['detect'].append(time.perf_counter() - t0)

            # ── 2. Keypoint extraction on cropped image ──────────────────────
            t1 = time.perf_counter()
            pred_det = self.extractor(inp_crop)   # dict: kpts, scores, descs
            timing['extract'].append(time.perf_counter() - t1)

            kpts2d   = pred_det['keypoints']    # [N, 2]
            desc2d   = pred_det['descriptors']  # [256, N]

            # ── 3. 2D-3D matching (GATsSPG) ──────────────────────────────────
            t2 = time.perf_counter()
            inp_data = {
                'keypoints2d':         kpts2d[np.newaxis].astype(np.float32),
                'keypoints3d':         kpts3d_b,
                'descriptors2d_query': desc2d[np.newaxis].astype(np.float32),
                'descriptors3d_db':    avg_desc3d_b,
                'descriptors2d_db':    clt_desc_b,
            }
            pred, _ = self.matcher_3d(inp_data)
            timing['match3d'].append(time.perf_counter() - t2)

            matches   = pred['matches0'].numpy().flatten().astype(np.int32)
            mscores   = pred['matching_scores0'].numpy().flatten()
            valid     = matches > -1
            mkpts2d   = kpts2d[valid]
            mkpts3d   = keypoints3d[matches[valid]]
            mconf     = mscores[valid]

            # ── 4. PnP pose estimation ────────────────────────────────────────
            t3 = time.perf_counter()
            pose_pred, pose_pred_homo, inliers = ransac_PnP(
                K_crop, mkpts2d, mkpts3d, scale=1000
            )
            timing['pnp'].append(time.perf_counter() - t3)

            pred_poses[frame_id] = (pose_pred, inliers)

            # ── 5. Visualise ──────────────────────────────────────────────────
            save_demo_image(
                pose_pred_homo,
                K,
                image_path=img_path,
                box3d_path=box3d_path,
                draw_box=len(inliers) > 6,
                save_path=osp.join(paths['vis_box_dir'], f'{frame_id}.jpg'),
                pose_homo=pose_pred_homo,
                draw_axes=True,
            )

        make_video(paths['vis_box_dir'], paths['demo_video_path'])
        print(f"\n[ONNX] Demo video saved to: {paths['demo_video_path']}")

        # Summarise timing
        for k, v in timing.items():
            print(f"  avg {k:10s}: {np.mean(v)*1000:.1f} ms/frame")

        return pred_poses, timing
