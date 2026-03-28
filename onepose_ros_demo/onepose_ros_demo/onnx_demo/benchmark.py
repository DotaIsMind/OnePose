"""
Benchmark: PyTorch (original) vs ONNX Runtime

Runs both pipelines on the same test sequence and produces:
  1. Side-by-side pose visualisation images
  2. Rotation / translation error between the two pipelines
  3. Timing comparison table
  4. A summary report saved to onnx_demo/benchmark_report.txt

Usage:
    python -m onnx_demo.benchmark
"""

from __future__ import annotations

import os
import sys
import time
import glob
import cv2
import numpy as np
import natsort
import os.path as osp
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_MODEL_DIR = Path(__file__).parent / "models"

# ── demo data paths ───────────────────────────────────────────────────────────
# DATA_ROOT    = str(PROJECT_ROOT / "data/demo/test_coffee")
# SEQ_DIR      = str(PROJECT_ROOT / "data/demo/test_coffee/test_coffee-test")
# SFM_DIR      = str(PROJECT_ROOT / "data/demo/test_coffee/sfm_model")

DATA_ROOT = Path("/raid/tengf/6d-pose-resource/OnePose/data/demo/test_coffee")
SEQ_DIR = str( DATA_ROOT / "test_coffee-test" )
SFM_DIR = str( DATA_ROOT / "sfm_model")

SP_ONNX  = str(ONNX_MODEL_DIR / "superpoint.onnx")
SG_ONNX  = str(ONNX_MODEL_DIR / "superglue.onnx")
GAT_ONNX = str(ONNX_MODEL_DIR / "gatsspg.onnx")

# SP_PTH   = str(PROJECT_ROOT / "data/models/extractors/SuperPoint/superpoint_v1.pth")
# SG_PTH   = str(PROJECT_ROOT / "data/models/matchers/SuperGlue/superglue_outdoor.pth")
# GAT_CKPT = str(PROJECT_ROOT / "data/models/checkpoints/onepose/GATsSPG.ckpt")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch inference (mirrors inference_demo.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_pytorch_inference(max_frames: int | None = None) -> Tuple[dict, dict]:
    """Run the original PyTorch pipeline and return (pred_poses, timing)."""
    import torch
    from torch.utils.data import DataLoader
    from utils.data_utils import get_K, pad_features3d_random, build_features3d_leaves
    from utils.path_utils import get_3d_box_path
    from utils.eval_utils import ransac_PnP
    from utils.vis_utils import save_demo_image, make_video
    from utils.model_io import load_network
    from src.models.GATsSPG_lightning_model import LitModelGATsSPG
    from src.models.extractors.SuperPoint.superpoint import SuperPoint
    from src.models.matchers.SuperGlue.superglue import SuperGlue
    from src.sfm.extract_features import confs as sp_confs
    from src.sfm.match_features import confs as sg_confs
    from src.datasets.normalized_dataset import NormalizedDataset
    from src.local_feature_2D_detector import LocalFeatureObjectDetector

    print("\n" + "=" * 60)
    print("  Running PyTorch inference …")
    print("=" * 60)

    # Load models
    sp_conf = sp_confs['superpoint']['conf']
    extractor = SuperPoint(sp_conf)
    extractor.eval()
    load_network(extractor, SP_PTH, force=True)

    sg_conf = sg_confs['superglue']['conf']
    matcher_2d = SuperGlue(sg_conf)
    matcher_2d.eval()
    load_network(matcher_2d, SG_PTH, force=True)

    lit = LitModelGATsSPG.load_from_checkpoint(GAT_CKPT, map_location='cpu')
    lit.eval()
    matcher_3d = lit.matcher

    # Paths
    anno_dir   = osp.join(SFM_DIR, "outputs_superpoint_superglue", "anno")
    sfm_ws_dir = osp.join(SFM_DIR, "outputs_superpoint_superglue", "sfm_ws", "model")
    vis_box_dir = osp.join(SEQ_DIR, "pred_vis_pytorch")
    os.makedirs(vis_box_dir, exist_ok=True)

    img_lists = natsort.natsorted(
        glob.glob(osp.join(SEQ_DIR, "color_full", "*.png"))
    )
    im_ids = sorted([int(osp.basename(p).replace('.png', '')) for p in img_lists])
    img_lists = [osp.join(osp.dirname(img_lists[0]), f'{i}.png') for i in im_ids]
    if max_frames:
        img_lists = img_lists[:max_frames]

    K, _ = get_K(osp.join(SEQ_DIR, "intrinsics.txt"))
    box3d_path = osp.join(DATA_ROOT, "box3d_corners.txt")
    bbox3d = np.loadtxt(box3d_path)

    detector = LocalFeatureObjectDetector(
        extractor, matcher_2d,
        sfm_ws_dir=sfm_ws_dir,
        output_results=False,
        detect_save_dir=osp.join(SEQ_DIR, "detector_vis"),
    )

    dataset = NormalizedDataset(img_lists, sp_confs['superpoint']['preprocessing'])
    loader  = DataLoader(dataset, num_workers=0)

    avg_data = np.load(osp.join(anno_dir, "anno_3d_average.npz"))
    clt_data = np.load(osp.join(anno_dir, "anno_3d_collect.npz"))
    idxs     = np.load(osp.join(anno_dir, "idxs.npy"))

    import torch
    keypoints3d = torch.Tensor(clt_data['keypoints3d'])
    num_3d = keypoints3d.shape[0]
    avg_desc3d, _ = pad_features3d_random(
        avg_data['descriptors3d'], avg_data['scores3d'], num_3d
    )
    clt_desc, _ = build_features3d_leaves(
        clt_data['descriptors3d'], clt_data['scores3d'], idxs, num_3d, 8
    )

    pred_poses: Dict[int, Tuple] = {}
    timing: Dict[str, list] = {'detect': [], 'extract': [], 'match3d': [], 'pnp': []}

    for frame_id, data in enumerate(tqdm(loader, desc="PyTorch inference")):
        with torch.no_grad():
            img_path = data['path'][0]
            inp = data['image']

            t0 = time.perf_counter()
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

            t1 = time.perf_counter()
            pred_det = extractor(inp_crop)
            pred_det = {k: v[0].cpu().numpy() for k, v in pred_det.items()}
            timing['extract'].append(time.perf_counter() - t1)

            kpts2d = pred_det['keypoints']
            desc2d = pred_det['descriptors']

            t2 = time.perf_counter()
            inp_data = {
                'keypoints2d':         torch.Tensor(kpts2d[None]),
                'keypoints3d':         keypoints3d[None],
                'descriptors2d_query': torch.Tensor(desc2d[None]),
                'descriptors3d_db':    avg_desc3d[None],
                'descriptors2d_db':    clt_desc[None],
                'image_size':          data['size'],
            }
            pred, _ = matcher_3d(inp_data)
            timing['match3d'].append(time.perf_counter() - t2)

            matches = pred['matches0'].detach().cpu().numpy()
            mscores = pred['matching_scores0'].detach().cpu().numpy()
            valid   = matches > -1
            mkpts2d = kpts2d[valid]
            mkpts3d = keypoints3d.numpy()[matches[valid]]

            t3 = time.perf_counter()
            pose_pred, pose_pred_homo, inliers = ransac_PnP(
                K_crop, mkpts2d, mkpts3d, scale=1000
            )
            timing['pnp'].append(time.perf_counter() - t3)

            pred_poses[frame_id] = (pose_pred, inliers)

            save_demo_image(
                pose_pred_homo, K,
                image_path=img_path,
                box3d_path=box3d_path,
                draw_box=len(inliers) > 6,
                save_path=osp.join(vis_box_dir, f'{frame_id}.jpg'),
                pose_homo=pose_pred_homo,
                draw_axes=True,
            )

    make_video(vis_box_dir, osp.join(SEQ_DIR, "demo_video_pytorch.mp4"))
    print(f"[PyTorch] Demo video saved to: {osp.join(SEQ_DIR, 'demo_video_pytorch.mp4')}")
    return pred_poses, timing


# ─────────────────────────────────────────────────────────────────────────────
# ONNX inference
# ─────────────────────────────────────────────────────────────────────────────

def run_onnx_inference(max_frames: int | None = None) -> Tuple[dict, dict]:
    """Run the ONNX pipeline and return (pred_poses, timing)."""
    from onnx_demo.pipeline import OnnxOnePosePipeline

    print("\n" + "=" * 60)
    print("  Running ONNX inference …")
    print("=" * 60)

    pipeline = OnnxOnePosePipeline(
        superpoint_onnx=SP_ONNX,
        superglue_onnx=SG_ONNX,
        gatsspg_onnx=GAT_ONNX,
        num_leaf=8,
        max_num_kp3d=2500,
    )

    # Temporarily limit frames if requested
    if max_frames is not None:
        import glob as _glob, natsort as _ns
        img_lists = _ns.natsorted(
            _glob.glob(osp.join(SEQ_DIR, "color_full", "*.png"))
        )[:max_frames]
        # Monkey-patch _get_paths to return limited list
        import onnx_demo.pipeline as _pl
        _orig = _pl._get_paths
        def _patched(data_root, data_dir, sfm_model_dir):
            lists, paths = _orig(data_root, data_dir, sfm_model_dir)
            return lists[:max_frames], paths
        _pl._get_paths = _patched
        pred_poses, timing = pipeline.run_sequence(DATA_ROOT, SEQ_DIR, SFM_DIR)
        _pl._get_paths = _orig
    else:
        pred_poses, timing = pipeline.run_sequence(DATA_ROOT, SEQ_DIR, SFM_DIR)

    return pred_poses, timing


# ─────────────────────────────────────────────────────────────────────────────
# Comparison metrics
# ─────────────────────────────────────────────────────────────────────────────

def _pose_error(pose_a: np.ndarray, pose_b: np.ndarray) -> Tuple[float, float]:
    """
    Compute rotation (degrees) and translation (cm) error between two poses.
    Both poses are [3, 4] or [4, 4].
    """
    if pose_a.shape[0] == 4:
        pose_a = pose_a[:3]
    if pose_b.shape[0] == 4:
        pose_b = pose_b[:3]

    R_a, t_a = pose_a[:, :3], pose_a[:, 3]
    R_b, t_b = pose_b[:, :3], pose_b[:, 3]

    t_err = np.linalg.norm(t_a - t_b) * 100   # metres → cm

    R_diff = R_a @ R_b.T
    trace  = np.clip(np.trace(R_diff), -1, 3)
    R_err  = np.degrees(np.arccos((trace - 1.0) / 2.0))

    return R_err, t_err


def compare_poses(pt_poses: dict, onnx_poses: dict) -> dict:
    """Compare per-frame poses from both pipelines."""
    common = sorted(set(pt_poses) & set(onnx_poses))
    R_errs, t_errs = [], []
    for fid in common:
        pt_pose   = pt_poses[fid][0]
        onnx_pose = onnx_poses[fid][0]
        if pt_pose is None or onnx_pose is None:
            continue
        r, t = _pose_error(pt_pose, onnx_pose)
        R_errs.append(r)
        t_errs.append(t)

    return {
        'n_frames':    len(common),
        'R_err_mean':  float(np.mean(R_errs))  if R_errs else float('nan'),
        'R_err_median':float(np.median(R_errs)) if R_errs else float('nan'),
        'R_err_max':   float(np.max(R_errs))    if R_errs else float('nan'),
        't_err_mean':  float(np.mean(t_errs))   if t_errs else float('nan'),
        't_err_median':float(np.median(t_errs)) if t_errs else float('nan'),
        't_err_max':   float(np.max(t_errs))    if t_errs else float('nan'),
        'R_errs':      R_errs,
        't_errs':      t_errs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_video(pt_vis_dir: str, onnx_vis_dir: str,
                          output_path: str):
    """Create a side-by-side comparison video."""
    pt_imgs   = natsort.natsorted(os.listdir(pt_vis_dir))
    onnx_imgs = natsort.natsorted(os.listdir(onnx_vis_dir))
    n = min(len(pt_imgs), len(onnx_imgs))
    if n == 0:
        print("[warn] No images found for comparison video.")
        return

    sample = cv2.imread(osp.join(pt_vis_dir, pt_imgs[0]))
    H, W = sample.shape[:2]
    out_W, out_H = W * 2 + 20, H + 60   # side-by-side + label bar

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 24, (out_W, out_H))

    for i in range(n):
        pt_img   = cv2.imread(osp.join(pt_vis_dir,   pt_imgs[i]))
        onnx_img = cv2.imread(osp.join(onnx_vis_dir, onnx_imgs[i]))
        if pt_img is None or onnx_img is None:
            continue

        canvas = np.ones((out_H, out_W, 3), dtype=np.uint8) * 30
        canvas[60:60+H, :W]       = pt_img
        canvas[60:60+H, W+20:]    = onnx_img

        cv2.putText(canvas, "PyTorch (original)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, "ONNX Runtime",
                    (W + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(canvas, f"Frame {i}",
                    (out_W // 2 - 50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 1)
        writer.write(canvas)

    writer.release()
    print(f"[Benchmark] Comparison video saved to: {output_path}")


def plot_error_curves(metrics: dict, output_path: str):
    """Plot per-frame rotation and translation errors."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        R_errs = metrics['R_errs']
        t_errs = metrics['t_errs']
        frames = list(range(len(R_errs)))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(frames, R_errs, 'b-o', markersize=3, linewidth=1)
        axes[0].axhline(metrics['R_err_mean'], color='r', linestyle='--',
                        label=f"mean={metrics['R_err_mean']:.2f}°")
        axes[0].set_ylabel("Rotation Error (degrees)")
        axes[0].set_title("PyTorch vs ONNX: Per-frame Rotation Error")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, t_errs, 'g-o', markersize=3, linewidth=1)
        axes[1].axhline(metrics['t_err_mean'], color='r', linestyle='--',
                        label=f"mean={metrics['t_err_mean']:.2f} cm")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Translation Error (cm)")
        axes[1].set_title("PyTorch vs ONNX: Per-frame Translation Error")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Benchmark] Error curves saved to: {output_path}")
    except Exception as e:
        print(f"[warn] Could not plot error curves: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(pt_timing: dict, onnx_timing: dict, metrics: dict,
                 report_path: str):
    lines = []
    lines.append("=" * 65)
    lines.append("  OnePose: PyTorch vs ONNX Runtime – Benchmark Report")
    lines.append("=" * 65)

    lines.append("\n── Timing Comparison (ms / frame) ──────────────────────────")
    header = f"{'Stage':<14} {'PyTorch':>12} {'ONNX':>12} {'Speedup':>10}"
    lines.append(header)
    lines.append("-" * 52)
    total_pt, total_onnx = 0.0, 0.0
    for stage in ['detect', 'extract', 'match3d', 'pnp']:
        pt_ms   = np.mean(pt_timing[stage])   * 1000
        onnx_ms = np.mean(onnx_timing[stage]) * 1000
        speedup = pt_ms / onnx_ms if onnx_ms > 0 else float('nan')
        total_pt   += pt_ms
        total_onnx += onnx_ms
        lines.append(f"{stage:<14} {pt_ms:>11.1f}  {onnx_ms:>11.1f}  {speedup:>9.2f}x")
    lines.append("-" * 52)
    total_speedup = total_pt / total_onnx if total_onnx > 0 else float('nan')
    lines.append(f"{'TOTAL':<14} {total_pt:>11.1f}  {total_onnx:>11.1f}  {total_speedup:>9.2f}x")

    lines.append("\n── Pose Error (PyTorch vs ONNX) ─────────────────────────────")
    lines.append(f"  Frames compared : {metrics['n_frames']}")
    lines.append(f"  Rotation  error : mean={metrics['R_err_mean']:.4f}°  "
                 f"median={metrics['R_err_median']:.4f}°  "
                 f"max={metrics['R_err_max']:.4f}°")
    lines.append(f"  Translation err : mean={metrics['t_err_mean']:.4f} cm  "
                 f"median={metrics['t_err_median']:.4f} cm  "
                 f"max={metrics['t_err_max']:.4f} cm")

    lines.append("\n── Interpretation ───────────────────────────────────────────")
    if metrics['R_err_mean'] < 1.0 and metrics['t_err_mean'] < 1.0:
        lines.append("  ✓ ONNX output is numerically equivalent to PyTorch.")
    elif metrics['R_err_mean'] < 5.0 and metrics['t_err_mean'] < 5.0:
        lines.append("  ~ ONNX output is close to PyTorch (minor numerical diff).")
    else:
        lines.append("  ✗ Significant difference detected – check model export.")

    lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report + "\n")
    print(f"\n[Benchmark] Report saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(max_frames: int | None = None):
    """
    Run full benchmark.

    Parameters
    ----------
    max_frames : limit the number of frames processed (None = all frames)
    """
    report_dir = Path(__file__).parent / "benchmark_results"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run both pipelines
    pt_poses,   pt_timing   = run_pytorch_inference(max_frames)
    onnx_poses, onnx_timing = run_onnx_inference(max_frames)

    # 2. Compare poses
    metrics = compare_poses(pt_poses, onnx_poses)

    # 3. Side-by-side comparison video
    pt_vis_dir   = osp.join(SEQ_DIR, "pred_vis_pytorch")
    onnx_vis_dir = osp.join(SEQ_DIR, "pred_vis_onnx")
    make_comparison_video(
        pt_vis_dir, onnx_vis_dir,
        str(report_dir / "comparison_video.mp4"),
    )

    # 4. Error curves
    plot_error_curves(metrics, str(report_dir / "error_curves.png"))

    # 5. Print & save report
    print_report(
        pt_timing, onnx_timing, metrics,
        str(report_dir / "benchmark_report.txt"),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OnePose PyTorch vs ONNX benchmark")
    parser.add_argument('--max_frames', type=int, default=None,
                        help="Limit number of frames (default: all)")
    args = parser.parse_args()
    main(max_frames=args.max_frames)
