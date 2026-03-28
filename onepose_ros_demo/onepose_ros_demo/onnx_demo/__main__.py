"""
onnx_demo main entry point.

Steps:
  1. Export PyTorch models to ONNX  (if not already done)
  2. Run ONNX inference on the demo sequence
  3. Run benchmark comparison (PyTorch vs ONNX)

Usage:
    python -m onnx_demo                        # full run
    python -m onnx_demo --skip_export         # skip model export
    python -m onnx_demo --max_frames 20       # quick test with 20 frames
    python -m onnx_demo --onnx_only           # ONNX inference only (no PyTorch)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__name__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_MODEL_DIR = Path(__name__).parent / "models"


def check_models_exist() -> bool:
    required = [
        ONNX_MODEL_DIR / "superpoint.onnx",
        ONNX_MODEL_DIR / "superglue.onnx",
        ONNX_MODEL_DIR / "gatsspg.onnx",
    ]
    return all(p.exists() for p in required)


def main():
    parser = argparse.ArgumentParser(
        description="OnePose ONNX Demo – export, infer, and benchmark"
    )
    parser.add_argument(
        '--skip_export', action='store_true',
        help="Skip ONNX model export (use existing .onnx files)"
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help="Limit number of frames for quick testing"
    )
    parser.add_argument(
        '--onnx_only', action='store_true',
        help="Run ONNX inference only (skip PyTorch benchmark)"
    )
    parser.add_argument(
        '--export_only', action='store_true',
        help="Only export models to ONNX, then exit"
    )
    args = parser.parse_args()

    # ── Step 1: Export models ─────────────────────────────────────────────────
    if not args.skip_export:
        if check_models_exist():
            print("[onnx_demo] ONNX models already exist – skipping export.")
            print("            Use --skip_export=False to force re-export.")
        else:
            print("[onnx_demo] Exporting models to ONNX …")
            from onnx_demo.export_models import main as export_main
            export_main()
    else:
        if not check_models_exist():
            print("[ERROR] ONNX models not found. Run without --skip_export first.")
            sys.exit(1)

    if args.export_only:
        print("[onnx_demo] Export complete. Exiting.")
        return

    # ── Step 2: Run benchmark or ONNX-only inference ─────────────────────────
    if args.onnx_only:
        print("\n[onnx_demo] Running ONNX inference only …")
        from onnx_demo.benchmark import (
            run_onnx_inference, DATA_ROOT, SEQ_DIR, SFM_DIR
        )
        onnx_poses, onnx_timing = run_onnx_inference(args.max_frames)

        import numpy as np
        print("\n── ONNX Timing Summary ──────────────────────────────────────")
        for stage, times in onnx_timing.items():
            print(f"  {stage:<12}: {np.mean(times)*1000:.1f} ms/frame")
        print("─" * 50)
        total = sum(np.mean(v) for v in onnx_timing.values())
        print(f"  {'TOTAL':<12}: {total*1000:.1f} ms/frame  "
              f"({1.0/total:.1f} FPS)")
    else:
        print("\n[onnx_demo] Running full benchmark (PyTorch + ONNX) …")
        from onnx_demo.benchmark import main as bench_main
        bench_main(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
