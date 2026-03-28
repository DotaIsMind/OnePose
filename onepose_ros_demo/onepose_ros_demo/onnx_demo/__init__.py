"""
onnx_demo - OnePose ONNX inference package

This package provides ONNX-based inference for the OnePose pipeline,
replacing PyTorch models with ONNX Runtime for CPU deployment.

Models:
    - SuperPoint: keypoint detection and description
    - SuperGlue: 2D-2D feature matching (for object detection)
    - GATsSPG: 2D-3D feature matching (for pose estimation)

Usage:
    from onnx_demo.pipeline import OnnxOnePosePipeline
    pipeline = OnnxOnePosePipeline(config)
    pipeline.run()
"""

__version__ = "1.0.0"
__author__ = "OnePose ONNX Demo"
