"""
launch/local_file.launch.py
============================
Launch the pose estimation node in **local_file** mode.

The node reads PNG images from
  <data_root>/test_coffee-test/color_full/
and publishes results on /pose_estimation_result.

Usage:
    ros2 launch onepose_ros_demo local_file.launch.py
    ros2 launch onepose_ros_demo local_file.launch.py publish_rate_hz:=5.0
    ros2 launch onepose_ros_demo local_file.launch.py loop_sequence:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path

# ── package root (one level up from launch/) – self-contained ─────────────────
_PKG_DIR      = Path(__name__).resolve().parent   # …/onepose_ros_demo/

_ONNX_DIR  =  _PKG_DIR / "onepose_ros_demo" / "onnx_demo" / "models"
_DATA_ROOT = _PKG_DIR / "data" / "demo" / "test_coffee"
_SEQ_DIR   = _PKG_DIR / "data" / "demo" / "test_coffee" / "test_coffee-test"
_SFM_DIR   = _PKG_DIR / "data" / "demo" / "test_coffee" / "sfm_model"


def generate_launch_description():
    return LaunchDescription([
        # ── launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "data_root",
            default_value=str(_DATA_ROOT),
            description="Path to the test_coffee root directory",
        ),
        DeclareLaunchArgument(
            "seq_dir",
            default_value=str(_SEQ_DIR),
            description="Path to the test sequence directory (contains color_full/)",
        ),
        DeclareLaunchArgument(
            "sfm_model_dir",
            default_value=str(_SFM_DIR),
            description="Path to the sfm_model directory",
        ),
        DeclareLaunchArgument(
            "superpoint_onnx",
            default_value=str(_ONNX_DIR / "superpoint.onnx"),
            description="Path to superpoint.onnx",
        ),
        DeclareLaunchArgument(
            "superglue_onnx",
            default_value=str(_ONNX_DIR / "superglue.onnx"),
            description="Path to superglue.onnx",
        ),
        DeclareLaunchArgument(
            "gatsspg_onnx",
            default_value=str(_ONNX_DIR / "gatsspg.onnx"),
            description="Path to gatsspg.onnx",
        ),
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="2.0",
            description="Frame processing rate (Hz) for local_file mode",
        ),
        DeclareLaunchArgument(
            "loop_sequence",
            default_value="false",
            description="Loop the image sequence indefinitely",
        ),

        # ── node ──────────────────────────────────────────────────────────────
        Node(
            package="onepose_ros_demo",
            executable="pose_estimation_node.py",
            name="pose_estimation_node",
            output="screen",
            parameters=[{
                "input_mode":       "local_file",
                "data_root":        LaunchConfiguration("data_root"),
                "seq_dir":          LaunchConfiguration("seq_dir"),
                "sfm_model_dir":    LaunchConfiguration("sfm_model_dir"),
                "superpoint_onnx":  LaunchConfiguration("superpoint_onnx"),
                "superglue_onnx":   LaunchConfiguration("superglue_onnx"),
                "gatsspg_onnx":     LaunchConfiguration("gatsspg_onnx"),
                "publish_rate_hz":  LaunchConfiguration("publish_rate_hz"),
                "loop_sequence":    LaunchConfiguration("loop_sequence"),
            }],
        ),
    ])
