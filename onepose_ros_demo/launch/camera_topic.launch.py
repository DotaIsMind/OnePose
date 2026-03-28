"""
launch/camera_topic.launch.py
==============================
Launch the pose estimation node in **camera_topic** mode.

The node subscribes to a live camera image topic and publishes results
on /pose_estimation_result.

Usage:
    ros2 launch onepose_ros_demo camera_topic.launch.py
    ros2 launch onepose_ros_demo camera_topic.launch.py \
        image_topic:=/my_camera/image_raw \
        camera_info_topic:=/my_camera/camera_info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path

# ── package root (one level up from launch/) – self-contained ─────────────────
_PKG_DIR      = Path(__name__).resolve().parent   # …/onepose_ros_demo/

_ONNX_DIR  =  _PKG_DIR / "onepose_ros_demo" / "onnx_demo" / "models"
_DATA_ROOT = _PKG_DIR / "data" / "test_coffee"
_SEQ_DIR   = _PKG_DIR / "data" / "test_coffee" / "test_coffee-test"
_SFM_DIR   = _PKG_DIR / "data" / "test_coffee" / "sfm_model"


def generate_launch_description():
    return LaunchDescription([
        # ── launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="ROS2 image topic to subscribe to",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera_info",
            description="ROS2 CameraInfo topic for intrinsics",
        ),
        DeclareLaunchArgument(
            "data_root",
            default_value=str(_DATA_ROOT),
            description="Path to the test_coffee root directory (for 3D annotations)",
        ),
        DeclareLaunchArgument(
            "seq_dir",
            default_value=str(_SEQ_DIR),
            description="Path to the test sequence directory (for intrinsics.txt)",
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

        # ── node ──────────────────────────────────────────────────────────────
        Node(
            package="onepose_ros_demo",
            executable="pose_estimation_node.py",
            name="pose_estimation_node",
            output="screen",
            parameters=[{
                "input_mode":          "camera_topic",
                "data_root":           LaunchConfiguration("data_root"),
                "seq_dir":             LaunchConfiguration("seq_dir"),
                "sfm_model_dir":       LaunchConfiguration("sfm_model_dir"),
                "superpoint_onnx":     LaunchConfiguration("superpoint_onnx"),
                "superglue_onnx":      LaunchConfiguration("superglue_onnx"),
                "gatsspg_onnx":        LaunchConfiguration("gatsspg_onnx"),
                "image_topic":         LaunchConfiguration("image_topic"),
                "camera_info_topic":   LaunchConfiguration("camera_info_topic"),
            }],
        ),
    ])
