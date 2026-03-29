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

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path

_PKG_NAME = "onepose_ros_demo"
# ament index: same workspace resolution as CMake find_package(onepose_ros_demo)
_PREFIX = Path(get_package_prefix(_PKG_NAME))
_SHARE = Path(get_package_share_directory(_PKG_NAME))
# ONNX bundled under lib/ (not share/); models path for superpoint/superglue/gatsspg
_ONNX_LIB_ROOT = _PREFIX / "lib" / _PKG_NAME / "onnx_demo"
_ONNX_DIR = _ONNX_LIB_ROOT / "models" / "onnx"

# Demo object data: installed to share/<pkg>/data/... via CMakeLists install(DIRECTORY data/ ...)
_PKG_DATA = _SHARE / "data"
_SRC_DATA = Path(__file__).resolve().parent.parent / "data"
_DATA_BASE = _PKG_DATA if _PKG_DATA.is_dir() else _SRC_DATA
_MARK = _DATA_BASE / "demo" / "mark_cup"
_DATA_ROOT = str(_MARK)
_SEQ_DIR = str(_MARK / "mark_cup-annotate")
_SFM_DIR = str(_MARK / "sfm_model")
# runtime.vis_dir: default under installed onnx_demo (writable if user chmods, else override)
_VIS_DIR = (_ONNX_LIB_ROOT / "outputs").resolve()


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
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="2.0",
            description="Unused in camera_topic mode; still passed to the node",
        ),
        DeclareLaunchArgument(
            "loop_sequence",
            default_value="false",
            description="Unused in camera_topic mode; still passed to the node",
        ),
        DeclareLaunchArgument(
            "num_leaf",
            default_value="8",
            description="GATsSPG num_leaf",
        ),
        DeclareLaunchArgument(
            "save_vis",
            default_value="true",
            description="Save visualization frames when True",
        ),
        DeclareLaunchArgument(
            "vis_save_dir",
            default_value="/tmp/onepose_ros_vis",
            description="Directory for saved visualization images",
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
                "publish_rate_hz":     ParameterValue(
                    LaunchConfiguration("publish_rate_hz"), value_type=float
                ),
                "loop_sequence":       ParameterValue(
                    LaunchConfiguration("loop_sequence"), value_type=bool
                ),
                "image_topic":         LaunchConfiguration("image_topic"),
                "camera_info_topic":   LaunchConfiguration("camera_info_topic"),
                "num_leaf":            ParameterValue(
                    LaunchConfiguration("num_leaf"), value_type=int
                ),
                "save_vis":            ParameterValue(
                    LaunchConfiguration("save_vis"), value_type=bool
                ),
                "vis_save_dir":        LaunchConfiguration("vis_save_dir"),
            }],
        ),
    ])
