"""
launch/local_file.launch.py
============================
Launch the pose estimation node in **local_file** mode.

Default paths match ``pipeline_online.yml`` semantics. Dataset roots come from
``get_package_share_directory('onepose_ros_demo')/data/...`` after install (or
the package source ``data/`` tree during development). ONNX paths use
``get_package_prefix`` → ``lib/<pkg>/onnx_demo/models/onnx``.

The node reads PNG images from ``<seq_dir>/color_full/`` and publishes on
``/pose_estimation_result``.

Usage:
    ros2 launch onepose_ros_demo local_file.launch.py
    ros2 launch onepose_ros_demo local_file.launch.py publish_rate_hz:=5.0
    ros2 launch onepose_ros_demo local_file.launch.py loop_sequence:=true
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
_VIS_DIR = str( "/tmp/pose_estimation_outputs")


def generate_launch_description():
    return LaunchDescription([
        # ── launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "data_root",
            default_value=str(_DATA_ROOT),
            description="data.data_root in pipeline_online.yml (object root, box3d_corners.txt)",
        ),
        DeclareLaunchArgument(
            "seq_dir",
            default_value=str(_SEQ_DIR),
            description="data.seq_dir in pipeline_online.yml (intrinsics.txt, color_full/)",
        ),
        DeclareLaunchArgument(
            "sfm_model_dir",
            default_value=str(_SFM_DIR),
            description="data.sfm_model_dir in pipeline_online.yml",
        ),
        DeclareLaunchArgument(
            "superpoint_onnx",
            default_value=str(_ONNX_DIR / "superpoint.onnx"),
            description="models.superpoint_onnx or <models.onnx_dir>/superpoint.onnx",
        ),
        DeclareLaunchArgument(
            "superglue_onnx",
            default_value=str(_ONNX_DIR / "superglue.onnx"),
            description="models.superglue_onnx or <models.onnx_dir>/superglue.onnx",
        ),
        DeclareLaunchArgument(
            "gatsspg_onnx",
            default_value=str(_ONNX_DIR / "gatsspg.onnx"),
            description="models.gatsspg_onnx or <models.onnx_dir>/gatsspg.onnx",
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
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="Unused in local_file mode; still passed to the node",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera_info",
            description="Unused in local_file mode; still passed to the node",
        ),
        DeclareLaunchArgument(
            "num_leaf",
            default_value="8",
            description="pipeline.num_leaf in pipeline_online.yml (GATsSPG)",
        ),
        DeclareLaunchArgument(
            "save_vis",
            default_value="true",
            description="Save visualization frames when True",
        ),
        DeclareLaunchArgument(
            "vis_save_dir",
            default_value=str(_VIS_DIR),
            description="runtime.vis_dir equivalent (default ./outputs → onnx_demo/outputs)",
        ),

        # ── node ──────────────────────────────────────────────────────────────
        Node(
            package="onepose_ros_demo",
            executable="pose_estimation_node.py",
            name="pose_estimation_node",
            output="screen",
            parameters=[{
                "input_mode":        "local_file",
                "data_root":         LaunchConfiguration("data_root"),
                "seq_dir":           LaunchConfiguration("seq_dir"),
                "sfm_model_dir":     LaunchConfiguration("sfm_model_dir"),
                "superpoint_onnx":   LaunchConfiguration("superpoint_onnx"),
                "superglue_onnx":    LaunchConfiguration("superglue_onnx"),
                "gatsspg_onnx":      LaunchConfiguration("gatsspg_onnx"),
                "publish_rate_hz":   ParameterValue(
                    LaunchConfiguration("publish_rate_hz"), value_type=float
                ),
                "loop_sequence":     ParameterValue(
                    LaunchConfiguration("loop_sequence"), value_type=bool
                ),
                "image_topic":       LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "num_leaf":          ParameterValue(
                    LaunchConfiguration("num_leaf"), value_type=int
                ),
                "save_vis":          ParameterValue(
                    LaunchConfiguration("save_vis"), value_type=bool
                ),
                "vis_save_dir":      LaunchConfiguration("vis_save_dir"),
            }],
        ),
    ])
