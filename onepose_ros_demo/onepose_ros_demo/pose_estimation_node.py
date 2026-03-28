#!/usr/bin/env python3
"""
pose_estimation_node.py
=======================
ROS2 node that wraps the OnePose ONNX inference pipeline and publishes
6-DoF pose estimation results.

Two input modes
---------------
1. **local_file** (default)
   Reads PNG images from  <project_root>/data/demo/test_coffee/test_coffee-test/color_full/
   and processes them sequentially at a configurable rate.

2. **camera_topic**
   Subscribes to  /camera/image_raw  (sensor_msgs/Image) for live frames.
   Camera intrinsics are read either from a parameter or from the
   /camera/camera_info  topic (sensor_msgs/CameraInfo).

Published topic
---------------
/pose_estimation_result  (geometry_msgs/PoseStamped)
  header.stamp    – ROS timestamp of the inference
  header.frame_id – "camera_optical_frame"
  pose.position   – translation (x, y, z) in metres
  pose.orientation– rotation as quaternion (x, y, z, w)

Usage examples
--------------
# local-file mode (default)
ros2 run onepose_ros_demo pose_estimation_node.py

# camera-topic mode
ros2 run onepose_ros_demo pose_estimation_node.py \
    --ros-args -p input_mode:=camera_topic

# override data paths
ros2 run onepose_ros_demo pose_estimation_node.py \
    --ros-args \
    -p data_root:=/path/to/test_coffee \
    -p seq_dir:=/path/to/test_coffee/test_coffee-test \
    -p sfm_model_dir:=/path/to/test_coffee/sfm_model
"""

from __future__ import annotations

import os
import sys
import glob
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
# # import natsort

# ── make sure the bundled onnx_demo and src packages are importable ───────────
# Directory layout (self-contained inside onepose_ros_demo/):
#   onepose_ros_demo/          ← _PKG_DIR  (also on sys.path for onnx_demo & src)
#     onepose_ros_demo/        ← _THIS_DIR (this file lives here)
#     onnx_demo/               ← bundled copy of onnx_demo
#     src/                     ← bundled copy of src/utils
_THIS_DIR     = Path(__name__).resolve().parent   # …/onepose_ros_demo/onepose_ros_demo/
_PKG_DIR      = _THIS_DIR.parent                  # …/onepose_ros_demo/
_PROJECT_ROOT = _PKG_DIR                          # self-contained: root IS the pkg dir
for _p in [str(_PKG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── ROS2 imports ──────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from onepose_ros_demo.msg import PoseEstimationResult

# ── project imports ───────────────────────────────────────────────────────────
from onnx_demo.onnx_models import SuperPointOnnx, GATsSPGOnnx
from onnx_demo.object_detector import LocalFeatureObjectDetectorOnnx
from onnx_demo.pipeline import (
    _pad_features3d,
    _build_features3d_leaves,
)
# from onnx_demo.utils.data_utils import get_K
from onnx_demo.utils.path_utils import get_3d_box_path
from onnx_demo.utils.eval_utils import ransac_PnP
from onnx_demo.utils.vis_utils import save_demo_image


# ── default paths (relative to the bundled package dir) ──────────────────────
_ONNX_DIR     = _PKG_DIR / "onnx_demo" / "models"
_DATA_ROOT    = _PKG_DIR / "data" / "demo" / "test_coffee"
_SEQ_DIR      = _DATA_ROOT / "test_coffee-test"
_SFM_DIR      = _DATA_ROOT / "sfm_model"


def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo

# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert ROS Image message → OpenCV grayscale numpy array
# ─────────────────────────────────────────────────────────────────────────────

def _ros_image_to_gray(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a uint8 grayscale numpy array."""
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("mono8", "8uc1"):
        img = data.reshape((msg.height, msg.width))
    elif enc in ("bgr8", "rgb8", "8uc3"):
        img = data.reshape((msg.height, msg.width, 3))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif enc in ("bgra8", "rgba8", "8uc4"):
        img = data.reshape((msg.height, msg.width, 4))
        if enc == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif enc in ("mono16", "16uc1"):
        if msg.is_bigendian:
            data = data.view(np.dtype('>u2'))
        else:
            data = data.view(np.uint16)
        img16 = data.reshape((msg.height, msg.width))
        img = (img16 >> 8).astype(np.uint8)
    else:
        # Fallback: try cv_bridge if available
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            raise ValueError(
                f"Unsupported image encoding '{msg.encoding}': {e}"
            )
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a temporary image file from a numpy array
# (the object detector needs a file path for some operations)
# ─────────────────────────────────────────────────────────────────────────────

_TMP_IMG_PATH = "/tmp/_onepose_ros_query.png"


def _save_tmp_image(gray: np.ndarray) -> str:
    cv2.imwrite(_TMP_IMG_PATH, gray)
    return _TMP_IMG_PATH


def _overlay_rt_on_image_top_left(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    x0: int = 8,
    y0: int = 20,
    line_height: int = 18,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw 3×3 rotation matrix R and translation vector t at the top-left with
    cv2.putText (white text, black outline for contrast). Returns a BGR image.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    black = (0, 0, 0)
    y = int(y0)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(-1)

    def _put_line(text: str) -> None:
        nonlocal y
        cv2.putText(
            out, text, (x0, y), font, font_scale, black, thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, text, (x0, y), font, font_scale, white, thickness, cv2.LINE_AA,
        )
        y += line_height

    _put_line("R:")
    for i in range(3):
        row = R[i]
        _put_line(
            f"  {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}"
        )
    _put_line(
        f"t: [{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]"
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core inference engine (stateful, wraps the ONNX pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class OnePoseEngine:
    """
    Stateful wrapper around the ONNX OnePose pipeline.

    Keeps track of the previous frame's pose so that the object detector
    can use the projection-based fast-detect path on subsequent frames.
    """

    def __init__(
        self,
        superpoint_onnx: str,
        superglue_onnx: str,
        gatsspg_onnx: str,
        sfm_ws_dir: str,
        avg_anno_3d_path: str,
        clt_anno_3d_path: str,
        idxs_path: str,
        box3d_path: str,
        data_root: str | None = None,
        num_leaf: int = 8,
        max_num_kp3d: int = 2500,
    ):
        # ── feature extractor ────────────────────────────────────────────────
        sp_cfg = {
            'nms_radius':         3,
            'keypoint_threshold': 0.005,
            'max_keypoints':      4096,
            'remove_borders':     4,
        }
        self.extractor  = SuperPointOnnx(superpoint_onnx, config=sp_cfg)
        self.matcher_3d = GATsSPGOnnx(gatsspg_onnx)
        self.num_leaf   = num_leaf

        # ── object detector ───────────────────────────────────────────────────
        self.detector = LocalFeatureObjectDetectorOnnx(
            superpoint_onnx_path=superpoint_onnx,
            superglue_onnx_path=superglue_onnx,
            sfm_ws_dir=sfm_ws_dir,
            output_results=False,
            detect_save_dir=None,
            data_dir=data_root,
        )

        # ── 3-D annotations ───────────────────────────────────────────────────
        avg_data = np.load(avg_anno_3d_path)
        clt_data = np.load(clt_anno_3d_path)
        idxs     = np.load(idxs_path)

        self.keypoints3d = clt_data['keypoints3d'].astype(np.float32)  # [M, 3]
        num_3d = self.keypoints3d.shape[0]

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

        self.avg_desc3d_b = avg_desc3d[np.newaxis]          # [1, 256, M]
        self.clt_desc_b   = clt_desc[np.newaxis]            # [1, 256, M*L]
        self.kpts3d_b     = self.keypoints3d[np.newaxis]    # [1, M, 3]

        # ── 3-D bounding box ─────────────────────────────────────────────────
        self.bbox3d = np.loadtxt(box3d_path)

        # ── state ─────────────────────────────────────────────────────────────
        self._prev_pose    = None
        self._prev_inliers = []
        self._frame_id     = 0

    def reset(self):
        """Reset inter-frame state (call when starting a new sequence)."""
        self._prev_pose    = None
        self._prev_inliers = []
        self._frame_id     = 0

    def infer(
        self,
        gray_img: np.ndarray,
        img_path: str,
        K: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list, int]:
        """
        Run one inference step.

        Parameters
        ----------
        gray_img : uint8 grayscale image  [H, W]
        img_path : path to the image file on disk (needed by the detector)
        K        : [3, 3] camera intrinsic matrix

        Returns
        -------
        pose_3x4   : [3, 4] rotation+translation  (None on failure)
        pose_4x4   : [4, 4] homogeneous pose       (None on failure)
        inliers    : list of PnP inlier indices
        frame_id   : current frame counter
        """
        frame_id = self._frame_id
        inp = (gray_img[np.newaxis, np.newaxis] / 255.0).astype(np.float32)

        # ── 1. Object detection ───────────────────────────────────────────────
        if frame_id == 0 or len(self._prev_inliers) < 8:
            bbox, inp_crop, K_crop = self.detector.detect(inp, img_path, K)
        else:
            bbox, inp_crop, K_crop = self.detector.previous_pose_detect(
                img_path, K, self._prev_pose, self.bbox3d
            )

        # ── 2. Keypoint extraction ────────────────────────────────────────────
        pred_det = self.extractor(inp_crop)
        kpts2d   = pred_det['keypoints']    # [N, 2]
        desc2d   = pred_det['descriptors']  # [256, N]

        # ── 3. 2D-3D matching ─────────────────────────────────────────────────
        inp_data = {
            'keypoints2d':         kpts2d[np.newaxis].astype(np.float32),
            'keypoints3d':         self.kpts3d_b,
            'descriptors2d_query': desc2d[np.newaxis].astype(np.float32),
            'descriptors3d_db':    self.avg_desc3d_b,
            'descriptors2d_db':    self.clt_desc_b,
        }
        pred, _ = self.matcher_3d(inp_data)

        matches  = pred['matches0'].numpy().flatten().astype(np.int32)
        mscores  = pred['matching_scores0'].numpy().flatten()
        valid    = matches > -1
        mkpts2d  = kpts2d[valid]
        mkpts3d  = self.keypoints3d[matches[valid]]

        # ── 4. PnP pose estimation ────────────────────────────────────────────
        pose_3x4, pose_4x4, inliers = ransac_PnP(
            K_crop, mkpts2d, mkpts3d, scale=1000
        )

        # ── update state ──────────────────────────────────────────────────────
        self._prev_pose    = pose_3x4
        self._prev_inliers = inliers if inliers is not None else []
        self._frame_id    += 1

        return pose_3x4, pose_4x4, self._prev_inliers, frame_id


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimationNode(Node):
    """
    ROS2 node for OnePose 6-DoF pose estimation.

    Parameters (ROS2 parameters)
    ----------------------------
    input_mode      : "local_file" | "camera_topic"   (default: "local_file")
    data_root       : path to test_coffee root dir
    seq_dir         : path to test sequence dir (contains color_full/)
    sfm_model_dir   : path to sfm_model dir
    superpoint_onnx : path to superpoint.onnx
    superglue_onnx  : path to superglue.onnx
    gatsspg_onnx    : path to gatsspg.onnx
    publish_rate_hz : frame rate for local_file mode  (default: 2.0)
    loop_sequence   : loop the local sequence forever  (default: False)
    image_topic     : image topic for camera_topic mode
                      (default: /camera/image_raw)
    camera_info_topic: camera info topic
                      (default: /camera/camera_info)
    num_leaf        : GATsSPG num_leaf  (default: 8)
    save_vis        : if True, save visualization via save_demo_image and Rt overlay
    vis_save_dir    : directory for saved frames when save_vis is True
    """

    def __init__(self):
        super().__init__("pose_estimation_node")

        # ── declare parameters ────────────────────────────────────────────────
        self.declare_parameter("input_mode",        "local_file")
        self.declare_parameter("data_root",         str(_DATA_ROOT))
        self.declare_parameter("seq_dir",           str(_SEQ_DIR))
        self.declare_parameter("sfm_model_dir",     str(_SFM_DIR))
        self.declare_parameter("superpoint_onnx",   str(_ONNX_DIR / "superpoint.onnx"))
        self.declare_parameter("superglue_onnx",    str(_ONNX_DIR / "superglue.onnx"))
        self.declare_parameter("gatsspg_onnx",      str(_ONNX_DIR / "gatsspg.onnx"))
        self.declare_parameter("publish_rate_hz",   2.0)
        self.declare_parameter("loop_sequence",     False)
        self.declare_parameter("image_topic",       "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("num_leaf",          8)
        self.declare_parameter("save_vis",          True)
        self.declare_parameter(
            "vis_save_dir",
            str(Path("/tmp/onepose_ros_vis").resolve()),
        )

        # ── read parameters ───────────────────────────────────────────────────
        self._input_mode   = self.get_parameter("input_mode").value
        self._data_root    = self.get_parameter("data_root").value
        self._seq_dir      = self.get_parameter("seq_dir").value
        self._sfm_dir      = self.get_parameter("sfm_model_dir").value
        self._sp_onnx      = self.get_parameter("superpoint_onnx").value
        self._sg_onnx      = self.get_parameter("superglue_onnx").value
        self._gat_onnx     = self.get_parameter("gatsspg_onnx").value
        self._rate_hz      = self.get_parameter("publish_rate_hz").value
        self._loop         = self.get_parameter("loop_sequence").value
        self._image_topic  = self.get_parameter("image_topic").value
        self._ci_topic     = self.get_parameter("camera_info_topic").value
        self._num_leaf     = self.get_parameter("num_leaf").value
        self._save_vis     = bool(self.get_parameter("save_vis").value)
        self._vis_save_dir = str(self.get_parameter("vis_save_dir").value)

        self.get_logger().info(
            f"[PoseEstimationNode] input_mode = {self._input_mode}"
        )

        # ── validate ONNX models ──────────────────────────────────────────────
        for label, path in [
            ("superpoint", self._sp_onnx),
            ("superglue",  self._sg_onnx),
            ("gatsspg",    self._gat_onnx),
        ]:
            if not Path(path).exists():
                self.get_logger().error(
                    f"ONNX model not found: {path}  "
                    f"(run 'python -m onnx_demo --export_only' first)"
                )
                raise FileNotFoundError(path)

        # ── build annotation paths ────────────────────────────────────────────
        anno_dir    = os.path.join(
            self._sfm_dir, "outputs_superpoint_superglue", "anno"
        )
        sfm_ws_dir  = os.path.join(
            self._sfm_dir, "outputs_superpoint_superglue", "sfm_ws", "model"
        )
        box3d_path  = get_3d_box_path(self._data_root)
        self._box3d_path = box3d_path

        # ── load camera intrinsics (local_file mode) ──────────────────────────
        intrin_path = os.path.join(self._seq_dir, "intrinsics.txt")
        if Path(intrin_path).exists():
            self._K, _ = get_K(intrin_path)
            self.get_logger().info(
                f"Loaded intrinsics from {intrin_path}"
            )
        else:
            self._K = None
            self.get_logger().warn(
                f"intrinsics.txt not found at {intrin_path}; "
                "will wait for /camera/camera_info"
            )

        # ── build inference engine ────────────────────────────────────────────
        self.get_logger().info("Loading ONNX models …")
        self._engine = OnePoseEngine(
            superpoint_onnx=self._sp_onnx,
            superglue_onnx=self._sg_onnx,
            gatsspg_onnx=self._gat_onnx,
            sfm_ws_dir=sfm_ws_dir,
            avg_anno_3d_path=os.path.join(anno_dir, "anno_3d_average.npz"),
            clt_anno_3d_path=os.path.join(anno_dir, "anno_3d_collect.npz"),
            idxs_path=os.path.join(anno_dir, "idxs.npy"),
            box3d_path=box3d_path,
            data_root=self._data_root,
            num_leaf=self._num_leaf,
        )
        self.get_logger().info("ONNX models loaded successfully.")

        # ── publishers ────────────────────────────────────────────────────────
        # geometry_msgs/PoseStamped (legacy):
        #   header.stamp     – ROS timestamp
        #   pose.position    – translation (x, y, z) in metres
        #   pose.orientation – rotation as quaternion (x, y, z, w)
        self._pub_pose_stamped = self.create_publisher(
            PoseStamped,
            "/pose_estimation_result",
            10,
        )
        # Custom PoseEstimationResult message:
        #   header            – timestamp + frame_id
        #   input_source      – "local_file" or "camera_topic"
        #   frame_id          – 0-based frame counter
        #   rotation_matrix   – 3×3 rotation matrix, row-major
        #   translation_vector– translation vector (metres)
        #   pose_matrix_4x4   – full 4×4 homogeneous pose, row-major
        #   num_inliers       – PnP RANSAC inlier count (-1 on failure)
        #   success           – true when pose estimation succeeded
        self._pub_custom = self.create_publisher(
            PoseEstimationResult,
            "/pose_estimation_result_custom",
            10,
        )

        # ── mode-specific setup ───────────────────────────────────────────────
        if self._input_mode == "local_file":
            self._setup_local_file_mode()
        elif self._input_mode == "camera_topic":
            self._setup_camera_topic_mode()
        else:
            self.get_logger().error(
                f"Unknown input_mode '{self._input_mode}'. "
                "Use 'local_file' or 'camera_topic'."
            )
            raise ValueError(self._input_mode)

    # ─────────────────────────────────────────────────────────────────────────
    # Local-file mode
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_local_file_mode(self):
        """Prepare image list and start a timer-driven processing loop."""
        color_dir = os.path.join(self._seq_dir, "color_full")
        img_paths = sorted(glob.glob(os.path.join(color_dir, "*.png")))

        if not img_paths:
            self.get_logger().error(
                f"No PNG images found in {color_dir}"
            )
            raise FileNotFoundError(color_dir)

        self.get_logger().info(
            f"[local_file] Found {len(img_paths)} images in {color_dir}"
        )

        self._img_paths  = img_paths
        self._img_index  = 0
        self._engine.reset()

        period = 1.0 / max(self._rate_hz, 0.1)
        self._timer = self.create_timer(period, self._local_file_callback)

    def _local_file_callback(self):
        """Timer callback: process one image from the local sequence."""
        if self._img_index >= len(self._img_paths):
            if self._loop:
                self.get_logger().info(
                    "[local_file] Sequence finished – looping."
                )
                self._img_index = 0
                self._engine.reset()
            else:
                self.get_logger().info(
                    "[local_file] Sequence finished. Stopping timer."
                )
                self._timer.cancel()
                return

        img_path = self._img_paths[self._img_index]
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self.get_logger().warn(f"Could not read image: {img_path}")
            self._img_index += 1
            return

        if self._K is None:
            self.get_logger().warn(
                "Camera intrinsics not available – skipping frame."
            )
            self._img_index += 1
            return

        self._run_inference_and_publish(
            gray_img=gray,
            img_path=img_path,
            K=self._K,
            source="local_file",
        )
        self._img_index += 1
        if self._img_index == 10:
            self._save_vis = False
            self.get_logger().info(
                "[local_file] Saving visualization for 10 frames."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Camera-topic mode
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_camera_topic_mode(self):
        """Subscribe to image and camera_info topics."""
        self._camera_K    = self._K   # may already be set from intrinsics.txt
        self._camera_lock = threading.Lock()

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to camera_info to get intrinsics dynamically
        self._ci_sub = self.create_subscription(
            CameraInfo,
            self._ci_topic,
            self._camera_info_callback,
            best_effort_qos,
        )

        # Subscribe to image topic
        self._img_sub = self.create_subscription(
            Image,
            self._image_topic,
            self._image_callback,
            best_effort_qos,
        )

        self.get_logger().info(
            f"[camera_topic] Subscribed to '{self._image_topic}' "
            f"and '{self._ci_topic}'"
        )

    def _camera_info_callback(self, msg: CameraInfo):
        """Extract K from CameraInfo message."""
        with self._camera_lock:
            if self._camera_K is None:
                K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                self._camera_K = K
                self.get_logger().info(
                    f"Received camera intrinsics from {self._ci_topic}:\n{K}"
                )

    def _image_callback(self, msg: Image):
        """Process an incoming image frame."""
        with self._camera_lock:
            K = self._camera_K

        if K is None:
            self.get_logger().warn(
                "Camera intrinsics not yet available – skipping frame. "
                f"Waiting for {self._ci_topic} …"
            )
            return

        try:
            gray = _ros_image_to_gray(msg)
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        img_path = _save_tmp_image(gray)

        self._run_inference_and_publish(
            gray_img=gray,
            img_path=img_path,
            K=K,
            source="camera_topic",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Shared inference + publish
    # ─────────────────────────────────────────────────────────────────────────

    def _run_inference_and_publish(
        self,
        gray_img: np.ndarray,
        img_path: str,
        K: np.ndarray,
        source: str,
    ):
        """Run the ONNX pipeline and publish the result."""
        t_start = time.perf_counter()

        try:
            pose_3x4, pose_4x4, inliers, frame_id = self._engine.infer(
                gray_img=gray_img,
                img_path=img_path,
                K=K,
            )
        except Exception as e:
            self.get_logger().error(
                f"Inference failed on frame {self._engine._frame_id}: {e}",
                throttle_duration_sec=2.0,
            )
            return

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # ── build message ─────────────────────────────────────────────────────
        msg = PoseStamped()

        # Header with current ROS time
        now = self.get_clock().now()
        msg.header.stamp    = now.to_msg()
        msg.header.frame_id = "camera_optical_frame"

        success = (
            pose_3x4 is not None
            and pose_4x4 is not None
            and inliers is not None
            and len(inliers) > 0
        )
        num_inliers = int(len(inliers)) if inliers is not None else -1

        t_vec = [0.0, 0.0, 0.0]
        if success:
            R_mat = pose_3x4[:, :3]   # [3, 3]
            t = pose_3x4[:, 3]        # [3]
            t_vec = t.flatten().tolist()

            # Convert rotation matrix to quaternion using cv2.Rodrigues
            rvec, _ = cv2.Rodrigues(R_mat)
            rvec = rvec.flatten()
            theta = np.linalg.norm(rvec)
            if theta < 1e-6:
                q = [0.0, 0.0, 0.0, 1.0]
            else:
                axis = rvec / theta
                half_theta = theta / 2.0
                sin_half = float(np.sin(half_theta))
                q = [
                    float(axis[0]) * sin_half, # x
                    float(axis[1]) * sin_half, # y
                    float(axis[2]) * sin_half, # z
                    float(np.cos(half_theta))  # w
                ]

            msg.pose.position.x = float(t_vec[0])
            msg.pose.position.y = float(t_vec[1])
            msg.pose.position.z = float(t_vec[2])

            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
            msg.pose.orientation.w = q[3]
        else:
            msg.pose.orientation.w = 1.0

        self._pub_pose_stamped.publish(msg)

        # ── build custom PoseEstimationResult message ────────────────────────
        custom_msg = PoseEstimationResult()
        custom_msg.header = msg.header
        custom_msg.frame_id = frame_id
        
        if success:
            # Rotation matrix 3x3 (row-major)
            R_flat = R_mat.flatten().tolist()
            custom_msg.rotation_matrix = R_flat
            
            # Translation vector
            custom_msg.translation_vector = t_vec
            
            # Pose matrix 4x4 (row-major)
            if pose_4x4 is not None:
                pose_4x4_flat = pose_4x4.flatten().tolist()
                custom_msg.pose_matrix_4x4 = pose_4x4_flat
            else:
                # Create identity 4x4 matrix if pose_4x4 is None
                identity = np.eye(4, dtype=np.float64).flatten().tolist()
                custom_msg.pose_matrix_4x4 = identity
        else:
            # Set default values when no pose
            identity_3x3 = np.eye(3, dtype=np.float64).flatten().tolist()
            custom_msg.rotation_matrix = identity_3x3
            custom_msg.translation_vector = [0.0, 0.0, 0.0]
            identity_4x4 = np.eye(4, dtype=np.float64).flatten().tolist()
            custom_msg.pose_matrix_4x4 = identity_4x4
        
        self._pub_custom.publish(custom_msg)

        # ── optional visualization save ───────────────────────────────────────
        if self._save_vis and success and pose_4x4 is not None:
            try:
                Path(self._vis_save_dir).mkdir(parents=True, exist_ok=True)
                save_path = os.path.join(
                    self._vis_save_dir, f"frame_{frame_id:04d}.jpg"
                )
                vis_img = save_demo_image(
                    pose_4x4.astype(np.float64),
                    np.asarray(K, dtype=np.float64),
                    image_path=img_path,
                    box3d_path=self._box3d_path,
                    draw_box=num_inliers > 6,
                    save_path=None,
                    pose_homo=pose_4x4.astype(np.float64),
                    draw_axes=True,
                )
                vis_img = _overlay_rt_on_image_top_left(vis_img, R_mat, np.asarray(t_vec))
                cv2.imwrite(save_path, vis_img)
            except Exception as e:
                self.get_logger().warn(f"save_vis failed: {e}")

        # ── log ───────────────────────────────────────────────────────────────
        status = "OK" if success else "FAILED"
        self.get_logger().info(
            f"[{source}] frame={frame_id:04d}  "
            f"inliers={num_inliers:3d}  "
            f"status={status}  "
            f"t={elapsed_ms:.1f}ms  "
            f"t_vec=[{t_vec[0]:.4f}, "
            f"{t_vec[1]:.4f}, "
            f"{t_vec[2]:.4f}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
