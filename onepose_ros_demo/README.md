# onepose_ros_demo

ROS2 (Jazzy) node that wraps the **OnePose ONNX inference pipeline** and
publishes 6-DoF pose estimation results on `/pose_estimation_result`.

---

## Package layout

```
onepose_ros_demo/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── msg/
│   └── PoseEstimationResult.msg   ← custom ROS2 message
├── onepose_ros_demo/
│   ├── __init__.py
│   └── pose_estimation_node.py    ← main ROS2 node
├── launch/
│   ├── local_file.launch.py       ← read from local data/
│   └── camera_topic.launch.py     ← subscribe to camera topic
└── config/
    └── params.yaml                ← default parameters
```

---

## Prerequisites

### 1. Conda environment

```bash
source /home/tengf/qrb_ros_simulation_ws/miniconda/bin/activate
conda activate onepose
```

### 2. ONNX models

The three ONNX models must exist before running the node:

```
onnx_demo/models/superpoint.onnx
onnx_demo/models/superglue.onnx
onnx_demo/models/gatsspg.onnx
```

Export them from the repo root (one-time step):

```bash
cd <repo_root>
python -m onnx_demo --export_only
```

### 3. ROS2 Jazzy

```bash
source /opt/ros/jazzy/setup.bash
```

---

## Build

```bash
# From your ROS2 workspace root (e.g. ~/ros2_ws)
cd ~/ros2_ws
colcon build --packages-select onepose_ros_demo --symlink-install
source install/setup.bash
```

---

## Run

### Mode 1 – Local file (reads from `data/demo/test_coffee`)

```bash
ros2 launch onepose_ros_demo local_file.launch.py
```

Optional overrides:

```bash
ros2 launch onepose_ros_demo local_file.launch.py \
    publish_rate_hz:=5.0 \
    loop_sequence:=true
```

### Mode 2 – Camera topic (live camera)

```bash
ros2 launch onepose_ros_demo camera_topic.launch.py \
    image_topic:=/camera/image_raw \
    camera_info_topic:=/camera/camera_info
```

---

## Inspect results

In a separate terminal (after sourcing ROS2 and the workspace):

```bash
# Print every message
ros2 topic echo /pose_estimation_result

# Check publish rate
ros2 topic hz /pose_estimation_result

# Inspect message type
ros2 interface show onepose_ros_demo/msg/PoseEstimationResult
```

---

## Custom message: `PoseEstimationResult`

| Field              | Type          | Description                                      |
|--------------------|---------------|--------------------------------------------------|
| `header`           | std_msgs/Header | Timestamp + frame_id                           |
| `input_source`     | string        | `"local_file"` or `"camera_topic"`               |
| `frame_id`         | int32         | 0-based frame counter                            |
| `rotation_matrix`  | float64[9]    | 3×3 rotation matrix, row-major                   |
| `translation_vector` | float64[3]  | Translation vector (metres)                      |
| `pose_matrix_4x4`  | float64[16]   | Full 4×4 homogeneous pose, row-major             |
| `num_inliers`      | int32         | PnP RANSAC inlier count (-1 on failure)          |
| `success`          | bool          | `true` when pose estimation succeeded            |

---

## Quick-start script

A convenience script is provided at `onepose_ros_demo/run_demo.sh`:

```bash
bash onepose_ros_demo/run_demo.sh
```
