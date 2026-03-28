# OnePose `onnx_demo` 说明

本包在 **ONNX Runtime** 上复现与 `inference_demo.py` 相近的推理流程，便于在无 GPU 或仅需部署推理的场景下使用。核心模块包括：

| 模块 | 作用 |
|------|------|
| `export_models.py` | 将 PyTorch 权重导出为 `.onnx` |
| `pipeline.py` | 整段序列位姿估计（`OnnxOnePosePipeline`） |
| `onnx_models.py` | SuperPoint / SuperGlue / GATsSPG 的 ORT 封装 |
| `object_detector.py` | 基于 SP+SG 的 2D 目标检测（ONNX） |
| `benchmark.py` | PyTorch 与 ONNX 的耗时与位姿误差对比 |

## 依赖

在项目根目录安装 OnePose 依赖，并额外安装：

```bash
pip install onnx onnxruntime
```

（亦见仓库根目录 `requirements.txt` 中的 `onnx`、`onnxruntime` 条目。）

导出 ONNX 时需要 **PyTorch** 与训练同款权重；运行 `pipeline` / `benchmark` 时需要 **OpenCV、NumPy** 及本仓库 `src/` 下的工具（如 `get_K`、`ransac_PnP`、`save_demo_image` 等）。

## 目录与模型文件

导出成功后，默认生成：

```
onnx_demo/models/
  superpoint.onnx
  superglue.onnx
  gatsspg.onnx
```

PyTorch 源权重默认路径（与 `export_models.py` 中一致）：

- `data/models/extractors/SuperPoint/superpoint_v1.pth`
- `data/models/matchers/SuperGlue/superglue_outdoor.pth`
- `data/models/checkpoints/onepose/GATsSPG.ckpt`

---

## 一、如何使用 `pipeline.py`

### 1. 类：`OnnxOnePosePipeline`

在**项目根目录**下执行 Python，保证能 `import src`：

```python
import sys
from pathlib import Path

ROOT = Path("/path/to/OnePose")  # 替换为仓库根目录
sys.path.insert(0, str(ROOT))

from onnx_demo.pipeline import OnnxOnePosePipeline

pipeline = OnnxOnePosePipeline(
    superpoint_onnx=str(ROOT / "onnx_demo/models/superpoint.onnx"),
    superglue_onnx=str(ROOT / "onnx_demo/models/superglue.onnx"),
    gatsspg_onnx=str(ROOT / "onnx_demo/models/gatsspg.onnx"),
    num_leaf=8,           # 与训练/推理配置一致，一般为 8
    max_num_kp3d=2500,    # 预留参数，当前 pipeline 逻辑与 inference_demo 对齐时可保持默认
)

pred_poses, timing = pipeline.run_sequence(
    data_root="/path/to/object_root",      # 含 box3d_corners.txt 的根目录
    seq_dir="/path/to/object_root/seq_name",  # 测试序列目录（见下）
    sfm_model_dir="/path/to/object_root/sfm_model",
)
```

### 2. `run_sequence` 参数含义

| 参数 | 说明 |
|------|------|
| `data_root` | 物体数据根目录，需包含 `box3d_corners.txt`（`get_3d_box_path` 会解析） |
| `seq_dir` | 当前测试序列目录，需包含 `color_full/*.png`、`intrinsics.txt` |
| `sfm_model_dir` | SfM 输出根目录，其下需有 `outputs_superpoint_superglue/anno/`（`anno_3d_average.npz`、`anno_3d_collect.npz`、`idxs.npy`）以及 `outputs_superpoint_superglue/sfm_ws/model/`（COLMAP 模型，供检测器用） |

### 3. 返回值

- `pred_poses`：`dict`，键为帧序号 `int`，值为 `(pose_3x4, inliers)`。
- `timing`：`dict`，键为 `detect` / `extract` / `match3d` / `pnp`，值为每帧耗时列表（秒）。

### 4. 输出文件（由 `run_sequence` 内部写入）

- 位姿可视化帧图：`{seq_dir}/pred_vis_onnx/{id}.jpg`
- 演示视频：`{seq_dir}/demo_video_onnx.mp4`
- 检测可视化目录：`{seq_dir}/detector_vis/`（若检测器有写盘逻辑）

### 5. 实现说明（与 PyTorch 对齐时需注意）

- GATsSPG 的 ONNX 图通常只包含 **三个描述子输入**（`descriptors2d_query`、`descriptors3d_db`、`descriptors2d_db`）；PyTorch 前向在空形状判断之后不再使用 2D/3D 关键点坐标，导出器可能去掉 `keypoints*`。`GATsSPGOnnx` 会按会话实际输入名喂数据，调用侧仍传入完整 `inp_data` 字典即可。

---

## 二、如何导出 ONNX（export）

### 方式 A：模块入口（推荐）

在**仓库根目录**执行：

```bash
python -m onnx_demo.export_models
```

将依次导出并校验三个模型到 `onnx_demo/models/`。

### 方式 B：通过 `python -m onnx_demo`

```bash
# 仅导出，不跑推理/基准
python -m onnx_demo --export_only
```

若 `onnx_demo/models/` 下已存在三个 `.onnx`，默认**不会**重复导出；需要重新导出时请**先删除**对应文件，或直接用法 A 覆盖。

### 源权重与输出对应关系

| 输出 | 源权重 |
|------|--------|
| `superpoint.onnx` | SuperPoint 密集骨干（NMS 等在 NumPy 侧完成） |
| `superglue.onnx` | SuperGlue（归一化后的关键点与描述子为输入） |
| `gatsspg.onnx` | `LitModelGATsSPG` 中的 2D-3D matcher |

导出过程需要安装 `onnx` 包（用于 `onnx.checker.check_model`）。

---

## 三、如何运行 Benchmark

`benchmark.py` 在**固定演示数据**上对比 PyTorch 与 ONNX：路径在文件内写死为 `data/demo/test_coffee`（`DATA_ROOT`、`SEQ_DIR`、`SFM_DIR`）。若使用自己的数据，需修改这些常量或自行封装调用 `run_pytorch_inference` / `run_onnx_inference`。

### 命令行

在仓库根目录：

```bash
# 全流程：先 PyTorch 再 ONNX，生成对比视频、误差曲线与报告
python -m onnx_demo.benchmark

# 只处理前 N 帧（调试/快速验证）
python -m onnx_demo.benchmark --max_frames 40
```

### 通过包入口（导出 + 基准）

```bash
python -m onnx_demo --skip_export          # 已有 onnx，直接跑 PyTorch vs ONNX 基准
python -m onnx_demo --max_frames 40        # 限制帧数
python -m onnx_demo --onnx_only            # 仅 ONNX 推理与计时，不跑 PyTorch
python -m onnx_demo --export_only          # 仅导出
```

### 输出位置（默认）

| 产物 | 路径 |
|------|------|
| 文本报告 | `onnx_demo/benchmark_results/benchmark_report.txt` |
| 并排对比视频 | `onnx_demo/benchmark_results/comparison_video.mp4` |
| 逐帧旋转/平移误差曲线 | `onnx_demo/benchmark_results/error_curves.png` |
| PyTorch 可视化视频 | `data/demo/test_coffee/test_coffee-test/demo_video_pytorch.mp4` |
| ONNX 可视化视频 | `data/demo/test_coffee/test_coffee-test/demo_video_onnx.mp4` |

报告中含各阶段平均耗时（ms/帧）、PyTorch 与 ONNX 位姿之间的旋转误差（度）与平移误差（厘米）等统计量。

---

## 常见问题

1. **强制重新导出 ONNX**  
   删除 `onnx_demo/models/*.onnx` 后重新执行 `python -m onnx_demo.export_models`，或使用 `--export_only`（在不存在 onnx 时会导出；若已存在则需先删文件）。

2. **Benchmark 找不到数据**  
   确认 `data/demo/test_coffee/...` 与 `sfm_model` 已按官方流程生成；或修改 `benchmark.py` 顶部路径指向你的序列。

3. **性能**  
   当前默认在 ONNX Runtime 中使用 **CPU**（`CPUExecutionProvider`）。生产环境可按目标平台切换 Execution Provider（如 CUDA、TensorRT）并调线程数。
