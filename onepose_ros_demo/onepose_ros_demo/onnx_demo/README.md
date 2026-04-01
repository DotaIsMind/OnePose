# OnePose `onnx_demo` 说明

本目录在 **ONNX Runtime（默认 `CPUExecutionProvider`）** 上提供从扫描数据预处理、离线 SfM、离线逐帧验证到**在线视频位姿估计**的完整链路。详细设计与 API 约定见 **[`camera_pipeline.md`](camera_pipeline.md)**。

---

## 端到端四脚本（Pose Estimation Pipeline）

下列脚本按**数据流顺序**组成部署流水线（均在 `test_onnx/onnx_demo/`）：

| 顺序 | 脚本 | 阶段 | 作用 |
|------|------|------|------|
| ① | [`parse_scanned_data.py`](parse_scanned_data.py) | 数据准备 | 解析 ARKit 等扫描目录，生成 `intrinsics.txt`、`box3d_corners.txt`、`color/`、`color_full/`、`poses/` 等 todo: test_<obj>数据改为test_<obj>-annotate的形式 |
| ② | [`run_single.py`](run_single.py) | 离线 SfM（每物体一次） | COLMAP 三角化 + 后处理，产出 `sfm_model_dir/outputs_superpoint_superglue/{anno,sfm_ws/model}` |
| ③ | [`pipeline_single.py`](pipeline_single.py) | 离线逐帧推理（验证） | 单文件整合 SP / SG / GAT 与检测器；`OnnxOnePosePipeline.run_sequence` 读 `color_full/*.png`，输出位姿与 `demo_video_onnx.mp4` |
| ④ | [`pipeline_online.py`](pipeline_online.py) | 在线视频 / 相机流 | `CameraPosePipeline` + YAML：`FrameInput`（灰度 + K）→ 每帧 `PoseEstimationResult` |

**依赖关系**：④ 依赖 ② 的 `sfm_model_dir` 与 ① 提供的 `data_root`（含 3D 框等）；③ 与 ④ 可并行用于验证，不互为前置。①② 为物体**首次部署**的离线阶段。

---

## 模块说明

| 模块 | 说明 |
|------|------|
| **`parse_scanned_data.py`** | 命令行：`--scanned_object_path`。处理 `*-annotate`（标注序列）与 `*-test`（测试序列）子目录，不写位姿估计，只准备目录与几何文件。 |
| **`run_single.py`** | 单文件 SfM，对应原 `run.py` 中 `type: sfm` 路径。推荐 **`--backend onnx`** 并指定 SuperPoint / SuperGlue 的 `.onnx`；`auto` 在缺少默认 ONNX 时会回退到 `torch_cpu`（需 PyTorch）。后处理中 3D 框过滤等为 NumPy 实现。 |
| **`pipeline_single.py`** | 整合原 `pipeline.py` / `onnx_models.py` / `object_detector.py` 思路：含 `SuperPointOnnx`、`SuperGlueOnnx`、`GATsSPGOnnx`、`LocalFeatureObjectDetectorOnnx`、`OnnxOnePosePipeline`。检测器匹配打包为 **NumPy**，不依赖 PyTorch。 |
| **`pipeline_online.py`** | 从 [`pipeline_online.yml`](pipeline_online.yml) 加载模型路径、数据根、`sfm_model_dir`、视频与运行时参数；`CameraPosePipeline.process_frame` 为单帧入口；`run_camera_loop` 驱动 `VideoFileSource`。检测器主路径仍通过临时灰度 PNG 与磁盘读图对齐（见 `camera_pipeline.md`）。 |
| **`export_models.py`** | 从 PyTorch 权重导出三个 `.onnx`（需安装 PyTorch）。 |
| **`benchmark.py`** | 固定演示数据上对比 PyTorch 与 ONNX 耗时与误差（可选依赖）。 |
| **`pipeline.py` / `onnx_models.py` / `object_detector.py`** | 历史拆分文件；日常推理优先使用 **`pipeline_single.py`**。 |

---

## 运行步骤

### 环境

```bash
pip install onnx onnxruntime opencv-python numpy scipy h5py omegaconf tqdm h5py natsort transforms3d
```

- **仅 CPU、不装 PyTorch**：可走 ① → ②（`--backend onnx`）→ ③ / ④；`src/utils/data_utils.py` 对 `torch` 为函数内延迟导入，几何函数（`get_K`、裁剪）不触发 PyTorch。  
- **导出 ONNX、`run_single --backend torch_cpu` 或 `benchmark` 的 PyTorch 分支**：需单独安装 PyTorch。

模型默认放在本目录 `models/`：

```
onnx_demo/models/
  superpoint.onnx
  superglue.onnx
  gatsspg.onnx
```

（导出见下文「导出 ONNX」。）

---

### 步骤 ①：解析扫描数据

```bash
cd test_onnx/onnx_demo
python parse_scanned_data.py --scanned_object_path /path/to/object_root
```

`object_root` 下应含 `*-annotate`、`*-test` 等子目录（见脚本内说明）。

---

### 步骤 ②：离线 SfM（生成 `sfm_model`）

```bash
python run_single.py \
  --data-dir "/path/to/object_root object_name-annotate" \
  --outputs-dir "/path/to/object_root/sfm_model" \
  --backend onnx \
  --detection-model ./models/superpoint.onnx \
  --matching-model ./models/superglue.onnx
```

将 `object_name-annotate` 换成实际 annotate 序列目录名；`outputs-dir` 会按对象名 format（与脚本 `--help` 一致）。**务必使用 `--backend onnx`** 若部署环境无 PyTorch。

---

### 步骤 ③：离线序列验证（可选）

编辑 `pipeline_single.py` 末尾 `main()` 中的 `data_root` / `seq_dir` / `sfm_model_dir` 与 `models/` 路径，或自行封装调用：

```bash
cd test_onnx/onnx_demo
python pipeline_single.py
# 仅前 N 帧调试：
python pipeline_single.py --max_frames 40
```

输出包括 `pred_vis_onnx/`、`demo_video_onnx.mp4` 等（见脚本内路径逻辑）。

---

### 步骤 ④：在线视频推理

1. 复制并编辑 [`pipeline_online.yml`](pipeline_online.yml)：设置 `input.video`、`data.*`、`models.onnx_dir` 或各 ONNX 绝对路径，必要时设置 `runtime.ref_width` / `ref_height` 与标定分辨率一致。  
2. 运行：

```bash
cd test_onnx/onnx_demo
python pipeline_online.py
python pipeline_online.py --config /path/to/custom.yml
```

可选：`runtime.vis_dir` 非空时写出 `detect/`、`match/`、`pose/` 序列图；`runtime.max_frames`、`frame_stride` 用于调试。

更完整的字段说明与任务记录见 **`camera_pipeline.md`** 第 11–12 节。

---

## 导出 ONNX

在**仓库根目录**（或按包结构）执行：

```bash
python -m onnx_demo.export_models
# 或
cd test_onnx/onnx_demo && python export_models.py
```

需要 PyTorch 与训练同款权重路径（见 `export_models.py` 内注释）。导出过程使用 `onnx` 包做 checker。

---

## Benchmark（可选）

在仓库根目录：

```bash
python -m onnx_demo.benchmark
python -m onnx_demo.benchmark --max_frames 40
python -m onnx_demo --onnx_only
```

默认数据路径写在 `benchmark.py` 内；使用自有数据需改常量或自行封装。输出见 `benchmark_results/` 与序列目录下的对比视频。

---

## 与旧版 `pipeline.py` 文档的关系

若文档仍写「`from onnx_demo.pipeline import OnnxOnePosePipeline`」，请改为 **`pipeline_single.OnnxOnePosePipeline`**，并将 `sys.path` 指向本目录（`onnx_demo`），以便 `import src.utils`。

---

## 常见问题

1. **`run_single` 意外使用 PyTorch**  
   使用 `--backend onnx` 并保证 `--detection-model` / `--matching-model` 指向存在的 `.onnx`；或设置环境变量 `ONEPOSE_SUPERPOINT_ONNX` / `ONEPOSE_SUPERGLUE_ONNX`，避免 `auto` 回退到 `torch_cpu`。

2. **内参与视频分辨率不一致**  
   在 `pipeline_online.yml` 的 `runtime` 中填写 `ref_width`、`ref_height`（标定时的宽高），运行时会缩放 `K`。

3. **性能**  
   默认 ONNX Runtime 为 **CPU**；可在 `_get_session` / 会话选项中更换 Execution Provider 或线程数（见 `pipeline_single.py`）。

4. **完整设计、YAML 字段、测试用例 ID**  
   见 **[`camera_pipeline.md`](camera_pipeline.md)**。
