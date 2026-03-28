# run-require 任务总结

## 目标回顾

1. 阅读并理解 `run.py` 中的 SfM 预处理流水线。
2. 将相关逻辑整合为单文件 `run_single.py`，对原先依赖 PyTorch 的部分改为 NumPy 或可在 CPU 上运行的替代实现。
3. 在指定 Conda 环境与 `test_coffee` 数据上运行官方 `run.py` 命令，处理遇到的问题。
4. 本文档记录问题与做法。

## `run.py` 流水线（简要）

Hydra 根据 `+preprocess=sfm_spp_spg_demo` 将 `type` 设为 `sfm`，入口 `main` 调用 `sfm(cfg)`：

1. **解析 `dataset.data_dir`**：形如 `"<root> <seq1> <seq2> ..."`，在 `<root>/<seq>/color/*.png` 收集图像。
2. **按 `sfm.down_ratio` 降采样**帧序号。
3. **`sfm_core`**：在 `outputs_superpoint_superglue/` 下执行 SuperPoint 特征、基于位姿的共视配对、SuperGlue 匹配、COLMAP 空模型与三角化。
4. **`postprocess`**：按轨迹长度与 `box3d_corners.txt` 过滤三维点、合并近邻点，并调用 `feature_process.get_kpt_ann` 写出 `anno/` 等标注。

其中 `src/sfm/postprocess/filter_points.filter_by_3d_box` 使用 PyTorch 张量做包围盒内点判断；`extract_features` / `match_features` 使用 `.cuda()`。

## 运行官方命令时的问题与处理

### Conda 激活方式

在部分非交互 shell 中，将 `source .../activate && conda activate onepose` 与 `cd`、`python` 写在同一行且未通过 `bash -lc` 包裹时，可能解析失败（例如把错误路径当成环境名）。

**做法**：使用：

```bash
bash -lc 'source /home/tengf/qrb_ros_simulation_ws/miniconda/bin/activate && conda activate onepose && cd /raid/tengf/6d-pose-resource/OnePose && ...'
```

在本环境中按上述方式执行后，`run.py` 对 `test_coffee` 的 SfM 与后处理已 **成功完成**（约两分钟级，取决于机器与 COLMAP）。

## `run_single.py` 的设计

| 模块 | 做法 |
|------|------|
| 3D 框过滤与点合并 | 在文件内用 **NumPy** 实现（等价于原 `filter_by_3d_box` 与 `merge`），避免为后处理再依赖 `torch`。 |
| 特征提取 / 匹配 | **`--backend torch_cpu`**：对 `torch.Tensor.cuda` / `nn.Module.cuda` 做 **no-op 补丁**，在 **CPU** 上调用原有 `extract_features.main` / `match_features.main`（与原版行为一致，仅设备从 CUDA 改为 CPU）。 |
| 可选 ONNX | **`--backend onnx` 或 `auto`（若默认 ONNX 路径下存在两个 `.onnx` 文件）**：用 **ONNX Runtime CPU** 跑 SuperPoint/SuperGlue，不经过 PyTorch 推理。需设置 `ONEPOSE_SUPERPOINT_ONNX` / `ONEPOSE_SUPERGLUE_ONNX` 或将模型放到 `data/models/onnx/superpoint_v1.onnx` 与 `superglue_outdoor.onnx`。 |

其余步骤（`pairs_from_poses`、`generate_empty`、`triangulation`、`feature_process`、`filter_tkl`）仍复用仓库中的 `src`，依赖 COLMAP 命令行。

## 验证

在已有 `sfm_model` 输出的情况下，使用：

`python run_single.py ... --no-redo --backend torch_cpu`

仅跳过 `sfm_core` 中 `redo` 分支，重新跑 **后处理**；在本仓库数据上运行成功，说明 NumPy 路径下的过滤与 `get_kpt_ann` 衔接正常。

## 使用提示

- 全量重跑与 `run.py` 一样会删除 `outputs_superpoint_superglue`（`redo=True` 时），请按需备份。
- 纯 CPU + 无 CUDA 时，优先使用 **`--backend torch_cpu`**（需安装 PyTorch CPU）；若希望推理阶段完全不经过 PyTorch，请准备 ONNX 并选择 **`--backend onnx`**。
