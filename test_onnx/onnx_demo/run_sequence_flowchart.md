# `run_sequence` 流程图（ONNX vs CPU）

```mermaid
flowchart TD
    A[读取序列与配置<br/>CPU] --> B[逐帧循环]

    B --> C[读灰度图 + 归一化<br/>Detect 前处理 / CPU]
    C --> D{首帧或跟踪丢失?}
    D -->|是| E[detector.detect()<br/>Detect 推理阶段]
    D -->|否| F[detector.previous_pose_detect()<br/>Detect 推理阶段]
    E --> G[得到 bbox, inp_crop, K_crop<br/>Detect 后处理 / CPU]
    F --> G

    G --> H[SuperPointOnnx 提特征<br/>Extract ONNX]
    H --> I[组装 matcher_3d 输入<br/>Match 前处理 / CPU]
    I --> J[GATsSPGOnnx 2D-3D 匹配<br/>Match ONNX]
    J --> K[解析 matches / mkpts2d / mkpts3d<br/>Match 后处理 / CPU]

    K --> L[构建 PnP 输入<br/>Pose 前处理 / CPU]
    L --> M[ransac_PnP 求位姿<br/>Pose Estimation 核心 / CPU]
    M --> N[保存可视化结果<br/>Pose 后处理 / CPU]
    N --> O{是否最后一帧?}
    O -->|否| B
    O -->|是| P[make_video + 输出统计 + 写 inference_core.log]

    classDef onnx fill:#e6f7ff,stroke:#1677ff,stroke-width:2px,color:#003a8c;
    classDef cpu fill:#fff7e6,stroke:#d46b08,stroke-width:1.5px,color:#613400;

    class H,J onnx;
    class A,B,C,D,E,F,G,I,K,L,M,N,O,P cpu;
```

## 标注说明

- 蓝色节点：ONNX 推理主耗时段（`SuperPointOnnx`、`GATsSPGOnnx`）。
- 橙色节点：CPU 前后处理（读图、数据打包、结果解析、PnP、可视化等）。
- `inference_core.log` 中的 `detect_* / match_* / pose_*` 与流程图中的三段对应。
