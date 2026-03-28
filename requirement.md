你需要完成1个任务：
1. 把运行inference_demo.py需要的模型转换为onnx，导出onnx model成功后再继续下一步
2. 测试onnx model和原始model的性能对比，输出pose estimation可视化效果和测量误差

任务目标：
1. 能够成功运行onnx_demo
2. 給出原模型文件和onnx_demo的性能对比：包括可视化pose estimation结果，测量误差

补充信息：
1. 模型文件位于./data/models目录下
2. 运行配置位于./config目录下

模型最终部署的运行环境：
- Ubuntu 24.04
- onnxruntime

你可以使用source /home/tengf/qrb_ros_simulation_ws/miniconda/bin/activate & conda activate onepose
激活Onepose的虚拟环境，并确定torch以及其他依赖的版本。


