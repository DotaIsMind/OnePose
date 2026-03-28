
完成以下任务：
1. Review run.py，确保你清晰的了解run Pipeline
2. 把相关的函数整合到一个文件下，保存为run_single.py, 遇到依赖pytorch的函数，改为使用numpy或其他方法，
我需要把代码运行在CPU上。

任务目标：
数据集位于：/raid/tengf/6d-pose-resource/OnePose/data/demo/test_coffee
使用source /home/tengf/qrb_ros_simulation_ws/miniconda/bin/activate && conda activate onepose激活虚拟环境
并运行
PROJECT_DIR=$(pwd)
OBJ_NAME=test_coffee
python $PROJECT_DIR/run.py \
    +preprocess="sfm_spp_spg_demo" \
    dataset.data_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-annotate" \
    dataset.outputs_dir="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model" \
，解决你遇到的问题，完成后新建一个summary.md总结你遇到的问题和使用的方法