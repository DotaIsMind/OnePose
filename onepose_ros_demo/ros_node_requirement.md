# onepose_ros_demo – ROS Node Requirement

Task lists:
1. review the pipeline.pt on ../onnx_demo path， ensure you are clear for running process
2. 把pipeline.py打包为一个ROS2 节点，完成以下功能：
    a. 能够以两种方式读取图片输入
        i. 从/camera/camera_info topic 订阅图像作为输入
        ii. 从本地data目录的test_coffee读取图片作为输入
    b. 完成pose estimation 后，发布ros2 ropic /pose_estimation_result
        i. /pose_estimation_result 包括旋转矩阵和偏移矩阵
        ii. /pose_estimation_result附带当前时间戳作为时间同步
3. 最终的代码文件保存在onepose_ros_demo目录下

任务目标：
1. 能够运行onepose_ros_demo
2. 能够使用ros2 topic echo /pose_estimation_result打印信息

运行环境：
ros2 jazzy
你可以使用source /home/tengf/qrb_ros_simulation_ws/miniconda/bin/activate
conda activate onepose
激活虚拟环境

```shell
# sfm preprocess:
python -m pdb $PROJECT_DIR/run.py     +preprocess="sfm_spp_spg_sample"     dataset.data_dir="$PROJECT_DIR/data/onepose_datasets/sample_data/0501-matchafranzzi-box matchafranzzi-4"     dataset.outputs_dir="$PROJECT_DIR/data/sfm_model/0501-matchafranzzi-box"

# inference demo on coffee data:
python $PROJECT_DIR/inference_demo.py     +experiment="test_demo"     input.data_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test"     input.sfm_model_dirs="$PROJECT_DIR/data/demo/$OBJ_NAME/sfm_model"     use_tracking=False

# inference demo on franzzi box data:
python $PROJECT_DIR/inference_demo.py     +experiment="test_demo"     input.data_dirs="./data/onepose_datasets/sample_data/0501-matchafranzzi-box matchafranzzi-4"     input.sfm_model_dirs="./data/sf
m_model/0501-matchafranzzi-box"     use_tracking=False
```