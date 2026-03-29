cmake_minimum_required(VERSION 3.8)
project(onepose_ros_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── find dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ── custom message definition ─────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseEstimationResult.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# ── install Python package (onepose_ros_demo/) ────────────────────────────────
# Use PYTHON_INSTALL_DIR (provided by ament_cmake_python) to install the Python
# package manually, bypassing ament_python_install_package() which conflicts with
# the presence of setup.py (causes duplicate CMP0002 CMake targets).
install(
  DIRECTORY onepose_ros_demo/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

# ── install executable scripts ────────────────────────────────────────────────
# Use an absolute path so the installed name is pose_estimation_node.py under
# lib/<pkg>/ (not lib/<pkg>/onepose_ros_demo/…), which ros2 launch expects.
install(
  PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/onepose_ros_demo/pose_estimation_node.py"
  DESTINATION lib/${PROJECT_NAME}
)

# ── install launch files ──────────────────────────────────────────────────────
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ── install config files ──────────────────────────────────────────────────────
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ── install demo / object datasets (under share so launch can use get_package_share_directory) ──
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
  install(
    DIRECTORY data/
    DESTINATION share/${PROJECT_NAME}/data
  )
endif()

# ── install bundled Python dependency packages ────────────────────────────────
# onnx_demo is a pure-Python package bundled inside this ROS package
# so that onepose_ros_demo can be built and run without the parent repo.
install(
  DIRECTORY onepose_ros_demo/onnx_demo/
  DESTINATION lib/${PROJECT_NAME}/onnx_demo
)

ament_package()
