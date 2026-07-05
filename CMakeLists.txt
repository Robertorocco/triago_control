cmake_minimum_required(VERSION 3.8)
project(triago_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ament build systems
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# --- PYTHON MODULE INSTALLATION ---
ament_python_install_package(${PROJECT_NAME})

# --- C++ NODE: head_pcl_perception_node ---
# Independent PCL/C++ cross-check pipeline for the head-camera tabletop
# perception problem (colleague's algorithm, ported -- see
# src/head_controller/head_pcl_perception_node.cpp for full context).
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation search)
# NOTE: no separate find_package(Eigen3) -- PCL already depends on Eigen and
# exports its include path via PCL_INCLUDE_DIRS below, which is sufficient
# for the plain Eigen::Vector3f/Matrix3f usage in this node. A standalone
# find_package(Eigen3 REQUIRED) can fail on some setups without the
# eigen3_cmake_module package installed; relying on PCL's own Eigen avoids
# that extra, unverified dependency.

add_executable(head_pcl_perception_node
  src/head_controller/head_pcl_perception_node.cpp
)

target_include_directories(head_pcl_perception_node PUBLIC
  ${PCL_INCLUDE_DIRS}
)
target_compile_definitions(head_pcl_perception_node PUBLIC ${PCL_DEFINITIONS})
target_link_libraries(head_pcl_perception_node
  ${PCL_LIBRARIES}
)

ament_target_dependencies(head_pcl_perception_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
)

install(TARGETS
  head_pcl_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

# --- EXECUTABLE NODE INSTALLATION ---
# All files here must contain a main() function and be executable
install(PROGRAMS
  scripts/visualize_live_shadow.py
  scripts/workspace_mapper.py
  scripts/qp_arm_teleop/base_controller.py
  scripts/qp_arm_teleop/plotter.py
  scripts/qp_arm_teleop/offline_plotter.py
  scripts/qp_arm_teleop/drift_evaluator_node.py
  scripts/qp_arm_teleop/keyboard_teleop.py
  scripts/qp_arm_teleop/main_shared_autonomy.py
  scripts/qp_arm_teleop/main_qp_controller.py
  scripts/qp_arm_teleop/trajectory_generator.py
  scripts/qp_arm_teleop/sa_coupling_debugger.py
  scripts/head_controller/main_head.py
  scripts/head_controller/head_plotter.py
  scripts/head_controller/calibration_audit.py
  scripts/head_controller/qp_head_visual_servo.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- URDF / CONFIG INSTALLATION ---
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()