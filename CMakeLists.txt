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

# --- EXECUTABLE NODE INSTALLATION ---
# All files here must contain a main() function and be executable
install(PROGRAMS
  scripts/visualize_live_shadow.py
  scripts/workspace_mapper.py
  scripts/qp_arm_teleop/base_controller.py
  scripts/qp_arm_teleop/plotter.py
  scripts/qp_arm_teleop/drift_evaluator_node.py
  scripts/qp_arm_teleop/keyboard_teleop.py
  scripts/qp_arm_teleop/main_shared_autonomy.py
  scripts/qp_arm_teleop/main_qp_controller.py
  scripts/qp_arm_teleop/trajectory_generator.py
  scripts/qp_arm_teleop/sa_coupling_debugger.py
  scripts/head_controller/main_head.py
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