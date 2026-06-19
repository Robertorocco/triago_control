import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Define the package and config file name
    # CHANGE 'triago_control' to your ACTUAL package name if different
    pkg_name = 'triago_control' 
    rviz_file_name = 'Recording_Rviz.rviz'

    # 2. Get the path to the config file
    # NOTE: This assumes you have installed the config folder in CMakeLists.txt
    # If you are running locally without building, use absolute path:
    # rviz_config_path = "/home/user/exchange/ros2-ws/src/triago_control/config/qp_debug.rviz"
    
    # Let's use the robust absolute path based on where you said your workspace is:
    rviz_config_path = os.path.join(
        os.getenv('HOME'), 
        'exchange/ros2-ws/src', 
        pkg_name, 
        'config', 
        rviz_file_name
    )

    print(f"Launching RViz with config: {rviz_config_path}")

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            parameters=[{'use_sim_time': True}]
        )
    ])