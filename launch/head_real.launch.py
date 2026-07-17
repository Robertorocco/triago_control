import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense_camera_cfg'),
                         'launch', 'rgbd.launch.py')
        ),
        launch_arguments={
            'camera_name': 'head_arm_rgbd',
            'depth_module.profile': '640,480,30',
            'rgb_camera.profile': '640,480,30',
            # main_head deprojects the depth image itself; the driver's own
            # coloured pointcloud is unused and only emits the "No stream match
            # for pointcloud chosen texture" warning -- turn it off.
            'pointcloud.enable': 'false',
        }.items(),
    )

    # Measured mount transform (arm_head_tool_link -> the camera driver's own
    # TF root) -- replaces the idealized CAD (0,0,0 / -90deg about Y) numbers.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='head_camera_mount_tf',
        arguments=[
            '-0.056783', '0.034171', '0.011676',
            '0.00914972', '-0.71833972', '0.00987883', '0.69556217',
            'arm_head_tool_link', 'head_arm_rgbd_link',
        ],
    )

    # main_head.py pointed at the real camera instead of head_control/config.py's
    # Gazebo sim defaults. Topic names confirmed against `ros2 topic list |
    # grep head_arm_rgbd` on the real robot.
    head_node = Node(
        package='triago_control',
        executable='main_head.py',
        name='main_head',
        output='screen',
        parameters=[{
            'color_topic': '/head_arm_rgbd/color/image_raw',
            'depth_topic': '/head_arm_rgbd/depth/image_rect_raw',
            'camera_info_topic': '/head_arm_rgbd/depth/camera_info',
            # Real hardware: the driver's optical-frame origin IS the sensor,
            # no Gazebo-render offset to correct -- disable the correction.
            'depth_center_frame': '',
            # Real objects are a black + a white/paper cylinder, not red/blue.
            'real_hardware_head': True,
        }],
    )

    return LaunchDescription([realsense_launch, static_tf, head_node])
