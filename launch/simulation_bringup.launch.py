#!/usr/bin/env python3
"""One-shot bring-up of a full TRIAGo simulation session: Gazebo world, TSID default
controllers, the mobile-base teleop, and the three RViz stations the operator watches.

Replaces the four-terminal manual sequence (see README / .kiro/context.md §11) with:

    ros2 launch triago_control simulation_bringup.launch.py world:=rack_world

Only `world` is normally set: it is forwarded to the Gazebo world file AND must match the
`world_name` ROS param the QP/shared-autonomy nodes are launched with, since world_loader.py
mirrors the same scene independently (they are NOT started here -- see the note below).

The controllers are started after CONTROLLER_DELAY_S: controller_manager only exists once
Gazebo has spawned the robot and its ros2_control plugin, so an immediate spawn races it.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Gazebo must have spawned the robot (and its ros2_control plugin) before the TSID
# controllers are loaded, otherwise controller_manager is not yet advertising.
CONTROLLER_DELAY_S = 10.0

# RViz is started after the controllers so the robot model resolves on first draw.
RVIZ_DELAY_S = CONTROLLER_DELAY_S + 4.0

# The robot-model filter only needs robot_state_publisher up (it waits/retries on its
# own service call up to 30s regardless), well before RViz needs its filtered topic.
ROBOT_MODEL_FILTER_DELAY_S = 3.0


def generate_launch_description():
    pkg = get_package_share_directory('triago_control')

    world = LaunchConfiguration('world')
    end_effector_right = LaunchConfiguration('end_effector_right')
    end_effector_left = LaunchConfiguration('end_effector_left')
    base_controller = LaunchConfiguration('base_controller')
    rviz_stations = LaunchConfiguration('rviz_stations')
    dim_mode = LaunchConfiguration('dim_mode')

    args = [
        DeclareLaunchArgument(
            'world', default_value='rack_world',
            description='Gazebo world file name (config/worlds/<world>.world). Launch the '
                        'QP and shared-autonomy nodes with the SAME -p world_name:=<world>.'),
        DeclareLaunchArgument(
            'end_effector_right', default_value='pal-pro-gripper',
            description='Right-arm end effector model.'),
        DeclareLaunchArgument(
            'end_effector_left', default_value='pal-pro-gripper',
            description='Left-arm end effector model.'),
        DeclareLaunchArgument(
            'base_controller', default_value='true', choices=['true', 'false'],
            description='Start the mobile-base teleop node.'),
        # NOT named `rviz`: triago_gazebo.launch.py declares its own `rviz` argument and
        # IncludeLaunchDescription shares the configuration scope, so a same-named
        # argument here is validated against ITS choices (['True','False']) too.
        DeclareLaunchArgument(
            'rviz_stations', default_value='true', choices=['true', 'false'],
            description='Start all three RViz stations (operator + overview + right tracking).'),
        DeclareLaunchArgument(
            'dim_mode', default_value='material', choices=['material', 'remove'],
            description="How study_robot_model_filter.py treats non-arm-chain links: "
                        "'material' dims them translucent, 'remove' hides them outright."),
    ]

    # 1. Gazebo + robot spawn.
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('triago_gazebo'),
            'launch', 'triago_gazebo.launch.py')),
        launch_arguments={
            'end_effector_right': end_effector_right,
            'end_effector_left': end_effector_left,
            'world_name': world,
            # Suppress its own RViz: the three stations below are the only ones wanted.
            'rviz': 'False',
        }.items())

    # 2. TSID default controllers (delayed -- see CONTROLLER_DELAY_S).
    controllers = TimerAction(period=CONTROLLER_DELAY_S, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('triago_controller_configuration'),
                'launch', 'tsid_default_controllers.launch.py')),
            launch_arguments={'use_sim_time': 'True'}.items()),
    ])

    # 3. Mobile-base teleop, once the controllers it commands are up.
    base_node = TimerAction(period=CONTROLLER_DELAY_S, actions=[
        Node(package='triago_control', executable='base_controller.py',
             name='base_controller', output='screen',
             parameters=[{'use_sim_time': True}],
             condition=IfCondition(base_controller)),
    ])

    # 4. Robot-model filter: republishes robot_description with non-arm-chain links dimmed
    #    or removed (dim_mode), since RViz's own per-link Alpha/Value overrides don't render
    #    for this robot's meshes (an Ogre/Collada material-sharing limitation, not fixable
    #    from a .rviz file). The three RViz stations below point their RobotModel display
    #    at this filtered topic instead of the real /robot_description.
    robot_model_filter = TimerAction(period=ROBOT_MODEL_FILTER_DELAY_S, actions=[
        Node(package='triago_control', executable='study_robot_model_filter.py',
             name='study_robot_model_filter', output='screen',
             parameters=[{'use_sim_time': True, 'dim_mode': dim_mode}]),
    ])

    # 5. Three RViz stations: operator (head-camera image + left-arm tracking view),
    #    overview (wide scene), right tracking (right-arm tracking view). Window placement
    #    lives in each .rviz file's `Window Geometry` block, so move/resize in RViz and
    #    save to persist it.
    def rviz_station(name, config):
        return Node(
            package='rviz2', executable='rviz2', name=name, output='screen',
            arguments=['-d', os.path.join(pkg, 'config', config)],
            parameters=[{'use_sim_time': ParameterValue(True, value_type=bool)}],
            condition=IfCondition(rviz_stations))

    rviz_group = TimerAction(period=RVIZ_DELAY_S, actions=[
        rviz_station('rviz2_operator', 'study_operator.rviz'),
        rviz_station('rviz2_overview', 'study_overview.rviz'),
        rviz_station('rviz2_right_tracking', 'study_right_tracking.rviz'),
    ])

    return LaunchDescription(
        args + [gazebo, controllers, base_node, robot_model_filter, rviz_group])
