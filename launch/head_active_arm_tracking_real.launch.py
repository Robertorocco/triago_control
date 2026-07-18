from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Measured mount transform (arm_head_tool_link -> the camera driver's own
    # TF root) -- same numbers as head_real.launch.py's static_tf and
    # head_active_arm_tracking.py's own _ensure_camera_frame() injection.
    # Independent of whatever name the camera driver itself publishes under
    # (PAL's default bringup, or a head_real.launch.py-style relaunch): this
    # only gives OTHER tooling (RViz, tf2_echo) a stable TF frame to look at.
    # head_active_arm_tracking.py's own kinematics never reads this TF frame
    # -- it injects the same offset directly into its Pinocchio model.
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

    # head_active_arm_tracking.py needs no camera image/depth topics at all --
    # it tracks the active teleop hand purely from /joint_states + FK. Unlike
    # head_real.launch.py, this deliberately does NOT launch or rename the
    # realsense driver: whatever camera is already running system-wide (e.g.
    # PAL's own default bringup) is left untouched, avoiding a device-busy
    # conflict from opening the same physical camera twice.
    tracker_node = Node(
        package='triago_control',
        executable='head_active_arm_tracking.py',
        name='head_active_arm_tracking',
        output='screen',
        parameters=[{
            'plot': False,
        }],
    )

    return LaunchDescription([static_tf, tracker_node])
