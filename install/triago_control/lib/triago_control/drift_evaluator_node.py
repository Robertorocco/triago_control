#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import TransformBroadcaster
import math
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def main(args=None):
    # Initializes ROS 2, spawns the executor thread, configures the GridSpec GUI, and blocks on the animation loop.
    rclpy.init(args=args)
    node = DriftEvaluatorNode()

    # ROS 2 spin must execute in a daemon thread; Matplotlib requires ownership of the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Setup Matplotlib Figure using GridSpec for the 4-subplot layout
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Rigorous Base Drift Evaluation (10s Window)')
    gs = GridSpec(3, 2, figure=fig)

    ax_x   = fig.add_subplot(gs[0, 0])
    ax_y   = fig.add_subplot(gs[1, 0])
    ax_yaw = fig.add_subplot(gs[2, 0])
    ax_xy  = fig.add_subplot(gs[:, 1])  # XY plane spans all rows on the right

    # Temporal Lines
    line_x,   = ax_x.plot([], [], 'r-', lw=2, label='X Drift [m]')
    line_y,   = ax_y.plot([], [], 'g-', lw=2, label='Y Drift [m]')
    line_yaw, = ax_yaw.plot([], [], 'b-', lw=2, label='Yaw Drift [deg]')
    
    temporal_axes = [ax_x, ax_y, ax_yaw]
    temporal_lines = [line_x, line_y, line_yaw]

    for ax in temporal_axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')

    ax_yaw.set_xlabel('Time [s]')

    # SE(2) XY Plane Setup
    ax_xy.grid(True, linestyle='--', alpha=0.7)
    ax_xy.set_title("SE(2) Phase Portrait")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect('equal', adjustable='datalim')
    
    # Plot the world origin as a static black dot
    ax_xy.plot(0.0, 0.0, 'ko', markersize=8, label='World Origin')
    
    # Placeholder for the robot base rectangle (we will update its vertices dynamically)
    base_rect, = ax_xy.plot([], [], 'c-', lw=2, label='TRIAGo Base')
    base_trajectory, = ax_xy.plot([], [], 'c--', lw=1, alpha=0.5)
    ax_xy.legend(loc='upper right')

    def _update_plot(frame, node, lines, axes, ax_xy, base_rect):
        # Dynamically updates temporal plots with 10^-2 clamping and rigidly transforms the SE(2) base footprint.
        if not node.times:
            return lines + [base_rect, base_trajectory]

        current_time = node.times[-1]
        window_start = max(0.0, current_time - 10.0)

        # 1. Update X, Y, Yaw Temporal Plots with mathematically bounded dynamic centering
        data_arrays = [node.x_data, node.y_data, node.yaw_data]
        min_half_ranges = [0.01, 0.01, 0.5] # 10^-2 m for translation, 0.5 deg for rotation

        for i, ax in enumerate(axes):
            lines[i].set_data(node.times, data_arrays[i])
            ax.set_xlim(window_start, window_start + 10.0)
            
            # Dynamic Y-Axis Clamping
            val_min, val_max = min(data_arrays[i]), max(data_arrays[i])
            center = (val_max + val_min) / 2.0
            actual_half_range = (val_max - val_min) / 2.0
            enforced_half_range = max(actual_half_range, min_half_ranges[i])
            
            ax.set_ylim(center - enforced_half_range, center + enforced_half_range)

        # 2. Update XY Plane
        base_trajectory.set_data(node.x_data, node.y_data)
        
        # Calculate rotated rectangle for the robot base footprint
        L, W = 0.54, 0.54  # Approximate physical dimensions of TRIAGo Pro omni-base
        x_t, y_t, yaw_t = node.x_data[-1], node.y_data[-1], math.radians(node.yaw_data[-1])
        
        # Local corners of the base
        dx = np.array([-L/2, L/2, L/2, -L/2, -L/2])
        dy = np.array([-W/2, -W/2, W/2, W/2, -W/2])
        
        # Apply 2D Rotation Matrix R(psi) and translate to (x_t, y_t)
        rx = x_t + dx * math.cos(yaw_t) - dy * math.sin(yaw_t)
        ry = y_t + dx * math.sin(yaw_t) + dy * math.cos(yaw_t)
        
        base_rect.set_data(rx, ry)
        
        # Autoscale XY plane around the trajectory
        ax_xy.relim()
        ax_xy.autoscale_view()

        return lines + [base_rect, base_trajectory]

    # Execute the live animation at 10 Hz
    ani = animation.FuncAnimation(
        fig, _update_plot, fargs=(node, temporal_lines, temporal_axes, ax_xy, base_rect), 
        interval=100, blit=False, save_count=50)

    fig.tight_layout()
    plt.show()

    # Graceful shutdown sequence
    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join(timeout=1.0)


class DriftEvaluatorNode(Node):
    def __init__(self):
        # Configures ROS 2 parameters, TF broadcasters, the Odometry subscription, and temporal memory buffers.
        super().__init__('drift_evaluator_live')

        ODOM_TOPIC = self.declare_parameter('odom_topic', '/mobile_base_controller/odom').value

        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._broadcast_world_anchor()
        self.dynamic_broadcaster = TransformBroadcaster(self)

        self.times = []
        self.x_data = []
        self.y_data = []
        self.yaw_data = []
        self.t0 = None

        self.odom_sub = self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_callback, 10)

        self.get_logger().info(f"Rigorous evaluation active on '{ODOM_TOPIC}'.")

    def _broadcast_world_anchor(self):
        # Broadcasts the identity matrix $I_{4 \times 4}$ as a permanent static anchor for the 'world' frame.
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id  = 'world'
        t.child_frame_id   = 'world_anchor'
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)

    def _odom_callback(self, msg: Odometry):
        # Extracts $SE(2)$ states from odometry, updates the TF tree, and aggressively prunes data older than 10 seconds.
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = msg.header.stamp
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id  = 'base_footprint'
        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation      = msg.pose.pose.orientation
        self.dynamic_broadcaster.sendTransform(tf_msg)

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_deg = math.degrees(math.atan2(siny, cosy))

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        current_time = time.time()
        if self.t0 is None:
            self.t0 = current_time
        
        t_elapsed = current_time - self.t0

        self.times.append(t_elapsed)
        self.x_data.append(x)
        self.y_data.append(y)
        self.yaw_data.append(yaw_deg)

        # Enforce strict 10-second memory pruning (assuming ~100Hz odom, buffer is roughly 1000 items)
        while self.times and (t_elapsed - self.times[0] > 10.0):
            self.times.pop(0)
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.yaw_data.pop(0)

if __name__ == '__main__':
    main()