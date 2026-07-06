#!/usr/bin/env python3
"""
head_look_at_table.py — aim the 7-DOF head camera at the table, with NO
perception pipeline attached.

WHY THIS EXISTS
    tabletop_perception_node.py (the port of triago_perception's C++ node)
    never commands the head — neither does its C++ original, nor anything
    else in the sibling `Triago_Project` repository (checked: the QP
    controller `qp_controller.cpp` always publishes an all-zero velocity
    command on the head topic on every code path; there is no head-aiming
    node anywhere in that stack). The only existing "point the camera at the
    table" implementation in this whole project is main_head.py's own
    control loop (HeadKinematics + LookAtController) — but main_head.py also
    runs its own upright-cylinder perception pipeline, which you don't want
    running concurrently with tabletop_perception_node.py (double camera
    processing, and both would fight over head_plotter.py's dashboard topics
    unless one is the bridge — see tabletop_perception_node.py's docstring).

    This script factors OUT just the control half of main_head.py: it reuses
    the exact same HeadKinematics + LookAtController classes (no re-derived
    QP, no copy-pasted control law) and runs ONLY the head-aiming loop, so it
    can be launched standalone alongside tabletop_perception_node.py.

WHAT IT DOES NOT DO
    * No camera subscription, no cloud processing, no markers, no telemetry.
    * No perception of any kind. It only moves the head.

RUN (two terminals, alongside a real /filtered_cloud source):
    ros2 run triago_control head_look_at_table.py
    ros2 run triago_control tabletop_perception_node.py

STOPPING: on shutdown (Ctrl+C), the head velocity command is zeroed once so
it doesn't keep drifting after the process exits.
"""

import os
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import triago_control.head_control.config as cfg
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.look_at_controller import LookAtController


class HeadLookAtTableNode(Node):
    def __init__(self):
        super().__init__("head_look_at_table")

        self.kin = HeadKinematics(self)
        self.controller = LookAtController(self.kin)

        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f"/{cfg.HEAD_CONTROLLER}/joint_velocity_cmd", 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)

        self.start_time = time.time()
        self.current_target = cfg.TABLE_TOP_CENTER_BASE.copy()

        self.create_timer(1.0 / cfg.CONTROL_RATE_HZ, self._control_tick)
        self.create_timer(cfg.CONSOLE_SUMMARY_PERIOD_S, self._console_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " head_look_at_table — aiming camera at the table (no perception)\n"
            "------------------------------------------------------------------\n"
            f"  Table top   : z={cfg.TABLE_TOP_Z_WORLD:.2f} m  "
            f"centre={cfg.TABLE_CENTER_BASE[:2]} (base frame)\n"
            f"  Scan        : {'ON' if cfg.ENABLE_SCAN else 'OFF'}\n"
            "==================================================================")

    def _joint_cb(self, msg: JointState):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.kin.update_joint_states(list(msg.name), list(msg.position), stamp_sec)

    def _control_tick(self):
        if not self.kin.is_ready():
            return
        T_cam_base, J_cam = self.kin.forward()
        t = time.time() - self.start_time
        self.current_target = self.controller.scan_target(t)
        dq = self.controller.compute(T_cam_base, J_cam, self.current_target)

        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub_head_cmd.publish(msg)

    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return
        aligned = "ALIGNED" if self.controller.is_aligned() else "slewing"
        self.get_logger().info(
            f"[HEAD] look-at err={self.controller.last_angle_deg:5.1f} deg "
            f"({aligned}) slack={self.controller.last_slack_norm:.3f} "
            f"target={np.round(self.current_target, 3)}")

    def stop(self):
        """Zero the head velocity command once, so it doesn't drift on exit."""
        msg = Float64MultiArray()
        msg.data = [0.0] * len(cfg.HEAD_JOINTS)
        self.pub_head_cmd.publish(msg)


def main():
    rclpy.init()
    node = HeadLookAtTableNode()

    node.get_logger().info("Fetching URDF from robot_state_publisher...")
    urdf_str = node.kin.fetch_urdf()
    if urdf_str is None:
        node.get_logger().error("No URDF — is robot_state_publisher running? Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".urdf") as f:
        f.write(urdf_str)
        urdf_path = f.name
    node.kin.build(urdf_path)
    os.remove(urdf_path)

    node.kin.switch_controllers()

    node.get_logger().info("Setup complete. Spinning (Ctrl+C to stop).")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
