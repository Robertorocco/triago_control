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

RUN (three terminals, see cloud_relay_node.py for why the relay is needed):
    ros2 run triago_control cloud_relay_node.py
    ros2 run triago_control head_look_at_table.py
    ros2 run triago_control tabletop_perception_node.py

TABLE POSITION -- WORLD MISMATCH WARNING (found 2026-07-06):
    cfg.TABLE_TOP_CENTER_BASE is hardcoded from THIS repo's own
    `movement_tutorial.world` (table centre x=1.00, top z=0.70). If you are
    instead running the sibling Triago_Project stack's `handover` world (its
    own default per qp_controller_sim.yaml), that world's "picking_table" is
    at a DIFFERENT pose (centre x=0.80, top z=0.50) -- aiming at the config
    default would then report a low look-at error while pointing at EMPTY
    SPACE next to the real table, not the table itself.
    To avoid silently aiming at the wrong world's table, this node does NOT
    just trust cfg.TABLE_TOP_CENTER_BASE -- table_x/table_y/table_z are
    exposed as ROS params so you can point it at YOUR world's real table
    without editing config.py. Defaults below are z=0.8m (per an explicit
    request to scan from that height) with x/y taken from config.py's x/y --
    VERIFY these against the world you are actually running and override if
    they don't match:
        ros2 run triago_control head_look_at_table.py --ros-args \
            -p table_x:=0.80 -p table_y:=0.0 -p table_z:=0.8

STOPPING: on shutdown (Ctrl+C), the head velocity command is zeroed once so
it doesn't keep drifting after the process exits.
"""

import os
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

import triago_control.head_control.config as cfg
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.look_at_controller import LookAtController


class HeadLookAtTableNode(Node):
    def __init__(self):
        super().__init__("head_look_at_table")

        self.kin = HeadKinematics(self)
        self.controller = LookAtController(self.kin)

        # --- Table target override (see module docstring — world mismatch
        # warning). Defaults: x/y from config.py's known table centre, z=0.8m
        # per explicit request. Overriding scan_target with a fixed base
        # point means head_look_at_table.py's own scan sweeps AROUND this
        # point (LookAtController.scan_target adds its waypoint offsets to
        # whatever base point we hand it — see _control_tick below).
        table_x = self.declare_parameter("table_x", float(cfg.TABLE_CENTER_BASE[0])).value
        table_y = self.declare_parameter("table_y", float(cfg.TABLE_CENTER_BASE[1])).value
        table_z = self.declare_parameter("table_z", 0.8).value
        self.table_target_base = np.array([table_x, table_y, table_z])

        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f"/{cfg.HEAD_CONTROLLER}/joint_velocity_cmd", 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)

        # TF, purely for the diagnostic cross-check below (mirrors
        # main_head.py's own [XFORM] diagnostic) — helps confirm whether the
        # camera is actually pointed where FK thinks it is.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.start_time = time.time()
        self.current_target = self.table_target_base.copy()

        self.create_timer(1.0 / cfg.CONTROL_RATE_HZ, self._control_tick)
        self.create_timer(cfg.CONSOLE_SUMMARY_PERIOD_S, self._console_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " head_look_at_table — aiming camera at the table (no perception)\n"
            "------------------------------------------------------------------\n"
            f"  Table target : {np.round(self.table_target_base, 3)} (base frame)\n"
            f"  config.py's own table pose (movement_tutorial.world): "
            f"{np.round(cfg.TABLE_TOP_CENTER_BASE, 3)}\n"
            "  If you are running a DIFFERENT world (e.g. Triago_Project's\n"
            "  'handover'), verify table_x/table_y/table_z above match ITS\n"
            "  real table and override with --ros-args -p table_x:=... if not.\n"
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
        # Re-derive the scan waypoint offset from LookAtController (keeps the
        # scan pattern identical to main_head.py) but apply it around OUR
        # table target rather than cfg.TABLE_TOP_CENTER_BASE — see the
        # world-mismatch note in the module docstring.
        offset = self.controller.scan_target(t) - cfg.TABLE_TOP_CENTER_BASE
        self.current_target = self.table_target_base + offset
        dq = self.controller.compute(T_cam_base, J_cam, self.current_target)

        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub_head_cmd.publish(msg)

    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return
        aligned = "ALIGNED" if self.controller.is_aligned() else "slewing"
        diag = ""
        # FK-vs-TF cross-check (mirrors main_head.py) — if these disagree,
        # robot_state_publisher's TF is not reflecting the live head config,
        # which would make ANY downstream cloud transform wrong too.
        fk_R, fk_t = self.kin.get_frame_in_base(cfg.CAMERA_OPTICAL_FRAME)
        try:
            tf = self.tf_buffer.lookup_transform(
                cfg.BASE_FRAME, cfg.CAMERA_OPTICAL_FRAME, Time(),
                timeout=Duration(seconds=0.05))
            q = tf.transform.rotation
            t = tf.transform.translation
            tf_t = np.array([t.x, t.y, t.z])
            if fk_t is not None:
                diag = f"\n       [XFORM] TF cam={np.round(tf_t,3)}  FK cam={np.round(fk_t,3)}"
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            pass
        self.get_logger().info(
            f"[HEAD] look-at err={self.controller.last_angle_deg:5.1f} deg "
            f"({aligned}) slack={self.controller.last_slack_norm:.3f} "
            f"target={np.round(self.current_target, 3)}" + diag)

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
