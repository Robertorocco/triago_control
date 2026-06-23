#!/usr/bin/env python3
"""
main_head.py — TRIAGo head: look at the table & detect the cylinders.

WHAT IT DOES
    1. Moves the 7-DOF head so the camera fixates the table top (with a gentle
       Lissajous scan to cover the whole surface and average out depth noise).
    2. Runs a *geometric* (no-ML, no-install) perception pipeline on the
       RealSense RGB-D stream:
           crop -> RANSAC table plane -> above-plane clustering ->
           upright-cylinder fit -> red/blue colour classification.
    3. Visualises everything three ways:
           - RViz markers (table box + top plane + cylinders + labels + look ray)
           - RViz PointCloud2 (the cropped coloured cloud the algorithm sees)
           - a low-frequency console report (status + performance, NO spam)

ARCHITECTURE
    All heavy lifting lives in the triago_control.head_control library. This
    node only wires the pieces together and owns the ROS timers:
        * control timer    @ CONTROL_RATE_HZ    -> FK + look-at QP + publish dq
        * perception timer  @ PERCEPTION_RATE_HZ -> pipeline + viz publish
        * console timer     @ CONSOLE_SUMMARY    -> human-readable status line

    The control loop owns Pinocchio (FK each tick); perception consumes a stored
    *copy* of the camera pose, so the two never fight over the model state.

IF NOTHING HAPPENS (camera): the most likely cause is wrong topic names. Find
    the real ones with:   ros2 topic list | grep -i camera
    then run:
        ros2 run triago_control main_head.py --ros-args \
            -p color_topic:=/your/color/image_raw \
            -p depth_topic:=/your/aligned_depth/image_raw \
            -p camera_info_topic:=/your/color/camera_info
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
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

import triago_control.head_control.config as cfg
from triago_control.head_control.camera_interface import CameraInterface
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.look_at_controller import LookAtController
from triago_control.head_control.perception_pipeline import PerceptionPipeline
from triago_control.head_control.visualization import (
    PerceptionVisualizer,
    make_pointcloud2,
)


class HeadPerceptionNode(Node):
    def __init__(self):
        super().__init__("main_head")

        # --- Library components ---------------------------------------
        self.kin = HeadKinematics(self)
        self.camera = CameraInterface(self)        # declares topic params + subs
        self.controller = LookAtController(self.kin)
        self.pipeline = PerceptionPipeline()
        self.viz = PerceptionVisualizer(frame_id=cfg.BASE_FRAME)

        # --- Publishers ------------------------------------------------
        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f"/{cfg.HEAD_CONTROLLER}/joint_velocity_cmd", 10
        )
        self.pub_cloud = self.create_publisher(PointCloud2, "/head_perception/cloud", 1)
        self.pub_markers = self.create_publisher(MarkerArray, "/head_perception/markers", 1)
        # Scalar telemetry for the plotter: [n_raw, n_crop, plane_z, look_err_deg,
        # slack, proc_ms]. Lets the plotter show cloud size / quality directly.
        self.pub_telemetry = self.create_publisher(
            Float64MultiArray, "/head_perception/telemetry", 10
        )

        # --- TF2 (correct camera pose at the depth frame's timestamp) --
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_warned = False
        self._diag_logged = False

        # --- Subscriptions ---------------------------------------------
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)

        # --- Shared state (control -> perception) ----------------------
        self.T_cam_base = None
        self.J_cam = None
        self.start_time = time.time()
        self.current_target = cfg.TABLE_TOP_CENTER_BASE.copy()
        self.latest_result = None
        self._camera_warned = False

        # --- Timers ----------------------------------------------------
        self.create_timer(1.0 / cfg.CONTROL_RATE_HZ, self._control_tick)
        self.create_timer(1.0 / cfg.PERCEPTION_RATE_HZ, self._perception_tick)
        self.create_timer(cfg.CONSOLE_SUMMARY_PERIOD_S, self._console_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " TRIAGo HEAD — table look-at + geometric cylinder detection\n"
            "------------------------------------------------------------------\n"
            f"  Color topic : {self.camera.color_topic}\n"
            f"  Depth topic : {self.camera.depth_topic}\n"
            f"  Info  topic : {self.camera.info_topic}\n"
            f"  Table top   : z={cfg.TABLE_TOP_Z_WORLD:.2f} m  "
            f"centre={cfg.TABLE_CENTER_BASE[:2]} (base frame)\n"
            f"  Scan        : {'ON' if cfg.ENABLE_SCAN else 'OFF'}\n"
            "==================================================================")

    # ================================================================== #
    # Callbacks                                                           #
    # ================================================================== #
    def _joint_cb(self, msg: JointState):
        # Convert ROS stamp to float seconds for the EMA velocity filter.
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.kin.update_joint_states(list(msg.name), list(msg.position), stamp_sec)

    # ================================================================== #
    # Control loop                                                        #
    # ================================================================== #
    def _control_tick(self):
        if not self.kin.is_ready():
            return

        # FK once per tick; share with perception.
        self.T_cam_base, self.J_cam = self.kin.forward()

        # Look-at target (with optional scan), then solve the QP.
        t = time.time() - self.start_time
        self.current_target = self.controller.scan_target(t)
        dq = self.controller.compute(self.T_cam_base, self.J_cam, self.current_target)

        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub_head_cmd.publish(msg)

    # ================================================================== #
    # Perception loop                                                     #
    # ================================================================== #
    def _perception_tick(self):
        if self.T_cam_base is None:        # control not running yet
            return

        if not self.camera.has_data():
            if not self._camera_warned:
                self.get_logger().warn(
                    "Waiting for camera data... "
                    f"(color={self.camera.n_color}, depth={self.camera.n_depth}, "
                    f"info={self.camera.n_info}). If these stay 0, the topic names "
                    "are wrong — see the header of main_head.py.")
                self._camera_warned = True
            return

        cloud = self.camera.get_point_cloud()
        if cloud is None:
            return
        points_optical, colors, stamp, frame_id = cloud
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        if not frame_id:
            return

        # --- Correct transform: TF lookup of base <- depth_frame AT the depth
        # frame's timestamp. This fixes both (a) the frame mismatch (color vs
        # depth optical) and (b) the timing skew while the head moves. ----
        R_cam_base, t_cam_base = self._lookup_transform(frame_id, stamp)
        if R_cam_base is None:
            return

        # One-shot diagnostic: confirm camera placement & data shapes.
        if not self._diag_logged:
            self.get_logger().info(
                f"[DIAG] depth_frame='{frame_id}'  raw_pts={len(points_optical)}  "
                f"cam_pos_base={np.round(t_cam_base, 3)}")
            self._diag_logged = True

        result = self.pipeline.process(points_optical, colors, R_cam_base, t_cam_base)
        self.latest_result = result

        # --- Publish PointCloud2 (cropped coloured cloud) --------------
        if result.cropped_points is not None and len(result.cropped_points) > 0:
            pc = make_pointcloud2(
                result.cropped_points, result.cropped_colors, cfg.BASE_FRAME, stamp
            )
            self.pub_cloud.publish(pc)

        # --- Publish markers -------------------------------------------
        markers = self.viz.build(result, self.current_target, t_cam_base, stamp)
        self.pub_markers.publish(markers)

        # --- Publish scalar telemetry for the plotter ------------------
        tel = Float64MultiArray()
        n_crop = len(result.cropped_points) if result.cropped_points is not None else 0
        plane_z = result.plane.height if result.plane is not None else float("nan")
        tel.data = [
            float(result.n_raw), float(n_crop), float(plane_z),
            float(self.controller.last_angle_deg), float(self.controller.last_slack_norm),
            float(result.proc_ms),
        ]
        self.pub_telemetry.publish(tel)

    def _lookup_transform(self, frame_id, stamp):
        """Return (R 3x3, t 3) for base_footprint <- frame_id at `stamp`.

        Falls back to the latest available transform if the exact stamp is not
        yet buffered. Returns (None, None) if TF is unavailable.
        """
        for query in (Time.from_msg(stamp), Time()):  # try exact time, then latest
            try:
                tf = self.tf_buffer.lookup_transform(
                    cfg.BASE_FRAME, frame_id, query, timeout=Duration(seconds=0.05)
                )
                q = tf.transform.rotation
                t = tf.transform.translation
                R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                return R, np.array([t.x, t.y, t.z])
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                continue
        if not self._tf_warned:
            self.get_logger().warn(
                f"TF lookup base<-{frame_id} failed (is robot_state_publisher up?).")
            self._tf_warned = True
        return None, None

    # ================================================================== #
    # Console report (low frequency — no per-tick spam)                   #
    # ================================================================== #
    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return

        r = self.latest_result
        aligned = "ALIGNED" if self.controller.is_aligned() else "slewing"
        slack_info = f"slack={self.controller.last_slack_norm:.3f}"
        head_line = (
            f"[HEAD] look-at err={self.controller.last_angle_deg:5.1f} deg ({aligned}) {slack_info}"
        )

        # Show joint positions vs limits so we can see what's stuck.
        q = self.kin.get_head_joint_positions()
        q_min, q_max = self.kin.get_head_joint_limits()
        margin_lo = q - q_min
        margin_hi = q_max - q
        # Mark joints that are within 0.05 rad of a limit with [!]
        joint_info = " ".join(
            f"j{i+1}={'[!]' if min(margin_lo[i], margin_hi[i]) < 0.05 else ''}{q[i]:+.2f}"
            for i in range(len(q))
        )

        if r is None:
            self.get_logger().info(head_line + " | perception: no frame yet\n       [JOINTS] " + joint_info)
            return

        plane_txt = (
            f"plane z={r.plane.height:.3f} m" if r.plane is not None else "NO TABLE"
        )
        obj_txt = ", ".join(
            f"{o.label}@({o.center[0]:.2f},{o.center[1]:.2f},{o.center[2]:.2f}) "
            f"r={o.radius*100:.1f}cm h={o.height*100:.1f}cm"
            for o in r.objects
        ) or "none"

        self.get_logger().info(
            head_line + "\n"
            f"       [PERCEPTION] raw={r.n_raw} crop={len(r.cropped_points) if r.cropped_points is not None else 0} "
            f"| {plane_txt} | proc={r.proc_ms:.1f} ms\n"
            f"       [OBJECTS] {obj_txt}\n"
            f"       [JOINTS] {joint_info}")


def main():
    rclpy.init()
    node = HeadPerceptionNode()

    # --- Phase 1: build kinematics from the live URDF -----------------
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

    # --- Phase 2: take over the head velocity controller --------------
    node.kin.switch_controllers()

    node.get_logger().info("Setup complete. Spinning (Ctrl+C to stop).")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
