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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

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
        self.kin.update_joint_states(list(msg.name), list(msg.position))

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
        points_optical, colors, stamp = cloud
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        # Snapshot the camera pose so a concurrent FK can't mutate it mid-run.
        T_cam_base = self.T_cam_base

        result = self.pipeline.process(points_optical, colors, T_cam_base)
        self.latest_result = result

        # --- Publish PointCloud2 (cropped coloured cloud) --------------
        if result.cropped_points is not None and len(result.cropped_points) > 0:
            pc = make_pointcloud2(
                result.cropped_points, result.cropped_colors, cfg.BASE_FRAME, stamp
            )
            self.pub_cloud.publish(pc)

        # --- Publish markers -------------------------------------------
        cam_pos_base = T_cam_base.translation
        markers = self.viz.build(result, self.current_target, cam_pos_base, stamp)
        self.pub_markers.publish(markers)

    # ================================================================== #
    # Console report (low frequency — no per-tick spam)                   #
    # ================================================================== #
    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return

        r = self.latest_result
        aligned = "ALIGNED" if self.controller.is_aligned() else "slewing"
        head_line = (
            f"[HEAD] look-at err={self.controller.last_angle_deg:5.1f} deg ({aligned})"
        )

        if r is None:
            self.get_logger().info(head_line + " | perception: no frame yet")
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
            f"       [OBJECTS] {obj_txt}")


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
