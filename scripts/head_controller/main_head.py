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
    This node computes control/perception ONLY -- no matplotlib, no plot
    windows. For real-hardware debugging (world belief, confidence,
    convergence-over-time), run scripts/head_controller/head_debug_plotter.py
    separately on a machine with a display; it subscribes to this node's
    /head_perception/cloud + cfg.DEBUG_JSON_TOPIC, nothing more is needed here.

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

import json
import os
import sys
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Empty, String
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

import triago_control.head_control.config as cfg
from triago_control.head_control.camera_interface import CameraInterface
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.look_at_controller import LookAtController
from triago_control.head_control.perception_pipeline import PerceptionPipeline
from triago_control.head_control.world_convergence import (
    WorldConvergenceMonitor, PerceivedWorld, PerceivedCylinder,
)
from triago_control.head_control.visualization import (
    PerceptionVisualizer,
    build_world_snapshot_markers,
    make_pointcloud2,
)


class HeadPerceptionNode(Node):
    def __init__(self):
        super().__init__("main_head")

        # Real hardware (config §11): height-based classification + slower,
        # more distant scan. Resolve BEFORE building anything that reads cfg.
        self.declare_parameter("real_hardware_head", cfg.REAL_HARDWARE_HEAD)
        cfg.REAL_HARDWARE_HEAD = self.get_parameter("real_hardware_head").value
        if cfg.REAL_HARDWARE_HEAD:
            cfg.HEAD_POSTURE_TARGET = cfg.HEAD_POSTURE_TARGET_REAL
            cfg.MAX_HEAD_VELOCITY = cfg.MAX_HEAD_VELOCITY_REAL
            cfg.LOOKAT_LAMBDA = cfg.LOOKAT_LAMBDA_REAL
            cfg.ACTIVE_VISION_PLATEAU_PATIENCE = cfg.ACTIVE_VISION_PLATEAU_PATIENCE_REAL
            cfg.ACTIVE_VISION_TIME_BUDGET_S = cfg.ACTIVE_VISION_TIME_BUDGET_S_REAL
            cfg.SCAN_WAYPOINTS = cfg.SCAN_WAYPOINTS_REAL

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
        # Debug-namespaced qdot cmd/measured for offline_plotter.py (7 floats,
        # HEAD_JOINTS order), independent of the real hardware topic name.
        self.pub_qdot_cmd = self.create_publisher(
            Float64MultiArray, "/head_perception/qdot_cmd", 10
        )
        self.pub_qdot_measured = self.create_publisher(
            Float64MultiArray, "/head_perception/qdot_measured", 10
        )
        self.pub_cloud = self.create_publisher(PointCloud2, "/head_perception/cloud", 1)
        self.pub_raw_cloud = self.create_publisher(PointCloud2, "/head_perception/raw_cloud", 1)
        # Above-plane candidate points ONLY (the exact input to clustering/
        # cylinder-fit) -- lets head_debug_plotter.py show what the detector
        # actually tries to fit a cylinder to, vs. what it decided is one.
        self.pub_above_cloud = self.create_publisher(PointCloud2, "/head_perception/above_cloud", 1)
        self.pub_markers = self.create_publisher(MarkerArray, "/head_perception/markers", 1)
        # Scalar telemetry for the plotter: [n_raw, n_crop, plane_z, look_err_deg,
        # slack, proc_ms]. Lets the plotter show cloud size / quality directly.
        self.pub_telemetry = self.create_publisher(
            Float64MultiArray, "/head_perception/telemetry", 10
        )
        # Real hardware only: full-detail JSON for head_debug_plotter.py (run
        # separately, on a machine with a display) -- see config.py §11.
        self.pub_debug_json = self.create_publisher(String, cfg.DEBUG_JSON_TOPIC, 10)
        # --- Perceived-world snapshot (camera estimate -> QP-CLF-CBF) ---------
        # LATCHED (TRANSIENT_LOCAL): published ONCE the estimate converges, so a
        # perceived-world QP controller started AFTER convergence still receives
        # the last snapshot on subscribe. See world_convergence.py / config §16.
        self.world_monitor = WorldConvergenceMonitor()
        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub_world = self.create_publisher(
            MarkerArray, cfg.PERCEIVED_WORLD_TOPIC, latched_qos
        )
        # Real hardware: same markers on a RViz-facing latched topic (config §16).
        self.pub_real_perception = self.create_publisher(
            MarkerArray, cfg.REAL_PERCEPTION_TOPIC, latched_qos
        )
        # Re-arm: an empty message re-observes and re-publishes a fresh snapshot.
        self.create_subscription(
            Empty, cfg.PERCEIVED_WORLD_RESCAN_TOPIC, self._rescan_cb, 1
        )

        # --- TF2 (correct camera pose at the depth frame's timestamp) --
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_warned = False
        self._diag_logged = False

        # --- Subscriptions ---------------------------------------------
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)

        # --- Depth optical-centre frame (SIM vs REAL — see config §1) --
        # Same fix as head_april_main: Gazebo renders from the camera LINK but
        # tags the depth image as the depth-optical frame, offset by ~2.5 cm.
        # Use the link position + optical orientation to remove the sim bias.
        self.declare_parameter("depth_center_frame", cfg.DEPTH_OPTICAL_CENTER_FRAME)
        self._depth_center_frame = self.get_parameter("depth_center_frame").value
        self._center_warned = False

        # --- Shared state (control -> perception) ----------------------
        self.T_cam_base = None
        self.J_cam = None
        self.start_time = time.time()
        self.current_target = cfg.TABLE_TOP_CENTER_BASE.copy()
        self.latest_result = None
        # Last TF-derived camera pose (for the FK-vs-TF cross-check diagnostic).
        self._last_tf_pos = None
        self._last_tf_R = None
        self._last_depth_frame = None
        self._last_vel_norm = 0.0
        self._last_integrated = False
        # Post-convergence: print the final QP payload once, then go quiet.
        self._converged_snapshot = None
        self._final_summary_printed = False

        # Real-hardware two-pose head motion (config §11): Phase 1 (standard
        # framing) -> Phase 2 (cylinder-midpoint close view), permanent once
        # entered. See _real_hardware_target/_try_enter_real_phase2.
        self._real_phase_start_t = time.time()
        self._real_phase2_active = False
        self._real_phase2_target = None
        self._real_retry_t = 0.0

        # Real hardware: the world is loaded MANUALLY (press ENTER), not on auto-
        # convergence -- a safety gate so the operator decides when the estimate
        # is handed to the CBF. A daemon stdin thread flips this flag; a timer on
        # the executor thread does the actual freeze+publish.
        self._load_requested = False
        self._console_state = None       # 'acquiring' | 'ready' (announce once per change)
        if cfg.REAL_HARDWARE_HEAD:
            threading.Thread(target=self._stdin_loop, daemon=True).start()
            self.create_timer(0.2, self._console_load_tick)
            # Launch-friendly trigger: stdin ENTER doesn't reach a node started by
            # `ros2 launch`, so also accept an Empty message (see config §16).
            self.create_subscription(
                Empty, cfg.PERCEIVED_WORLD_LOAD_TOPIC, self._load_topic_cb, 1)
            # Keep the RViz world topic alive at 1 Hz so a DEFAULT (volatile) RViz
            # MarkerArray display shows it -- a latched publish alone is only seen
            # by transient-local subscribers (the CBF/autonomy consumers).
            self.create_timer(1.0, self._republish_real_perception)

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
            f"  Centre frm  : {self._depth_center_frame} (depth optical-centre position)\n"
            f"  Head motion : {'ACTIVE VISION (perception-led)' if cfg.ENABLE_ACTIVE_VISION else ('fixed scan' if cfg.ENABLE_SCAN else 'fixed look-at')}\n"
            + ("  LOAD WORLD  : freeze + publish to CBF + autonomy (+ RViz) via\n"
               "                ENTER here (ros2 run only), OR any time:\n"
               "                ros2 topic pub --once /perceived_world/load "
               "std_msgs/msg/Empty \"{}\"\n"
               if cfg.REAL_HARDWARE_HEAD else "")
            + "==================================================================")

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

        # Real hardware: exactly two fixed dwell postures (config §11), no
        # scanning -- bypasses the active-vision SWEEP/REFINE/HOLD state
        # machine entirely. Sim is untouched (still the full active-vision
        # system, perception-driven look-at target / fixed scan fallback).
        if cfg.REAL_HARDWARE_HEAD:
            self.current_target = self._real_hardware_target(self.latest_result)
        else:
            t = time.time() - self.start_time
            self.current_target = self.controller.active_vision_target(t, self.latest_result)

        dq = self.controller.compute(self.T_cam_base, self.J_cam, self.current_target)

        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub_head_cmd.publish(msg)

        # Debug duplicates for offline_plotter.py (see the publisher comment above).
        self.pub_qdot_cmd.publish(Float64MultiArray(data=[float(x) for x in dq]))
        v_measured = self.kin.get_head_joint_velocities()
        self.pub_qdot_measured.publish(
            Float64MultiArray(data=[float(x) for x in v_measured]))

    # ================================================================== #
    # Perception loop                                                     #
    # ================================================================== #
    def _perception_tick(self):
        if self.T_cam_base is None:        # control not running yet
            return

        if not self.camera.has_data():
            self.get_logger().warn(
                "Waiting for camera data... "
                f"(color={self.camera.n_color}, depth={self.camera.n_depth}, "
                f"info={self.camera.n_info}). If these stay 0 for a while, either "
                "the topic names are wrong or the camera driver isn't running on "
                "this machine (see the header of main_head.py).",
                throttle_duration_sec=10.0)
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
        # depth optical) and (b) the timing skew while the head moves.
        # ORIENTATION from the depth-optical frame; POSITION from the true
        # optical centre (camera link in sim — Gazebo renders there, see §7.2).
        R_cam_base, t_cam_base = self._lookup_transform(frame_id, stamp)
        if R_cam_base is None:
            return

        if self._depth_center_frame and self._depth_center_frame != frame_id:
            _, t_center = self._lookup_transform(self._depth_center_frame, stamp)
            if t_center is not None:
                t_cam_base = t_center       # link position, optical orientation
            elif not self._center_warned:
                self.get_logger().warn(
                    f"depth_center_frame '{self._depth_center_frame}' not in TF; "
                    f"falling back to '{frame_id}' position (expect ~cm sim offset).")
                self._center_warned = True
        self._last_tf_pos = t_cam_base
        self._last_tf_R = R_cam_base
        self._last_depth_frame = frame_id

        # One-shot diagnostic: confirm camera placement & data shapes.
        if not self._diag_logged:
            self.get_logger().info(
                f"[DIAG] depth_frame='{frame_id}'  raw_pts={len(points_optical)}  "
                f"cam_pos_base={np.round(t_cam_base, 3)}")
            self._diag_logged = True

        # Publish the FULL raw cloud (transformed to base) so you can SEE in
        # RViz where the points actually land relative to the robot model.
        raw_base = points_optical @ R_cam_base.T + t_cam_base
        raw_pc = make_pointcloud2(
            raw_base.astype(np.float32), colors, cfg.BASE_FRAME, stamp
        )
        self.pub_raw_cloud.publish(raw_pc)

        # Snapshot the camera pose so a concurrent FK can't mutate it mid-run.
        # Velocity-gate accumulation: only fuse when the head is settled, else
        # a moving head smears the fused map (NO TABLE / stretched cylinders).
        head_vel = self.kin.get_head_joint_velocities()
        vel_norm = float(np.linalg.norm(head_vel))
        allow_integrate = vel_norm < cfg.INTEGRATE_VEL_THRESH
        self._last_vel_norm = vel_norm
        self._last_integrated = allow_integrate

        result = self.pipeline.process(
            points_optical, colors, R_cam_base, t_cam_base,
            allow_integrate=allow_integrate, allow_track_update=allow_integrate,
            explore_phase=self.controller.phase,
        )
        self.latest_result = result

        # Freeze + publish the snapshot once confidence, coverage, and drift
        # rate all clear threshold over a settled window (world_convergence.py).
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9
        snapshot = self.world_monitor.update(
            result, allow_update=allow_integrate, stamp_sec=stamp_sec)
        # Sim auto-publishes on convergence; real hardware waits for the manual
        # ENTER trigger (see _console_load_tick) so the operator gates the load.
        if snapshot is not None and not cfg.REAL_HARDWARE_HEAD:
            self._publish_world_snapshot(snapshot, stamp)

        # --- Publish PointCloud2 (cropped coloured cloud) --------------
        if result.cropped_points is not None and len(result.cropped_points) > 0:
            pc = make_pointcloud2(
                result.cropped_points, result.cropped_colors, cfg.BASE_FRAME, stamp
            )
            self.pub_cloud.publish(pc)

        # Above-plane candidate points (real-hardware debug: what gets clustered).
        if (cfg.REAL_HARDWARE_HEAD and result.above_points is not None
                and len(result.above_points) > 0):
            above_cols = (result.above_colors if result.above_colors is not None
                          else np.zeros((len(result.above_points), 3), dtype=np.uint8))
            self.pub_above_cloud.publish(make_pointcloud2(
                result.above_points.astype(np.float32), above_cols, cfg.BASE_FRAME, stamp))

        # --- Publish markers -------------------------------------------
        markers = self.viz.build(result, self.current_target, t_cam_base, stamp)
        self.pub_markers.publish(markers)

        # --- Publish scalar telemetry for the plotter ------------------
        tel = Float64MultiArray()
        n_crop = len(result.cropped_points) if result.cropped_points is not None else 0
        plane_z = result.plane.height if result.plane is not None else float("nan")
        # Per-colour confidence (so the plotter can show estimation quality).
        red_conf = next((o.confidence for o in result.objects if o.color_name == "red"), 0.0)
        blue_conf = next((o.confidence for o in result.objects if o.color_name == "blue"), 0.0)
        # [9]=converge_progress [10]=converged [11]=phase (0 SWEEP/1 REFINE/2 HOLD)
        # [12]=weakest arc coverage [13]=drift rate m/s (-1=NaN) [14]=head qdot norm
        drift_rate = self.world_monitor.max_drift_rate
        tel.data = [
            float(result.n_raw), float(n_crop), float(plane_z),
            float(self.controller.last_angle_deg), float(self.controller.last_slack_norm),
            float(result.proc_ms), float(red_conf), float(blue_conf),
            float(result.map_size),
            float(self.world_monitor.converge_progress(stamp_sec)),
            1.0 if self.world_monitor.converged else 0.0,
            float(self.controller.phase),
            float(self.controller.weakest_coverage),
            float(drift_rate) if np.isfinite(drift_rate) else -1.0,
            float(self._last_vel_norm),
        ]
        self.pub_telemetry.publish(tel)

        # --- Real hardware only: full-detail JSON for head_debug_plotter.py --
        if cfg.REAL_HARDWARE_HEAD:
            # Same red=right/blue=left side convention a LOAD would apply (see
            # _rank_by_side) -- overrides the live per-object height-based
            # colour_name so the debug view never disagrees with what gets
            # loaded to the CBF. id(o)-keyed since TrackedObject isn't hashable
            # by value and result.objects is already <= WORLD_EXPECTED_CYLINDERS.
            side_color = {id(o): c for o, c in
                          zip(self._rank_by_side(result.objects), ["red", "blue"])}
            payload = {
                "stamp": stamp_sec,
                "phase": {0: "SWEEP", 1: "REFINE", 2: "HOLD"}.get(self.controller.phase, "?"),
                "look_err_deg": self.controller.last_angle_deg,
                "aligned": self.controller.is_aligned(),
                "slack_norm": self.controller.last_slack_norm,
                "n_raw": result.n_raw,
                "n_crop": n_crop,
                "plane_z": result.plane.height if result.plane is not None else None,
                "table_center": result.table_center.tolist() if result.table_center is not None else None,
                "table_size": result.table_size.tolist() if result.table_size is not None else None,
                "static_prior_center": cfg.TABLE_CENTER_BASE.tolist(),
                "static_prior_size": cfg.TABLE_SIZE.tolist(),
                "gate_bounds": (self.pipeline.table_xy_bounds.tolist()
                                if self.pipeline.table_xy_bounds is not None else None),
                "ee_x_cutoff": self.pipeline.ee_x_cutoff,
                "converge_progress": self.world_monitor.converge_progress(stamp_sec),
                "converged": self.world_monitor.converged,
                "drift_rate": float(drift_rate) if np.isfinite(drift_rate) else None,
                "objects": [
                    {
                        "id": o.id,
                        "label": f"{side_color.get(id(o), o.color_name)}_cylinder",
                        "color_name": side_color.get(id(o), o.color_name),
                        "center": o.center.tolist(), "radius": o.radius, "height": o.height,
                        "arc_coverage": o.arc_coverage, "vertical_coverage": o.vertical_coverage,
                        "confidence": o.confidence,
                        "fit_rms": o.best_fit_rms, "frames_unseen": o.frames_unseen,
                    }
                    for o in result.objects
                ],
            }
            self.pub_debug_json.publish(String(data=json.dumps(payload)))

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
    # Perceived-world snapshot (camera estimate -> QP-CLF-CBF)            #
    # ================================================================== #
    # ================================================================== #
    # Manual world load (real hardware: press ENTER to freeze + publish)  #
    # ================================================================== #
    def _stdin_loop(self):
        """Daemon thread: block on stdin; each ENTER requests a world load."""
        while rclpy.ok():
            line = sys.stdin.readline()
            if line == "":          # EOF (stdin closed) -> stop watching
                break
            self._load_requested = True

    def _console_load_tick(self):
        """Executor-thread poll of the ENTER flag -> freeze + publish the world."""
        if not self._load_requested:
            return
        self._load_requested = False
        self._load_world_from_console()

    def _load_topic_cb(self, _msg):
        """Launch-friendly equivalent of pressing ENTER (config §16)."""
        self._load_requested = True

    def _republish_real_perception(self):
        """1 Hz keep-alive of the RViz world topic (re-stamped so it stays fresh
        for any Fixed Frame). Machine consumers use the once-latched
        /perceived_world/snapshot instead; this is purely for the display."""
        if self._converged_snapshot is None:
            return
        stamp = self.get_clock().now().to_msg()
        markers = build_world_snapshot_markers(
            self._converged_snapshot, cfg.BASE_FRAME, stamp,
            radius_inflation=cfg.CYL_RADIUS_INFLATION)
        self.pub_real_perception.publish(markers)

    @staticmethod
    def _rank_by_side(objs):
        """Confidence-first (keep the WORLD_EXPECTED_CYLINDERS most confident
        tracks, dropping spurious extras), THEN sorted by Y for a side
        assignment: red = RIGHT (more-negative Y), blue = LEFT (+Y), matching
        base_footprint's +Y=left (REP-103, verified against this robot's own
        URDF: arm_right sits at Y=-0.159, arm_left at Y=+0.159). Used for BOTH
        the world-load snapshot AND the live debug-plotter telemetry, so what
        you watch live never disagrees with what a LOAD would produce."""
        ranked = list(objs)
        if len(ranked) > cfg.WORLD_EXPECTED_CYLINDERS:
            ranked = sorted(ranked, key=lambda o: o.confidence, reverse=True)
            ranked = ranked[:cfg.WORLD_EXPECTED_CYLINDERS]
        return sorted(ranked, key=lambda o: float(o.center[1]))

    def _perceived_world_from_result(self, result):
        """Freeze the CURRENT estimate into a PerceivedWorld. Forces exactly one
        'red' + one 'blue' cylinder -- the real objects are black/white, but
        downstream (CBF + shared autonomy) keys on the red/blue slots. Side
        convention: see _rank_by_side."""
        if result is None or result.table_center is None or result.table_size is None:
            return None
        objs = self._rank_by_side(result.objects)
        colors = ["red", "blue"]                                # right=red, left=blue
        cyls = [
            PerceivedCylinder(
                color_name=colors[i], center=np.asarray(o.center, float).copy(),
                radius=float(o.radius), height=float(o.height),
                confidence=float(getattr(o, "confidence", 0.0)))
            for i, o in enumerate(objs)
        ]
        return PerceivedWorld(
            table_center=np.asarray(result.table_center, float).copy(),
            table_size=np.asarray(result.table_size, float).copy(),
            cylinders=cyls)

    def _load_world_from_console(self):
        """Build the perceived world from the latest estimate and latch-publish it
        on BOTH the machine contract (CBF + shared autonomy) and the RViz topic."""
        world = self._perceived_world_from_result(self.latest_result)
        if world is None or not world.cylinders:
            self.get_logger().warn(
                "[LOAD] No table + object estimate yet -- aim at the table and "
                "wait for detections before pressing ENTER.")
            return
        stamp = self.get_clock().now().to_msg()
        markers = build_world_snapshot_markers(
            world, cfg.BASE_FRAME, stamp, radius_inflation=cfg.CYL_RADIUS_INFLATION)
        self.pub_world.publish(markers)              # -> CBF + shared autonomy
        self.pub_real_perception.publish(markers)    # -> RViz
        self._converged_snapshot = world
        summary = ", ".join(
            f"{c.color_name} r={c.radius*100:.1f}cm h={c.height*100:.1f}cm "
            f"@({c.center[0]:.2f},{c.center[1]:.2f},{c.center[2]:.2f})"
            for c in world.cylinders)
        self.get_logger().info(
            "\n============================================================\n"
            " WORLD LOADED (manual) -- latched to the safety controller.\n"
            f"   topics: {cfg.PERCEIVED_WORLD_TOPIC} (CBF/autonomy) + "
            f"{cfg.REAL_PERCEPTION_TOPIC} (RViz)\n"
            f"   table  centre={np.round(world.table_center,3)} "
            f"size={np.round(world.table_size,3)}\n"
            f"   objects: {summary}\n"
            " Press ENTER again to re-load with the latest estimate.\n"
            "============================================================")

    def _rescan_cb(self, _msg):
        """Re-arm the convergence monitor so it re-observes and re-publishes."""
        self.world_monitor.reset()
        self.pipeline.reset_table_footprint()
        self._final_summary_printed = False   # resume console until it re-converges
        self._converged_snapshot = None
        # Fresh Phase 1 -> Phase 2 cycle too, in case the table/cylinders moved.
        if cfg.REAL_HARDWARE_HEAD:
            cfg.HEAD_POSTURE_TARGET = cfg.HEAD_POSTURE_TARGET_REAL
            self._real_phase_start_t = time.time()
        self._real_phase2_active = False
        self._real_phase2_target = None
        self._real_retry_t = 0.0
        self.get_logger().info(
            "[PerceivedWorld] Re-arm requested — re-observing; will re-publish "
            "the snapshot once the estimate re-converges.")

    def _publish_world_snapshot(self, snapshot, stamp):
        """Build + latch-publish the converged perceived world for the QP stack."""
        self._converged_snapshot = snapshot   # kept for the final console summary
        markers = build_world_snapshot_markers(
            snapshot, cfg.BASE_FRAME, stamp,
            radius_inflation=cfg.CYL_RADIUS_INFLATION,
        )
        self.pub_world.publish(markers)
        summary = ", ".join(
            f"{c.color_name} r={c.radius * 100:.1f}cm h={c.height * 100:.1f}cm "
            f"@({c.center[0]:.2f},{c.center[1]:.2f},{c.center[2]:.2f}) "
            f"conf={c.confidence * 100:.0f}%"
            for c in snapshot.cylinders
        ) or "none"
        self.get_logger().info(
            "\n============================================================\n"
            " PERCEIVED WORLD CONVERGED — latched snapshot published on\n"
            f"   {cfg.PERCEIVED_WORLD_TOPIC}\n"
            f"   table  centre={np.round(snapshot.table_center, 3)} "
            f"size={np.round(snapshot.table_size, 3)}\n"
            f"   objects: {summary}\n"
            " The perceived-world QP controller can now build its CBF from this.\n"
            "------------------------------------------------------------\n"
            " [DIAGNOSTIC, sim-only -- compares the FROZEN snapshot against the\n"
            "  known Gazebo GT (config.py §14); the estimator itself never reads\n"
            "  GT_*, this is purely 'how accurate was the world we just handed\n"
            "  to the safety controller']\n"
            f"{self._gt_diagnostic_summary(snapshot)}\n"
            "============================================================")

    def _gt_diagnostic_summary(self, snapshot) -> str:
        """DIAGNOSTIC ONLY (sim ground truth, config.py §14): how close is the
        FROZEN snapshot -- the exact numbers that became the CBF -- to the known
        answer? Never fed back into estimation, convergence, or the CBF itself."""
        gt_by_color = {
            "red": (cfg.GT_RED_CENTER, cfg.GT_RED_RADIUS, cfg.GT_RED_HEIGHT),
            "blue": (cfg.GT_BLUE_CENTER, cfg.GT_BLUE_RADIUS, cfg.GT_BLUE_HEIGHT),
        }
        lines = []
        for c in snapshot.cylinders:
            gt = gt_by_color.get(c.color_name)
            if gt is None:
                continue
            gt_center, gt_radius, gt_height = gt
            pos_err_cm = float(np.linalg.norm(c.center - gt_center)) * 100.0
            r_err_cm = (c.radius - gt_radius) * 100.0
            h_err_cm = (c.height - gt_height) * 100.0
            lines.append(
                f"   {c.color_name:5s} cylinder: pos_err={pos_err_cm:5.2f} cm  "
                f"radius_err={r_err_cm:+5.2f} cm  height_err={h_err_cm:+5.2f} cm")

        tpos_err_cm = float(np.linalg.norm(snapshot.table_center - cfg.TABLE_CENTER_WORLD)) * 100.0
        tsize_err_cm = (snapshot.table_size - cfg.TABLE_SIZE) * 100.0
        lines.append(
            f"   table          : pos_err={tpos_err_cm:5.2f} cm  "
            f"size_err=[x{tsize_err_cm[0]:+.2f}, y{tsize_err_cm[1]:+.2f}, "
            f"z{tsize_err_cm[2]:+.2f}] cm")
        return "\n".join(lines)

    def _print_final_summary(self):
        """Print the EXACT data handed to the QP-CLF-CBF stack (the frozen snapshot),
        once, in a compact shareable block. Console goes quiet after this."""
        snap = self._converged_snapshot
        if snap is None:
            return
        lines = [
            "\n========================================================================",
            " FINAL PERCEIVED WORLD  (frozen snapshot sent to the QP-CLF-CBF stack)",
            "------------------------------------------------------------------------",
            f"   table   : center=[{snap.table_center[0]:.4f}, {snap.table_center[1]:.4f}, "
            f"{snap.table_center[2]:.4f}] m   size=[{snap.table_size[0]:.4f}, "
            f"{snap.table_size[1]:.4f}, {snap.table_size[2]:.4f}] m",
        ]
        for c in snap.cylinders:
            lines.append(
                f"   {c.color_name:5s} : center=[{c.center[0]:.4f}, {c.center[1]:.4f}, "
                f"{c.center[2]:.4f}] m   radius={c.radius*100:.2f} cm   "
                f"height={c.height*100:.2f} cm   conf={c.confidence*100:.0f}%")
        lines.append(
            "------------------------------------------------------------------------")
        lines.append(" [sim-only accuracy vs Gazebo GT — estimator never reads GT]:")
        lines.append(self._gt_diagnostic_summary(snap))
        lines.append(
            " Console now quiet. Re-arm with:  ros2 topic pub --once "
            f"{cfg.PERCEIVED_WORLD_RESCAN_TOPIC} std_msgs/Empty '{{}}'")
        lines.append(
            "========================================================================")
        self.get_logger().info("\n".join(lines))

    # ================================================================== #
    # Console report (low frequency — no per-tick spam)                   #
    # ================================================================== #
    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return

        # Real hardware: manual (ENTER/topic) load, so route here BEFORE the
        # sim auto-convergence summary (which would otherwise fire + print sim GT).
        if cfg.REAL_HARDWARE_HEAD:
            self._console_tick_real(self.latest_result)
            return

        # Frozen: print the QP payload once, then stop the periodic status log.
        if self.world_monitor.converged:
            if not self._final_summary_printed:
                self._print_final_summary()
                self._final_summary_printed = True
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
            f"r={o.radius*100:.1f}cm h={o.height*100:.1f}cm "
            f"[cov={o.arc_coverage*100:.0f}% conf={o.confidence*100:.0f}%]"
            for o in r.objects
        ) or "none"

        # --- Decisive diagnostic: where does TF say the camera is, vs Pinocchio
        # FK for the SAME depth frame? If they disagree, robot_state_publisher's
        # TF is not reflecting the live head config (= the transform bug). Also
        # show the detected table-plane centroid: it should be near (1.0, 0.0).
        diag = ""
        if self._last_depth_frame is not None and self._last_tf_pos is not None:
            fk_R, fk_t = self.kin.get_frame_in_base(self._last_depth_frame)
            tf_t = self._last_tf_pos
            if fk_t is not None:
                tf_rpy = np.degrees(Rot.from_matrix(self._last_tf_R).as_euler("xyz"))
                fk_rpy = np.degrees(Rot.from_matrix(fk_R).as_euler("xyz"))
                diag += (f"\n       [XFORM] TF cam={np.round(tf_t,3)} rpy={np.round(tf_rpy,1)}  "
                         f"FK cam={np.round(fk_t,3)} rpy={np.round(fk_rpy,1)}")
            else:
                diag += f"\n       [XFORM] TF cam={np.round(tf_t,3)}  (FK: frame not in model)"
        if r.plane_centroid is not None:
            diag += (f"\n       [PLANE-CENTROID] {np.round(r.plane_centroid,3)} "
                     f"(expect ~[1.0, 0.0, 0.70])")

        # Active-vision + convergence status (perception-driven head motion).
        phase_txt = {0: "SWEEP", 1: "REFINE", 2: "HOLD"}.get(self.controller.phase, "?")
        drift = self.world_monitor.max_drift_rate
        drift_txt = f"{drift*1000:.1f}mm/s" if np.isfinite(drift) else "n/a"
        conv_txt = ("CONVERGED" if self.world_monitor.converged
                    else f"{self.world_monitor.converge_progress():.0%} window")
        avision = (f"\n       [AVISION] phase={phase_txt} "
                   f"weakest_cov={self.controller.weakest_coverage*100:.0f}% "
                   f"drift={drift_txt} conv={conv_txt}")

        self.get_logger().info(
            head_line + "\n"
            f"       [PERCEPTION] raw={r.n_raw} crop={len(r.cropped_points) if r.cropped_points is not None else 0} "
            f"map={r.map_size} | {plane_txt} | proc={r.proc_ms:.1f} ms | "
            f"head_vel={self._last_vel_norm:.3f} {'FUSING' if self._last_integrated else 'moving'}\n"
            f"       [OBJECTS] {obj_txt}\n"
            f"       [JOINTS] {joint_info}" + avision + diag)

    # ================================================================== #
    # Real-hardware console (quiet: announce readiness ONCE, then wait)    #
    # ================================================================== #
    def _console_tick_real(self, r):
        """Real hardware: no per-tick status spam. Print ONE line when the
        estimate becomes ready-to-load (table + expected objects + aligned),
        then stay silent until the operator loads the world. Live diagnostics
        live in head_debug_plotter.py."""
        if self._converged_snapshot is not None:
            return                                  # world loaded -> quiet

        n_obj = len(r.objects) if r is not None else 0
        ready = (r is not None and r.plane is not None
                 and n_obj >= cfg.WORLD_EXPECTED_CYLINDERS
                 and self.controller.is_aligned())
        state = "ready" if ready else "acquiring"
        if state == self._console_state:
            return                                  # no change -> stay silent
        self._console_state = state
        if ready:
            self.get_logger().info(
                f"\n[READY] Table + {n_obj} object(s) detected, head aligned.\n"
                "        Waiting for your LOAD command:\n"
                f"          ros2 topic pub --once {cfg.PERCEIVED_WORLD_LOAD_TOPIC} "
                "std_msgs/msg/Empty \"{}\"\n"
                "        (or press ENTER if main_head was started with `ros2 run`).")
        else:
            self.get_logger().info(
                "[ACQUIRING] Aiming the head + detecting the table/objects... "
                "will report when ready to load.")

    # ================================================================== #
    # Real-hardware two-pose head motion (no scanning)                     #
    # ================================================================== #
    def _real_hardware_target(self, result):
        """Real hardware ONLY (config §11): exactly two fixed dwell postures,
        no scanning, no per-object refine loop -- bypasses
        LookAtController.active_vision_target()/scan_target() entirely (sim
        is untouched and still uses that full active-vision system).

          Phase 1 (t < REAL_PHASE1_DURATION_S): standard framing, aimed at the
            live table centroid (falls back to the static prior before the
            first estimate) -- roughly localizes the table + both cylinders.
          Phase 2 (t >= REAL_PHASE1_DURATION_S): ONE permanent switch to a
            closer view fixated on the MIDPOINT between the two detected
            cylinders. Retries every REAL_PHASE2_RETRY_COOLDOWN_S if not
            enough detections / unreachable yet -- never reverts once it
            succeeds (a prior dwell-then-revert design let continued fusion
            from the farther view re-converge the refined estimate back
            toward its old, less accurate values).

        controller.phase is driven manually here (0 in Phase 1, 2 in Phase 2)
        so perception_pipeline's table-footprint freeze (explore_phase==2)
        and the debug-plotter's phase readout both still mean what they
        always did, even though the SWEEP/REFINE state machine is bypassed.
        """
        if self._real_phase2_active:
            self.controller.phase = 2
            return self._real_phase2_target

        self.controller.phase = 0
        t = time.time() - self._real_phase_start_t
        if (cfg.ENABLE_CLOSE_INSPECT and t >= cfg.REAL_PHASE1_DURATION_S
                and t - self._real_retry_t >= cfg.REAL_PHASE2_RETRY_COOLDOWN_S):
            self._real_retry_t = t
            self._try_enter_real_phase2(result)
            if self._real_phase2_active:
                self.controller.phase = 2
                return self._real_phase2_target

        if result is not None and result.table_center is not None:
            return np.asarray(result.table_center, dtype=float)
        return cfg.TABLE_TOP_CENTER_BASE.copy()

    def _try_enter_real_phase2(self, result):
        """Attempt the one-time, permanent switch to the cylinder-midpoint
        closer view. No-op (retried later, see REAL_PHASE2_RETRY_COOLDOWN_S)
        if fewer than 2 cylinders are seen yet or the posture is unreachable
        -- never forces an unsafe/unreachable posture."""
        objs = self._rank_by_side(result.objects) if result is not None else []
        if len(objs) < 2:
            self.get_logger().warn(
                "[PHASE2] Fewer than 2 cylinders detected yet -- will retry.")
            return
        c0 = np.asarray(objs[0].center, dtype=float)
        c1 = np.asarray(objs[1].center, dtype=float)
        midpoint = 0.5 * (c0 + c1)

        elev = np.deg2rad(cfg.CLOSE_INSPECT_ELEVATION_DEG)
        direction_unit = np.array([-np.cos(elev), 0.0, np.sin(elev)])
        standoff = cfg.CLOSE_INSPECT_MAX_STANDOFF_M

        half_extent = 0.5 * float(np.linalg.norm(c1 - c0)) + cfg.CLOSE_INSPECT_OVERHANG_MARGIN_M
        intr = self.camera.get_scaled_intrinsics()
        dbg = self.camera.get_intrinsics_debug()
        depth_wh = dbg.get("depth_wh") if dbg is not None else None
        if intr is not None and depth_wh and depth_wh[0] and depth_wh[1]:
            fx, fy, _, _ = intr
            W, H = depth_wh
            # Conservative: treat half_extent as a radius that must project
            # within the margin fraction of BOTH image axes (we don't know the
            # camera's exact roll relative to the cylinders here, so this
            # bounds whichever axis actually captures it).
            d_x = 2.0 * fx * half_extent / (cfg.CLOSE_INSPECT_FOV_MARGIN_FRAC * W)
            d_y = 2.0 * fy * half_extent / (cfg.CLOSE_INSPECT_FOV_MARGIN_FRAC * H)
            needed = max(d_x, d_y)
            standoff = float(np.clip(needed, cfg.CLOSE_INSPECT_MIN_STANDOFF_M,
                                     cfg.CLOSE_INSPECT_MAX_STANDOFF_M))
        else:
            self.get_logger().warn(
                "[PHASE2] Camera intrinsics unavailable -- using the max standoff.")

        cand = midpoint + direction_unit * standoff
        q = self.kin.solve_posture_for_position(cand)
        if q is None:
            self.get_logger().warn("[PHASE2] Closer view unreachable -- will retry.")
            return

        cfg.HEAD_POSTURE_TARGET = q
        self._real_phase2_target = midpoint
        self._real_phase2_active = True
        self.get_logger().info(
            f"\n[PHASE2] LOCKED -- fixating the midpoint between the 2 "
            f"cylinders @{np.round(midpoint, 2)}, elevation="
            f"{cfg.CLOSE_INSPECT_ELEVATION_DEG:.0f}deg, standoff={standoff:.2f}m, "
            f"cam target={np.round(cand, 2)}.\n"
            "          Staying here permanently -- re-arm with a rescan to "
            "return to Phase 1.")


def _fetch_ee_x_cutoff(node, timeout_s):
    """Real hardware only, 1-shot: read main_qp_controller.py's /qp_debug/ee_real
    once to get max(EE_right.x, EE_left.x) -- the scene-cut boundary (config
    §11). Returns None (cut disabled) if nothing arrives within timeout_s,
    e.g. the arm QP controller isn't running alongside main_head.py."""
    received = {}

    def cb(msg):
        if "data" not in received:
            received["data"] = msg.data

    sub = node.create_subscription(Float64MultiArray, cfg.EE_STATE_TOPIC, cb, 10)
    start = time.time()
    while "data" not in received and (time.time() - start) < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)

    if "data" not in received or len(received["data"]) < 9:
        node.get_logger().warn(
            f"No usable message on {cfg.EE_STATE_TOPIC} within {timeout_s:.0f}s "
            "(is main_qp_controller.py running?) -- EE-based scene cut disabled.")
        return None
    d = received["data"]
    x_cutoff = max(d[0], d[6]) + cfg.EE_X_CUTOFF_MARGIN
    node.get_logger().info(
        f"[Init] EE-based scene cut: right.x={d[0]:.2f} left.x={d[6]:.2f} "
        f"-> excluding x < {x_cutoff:.2f}m from the analysis.")
    return x_cutoff


def main():
    # NO signal handlers: rclpy's default SIGINT handler shuts down the whole
    # context BEFORE our finally below runs, so restore_controllers() would
    # find an already-dead context on Ctrl-C. Disabling it means Ctrl-C just
    # raises a plain KeyboardInterrupt and WE control shutdown ordering.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = HeadPerceptionNode()

    if cfg.REAL_HARDWARE_HEAD:
        node.pipeline.set_ee_x_cutoff(
            _fetch_ee_x_cutoff(node, cfg.EE_STATE_WAIT_TIMEOUT_S))

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
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.kin.restore_controllers()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
