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
from triago_control.head_control.view_planner import ActivePerceptionPlanner
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
        self.planner = ActivePerceptionPlanner()   # active-perception standoff
        self.viz = PerceptionVisualizer(frame_id=cfg.BASE_FRAME)

        # --- Publishers ------------------------------------------------
        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f"/{cfg.HEAD_CONTROLLER}/joint_velocity_cmd", 10
        )
        self.pub_cloud = self.create_publisher(PointCloud2, "/head_perception/cloud", 1)
        self.pub_raw_cloud = self.create_publisher(PointCloud2, "/head_perception/raw_cloud", 1)
        self.pub_markers = self.create_publisher(MarkerArray, "/head_perception/markers", 1)
        # Scalar telemetry for the plotter: [n_raw, n_crop, plane_z, look_err_deg,
        # slack, proc_ms]. Lets the plotter show cloud size / quality directly.
        self.pub_telemetry = self.create_publisher(
            Float64MultiArray, "/head_perception/telemetry", 10
        )
        # Active-perception standoff telemetry for plotting/inspection:
        # [range, d_star, d_star_raw, max_edge_occ, occL, occR, occT, occB,
        #  fill, mean_r_px, worst_rms, action_code]. action_code:
        # 0=HOLD 1=APPROACH 2=RETREAT 3=INIT.
        self.pub_view = self.create_publisher(
            Float64MultiArray, "/head_perception/view_debug", 10
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
        # Last TF-derived camera pose (for the FK-vs-TF cross-check diagnostic).
        self._last_tf_pos = None
        self._last_tf_R = None
        self._last_depth_frame = None
        self._last_vel_norm = 0.0
        self._last_integrated = False

        # --- Bias-vs-range diagnostic (READ-ONLY audit, per project rule:
        # ground-truth constants are only ever used here for COMPARISON, never
        # fed back into the perception output). Accumulates raw, PRE-TRACKER
        # per-frame detections vs. GT_RED_CENTER/GT_BLUE_CENTER together with
        # the camera-to-object range at that instant, so we can fit
        #     bias = intercept + slope * range
        # A nonzero INTERCEPT with near-zero SLOPE indicates a constant
        # translational error (e.g. a fixed offset in the extrinsic chain,
        # independent of viewing distance). A nonzero SLOPE indicates an
        # angular/rotational error or an intrinsics (cx/cy) error (bias grows
        # proportionally with distance). This is diagnostic-only console
        # output; it never alters the actual perception result.
        self._bias_samples = {"red": [], "blue": []}   # color -> list[(range, dx, dy)]

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
            allow_integrate=allow_integrate, allow_track_update=allow_integrate
        )
        self.latest_result = result
        self._collect_bias_samples(result, t_cam_base)

        # --- Active perception: adapt the camera standoff distance -----
        # Decide (from the observed cloud framing + object resolution, NO scene
        # ground truth) whether to move closer / farther, and hand the desired
        # standoff to the look-at QP as a soft range task.
        if cfg.ENABLE_ACTIVE_VIEW:
            intr = self.camera.get_scaled_intrinsics()
            dbg = self.camera.get_intrinsics_debug()
            wh = dbg.get("depth_wh") if dbg else None
            if intr is not None and wh is not None:
                d_star = self.planner.update(
                    result, R_cam_base, t_cam_base,
                    self.current_target, intr, wh
                )
                self.controller.set_standoff(d_star)
                self._publish_view_debug()

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
        # Per-colour confidence (so the plotter can show estimation quality).
        red_conf = next((o.confidence for o in result.objects if o.color_name == "red"), 0.0)
        blue_conf = next((o.confidence for o in result.objects if o.color_name == "blue"), 0.0)
        tel.data = [
            float(result.n_raw), float(n_crop), float(plane_z),
            float(self.controller.last_angle_deg), float(self.controller.last_slack_norm),
            float(result.proc_ms), float(red_conf), float(blue_conf),
            float(result.map_size),
        ]
        self.pub_telemetry.publish(tel)

    def _collect_bias_samples(self, result, t_cam_base):
        """Accumulate (range, dx, dy) samples from RAW (pre-tracker) detections
        vs. the known GT centers, for the bias-vs-range regression diagnostic.

        READ-ONLY: this compares against cfg.GT_RED_CENTER/GT_BLUE_CENTER
        purely for console reporting. It does not feed back into
        `result.objects`, `result.raw_detections`, or any published topic that
        drives control/collision — see the module rule on ground-truth usage.
        Uses raw_detections (this frame's fresh fit) rather than the
        EMA-tracked `objects`, since the tracker's smoothing autocorrelates
        consecutive samples and would mask a genuine range-dependent trend.
        """
        gt = {"red": cfg.GT_RED_CENTER, "blue": cfg.GT_BLUE_CENTER}
        for det in result.raw_detections:
            if det.color_name not in gt:
                continue
            rng = float(np.linalg.norm(det.center - t_cam_base))
            dx = float(det.center[0] - gt[det.color_name][0])
            dy = float(det.center[1] - gt[det.color_name][1])
            self._bias_samples[det.color_name].append((rng, dx, dy))
            # Cap memory: keep only the most recent 500 samples per color.
            if len(self._bias_samples[det.color_name]) > 500:
                self._bias_samples[det.color_name].pop(0)

    def _bias_regression_report(self) -> str:
        """Fit bias = intercept + slope*range per axis/color and render a
        console block distinguishing a constant (translational) error from a
        range-scaling (rotational/intrinsics) error. READ-ONLY diagnostic.
        """
        lines = ["       +== BIAS-VS-RANGE DIAGNOSTIC (raw detections vs GT, read-only) =+"]
        any_data = False
        for color in ("red", "blue"):
            samples = self._bias_samples[color]
            if len(samples) < 5:
                lines.append(f"       |   {color:<5s}: not enough samples yet ({len(samples)})            |")
                continue
            any_data = True
            arr = np.array(samples)   # (N, 3): range, dx, dy
            rng_arr = arr[:, 0]
            for axis_name, col in (("dx", 1), ("dy", 2)):
                y = arr[:, col]
                # Least-squares line fit: y = a + b*rng
                A = np.column_stack([np.ones_like(rng_arr), rng_arr])
                try:
                    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
                    intercept, slope = sol
                except np.linalg.LinAlgError:
                    intercept, slope = float(y.mean()), 0.0
                verdict = ("CONST" if abs(slope) < 0.01 else
                           ("SCALES w/ range" if abs(slope) > 0.02 else "mixed"))
                lines.append(
                    f"       |   {color:<5s} {axis_name}: {intercept*100:+6.2f}cm "
                    f"+ {slope*100:+6.2f}cm/m * range  [{verdict:<16s}] n={len(samples):<4d}|")
        if not any_data:
            lines.append("       |   (waiting for raw detections...)                            |")
        lines.append("       +================================================================+")
        return "\n" + "\n".join(lines)

    def _publish_view_debug(self):
        """Publish the active-perception standoff telemetry array."""
        m = self.planner.metrics
        if not m:
            return
        occ = m.get("edge_occ", {})
        action_code = {"HOLD": 0.0, "APPROACH": 1.0, "RETREAT": 2.0, "INIT": 3.0}
        msg = Float64MultiArray()
        msg.data = [
            float(m.get("range", 0.0)),
            float(m.get("d_star", 0.0)),
            float(m.get("d_star_raw", 0.0)),
            float(m.get("max_edge", 0.0)),
            float(occ.get("L", 0.0)), float(occ.get("R", 0.0)),
            float(occ.get("T", 0.0)), float(occ.get("B", 0.0)),
            float(m.get("fill", 0.0)),
            float(m.get("mean_r_px", 0.0)),
            float(m.get("worst_rms", 0.0)),
            float(action_code.get(m.get("action", "HOLD"), 0.0)),
        ]
        self.pub_view.publish(msg)

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
            diag += (f"\n       [PLANE-CENTROID] mean={np.round(r.plane_centroid,3)}  "
                     f"bbox={np.round(r.plane_bbox_center,3) if r.plane_bbox_center is not None else 'N/A'}")
            diag += (f"\n       [TABLE-EXTENT] X={r.table_extent_x:.3f}m (expect {cfg.TABLE_SIZE[0]:.2f}) "
                     f"Y={r.table_extent_y:.3f}m (expect {cfg.TABLE_SIZE[1]:.2f})")
            # Intrinsics sanity: if the observed Y-extent of the table is
            # significantly different from the known physical width, the
            # intrinsics (fx/fy) are probably wrong (lateral scale error).
            if r.table_extent_y > 0.05:
                y_ratio = r.table_extent_y / cfg.TABLE_SIZE[1]
                if y_ratio < 0.5:
                    diag += f"  [!Y-extent only {y_ratio*100:.0f}% of physical — partial view or fx wrong]"
                elif y_ratio > 1.3:
                    diag += f"  [!Y-extent {y_ratio*100:.0f}% of physical — fx too HIGH?]"

        self.get_logger().info(
            head_line + "\n"
            f"       [PERCEPTION] raw={r.n_raw} crop={len(r.cropped_points) if r.cropped_points is not None else 0} "
            f"map={r.map_size} | {plane_txt} | proc={r.proc_ms:.1f} ms | "
            f"head_vel={self._last_vel_norm:.3f} {'FUSING' if self._last_integrated else 'moving'}\n"
            f"       [OBJECTS] {obj_txt}\n"
            f"       [JOINTS] {joint_info}" + diag
            + self._active_view_panel()
            + self._bias_regression_report())

    # ------------------------------------------------------------------ #
    # Active-perception debug window (shareable console panel)             #
    # ------------------------------------------------------------------ #
    def _active_view_panel(self) -> str:
        """Render the active-perception state as a boxed console 'window'.

        This is the panel to screenshot and share: it shows, at a glance,
        WHETHER the head decided to move closer/farther and WHY — the framing
        (per-edge border occupancy + contained verdict) and the object
        resolution (apparent radius in px, rim-fit RMS) that drive the
        decision.
        """
        if not cfg.ENABLE_ACTIVE_VIEW:
            return ""
        m = self.planner.metrics
        if not m:
            return ("\n       +-- ACTIVE PERCEPTION -----------------------------------------+"
                    "\n       |  waiting for first perception frame...                       |"
                    "\n       +--------------------------------------------------------------+")

        action = m.get("action", "HOLD")
        arrow = {"APPROACH": "vv CLOSER", "RETREAT": "^^ FARTHER",
                 "HOLD": "== HOLD", "INIT": ".. INIT"}.get(action, action)
        rng = m.get("range", 0.0)
        d_star = m.get("d_star", 0.0)
        d_raw = m.get("d_star_raw", 0.0)
        err = rng - d_star
        occ = m.get("edge_occ", {"L": 0, "R": 0, "T": 0, "B": 0})
        hi = cfg.VIEW_BORDER_HIGH
        lo = cfg.VIEW_BORDER_LOW

        def emark(val):
            if val > hi:
                return "CLIP"
            if val < lo:
                return " ok "
            return " .. "

        contain = ("CONTAINED" if m.get("contained") else
                   ("CLIPPING" if m.get("clipping") else "partial"))
        W, H = m.get("img_wh", (0, 0))

        lines = []
        lines.append("       +== ACTIVE PERCEPTION (next-best-view standoff) ===============+")
        lines.append(f"       | decision : {arrow:<10s}                                       |")
        lines.append(f"       | reason   : {m.get('reason', '')[:49]:<49s} |")
        lines.append( "       |--------------------------------------------------------------|")
        lines.append(f"       | range r  = {rng:5.3f} m   d* = {d_star:5.3f} m (raw {d_raw:5.3f})   "
                     f"err {err:+5.3f} |")
        lines.append(f"       | clamp    [{cfg.VIEW_D_STAR_MIN:.2f},{cfg.VIEW_D_STAR_MAX:.2f}] m"
                     f"   range-slack {self.controller.last_range_slack:+.3f}          |")
        lines.append( "       |-- FRAMING (border occupancy, %; CLIP>{:.0f} ok<{:.0f}) ----------|"
                     .format(hi * 100, lo * 100))
        lines.append(f"       |   L {occ['L']*100:4.1f}%{emark(occ['L'])}  R {occ['R']*100:4.1f}%{emark(occ['R'])}"
                     f"  T {occ['T']*100:4.1f}%{emark(occ['T'])}  B {occ['B']*100:4.1f}%{emark(occ['B'])}   |")
        lines.append(f"       |   verdict: {contain:<10s}  fill {m.get('fill',0)*100:4.1f}%  "
                     f"img {W}x{H}  n_proj {m.get('n_proj',0):>6d} |")
        lines.append( "       |-- RESOLUTION (per object) -----------------------------------|")
        objs = m.get("objects", [])
        if not objs:
            lines.append("       |   (no objects tracked yet)                                   |")
        else:
            tgt = cfg.VIEW_RES_RADIUS_PX_TARGET
            for o in objs:
                flag = "LOW " if o["r_px"] < tgt else "good"
                lines.append(
                    f"       |   {o['label'][:14]:<14s} r_px {o['r_px']:5.1f}({flag} tgt {tgt:.0f})  "
                    f"rms {o['fit_rms']*1e3:4.1f}mm  cov {o['coverage']*100:3.0f}% |")
        lines.append( "       +==============================================================+")
        return "\n" + "\n".join(lines)


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
