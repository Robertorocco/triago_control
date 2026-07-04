#!/usr/bin/env python3
"""
calibration_audit.py — isolate CALIBRATION error from ALGORITHM (fit) error
in the head camera's cylinder-perception pipeline.

WHY THIS SCRIPT EXISTS
    main_head.py reports ~2-3cm error on the two known cylinders (r=2cm,
    h=15cm). The 2026-07-02 accuracy pass already pushed the *fitting*
    algorithm's own bias down to ~1-4mm (rim extraction + Hyper circle fit,
    verified numerically against synthetic point clouds — see
    head_control/object_detector.py's module docstring and .kiro/context.md
    §5.10). A 2-3cm residual is 5-20x larger than that — too big to be a
    fitting artifact, and in fact LARGER than the cylinder's own 2cm radius.
    That last fact is the key diagnostic: a raw, unfitted centroid of points
    that are actually ON the object cannot be biased by more than roughly the
    object's own radius (partial-view occlusion biases the centroid TOWARD
    the visible side, at most). So if even the RAW centroid (no clustering,
    no circle fit — just np.mean of points in a generous box around the known
    location) is off by more than ~1cm, that is near-proof of a rigid
    extrinsic/intrinsic bias upstream of the algorithm, not a fitting bug.

WHAT THIS SCRIPT CHECKS (in order, cheapest/most-fundamental first)
    1. INTRINSICS SANITY — does camera_info's calibrated resolution match the
       actual depth image resolution? (a mismatch silently triggers a rescale
       in camera_interface.py — confirms it's doing the right thing, or catches
       it doing the wrong thing). Also prints D (distortion) and fx/fy/cx/cy.
    2. EXTRINSIC SELF-CONSISTENCY — Pinocchio FK (used by the controller) vs
       TF (used to place the point cloud) for the SAME camera optical frame,
       at the SAME instant. These MUST agree closely (sub-mm/sub-degree) if
       robot_state_publisher is correctly reflecting the live joint state. A
       disagreement here is a SOFTWARE bug (stale TF, wrong frame, timing
       skew) — fixable in code, not a physical recalibration.
    3. TABLE PLANE CHECK — RANSAC plane height + inlier centroid vs the known
       SDF table pose. The table is large & flat, so this is the most
       reliable single "is the whole scene rigidly shifted?" signal: if the
       table checks out but the cylinders don't, suspect the algorithm; if
       the table is ALSO off, suspect a rigid extrinsic bias affecting
       everything downstream of the transform.
    4. PER-OBJECT RAW CENTROID vs GROUND TRUTH — for each known cylinder,
       take every above-plane point within a generous box around its KNOWN
       location (no clustering, no circle fit) and report the raw mean vs
       ground truth. This isolates calibration error from the fitting
       algorithm's error (item 5 below).
    5. PER-OBJECT PIPELINE (FITTED) ESTIMATE vs GROUND TRUTH — runs the exact
       SAME PerceptionPipeline main_head.py uses, for a side-by-side number:
       "raw centroid error" vs "fitted error". If they're similar magnitude,
       calibration dominates (fixing the fit won't help much further). If the
       fitted error is much larger than the raw centroid error, the fit is
       adding its own (separate, algorithmic) error on top.

    A running EMA of each error vector is kept (not just instantaneous) since
    a single frame's depth noise can be a few mm — the diagnosis should be
    read once several samples have accumulated (the script prints a sample
    counter so you know when to trust it).

USAGE
    ros2 run triago_control calibration_audit.py
    (needs the same running scene as main_head.py: Gazebo + robot_state_publisher
    + the head camera topics. Does NOT command the head — read-only, safe to
    run alongside main_head.py.)

READING THE OUTPUT — a rule of thumb, not a strict decision procedure:
    - FK vs TF disagree (>5mm / >0.5deg)   -> SOFTWARE bug in the extrinsic
      chain (stale TF, wrong frame lookup, timing). Fix in code.
    - FK/TF agree, but table plane centroid/height is off by ~cylinder-error
      magnitude -> RIGID extrinsic bias (wrong URDF offset for the camera
      joint, or wrong optical-frame convention) OR intrinsic scaling. Affects
      the WHOLE scene uniformly. Fixable with ONE calibration constant
      (cfg.PERCEPTION_XYZ_OFFSET already exists for exactly this) if the
      offset is the same for the table AND both cylinders.
    - Table plane checks out (sub-cm), but the RAW cylinder centroids are
      still off by ~2-3cm -> NOT explainable by the transform chain (a
      correct transform places ALL points correctly, including cylinder
      points) — re-open the algorithm side (unexpected occlusion, wrong GT
      assumption, association mixing red/blue, etc.)
    - Table AND raw centroids check out, but the FITTED estimate is still off
      by more than the raw centroid -> the circle fit itself is adding error
      on THIS specific scene (re-open object_detector.py, but note this
      contradicts the numerically-verified sub-cm result from §5.10 — look
      for a scene-specific cause: unexpected occlusion pattern, colour
      misclassification, wrong cluster boundary, etc.)
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
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

import triago_control.head_control.config as cfg
from triago_control.head_control.camera_interface import CameraInterface
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.perception_pipeline import PerceptionPipeline


class EmaVec:
    """Tiny running-average helper (per-component EMA) with a sample counter."""

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None
        self.n = 0

    def update(self, v):
        v = np.asarray(v, dtype=float)
        if self.value is None:
            self.value = v.copy()
        else:
            self.value = self.alpha * v + (1.0 - self.alpha) * self.value
        self.n += 1
        return self.value


class CalibrationAuditNode(Node):
    def __init__(self):
        super().__init__("calibration_audit")

        self.kin = HeadKinematics(self)
        self.camera = CameraInterface(self)
        self.pipeline = PerceptionPipeline()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)

        # EMA trackers per diagnostic quantity.
        self.ema_fk_tf_pos_err = EmaVec()
        self.ema_fk_tf_ang_err = EmaVec()
        self.ema_plane_centroid_err = EmaVec()
        self.ema_plane_height_err = EmaVec()
        self.ema_raw_err = {"red": EmaVec(), "blue": EmaVec()}
        self.ema_fit_err = {"red": EmaVec(), "blue": EmaVec()}

        self._intrinsics_logged = False
        self.create_timer(2.0, self._audit_tick)

        self.get_logger().info(
            "\n==================================================================\n"
            " CALIBRATION AUDIT — isolating extrinsic/intrinsic bias from fit bias\n"
            " (read-only: does NOT command the head; run main_head.py separately)\n"
            "==================================================================")

    def _joint_cb(self, msg: JointState):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.kin.update_joint_states(list(msg.name), list(msg.position), stamp_sec)

    # ------------------------------------------------------------------ #
    def _lookup_tf(self, frame_id, stamp):
        for query in (Time.from_msg(stamp), Time()):
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
        return None, None

    # ------------------------------------------------------------------ #
    def _audit_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("[AUDIT] waiting for /joint_states (head joints)...")
            return
        if not self.camera.has_data():
            self.get_logger().info(
                f"[AUDIT] waiting for camera data (color={self.camera.n_color}, "
                f"depth={self.camera.n_depth}, info={self.camera.n_info})...")
            return

        # === 1. INTRINSICS SANITY ==================================== #
        if not self._intrinsics_logged:
            intr = self.camera.get_intrinsics_debug()
            if intr is not None:
                mismatch = (intr["info_wh"] is not None and intr["depth_wh"] is not None
                            and intr["info_wh"] != intr["depth_wh"])
                self.get_logger().info(
                    "\n[1. INTRINSICS]\n"
                    f"   fx={intr['fx']:.2f} fy={intr['fy']:.2f} cx={intr['cx']:.2f} cy={intr['cy']:.2f}\n"
                    f"   camera_info resolution = {intr['info_wh']}\n"
                    f"   depth image resolution = {intr['depth_wh']}\n"
                    f"   color image resolution = {intr['color_wh']}\n"
                    f"   distortion D           = {np.round(intr['D'], 5) if intr['D'] else None}\n"
                    + ("   [WARN] info/depth resolution MISMATCH -> camera_interface.py is "
                       "RESCALING intrinsics on every frame. Verify that rescale is intended.\n"
                       if mismatch else
                       "   [OK] camera_info resolution matches the depth image (no rescale needed).\n")
                )
                self._intrinsics_logged = True

        # === 2. EXTRINSIC SELF-CONSISTENCY (FK vs TF) ================= #
        T_cam_base_fk, _ = self.kin.forward()
        frame_id = self.camera.get_depth_frame_id()
        cloud = self.camera.get_point_cloud()
        if cloud is None:
            return
        points_optical, colors, stamp, frame_id = cloud
        if stamp is None or not frame_id:
            return
        R_tf, t_tf = self._lookup_tf(frame_id, stamp)
        if R_tf is None:
            self.get_logger().warn(f"[AUDIT] TF lookup base<-{frame_id} failed.")
            return

        fk_pos = T_cam_base_fk.translation
        fk_R = T_cam_base_fk.rotation
        pos_err = fk_pos - t_tf
        R_gap = fk_R @ R_tf.T
        tr = np.clip((np.trace(R_gap) - 1.0) / 2.0, -1.0, 1.0)
        ang_err_deg = float(np.degrees(np.arccos(tr)))
        ema_pos = self.ema_fk_tf_pos_err.update(pos_err)
        ema_ang = self.ema_fk_tf_ang_err.update([ang_err_deg])

        # === 3. TABLE PLANE CHECK ====================================== #
        result = self.pipeline.process(points_optical, colors, R_tf, t_tf,
                                        allow_integrate=False, allow_track_update=True)

        plane_report = "NO TABLE DETECTED"
        if result.plane is not None:
            height_err = result.plane.height - cfg.TABLE_TOP_Z_WORLD
            ema_h = self.ema_plane_height_err.update([height_err])
            plane_report = (f"height={result.plane.height:.4f}m  "
                             f"(GT={cfg.TABLE_TOP_Z_WORLD:.4f}m, "
                             f"err={height_err*1000:+.1f}mm, EMA={ema_h[0]*1000:+.1f}mm)")
            if result.plane_centroid is not None:
                c_err = result.plane_centroid[:2] - cfg.TABLE_CENTER_BASE[:2]
                ema_c = self.ema_plane_centroid_err.update(c_err)
                plane_report += (f"\n   centroid_xy={np.round(result.plane_centroid[:2],4)} "
                                  f"(GT={np.round(cfg.TABLE_CENTER_BASE[:2],4)}, "
                                  f"err_xy=({c_err[0]*1000:+.1f},{c_err[1]*1000:+.1f})mm, "
                                  f"EMA=({ema_c[0]*1000:+.1f},{ema_c[1]*1000:+.1f})mm)")

        # === 4 & 5. PER-OBJECT RAW CENTROID + FITTED ESTIMATE ========== #
        gt = {
            "red": (cfg.GT_RED_CENTER, cfg.GT_RED_RADIUS, cfg.GT_RED_HEIGHT),
            "blue": (cfg.GT_BLUE_CENTER, cfg.GT_BLUE_RADIUS, cfg.GT_BLUE_HEIGHT),
        }
        obj_lines = []
        above = result.above_points
        for color_name, (center_gt, radius_gt, height_gt) in gt.items():
            line = [f"   [{color_name.upper()}] GT center={np.round(center_gt,4)}"]
            # --- raw centroid: points within a generous box, no fit at all.
            if above is not None and len(above) > 0:
                box = 0.10  # 10cm half-width -- comfortably > 2-3cm expected error, << 40cm object separation
                m = ((np.abs(above[:, 0] - center_gt[0]) < box)
                     & (np.abs(above[:, 1] - center_gt[1]) < box))
                pts_near = above[m]
                if len(pts_near) >= 5:
                    raw_centroid = pts_near.mean(axis=0)
                    raw_err = raw_centroid - center_gt
                    ema_raw = self.ema_raw_err[color_name].update(raw_err)
                    line.append(
                        f"raw_centroid={np.round(raw_centroid,4)} n={len(pts_near)} "
                        f"err=({raw_err[0]*1000:+.1f},{raw_err[1]*1000:+.1f},{raw_err[2]*1000:+.1f})mm "
                        f"|err|={np.linalg.norm(raw_err)*1000:.1f}mm "
                        f"EMA|err|={np.linalg.norm(ema_raw)*1000:.1f}mm (n={self.ema_raw_err[color_name].n})")
                else:
                    line.append(f"raw_centroid: only {len(pts_near)} pts in box, skipping")
            else:
                line.append("raw_centroid: no above-plane points this frame")

            # --- fitted estimate from the real pipeline (nearest object of that colour).
            candidates = [o for o in result.objects if o.color_name == color_name]
            if candidates:
                obj = min(candidates, key=lambda o: np.linalg.norm(o.center[:2] - center_gt[:2]))
                fit_err = obj.center - center_gt
                ema_fit = self.ema_fit_err[color_name].update(fit_err)
                line.append(
                    f"fitted_center={np.round(obj.center,4)} r={obj.radius*100:.1f}cm(GT={radius_gt*100:.1f}) "
                    f"h={obj.height*100:.1f}cm(GT={height_gt*100:.1f}) "
                    f"err=({fit_err[0]*1000:+.1f},{fit_err[1]*1000:+.1f},{fit_err[2]*1000:+.1f})mm "
                    f"|err|={np.linalg.norm(fit_err)*1000:.1f}mm "
                    f"EMA|err|={np.linalg.norm(ema_fit)*1000:.1f}mm (n={self.ema_fit_err[color_name].n})")
            else:
                line.append("fitted: not currently detected/tracked")
            obj_lines.append("\n      ".join(line))

        self.get_logger().info(
            "\n[2. EXTRINSIC FK-vs-TF]  (must be ~0 -- else it's a SOFTWARE bug)\n"
            f"   pos_err=({pos_err[0]*1000:+.1f},{pos_err[1]*1000:+.1f},{pos_err[2]*1000:+.1f})mm "
            f"EMA=({ema_pos[0]*1000:+.1f},{ema_pos[1]*1000:+.1f},{ema_pos[2]*1000:+.1f})mm  "
            f"ang_err={ang_err_deg:.3f}deg EMA={ema_ang[0]:.3f}deg (n={self.ema_fk_tf_pos_err.n})\n"
            "\n[3. TABLE PLANE]  (whole-scene rigid-shift indicator)\n"
            f"   {plane_report}\n"
            "\n[4+5. CYLINDERS]  (raw centroid = calibration signal; fitted = + algorithm)\n"
            + "\n".join(obj_lines)
        )


def main():
    rclpy.init()
    node = CalibrationAuditNode()

    node.get_logger().info("Fetching URDF from robot_state_publisher...")
    urdf_str = node.kin.fetch_urdf()
    if urdf_str is None:
        node.get_logger().error("No URDF -- is robot_state_publisher running? Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".urdf") as f:
        f.write(urdf_str)
        urdf_path = f.name
    node.kin.build(urdf_path)
    os.remove(urdf_path)

    node.get_logger().info("Setup complete. Auditing every 2s (Ctrl+C to stop).")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
