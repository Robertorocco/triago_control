#!/usr/bin/env python3
"""
head_april_main.py — TRIAGo head: look at the table & reconstruct object poses
from a single AprilTag (fiducial), instead of the geometric RGB-D cylinder fit.

WHY THIS EXISTS
    main_head.py estimates each cylinder pose directly from the depth cloud
    (RANSAC plane + rim/circle fit). That works but carries a few-centimetre
    error dominated by the head-camera extrinsic chain and stereo depth noise.
    This node takes a different route: it localizes ONE fiducial of known
    geometry (an AprilTag) with the ViSP `visp_apriltag` detector, and
    reconstructs every scene object from a KNOWN rigid transform relative to
    that tag. The only quantity estimated from the image is the camera->tag
    pose; object geometry is a model prior.

RECONSTRUCTION
    For each object k:
        base_M_obj_k = base_M_cam . c_M_tag . tag_M_obj_k
    where
        base_M_cam  : TF (base_footprint <- color optical frame), at the
                      detection timestamp — same correctness trick main_head
                      uses (right frame, right time while the head moves).
        c_M_tag     : from the visp_apriltag detection (camera-optical -> tag).
        tag_M_obj_k : KNOWN, = world_M_tag^-1 . world_M_obj_k, derived once at
                      startup from cfg.APRILTAG_SCENE (see config.py §15 for why
                      this is a legitimate fiducial model, NOT ground-truth
                      injection: the node never reads any object's absolute
                      world pose at runtime).

CONTROL
    The look-at control loop is reused verbatim from main_head.py (HeadKinematics
    + LookAtController + controller switching): the head keeps the table (and
    thus the tag) centred in the camera.

VISUALISATION / INSPECTION
    Publishes the SAME topics head_plotter.py already consumes, so the existing
    live dashboard shows the reconstruction quality vs ground truth with NO
    plotter change:
        /head_perception/markers    (MarkerArray: red/blue cylinders ns=objects,
                                      table ns=table_top, + a tag frame ns=apriltag)
        /head_perception/telemetry  (Float64MultiArray, same 9-slot layout)

PREREQUISITE — run the ViSP AprilTag detector alongside this node:
    ros2 run visp_apriltag visp_apriltag_node --ros-args \
        -p rgb_image_topic_name:=/gripper_head_camera_rgbd/color/image_raw \
        -p rgb_camera_info_topic_name:=/gripper_head_camera_rgbd/color/camera_info \
        -p tag_family:=36h11 -p pose_method:=homography_virtual_vs \
        -p tag_size_keys:="[-1]" -p tag_size_values:="[0.12]"
    (topics / family / size are mirrored in config.py §15.)
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
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation as Rot

import tf2_ros

import triago_control.head_control.config as cfg
from triago_control.head_control.head_kinematics import HeadKinematics
from triago_control.head_control.look_at_controller import LookAtController

# The AprilTag detection message lives in the visp_tracker_common package
# (built from the visp_tk repo). Import lazily-friendly so the failure message
# is actionable if the package is not on the path.
try:
    from visp_tracker_common.msg import AprilTagDetectionArray
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import visp_tracker_common.msg.AprilTagDetectionArray. "
        "Build and source the visp_tk packages (visp_common, "
        "visp_tracker_common, visp_apriltag) in your workspace first — see "
        "the visp-apriltag repo README."
    ) from exc


# ====================================================================== #
# Small SE(3) helpers (numpy 4x4)                                         #
# ====================================================================== #
def mat_from_rpy_t(rpy, t):
    """Homogeneous 4x4 from an (roll,pitch,yaw) and a translation (3,)."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = t
    return T


def mat_from_pose(pose):
    """Homogeneous 4x4 from a geometry_msgs/Pose."""
    q = pose.orientation
    p = pose.position
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def quat_from_mat(R):
    """(x,y,z,w) quaternion from a 3x3 rotation matrix."""
    return Rot.from_matrix(R).as_quat()


class HeadAprilNode(Node):
    def __init__(self):
        super().__init__("head_april_main")

        # --- Library components (control side, reused from main_head) ---
        self.kin = HeadKinematics(self)
        self.controller = LookAtController(self.kin)

        # --- Publishers ------------------------------------------------
        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f"/{cfg.HEAD_CONTROLLER}/joint_velocity_cmd", 10
        )
        self.pub_markers = self.create_publisher(MarkerArray, "/head_perception/markers", 1)
        self.pub_telemetry = self.create_publisher(
            Float64MultiArray, "/head_perception/telemetry", 10
        )

        # --- TF2 (base_footprint <- color optical frame) ---------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_warned = False

        # --- AprilTag scene model (tag-relative transforms) ------------
        # world_M_tag defines the ViSP tag frame in the world; tag_M_obj is
        # derived ONCE and is the only scene knowledge used at runtime.
        self.world_M_tag = mat_from_rpy_t(cfg.APRILTAG_RPY_WORLD, cfg.APRILTAG_CENTER_WORLD)
        tag_M_world = np.linalg.inv(self.world_M_tag)
        self.tag_M_obj = {}
        for name, spec in cfg.APRILTAG_SCENE.items():
            world_M_obj = mat_from_rpy_t(spec["rpy"], spec["center"])
            self.tag_M_obj[name] = tag_M_world @ world_M_obj

        # --- Subscriptions ---------------------------------------------
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)
        # The visp_apriltag node publishes tags_info with
        # KeepLast(5).best_effort().transient_local() (see visp_tracker_common
        # BaseTracker). A default RELIABLE subscriber is QoS-incompatible with a
        # BEST_EFFORT publisher and silently receives NOTHING — so match it.
        qos_tags = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            AprilTagDetectionArray, cfg.APRILTAG_DETECTIONS_TOPIC,
            self._apriltag_cb, qos_tags
        )

        # --- Shared state ----------------------------------------------
        self.T_cam_base = None
        self.J_cam = None
        self.start_time = time.time()
        self.current_target = cfg.TABLE_TOP_CENTER_BASE.copy()

        # Latest tag observation (thread: the callback writes, the perception
        # timer reads). Store the raw c_M_tag, its stamp and frame.
        self._tag_c_M_tag = None
        self._tag_stamp = None
        self._tag_frame = None
        self._tag_recv_time = 0.0
        self._n_tags_last = 0
        self._last_recon = {}          # name -> (center base, R base) reconstructed
        self._last_proc_ms = 0.0
        self._camera_warned = False

        # --- Tag-localization diagnostic (READ-ONLY audit) -------------
        # Splits the systematic reconstruction bias into its two possible
        # sources — the TF camera extrinsic vs. ViSP's tag-pose estimate —
        # by comparing the CAMERA-ESTIMATED tag pose in base against the KNOWN
        # tag pose (self.world_M_tag). This is comparison-only tooling: the
        # known pose is NEVER fed back into the reconstruction (that would be a
        # forbidden ground-truth fudge). Toggle with `-p tag_diag:=false`.
        # Logged at the console cadence (not per-frame) to avoid spam.
        self.declare_parameter("tag_diag", True)
        self._tag_diag = self.get_parameter("tag_diag").value
        self._diag_cache = None        # (base_M_cam, c_M_tag, base_M_tag)

        # --- Timers ----------------------------------------------------
        self.create_timer(1.0 / cfg.CONTROL_RATE_HZ, self._control_tick)
        self.create_timer(1.0 / cfg.PERCEPTION_RATE_HZ, self._perception_tick)
        self.create_timer(cfg.CONSOLE_SUMMARY_PERIOD_S, self._console_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " TRIAGo HEAD — table look-at + AprilTag pose reconstruction\n"
            "------------------------------------------------------------------\n"
            f"  Tag topic   : {cfg.APRILTAG_DETECTIONS_TOPIC}\n"
            f"  Tag id      : {cfg.APRILTAG_ID}  family {cfg.APRILTAG_FAMILY}  "
            f"size {cfg.APRILTAG_SIZE} m\n"
            f"  Optical frm : {cfg.APRILTAG_CAMERA_OPTICAL_FRAME}\n"
            f"  Reconstruct : {', '.join(cfg.APRILTAG_SCENE.keys())}\n"
            "  Detector    : run `visp_apriltag_node` (see header of this file)\n"
            "==================================================================")

    # ================================================================== #
    # Callbacks                                                           #
    # ================================================================== #
    def _joint_cb(self, msg: JointState):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.kin.update_joint_states(list(msg.name), list(msg.position), stamp_sec)

    def _apriltag_cb(self, msg: AprilTagDetectionArray):
        """Latch the reference tag's camera->tag pose (id == cfg.APRILTAG_ID)."""
        self._n_tags_last = len(msg.detections)
        for det in msg.detections:
            if det.id == cfg.APRILTAG_ID:
                self._tag_c_M_tag = mat_from_pose(det.pose)
                # Prefer the per-detection header; fall back to the array header.
                stamp = det.header.stamp if det.header.frame_id else msg.header.stamp
                frame = det.header.frame_id or msg.header.frame_id
                self._tag_stamp = stamp
                self._tag_frame = frame or cfg.APRILTAG_CAMERA_OPTICAL_FRAME
                self._tag_recv_time = time.time()
                return

    # ================================================================== #
    # Control loop (identical to main_head.py)                            #
    # ================================================================== #
    def _control_tick(self):
        if not self.kin.is_ready():
            return
        self.T_cam_base, self.J_cam = self.kin.forward()
        t = time.time() - self.start_time
        self.current_target = self.controller.scan_target(t)
        dq = self.controller.compute(self.T_cam_base, self.J_cam, self.current_target)
        msg = Float64MultiArray()
        msg.data = [float(x) for x in dq]
        self.pub_head_cmd.publish(msg)

    # ================================================================== #
    # Perception loop — AprilTag reconstruction                           #
    # ================================================================== #
    def _perception_tick(self):
        if not self.kin.is_ready():
            return

        # Is there a fresh tag observation?
        fresh = (
            self._tag_c_M_tag is not None
            and (time.time() - self._tag_recv_time) < cfg.APRILTAG_DETECTION_TIMEOUT_S
        )
        if not fresh:
            if not self._camera_warned:
                self.get_logger().warn(
                    f"No fresh tag id {cfg.APRILTAG_ID} on "
                    f"'{cfg.APRILTAG_DETECTIONS_TOPIC}'. Is visp_apriltag_node "
                    "running and pointed at the color image? (see this file's header)")
                self._camera_warned = True
            self._publish_telemetry(reconstructed=set(), proc_ms=0.0)
            return

        t0 = time.perf_counter()
        c_M_tag = self._tag_c_M_tag
        stamp = self._tag_stamp
        frame = self._tag_frame

        # base_M_cam from TF at the detection timestamp (correct frame + time).
        base_M_cam = self._lookup_transform(frame, stamp)
        if base_M_cam is None:
            self._publish_telemetry(reconstructed=set(), proc_ms=0.0)
            return

        base_M_tag = base_M_cam @ c_M_tag
        self._diag_cache = (base_M_cam, c_M_tag, base_M_tag)   # read-only audit

        # Reconstruct every scene object from its known tag-relative transform.
        recon = {}
        for name, tag_M_obj in self.tag_M_obj.items():
            base_M_obj = base_M_tag @ tag_M_obj
            recon[name] = (base_M_obj[:3, 3].copy(), base_M_obj[:3, :3].copy())
        self._last_recon = recon

        proc_ms = (time.perf_counter() - t0) * 1e3
        self._last_proc_ms = proc_ms

        # Publish markers (head_plotter reads these) + a tag-frame marker.
        self._publish_markers(base_M_tag, recon, stamp)
        self._publish_telemetry(reconstructed=set(recon.keys()), proc_ms=proc_ms)

    def _lookup_transform(self, frame_id, stamp):
        """Return base_footprint <- frame_id as a 4x4, or None."""
        for query in (Time.from_msg(stamp), Time()):     # exact time, then latest
            try:
                tf = self.tf_buffer.lookup_transform(
                    cfg.BASE_FRAME, frame_id, query, timeout=Duration(seconds=0.05)
                )
                q = tf.transform.rotation
                tr = tf.transform.translation
                T = np.eye(4)
                T[:3, :3] = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                T[:3, 3] = [tr.x, tr.y, tr.z]
                return T
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                continue
        if not self._tf_warned:
            self.get_logger().warn(
                f"TF lookup base<-{frame_id} failed (is robot_state_publisher up?).")
            self._tf_warned = True
        return None

    # ================================================================== #
    # Visualisation                                                       #
    # ================================================================== #
    def _publish_markers(self, base_M_tag, recon, stamp):
        markers = MarkerArray()
        mid = 0

        # --- The reconstructed objects ---------------------------------
        for name, (center, R) in recon.items():
            spec = cfg.APRILTAG_SCENE[name]
            quat = quat_from_mat(R)

            if spec["kind"] == "cylinder":
                m = Marker()
                m.header.frame_id = cfg.BASE_FRAME
                m.header.stamp = stamp
                m.ns = "objects"                 # <- head_plotter classifies these
                m.id = mid; mid += 1
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x = float(center[0])
                m.pose.position.y = float(center[1])
                m.pose.position.z = float(center[2])
                m.pose.orientation.x = float(quat[0])
                m.pose.orientation.y = float(quat[1])
                m.pose.orientation.z = float(quat[2])
                m.pose.orientation.w = float(quat[3])
                m.scale.x = float(2.0 * spec["radius"])
                m.scale.y = float(2.0 * spec["radius"])
                m.scale.z = float(spec["height"])
                self._set_color(m, spec["color"], alpha=0.9)
                markers.markers.append(m)

                # a small text label above the cylinder
                markers.markers.append(
                    self._text_marker(mid, name, center + np.array([0, 0, spec["height"]]),
                                      stamp))
                mid += 1

            elif spec["kind"] == "table":
                m = Marker()
                m.header.frame_id = cfg.BASE_FRAME
                m.header.stamp = stamp
                m.ns = "table_top"               # <- head_plotter reads pose.z
                m.id = mid; mid += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = float(center[0])
                m.pose.position.y = float(center[1])
                m.pose.position.z = float(center[2])
                m.pose.orientation.x = float(quat[0])
                m.pose.orientation.y = float(quat[1])
                m.pose.orientation.z = float(quat[2])
                m.pose.orientation.w = float(quat[3])
                m.scale.x = float(spec["size_xy"][0])
                m.scale.y = float(spec["size_xy"][1])
                m.scale.z = 0.005
                self._set_color(m, "table", alpha=0.35)
                markers.markers.append(m)

        # --- The tag frame itself (for RViz; head_plotter ignores it) --
        tag_pos = base_M_tag[:3, 3]
        tag_q = quat_from_mat(base_M_tag[:3, :3])
        tag = Marker()
        tag.header.frame_id = cfg.BASE_FRAME
        tag.header.stamp = stamp
        tag.ns = "apriltag"
        tag.id = mid; mid += 1
        tag.type = Marker.CUBE
        tag.action = Marker.ADD
        tag.pose.position.x = float(tag_pos[0])
        tag.pose.position.y = float(tag_pos[1])
        tag.pose.position.z = float(tag_pos[2])
        tag.pose.orientation.x = float(tag_q[0])
        tag.pose.orientation.y = float(tag_q[1])
        tag.pose.orientation.z = float(tag_q[2])
        tag.pose.orientation.w = float(tag_q[3])
        tag.scale.x = float(cfg.APRILTAG_SIZE)
        tag.scale.y = float(cfg.APRILTAG_SIZE)
        tag.scale.z = 0.002
        tag.color.r = 0.1; tag.color.g = 0.1; tag.color.b = 0.1; tag.color.a = 1.0
        markers.markers.append(tag)

        self.pub_markers.publish(markers)

    def _text_marker(self, mid, text, pos, stamp):
        m = Marker()
        m.header.frame_id = cfg.BASE_FRAME
        m.header.stamp = stamp
        m.ns = "labels"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2] + 0.05)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.04
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 1.0; m.color.a = 1.0
        m.text = text
        return m

    @staticmethod
    def _set_color(marker, color_name, alpha=1.0):
        if color_name == "red":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.1, 0.1
        elif color_name == "blue":
            marker.color.r, marker.color.g, marker.color.b = 0.1, 0.1, 1.0
        else:  # table / other
            marker.color.r, marker.color.g, marker.color.b = 0.87, 0.72, 0.53
        marker.color.a = float(alpha)

    # ================================================================== #
    # Telemetry (same 9-slot layout head_plotter expects)                 #
    # ================================================================== #
    def _publish_telemetry(self, reconstructed, proc_ms):
        # [n_raw, n_crop, plane_z, look_err_deg, slack, proc_ms,
        #  red_conf, blue_conf, map_size]
        plane_z = float("nan")
        if "work_table" in self._last_recon:
            plane_z = float(self._last_recon["work_table"][0][2])
        red_conf = 1.0 if "red_cylinder" in reconstructed else 0.0
        blue_conf = 1.0 if "blue_cylinder" in reconstructed else 0.0
        tel = Float64MultiArray()
        tel.data = [
            float(self._n_tags_last),                 # n_raw  -> "# tags seen"
            0.0,                                       # n_crop (unused here)
            plane_z,                                   # reconstructed table top z
            float(self.controller.last_angle_deg),    # look-at error [deg]
            float(self.controller.last_slack_norm),   # look-at slack
            float(proc_ms),                            # reconstruction time [ms]
            float(red_conf),
            float(blue_conf),
            0.0,                                       # map_size (unused here)
        ]
        self.pub_telemetry.publish(tel)

    # ================================================================== #
    # Console report                                                      #
    # ================================================================== #
    def _console_tick(self):
        if not self.kin.is_ready():
            self.get_logger().info("Waiting for /joint_states (head joints)...")
            return
        aligned = "ALIGNED" if self.controller.is_aligned() else "slewing"
        head_line = (
            f"[HEAD] look-at err={self.controller.last_angle_deg:5.1f} deg ({aligned}) "
            f"slack={self.controller.last_slack_norm:.3f}"
        )
        fresh = (
            self._tag_c_M_tag is not None
            and (time.time() - self._tag_recv_time) < cfg.APRILTAG_DETECTION_TIMEOUT_S
        )
        if not fresh:
            self.get_logger().info(
                head_line + f" | AprilTag: no fresh id {cfg.APRILTAG_ID} "
                f"(tags in view: {self._n_tags_last})")
            return
        obj_txt = ", ".join(
            f"{name}@({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
            for name, (c, _) in self._last_recon.items()
        ) or "none"
        self.get_logger().info(
            head_line + "\n"
            f"       [APRILTAG] id {cfg.APRILTAG_ID} seen | recon {self._last_proc_ms:.2f} ms\n"
            f"       [RECON] {obj_txt}")

        if self._tag_diag and self._diag_cache is not None:
            self._log_tag_diagnostic(*self._diag_cache)

    # ================================================================== #
    # Tag-localization diagnostic (READ-ONLY audit — see __init__ note)   #
    # ================================================================== #
    def _log_tag_diagnostic(self, base_M_cam, c_M_tag, base_M_tag):
        """Log where the systematic reconstruction bias comes from.

        The whole scene hangs rigidly off the tag, so a constant offset on
        every object equals the error on the ESTIMATED tag pose:
            base_M_tag = base_M_cam . c_M_tag
                         └ TF extrinsic ┘  └ ViSP ┘
        We compare that estimate against the KNOWN tag pose (self.world_M_tag,
        base==world) and also print, in the CAMERA frame, ViSP's raw c_M_tag
        vs. the c_M_tag that the TF extrinsic + known tag pose imply
        (`c_M_tag_ideal = base_M_cam^-1 . world_M_tag`). Reading:
          * If ViSP's c_M_tag ~= c_M_tag_ideal  -> ViSP agrees with (TF+truth);
            the residual is dominated by the TF extrinsic.
          * If they differ, look at the split: a |range| (Z) mismatch points at
            tag_size / focal-length (intrinsics) scale; a lateral (X/Y)
            mismatch points at principal point (cx,cy) or a lateral extrinsic.
        NOTHING here is fed back into the reconstruction.
        """
        true_base_M_tag = self.world_M_tag           # known tag pose (base==world)

        cam_t = base_M_cam[:3, 3]
        cam_rpy = np.degrees(Rot.from_matrix(base_M_cam[:3, :3]).as_euler("xyz"))

        ct = c_M_tag[:3, 3]
        rng = float(np.linalg.norm(ct))

        c_M_tag_ideal = np.linalg.inv(base_M_cam) @ true_base_M_tag
        cti = c_M_tag_ideal[:3, 3]
        rng_i = float(np.linalg.norm(cti))
        dcam = ct - cti                              # ViSP - ideal, in camera frame

        est_t = base_M_tag[:3, 3]
        est_yaw = float(np.degrees(Rot.from_matrix(base_M_tag[:3, :3]).as_euler("xyz"))[2])
        true_t = true_base_M_tag[:3, 3]
        true_yaw = float(np.degrees(Rot.from_matrix(true_base_M_tag[:3, :3]).as_euler("xyz"))[2])
        dpos = est_t - true_t

        self.get_logger().info(
            "       [TAG-DIAG] (read-only; frame='%s')\n" % self._tag_frame +
            f"         extrinsic base_M_cam : t={np.round(cam_t, 4)} rpy_deg={np.round(cam_rpy, 1)}\n"
            f"         ViSP  c_M_tag        : t={np.round(ct, 4)} range={rng:.4f} m\n"
            f"         ideal c_M_tag(TF+GT) : t={np.round(cti, 4)} range={rng_i:.4f} m\n"
            f"         ViSP-ideal (cam frm) : dt={np.round(dcam, 4)} d|range|={rng - rng_i:+.4f} m\n"
            f"         tag in base  EST     : t={np.round(est_t, 4)} yaw={est_yaw:+.1f} deg\n"
            f"         tag in base  KNOWN   : t={np.round(true_t, 4)} yaw={true_yaw:+.1f} deg\n"
            f"         >> TAG POS ERROR     : {np.round(dpos, 4)} m  "
            f"|{np.linalg.norm(dpos) * 100:.2f} cm|  yaw_err={est_yaw - true_yaw:+.2f} deg")


def main():
    rclpy.init()
    node = HeadAprilNode()

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
