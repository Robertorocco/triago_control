#!/usr/bin/env python3
# trajectory_generator.py
"""
Open-loop cartesian trajectory generator for QP-CLF-CBF robustness testing.

This node feeds `main_qp_controller.py` with smooth, repeatable quintic
trajectories so the safety controller can be stress-tested in a controlled way:
slow free-space motions first, then bimanual motions, then deliberately
collision- or workspace-violating targets that the CBF must arbitrate.

Design contract (why the QP controller stays untouched)
-------------------------------------------------------
The generator ONLY publishes on the standard cartesian-reference topics:

    /arm_right/cartesian_reference   Float64MultiArray
    /arm_left/cartesian_reference    Float64MultiArray
        layout: [x, y, z, roll, pitch, yaw, xdot, ydot, zdot, wx, wy, wz, (task_dim)]

This is exactly the protocol `main_qp_controller.ref_cb_right/left` already
parses (>=12 floats -> 6-DOF, optional 13th float -> task dimension, 6-float ->
position only). The controller cannot tell this source apart from the keyboard
teleop or the shared-autonomy node, so it requires NO modification.

Everything that defines a test -- endpoints, categories, the DYNAMIC_TRAJECTORY
flag, timing -- lives in `config/trajectory_endpoints.yaml`. Editing that file
is all that is needed to change behaviour.

Two spatial-path modes are supported for the active arm(s):
  - 2-point (absolute/swap/swap_perturbed): straight line, start -> target.
  - 'path' (waypoints_right/waypoints_left in the preset): a smooth cubic
    spline threaded through the sampled start + every listed waypoint, still
    driven by the SAME quintic tau(t) timing profile -- see
    `_build_path_spline`/timer_callback. Lets a preset define a curved,
    multi-point shape (e.g. an 'S' around workspace obstacles) instead of
    only a start/end pair.

Pipeline
--------
1. WAITING  : sample the real start pose from /qp_debug/ee_real for `delay_start`
              seconds (position + orientation per hand).
2. TRACKING : interpolate start -> target with a quintic (zero vel/acc at the
              ends). If `dynamic_trajectory` is on, a virtual clock integrates
              dt * sigma, where sigma shrinks with the CBF shadow price
              (/qp_debug/lambda_cbf) so the reference yields to the safety filter.
3. REGULATION: hold the final target with zero feed-forward velocity.

Telemetry mirrors the legacy generator so the existing dashboard keeps working:
    /trajectory/phase           (String  : 'S' / 'T' / 'R')
    /trajectory/phase_marker    (Marker  : RViz text)
    /trajectory/reference_state (Float64MultiArray, 12: [x_r,xdot_r,x_l,xdot_l])
    /trajectory/time_scale      (Float64 : sigma)
"""

import os

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, String, Bool
from visualization_msgs.msg import Marker

import triago_control.qp_controller.config as cfg

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - ament always present under ROS 2
    get_package_share_directory = None

# Optional keyboard E-stop removed — pynput hangs in headless environments.
# Use Ctrl-C to stop the node instead.


def euler_to_quaternion(roll, pitch, yaw):
    """Fixed-axis RPY -> quaternion [x, y, z, w]."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return np.array([qx, qy, qz, qw])


def quaternion_to_rpy(q):
    """Quaternion [x, y, z, w] -> RPY [roll, pitch, yaw]."""
    x, y, z, w = q
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions q0 and q1 at fraction t.
    Both quaternions are [x, y, z, w]. Returns interpolated quaternion."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # If very close, use linear interpolation to avoid division by zero
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def angular_velocity_from_slerp(q0, q1, s_dot_sigma, duration):
    """Approximate angular velocity [wx, wy, wz] for the SLERP interpolation.
    Uses the axis-angle between q0 and q1 scaled by s_dot * sigma."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return np.zeros(3)
    # Relative quaternion: q_rel = q1 * q0_inv
    # q0_inv for unit quaternion = [-x, -y, -z, w]
    q0_inv = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
    # q_rel = q1 * q0_inv (Hamilton product)
    q_rel = _quat_mult(q1, q0_inv)
    # Convert to axis-angle
    angle = 2.0 * np.arccos(np.clip(q_rel[3], -1.0, 1.0))
    if abs(angle) < 1e-10:
        return np.zeros(3)
    axis = q_rel[:3] / np.sin(angle / 2.0)
    # Angular velocity = axis * angle * s_dot * sigma (chain rule)
    return axis * angle * s_dot_sigma


def _quat_mult(q1, q2):
    """Hamilton product of two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _default_config_path():
    """Locate the installed trajectory_endpoints.yaml (share dir), else source."""
    fname = 'trajectory_endpoints.yaml'

    # 1. Ament share directory (the correct installed location).
    if get_package_share_directory is not None:
        try:
            share = get_package_share_directory('triago_control')
            candidate = os.path.join(share, 'config', fname)
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass

    # 2. Fallback: relative to this script inside the *source* tree.
    #    <pkg_root>/scripts/qp_arm_teleop/trajectory_generator.py
    #    -> <pkg_root>/config/trajectory_endpoints.yaml
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.normpath(os.path.join(here, '..', '..', 'config', fname))
    if os.path.exists(guess):
        return guess

    # 3. Last-ditch: check common colcon workspace layout.
    #    Installed at: <ws>/install/<pkg>/lib/<pkg>/trajectory_generator.py
    #    Config at:    <ws>/install/<pkg>/share/<pkg>/config/...
    guess2 = os.path.normpath(os.path.join(here, '..', 'share',
                                           'triago_control', 'config', fname))
    if os.path.exists(guess2):
        return guess2

    # Return the ament-based guess anyway (will produce a clear error later).
    return guess


class TrajectoryGenerator(Node):
    """Publishes quintic cartesian references defined by a YAML endpoint file."""

    def __init__(self):
        super().__init__('trajectory_generator')

        # --- Resolve + load the endpoint/flags file -----------------------
        self.declare_parameter('config_file', '')
        cfg_path = self.get_parameter('config_file').get_parameter_value().string_value
        if not cfg_path:
            cfg_path = _default_config_path()
        self.config_path = cfg_path
        self.cfg = self._load_config(cfg_path)

        # --- Global behaviour pulled from the YAML ------------------------
        self.ref_frame = str(self.cfg.get('ref_frame', 'base_footprint'))
        self.duration = float(self.cfg.get('duration', 15.0))
        self.delay_start = float(self.cfg.get('delay_start', 1.0))
        self.cube_size = float(self.cfg.get('cube_size', 0.1))
        self.dynamic_trajectory = bool(self.cfg.get('dynamic_trajectory', True))
        self.control_orientation = bool(self.cfg.get('control_orientation', True))
        self.task_dimension = float(self.cfg.get('task_dimension', 6.0))

        self.declare_parameter('preset', '')
        preset_override = self.get_parameter('preset').get_parameter_value().string_value
        self.preset_name = preset_override if preset_override else str(self.cfg.get('active_preset', ''))
        self.preset = self._resolve_preset(self.preset_name)
        self.arms = str(self.preset.get('arms', 'both')).lower()
        self.mode = str(self.preset.get('mode', 'absolute')).lower()

        # --- Subscribers --------------------------------------------------
        # /qp_debug/ee_real layout (18 floats):
        #   [p_r(3), v_r(3), p_l(3), v_l(3), rpy_r(3), rpy_l(3)]
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real',
                                 self.ee_real_callback, 10)
        # [lambda_cbf_R, lambda_cbf_L]; must match main_qp_controller.py's
        # Float64MultiArray type or rosbag2 refuses to record the topic.
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_cbf',
                                 self.lambda_cb, 10)

        # --- Publishers ---------------------------------------------------
        self.pub_ref_right = self.create_publisher(
            Float64MultiArray, '/arm_right/cartesian_reference', 10)
        self.pub_ref_left = self.create_publisher(
            Float64MultiArray, '/arm_left/cartesian_reference', 10)
        self.pub_phase = self.create_publisher(String, '/trajectory/phase', 10)
        self.pub_rviz_marker = self.create_publisher(
            Marker, '/trajectory/phase_marker', 10)
        self.pub_dashboard = self.create_publisher(
            Float64MultiArray, '/trajectory/reference_state', 10)
        self.pub_time_scale = self.create_publisher(
            Float64, '/trajectory/time_scale', 10)
        # Offline-recording trigger: True on WAITING->TRACKING (t=0), False on
        # TRACKING->REGULATION. See cfg.OFFLINE_RECORD_TRIGGER_TOPIC / offline_plotter.py.
        self.pub_record_trigger = self.create_publisher(
            Bool, cfg.OFFLINE_RECORD_TRIGGER_TOPIC, 10)

        # --- State --------------------------------------------------------
        self.t0 = self._now()
        self.last_loop_time = self.t0
        self.virtual_time = 0.0
        self.lambda_cbf = 0.0
        self.current_sigma = 1.0
        self.targets_generated = False
        self.should_stop = False
        self.data_received = False
        self.current_phase = ""

        # Hold in WAITING until the recorder subscribes (VOLATILE trigger edge
        # would be missed otherwise), bounded by OFFLINE_RECORD_WAIT_TIMEOUT_S.
        self._recorder_ready = False
        self._settle_done_time = None
        self._recorder_wait_logged = False

        self.pos_r = np.zeros(3); self.pos_l = np.zeros(3)
        self.rpy_r = np.zeros(3); self.rpy_l = np.zeros(3)
        self.start_right = np.zeros(3); self.start_left = np.zeros(3)
        self.end_right = np.zeros(3); self.end_left = np.zeros(3)
        self.start_rpy_r = np.zeros(3); self.start_rpy_l = np.zeros(3)
        self.end_rpy_r = np.zeros(3); self.end_rpy_l = np.zeros(3)
        # Quaternion caches for SLERP (computed once at target generation)
        self.q_start_r = np.array([0, 0, 0, 1.0])
        self.q_start_l = np.array([0, 0, 0, 1.0])
        self.q_end_r = np.array([0, 0, 0, 1.0])
        self.q_end_l = np.array([0, 0, 0, 1.0])
        # Optional smooth multi-waypoint path per arm (mode: 'path' presets
        # only). None -> the original straight-line start->target formula.
        self.path_spline_r = None
        self.path_spline_l = None

        # --- Diagnostics counters -----------------------------------------
        self.ee_msg_count = 0       # how many /qp_debug/ee_real msgs arrived
        self.ee_last_len = 0        # length of the last ee_real msg
        self.lambda_msg_count = 0   # how many /qp_debug/lambda_cbf msgs arrived
        self.first_ee_logged = False
        self.ref_pub_count = 0      # how many reference msgs we have published
        self.orientation_warned = False

        # --- Optional keyboard E-stop removed (pynput hangs headless) ----
        # Use Ctrl-C to stop the node.

        self.timer = self.create_timer(0.01, self.timer_callback)
        # Watchdog/diagnostics at 1 Hz: tells you *why* nothing is moving.
        self.diag_timer = self.create_timer(1.0, self._diagnostics_callback)
        self.last_print_time = self.t0

        self._print_banner()

    def _now(self):
        """Wall-clock time [s]."""
        return self.get_clock().now().nanoseconds / 1e9

    # =====================================================================
    # CONFIG LOADING
    # =====================================================================
    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[trajectory_generator] config file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or 'presets' not in data:
            raise ValueError(
                f"[trajectory_generator] malformed config (no 'presets'): {path}")
        return data

    def _resolve_preset(self, name):
        presets = self.cfg.get('presets', {})
        if name not in presets:
            available = ', '.join(sorted(presets.keys()))
            raise KeyError(
                f"[trajectory_generator] active_preset '{name}' not found. "
                f"Available: {available}")
        return presets[name]

    def _print_banner(self):
        # Force the node logger to show INFO (in case ros2 run suppresses it).
        try:
            from rclpy.logging import LoggingSeverity
            self.get_logger().set_level(LoggingSeverity.INFO)
        except Exception:
            pass  # Older distros may not have this; print() below always works.

        # Use print(flush=True) so output appears IMMEDIATELY in all terminals.
        print("=" * 60, flush=True)
        print(" TRAJECTORY GENERATOR", flush=True)
        print("=" * 60, flush=True)
        print(f"  config        : {self.config_path}", flush=True)
        print(f"  preset        : {self.preset_name} "
              f"[{self.preset.get('category', 'n/a')}]", flush=True)
        print(f"  description   : {self.preset.get('description', '').strip()}",
              flush=True)
        print(f"  arms          : {self.arms}   mode: {self.mode}", flush=True)
        print(f"  duration      : {self.duration:.1f}s   "
              f"delay: {self.delay_start:.1f}s", flush=True)
        print(f"  dynamic_traj  : {self.dynamic_trajectory}", flush=True)
        print(f"  orientation   : {self.control_orientation} "
              f"(task_dim={self.task_dimension})", flush=True)
        print("=" * 60, flush=True)
        print(f"[INIT] Waiting for /qp_debug/ee_real to sample start pose. "
              f"Arms will NOT move until that topic arrives.", flush=True)

    # =====================================================================
    # DIAGNOSTICS WATCHDOG (1 Hz)
    # =====================================================================
    def _diagnostics_callback(self):
        """Print (flushed) exactly why the trajectory may not be running."""
        n_pub_ee = self.count_publishers('/qp_debug/ee_real')
        n_pub_lambda = self.count_publishers('/qp_debug/lambda_cbf')
        n_sub_r = self.count_subscribers('/arm_right/cartesian_reference')
        n_sub_l = self.count_subscribers('/arm_left/cartesian_reference')

        # 1) Not moving because we never sampled a start pose.
        if not self.data_received:
            if self.ee_msg_count == 0:
                print(f"[WARN] WAITING: no /qp_debug/ee_real received yet "
                      f"({n_pub_ee} publisher(s) detected). "
                      f"Is main_qp_controller.py running?", flush=True)
            else:
                print(f"[WARN] RECEIVING /qp_debug/ee_real but "
                      f"length={self.ee_last_len} (<9 floats). "
                      f"Cannot sample start position.", flush=True)
            return

        # 2) Data is flowing -> report status.
        if self.dynamic_trajectory and self.lambda_msg_count == 0:
            print(f"[WARN] dynamic_trajectory=ON but no /qp_debug/lambda_cbf "
                  f"({n_pub_lambda} pub). sigma stays 1.0.", flush=True)

        if self.ref_pub_count > 0 and (n_sub_r == 0 and n_sub_l == 0):
            print("[WARN] Publishing refs but NOBODY subscribed to "
                  "/arm_{{right,left}}/cartesian_reference!", flush=True)

        print(f"[STATUS] phase={self.current_phase or '-'} "
              f"v_time={self.virtual_time:.2f}/{self.duration:.0f}s "
              f"sigma={self.current_sigma:.2f} lambda={self.lambda_cbf:.2f} "
              f"| ee_msgs={self.ee_msg_count}(len={self.ee_last_len}) "
              f"refs_pub={self.ref_pub_count} subs(R/L)={n_sub_r}/{n_sub_l}",
              flush=True)

    # =====================================================================
    # SUBSCRIBER CALLBACKS
    # =====================================================================
    def lambda_cb(self, msg):
        """[lambda_cbf_R, lambda_cbf_L] -- the worse (max) of the two
        per-arm CBF shadow prices drives the dynamic time scaling (same
        worst-case convention as qp_formulator.py's last_lambda_col)."""
        if len(msg.data) >= 2:
            self.lambda_cbf = float(max(msg.data[0], msg.data[1]))
        elif len(msg.data) == 1:
            self.lambda_cbf = float(msg.data[0])
        else:
            return
        self.lambda_msg_count += 1

    def ee_real_callback(self, msg):
        """Sample the real EE state (positions + orientation) from the QP node."""
        self.ee_msg_count += 1
        self.ee_last_len = len(msg.data)

        if not self.first_ee_logged:
            self.first_ee_logged = True
            print(f"[OK] First /qp_debug/ee_real received "
                  f"(length={self.ee_last_len}).", flush=True)
            if self.ee_last_len < 18 and self.control_orientation \
                    and not self.orientation_warned:
                self.orientation_warned = True
                print(f"[WARN] control_orientation=True but ee_real has only "
                      f"{self.ee_last_len} floats (<18): no RPY available. "
                      f"Holding orientation at zero RPY.", flush=True)

        if len(msg.data) >= 9:
            self.pos_r = np.array(msg.data[0:3])
            self.pos_l = np.array(msg.data[6:9])
            self.data_received = True
        if len(msg.data) >= 18:
            self.rpy_r = np.array(msg.data[12:15])
            self.rpy_l = np.array(msg.data[15:18])

    # =====================================================================
    # MAIN LOOP
    # =====================================================================
    def timer_callback(self):
        if self.should_stop or not self.data_received:
            return

        current_time = self._now()
        dt = current_time - self.last_loop_time
        self.last_loop_time = current_time
        t_total = current_time - self.t0

        # Clamp an outlier dt so a host stall can't snap virtual_time forward.
        if dt > 0.1 or dt <= 0.0:
            dt = 0.01

        # --- PHASE 1: WAITING / SETUP ------------------------------------
        # The record-start signal (update_phase 'S'->'T', below) is a single
        # VOLATILE std_msgs/Bool -- a subscriber that joins after it is published
        # never receives it. Same-host (sim) DDS discovery is instantaneous so the
        # plotter always caught it; across a network (real robot <-> docker over
        # Cyclone) discovery takes seconds, longer than delay_start, so the edge
        # fired before offline_plotter had subscribed and nothing was recorded.
        # Fix: after the settle window, keep holding in WAITING until the recorder
        # is discovered (get_subscription_count() > 0) -- guaranteeing it is both
        # subscribed and, being RELIABLE, delivered the edge -- bounded by
        # cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S so a recorder-less run still proceeds.
        settle_done = t_total >= self.delay_start
        if settle_done and not self._recorder_ready:
            if self._settle_done_time is None:
                self._settle_done_time = current_time
            wait_disabled = cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S <= 0.0
            has_recorder = self.pub_record_trigger.get_subscription_count() > 0
            waited = current_time - self._settle_done_time
            if wait_disabled or has_recorder:
                self._recorder_ready = True
                if has_recorder and not wait_disabled:
                    print("[STATE] Offline recorder subscribed on "
                          f"'{cfg.OFFLINE_RECORD_TRIGGER_TOPIC}' -- starting motion.",
                          flush=True)
            elif waited >= cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S:
                self._recorder_ready = True   # gave up waiting -- run anyway
                print("[WARN] No offline recorder on "
                      f"'{cfg.OFFLINE_RECORD_TRIGGER_TOPIC}' after "
                      f"{cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S:.0f}s -- starting anyway "
                      "(this trial will NOT be recorded).", flush=True)
            elif not self._recorder_wait_logged:
                self._recorder_wait_logged = True
                print("[STATE] Settle window done; holding in WAITING for an "
                      f"offline recorder on '{cfg.OFFLINE_RECORD_TRIGGER_TOPIC}' "
                      f"(up to {cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S:.0f}s)...",
                      flush=True)

        if (not settle_done) or (not self._recorder_ready):
            self.start_right = self.pos_r.copy()
            self.start_left = self.pos_l.copy()
            self.start_rpy_r = self.rpy_r.copy()
            self.start_rpy_l = self.rpy_l.copy()
            self.virtual_time = 0.0
            self.publish_references(self.start_right, np.zeros(3),
                                    self.start_rpy_r, np.zeros(3),
                                    self.start_left, np.zeros(3),
                                    self.start_rpy_l, np.zeros(3))
            self.update_phase('S', "PHASE: WAITING", 1.0, 0.8, 0.0)
            return

        # --- TRANSITION: generate the targets once -----------------------
        if not self.targets_generated:
            self.end_right, self.end_left = self._compute_targets()
            self.end_rpy_r, self.end_rpy_l = self._compute_orientation_targets()
            # Pre-compute quaternions for SLERP
            self.q_start_r = euler_to_quaternion(*self.start_rpy_r)
            self.q_start_l = euler_to_quaternion(*self.start_rpy_l)
            self.q_end_r = euler_to_quaternion(*self.end_rpy_r)
            self.q_end_l = euler_to_quaternion(*self.end_rpy_l)
            self.path_spline_r = self._build_path_spline('right')
            self.path_spline_l = self._build_path_spline('left')
            self.targets_generated = True
            print(f"[STATE] Targets generated (preset '{self.preset_name}', "
                  f"mode '{self.mode}')", flush=True)
            if self.path_spline_r is not None or self.path_spline_l is not None:
                print(f"  Path waypoints: right="
                      f"{len(self.preset.get('waypoints_right', []))} "
                      f"left={len(self.preset.get('waypoints_left', []))}",
                      flush=True)
            print(f"  Right: pos {self.start_right.round(3)} "
                  f"-> {self.end_right.round(3)}", flush=True)
            print(f"         rpy {np.degrees(self.start_rpy_r).round(1)} "
                  f"-> {np.degrees(self.end_rpy_r).round(1)} deg", flush=True)
            print(f"  Left : pos {self.start_left.round(3)} "
                  f"-> {self.end_left.round(3)}", flush=True)
            print(f"         rpy {np.degrees(self.start_rpy_l).round(1)} "
                  f"-> {np.degrees(self.end_rpy_l).round(1)} deg", flush=True)

        # --- 1. TIME SCALING ---------------------------------------------
        if self.dynamic_trajectory:
            k_lambda = 3.0    # how aggressively the clock slows near obstacles
            sigma_min = 0.20  # floor speed so the reference never fully stalls
            target_sigma = sigma_min + (1.0 - sigma_min) * np.exp(
                -k_lambda * self.lambda_cbf)
            # Low-pass filter to avoid reference jerk (~0.5s ramp at 100 Hz).
            filter_alpha = 0.95
            self.current_sigma = (filter_alpha * self.current_sigma +
                                  (1.0 - filter_alpha) * target_sigma)
        else:
            self.current_sigma = 1.0
        sigma = self.current_sigma

        self.pub_time_scale.publish(Float64(data=float(sigma)))

        # --- 2. INTEGRATE THE VIRTUAL CLOCK ------------------------------
        self.virtual_time += sigma * dt

        if self.virtual_time >= self.duration:
            # --- REGULATION ---
            x_ref_r, xdot_ref_r = self.end_right, np.zeros(3)
            x_ref_l, xdot_ref_l = self.end_left, np.zeros(3)
            rpy_ref_r = self.end_rpy_r
            rpy_ref_l = self.end_rpy_l
            w_ref_r = np.zeros(3)
            w_ref_l = np.zeros(3)
            if current_time - self.last_print_time > 1.0:
                print(f"[HOLDING] dist R: "
                      f"{np.linalg.norm(x_ref_r - self.pos_r):.3f}m | "
                      f"dist L: {np.linalg.norm(x_ref_l - self.pos_l):.3f}m | "
                      f"lambda: {self.lambda_cbf:.2f}", flush=True)
                self.last_print_time = current_time
            self.update_phase('R', "PHASE: REGULATION", 0.0, 0.5, 1.0)
        else:
            # --- TRACKING (quintic) ---
            tau = self.virtual_time / self.duration
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration

            # Position interpolation -- straight line, unless a 'path' preset
            # built a smooth spline through intermediate waypoints for this
            # arm (see _build_path_spline). Either way `s` is the SAME
            # quintic-in-tau scalar, so timing/dynamic_trajectory behaviour
            # is identical; only the spatial shape changes.
            if self.path_spline_r is not None:
                x_ref_r = self.path_spline_r(s)
                xdot_ref_r = self.path_spline_r(s, 1) * (s_dot * sigma)
            else:
                x_ref_r = self.start_right + s * (self.end_right - self.start_right)
                xdot_ref_r = (s_dot * sigma) * (self.end_right - self.start_right)
            if self.path_spline_l is not None:
                x_ref_l = self.path_spline_l(s)
                xdot_ref_l = self.path_spline_l(s, 1) * (s_dot * sigma)
            else:
                x_ref_l = self.start_left + s * (self.end_left - self.start_left)
                xdot_ref_l = (s_dot * sigma) * (self.end_left - self.start_left)

            # Orientation interpolation (SLERP with same quintic scalar)
            q_r = slerp(self.q_start_r, self.q_end_r, s)
            q_l = slerp(self.q_start_l, self.q_end_l, s)
            rpy_ref_r = quaternion_to_rpy(q_r)
            rpy_ref_l = quaternion_to_rpy(q_l)
            # Angular velocity feed-forward
            w_ref_r = angular_velocity_from_slerp(
                self.q_start_r, self.q_end_r, s_dot * sigma, self.duration)
            w_ref_l = angular_velocity_from_slerp(
                self.q_start_l, self.q_end_l, s_dot * sigma, self.duration)

            if current_time - self.last_print_time > 0.5:
                print(f"[TRACKING] v-time: {self.virtual_time:.1f}s | "
                      f"speed: {sigma * 100:.0f}% | "
                      f"lambda: {self.lambda_cbf:.2f} | "
                      f"err R: {np.linalg.norm(x_ref_r - self.pos_r):.3f}m",
                      flush=True)
                self.last_print_time = current_time
            self.update_phase('T', "PHASE: TRACKING", 0.0, 1.0, 0.0)

        self.publish_references(x_ref_r, xdot_ref_r, rpy_ref_r, w_ref_r,
                                x_ref_l, xdot_ref_l, rpy_ref_l, w_ref_l)
        self.publish_dashboard(x_ref_r, xdot_ref_r, x_ref_l, xdot_ref_l)

    # =====================================================================
    # TARGET COMPUTATION
    # =====================================================================
    def _compute_targets(self):
        """Resolve per-hand end targets from the active preset + sampled starts."""
        if self.mode == 'swap':
            end_r = self.start_left.copy()
            end_l = self.start_right.copy()
        elif self.mode == 'swap_perturbed':
            half = self.cube_size / 2.0
            end_r = self.start_left + np.random.uniform(-half, half, 3)
            end_l = self.start_right + np.random.uniform(-half, half, 3)
        elif self.mode == 'path':
            # Final target = last listed waypoint; the intermediate points
            # are consumed separately by _build_path_spline.
            if 'waypoints_right' in self.preset:
                end_r = np.array(self.preset['waypoints_right'][-1], dtype=float)
            else:
                end_r = self.start_right.copy()
            if 'waypoints_left' in self.preset:
                end_l = np.array(self.preset['waypoints_left'][-1], dtype=float)
            else:
                end_l = self.start_left.copy()
        else:  # 'absolute'
            end_r = np.array(self.preset.get('right', self.start_right), dtype=float)
            end_l = np.array(self.preset.get('left', self.start_left), dtype=float)

        # Inactive arm holds its sampled start pose (regulated in place).
        if self.arms == 'right':
            end_l = self.start_left.copy()
        elif self.arms == 'left':
            end_r = self.start_right.copy()
        return end_r, end_l

    def _build_path_spline(self, side):
        """For mode=='path' presets: a cubic spline through
        [sampled_start] + waypoints_<side> (last entry == the resolved
        end_right/end_left target), arc-length parametrized onto s in [0, 1]
        so it plugs directly into the existing quintic tau(t) timing (s below
        is the SAME scalar the straight-line formula used). Returns None for
        every 2-point preset (absolute/swap/swap_perturbed) or an inactive
        arm -- those keep the original straight-line interpolation untouched.
        """
        if self.mode != 'path':
            return None
        if (side == 'right' and self.arms == 'left') or \
           (side == 'left' and self.arms == 'right'):
            return None
        key = f'waypoints_{side}'
        if key not in self.preset:
            return None
        start = self.start_right if side == 'right' else self.start_left
        pts = np.array([start] + [np.array(p, dtype=float)
                                   for p in self.preset[key]])
        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        if cum[-1] < 1e-9:
            return None  # degenerate: every waypoint coincides with start
        knots = cum / cum[-1]
        return CubicSpline(knots, pts, axis=0)

    def _compute_orientation_targets(self):
        """Resolve per-hand target RPY from the active preset.
        If rpy_right/rpy_left is specified in the YAML, use it.
        Otherwise, hold the sampled start orientation (no rotation)."""
        if 'rpy_right' in self.preset:
            end_rpy_r = np.array(self.preset['rpy_right'], dtype=float)
        else:
            end_rpy_r = self.start_rpy_r.copy()

        if 'rpy_left' in self.preset:
            end_rpy_l = np.array(self.preset['rpy_left'], dtype=float)
        else:
            end_rpy_l = self.start_rpy_l.copy()

        # Inactive arm holds its start orientation.
        if self.arms == 'right':
            end_rpy_l = self.start_rpy_l.copy()
        elif self.arms == 'left':
            end_rpy_r = self.start_rpy_r.copy()

        return end_rpy_r, end_rpy_l

    # =====================================================================
    # PUBLISH HELPERS
    # =====================================================================
    def publish_references(self, x_r, xdot_r, rpy_r, w_r,
                               x_l, xdot_l, rpy_l, w_l):
        """Pack and send the cartesian reference on the standard QP contract."""
        msg_r = Float64MultiArray()
        msg_l = Float64MultiArray()
        if self.control_orientation:
            # 6-DOF: [x,y,z, r,p,y, xdot,ydot,zdot, wx,wy,wz, task_dim]
            msg_r.data = (x_r.tolist() + rpy_r.tolist() +
                          xdot_r.tolist() + w_r.tolist() +
                          [float(self.task_dimension)])
            msg_l.data = (x_l.tolist() + rpy_l.tolist() +
                          xdot_l.tolist() + w_l.tolist() +
                          [float(self.task_dimension)])
        else:
            # Position-only fallback (6-float branch in the controller).
            msg_r.data = x_r.tolist() + xdot_r.tolist()
            msg_l.data = x_l.tolist() + xdot_l.tolist()
        self.pub_ref_right.publish(msg_r)
        self.pub_ref_left.publish(msg_l)
        self.ref_pub_count += 1

    def publish_dashboard(self, x_r, xdot_r, x_l, xdot_l):
        """12-element [x_r, xdot_r, x_l, xdot_l] layout for the legacy plotter."""
        msg = Float64MultiArray()
        msg.data = x_r.tolist() + xdot_r.tolist() + x_l.tolist() + xdot_l.tolist()
        self.pub_dashboard.publish(msg)

    def update_phase(self, phase_char, text, r, g, b):
        if self.current_phase != phase_char:
            previous_phase = self.current_phase
            self.current_phase = phase_char
            self.pub_phase.publish(String(data=phase_char))
            self.publish_rviz_marker(text, r, g, b)
            print(f"[PHASE] Switched to {text}", flush=True)

            # Offline-recording trigger (see cfg.OFFLINE_RECORD_TRIGGER_TOPIC):
            # rising edge on entering TRACKING ('T' -- the quintic motion is
            # actually running, this defines t=0 for the recorded trial);
            # falling edge on LEAVING TRACKING for any other phase (normally
            # 'R', REGULATION -- the commanded motion has concluded).
            if phase_char == 'T':
                self.pub_record_trigger.publish(Bool(data=True))
            elif previous_phase == 'T':
                self.pub_record_trigger.publish(Bool(data=False))

    def publish_rviz_marker(self, text, r, g, b):
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "phase_indicator"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.4
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.1
        marker.scale.x = 0.15; marker.scale.y = 0.15; marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = float(r); marker.color.g = float(g); marker.color.b = float(b)
        marker.text = text
        self.pub_rviz_marker.publish(marker)




def main():
    rclpy.init()
    try:
        node = TrajectoryGenerator()
    except Exception as e:
        # Print to stderr so it's never buffered / swallowed.
        import sys
        print(f"\n[FATAL] trajectory_generator failed to initialize:\n  {e}\n",
              file=sys.stderr, flush=True)
        rclpy.shutdown()
        raise

    # Wall-clock control loop by design -- do not set use_sim_time here.

    try:
        while rclpy.ok() and not node.should_stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
