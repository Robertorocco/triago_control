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
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64, String
from visualization_msgs.msg import Marker

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - ament always present under ROS 2
    get_package_share_directory = None

# Optional keyboard E-stop ('0' to freeze). Headless robots may lack pynput /
# an X server, so the import is best-effort and the node runs fine without it.
try:
    from pynput import keyboard as _kb
except Exception:  # pragma: no cover
    _kb = None


def euler_to_quaternion(roll, pitch, yaw):
    """Fixed-axis RPY -> quaternion [x, y, z, w] (for the RViz pose marker)."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return [qx, qy, qz, qw]


def _default_config_path():
    """Locate the installed trajectory_endpoints.yaml (share dir), else source."""
    if get_package_share_directory is not None:
        try:
            share = get_package_share_directory('triago_control')
            candidate = os.path.join(share, 'config', 'trajectory_endpoints.yaml')
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass
    # Fallback: walk up from this file to <pkg>/config/trajectory_endpoints.yaml
    here = os.path.dirname(os.path.abspath(__file__))
    guess = os.path.normpath(os.path.join(here, '..', '..', 'config',
                                          'trajectory_endpoints.yaml'))
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

        self.preset_name = str(self.cfg.get('active_preset', ''))
        self.preset = self._resolve_preset(self.preset_name)
        self.arms = str(self.preset.get('arms', 'both')).lower()
        self.mode = str(self.preset.get('mode', 'absolute')).lower()

        # --- Subscribers --------------------------------------------------
        # /qp_debug/ee_real layout (18 floats):
        #   [p_r(3), v_r(3), p_l(3), v_l(3), rpy_r(3), rpy_l(3)]
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real',
                                 self.ee_real_callback, 10)
        self.create_subscription(Float64, '/qp_debug/lambda_cbf',
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

        # --- State --------------------------------------------------------
        self.t0 = time.time()
        self.last_loop_time = time.time()
        self.virtual_time = 0.0
        self.lambda_cbf = 0.0
        self.current_sigma = 1.0
        self.targets_generated = False
        self.should_stop = False
        self.data_received = False
        self.current_phase = ""

        self.pos_r = np.zeros(3); self.pos_l = np.zeros(3)
        self.rpy_r = np.zeros(3); self.rpy_l = np.zeros(3)
        self.start_right = np.zeros(3); self.start_left = np.zeros(3)
        self.end_right = np.zeros(3); self.end_left = np.zeros(3)
        self.hold_rpy_r = np.zeros(3); self.hold_rpy_l = np.zeros(3)

        # --- Optional keyboard E-stop ('0') -------------------------------
        self.listener = None
        if _kb is not None:
            try:
                self.listener = _kb.Listener(on_press=self.on_key_press)
                self.listener.start()
            except Exception as e:  # pragma: no cover
                self.get_logger().warn(f"[INIT] Keyboard listener disabled: {e}")

        self.timer = self.create_timer(0.01, self.timer_callback)
        self.last_print_time = time.time()

        self._print_banner()

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
        print("\n" + "=" * 60)
        print(" TRAJECTORY GENERATOR")
        print("=" * 60)
        print(f"  config        : {self.config_path}")
        print(f"  preset        : {self.preset_name} "
              f"[{self.preset.get('category', 'n/a')}]")
        print(f"  description   : {self.preset.get('description', '').strip()}")
        print(f"  arms          : {self.arms}   mode: {self.mode}")
        print(f"  duration      : {self.duration:.1f}s   delay: {self.delay_start:.1f}s")
        print(f"  dynamic_traj  : {self.dynamic_trajectory}")
        print(f"  orientation   : {self.control_orientation} "
              f"(task_dim={self.task_dimension})")
        print("=" * 60)
        print(f"[INIT] Waiting {self.delay_start:.1f}s for direct kinematics "
              f"(/qp_debug/ee_real)...\n")

    # =====================================================================
    # SUBSCRIBER CALLBACKS
    # =====================================================================
    def lambda_cb(self, msg):
        """Latest CBF shadow price; drives the dynamic time scaling."""
        self.lambda_cbf = msg.data

    def ee_real_callback(self, msg):
        """Sample the real EE state (positions + orientation) from the QP node."""
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

        current_time = time.time()
        dt = current_time - self.last_loop_time
        self.last_loop_time = current_time
        t_total = current_time - self.t0

        # --- PHASE 1: WAITING / SETUP ------------------------------------
        if t_total < self.delay_start:
            self.start_right = self.pos_r.copy()
            self.start_left = self.pos_l.copy()
            self.hold_rpy_r = self.rpy_r.copy()
            self.hold_rpy_l = self.rpy_l.copy()
            self.virtual_time = 0.0
            self.publish_references(self.start_right, np.zeros(3),
                                    self.start_left, np.zeros(3))
            self.update_phase('S', "PHASE: WAITING", 1.0, 0.8, 0.0)
            return

        # --- TRANSITION: generate the targets once -----------------------
        if not self.targets_generated:
            self.end_right, self.end_left = self._compute_targets()
            self.targets_generated = True
            print("\n[STATE] Targets generated "
                  f"(preset '{self.preset_name}', mode '{self.mode}')")
            print(f"  Right: start {self.start_right.round(3)} "
                  f"-> end {self.end_right.round(3)}")
            print(f"  Left : start {self.start_left.round(3)} "
                  f"-> end {self.end_left.round(3)}\n")

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
            if current_time - self.last_print_time > 1.0:
                print(f"[HOLDING] dist R: "
                      f"{np.linalg.norm(x_ref_r - self.pos_r):.3f}m | "
                      f"dist L: {np.linalg.norm(x_ref_l - self.pos_l):.3f}m | "
                      f"lambda: {self.lambda_cbf:.2f}")
                self.last_print_time = current_time
            self.update_phase('R', "PHASE: REGULATION", 0.0, 0.5, 1.0)
        else:
            # --- TRACKING (quintic) ---
            tau = self.virtual_time / self.duration
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
            x_ref_r = self.start_right + s * (self.end_right - self.start_right)
            xdot_ref_r = (s_dot * sigma) * (self.end_right - self.start_right)
            x_ref_l = self.start_left + s * (self.end_left - self.start_left)
            xdot_ref_l = (s_dot * sigma) * (self.end_left - self.start_left)
            if current_time - self.last_print_time > 0.5:
                print(f"[TRACKING] v-time: {self.virtual_time:.1f}s | "
                      f"speed: {sigma * 100:.0f}% | "
                      f"lambda: {self.lambda_cbf:.2f} | "
                      f"err R: {np.linalg.norm(x_ref_r - self.pos_r):.3f}m")
                self.last_print_time = current_time
            self.update_phase('T', "PHASE: TRACKING", 0.0, 1.0, 0.0)

        self.publish_references(x_ref_r, xdot_ref_r, x_ref_l, xdot_ref_l)
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
        else:  # 'absolute'
            end_r = np.array(self.preset.get('right', self.start_right), dtype=float)
            end_l = np.array(self.preset.get('left', self.start_left), dtype=float)

        # Inactive arm holds its sampled start pose (regulated in place).
        if self.arms == 'right':
            end_l = self.start_left.copy()
        elif self.arms == 'left':
            end_r = self.start_right.copy()
        return end_r, end_l

    # =====================================================================
    # PUBLISH HELPERS
    # =====================================================================
    def publish_references(self, x_r, xdot_r, x_l, xdot_l):
        """Pack and send the cartesian reference on the standard QP contract."""
        msg_r = Float64MultiArray()
        msg_l = Float64MultiArray()
        if self.control_orientation:
            # 6-DOF: position + held start orientation + lin vel + zero ang vel.
            msg_r.data = (x_r.tolist() + self.hold_rpy_r.tolist() +
                          xdot_r.tolist() + [0.0, 0.0, 0.0] +
                          [float(self.task_dimension)])
            msg_l.data = (x_l.tolist() + self.hold_rpy_l.tolist() +
                          xdot_l.tolist() + [0.0, 0.0, 0.0] +
                          [float(self.task_dimension)])
        else:
            # Position-only fallback (6-float branch in the controller).
            msg_r.data = x_r.tolist() + xdot_r.tolist()
            msg_l.data = x_l.tolist() + xdot_l.tolist()
        self.pub_ref_right.publish(msg_r)
        self.pub_ref_left.publish(msg_l)

    def publish_dashboard(self, x_r, xdot_r, x_l, xdot_l):
        """12-element [x_r, xdot_r, x_l, xdot_l] layout for the legacy plotter."""
        msg = Float64MultiArray()
        msg.data = x_r.tolist() + xdot_r.tolist() + x_l.tolist() + xdot_l.tolist()
        self.pub_dashboard.publish(msg)

    def update_phase(self, phase_char, text, r, g, b):
        if self.current_phase != phase_char:
            self.current_phase = phase_char
            self.pub_phase.publish(String(data=phase_char))
            self.publish_rviz_marker(text, r, g, b)
            print(f"[PHASE] Switched to {text}")

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

    def on_key_press(self, key):
        try:
            if key.char == '0':
                print("\n[STOP] '0' pressed - freezing motion.", flush=True)
                self.should_stop = True
                self.publish_references(self.pos_r, np.zeros(3),
                                        self.pos_l, np.zeros(3))
        except AttributeError:
            pass


def main():
    rclpy.init()
    node = TrajectoryGenerator()
    try:
        while rclpy.ok() and not node.should_stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node.listener is not None:
            try:
                node.listener.stop()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
