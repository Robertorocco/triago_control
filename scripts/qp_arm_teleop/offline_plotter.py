#!/usr/bin/env python3
# offline_plotter.py
"""
Static, publication-quality figures for the QP-CLF-CBF pipeline.

Where plotter.py is a LIVE, ever-scrolling dashboard (rolling 50 s buffers,
redrawn 10x/s, meant to be watched while the robot moves), this node records
ONE COMPLETE TRIAL from t=0 and, once it ends, renders the same telemetry as
a fixed set of formal figures (PDF + PNG, 300 dpi) suitable for inclusion in
a paper. No live window is ever shown -- this script only ever WRITES files.

Recording model (generic, source-agnostic trigger)
----------------------------------------------------
This node never decides what "a trial" is. It is driven entirely by a single
boolean topic, `cfg.OFFLINE_RECORD_TRIGGER_TOPIC`
(default: '/offline_plotter/record_trigger', std_msgs/Bool):

    True  -> start (rising edge from idle) or continue recording. On a rising
             edge, t=0 is anchored to THIS instant -- not node startup, not
             the first data sample -- so the saved time axis always starts
             exactly at 0 regardless of how long this node has been running.
    False -> the commanded motion has concluded. Recording CONTINUES for a
             further `cfg.OFFLINE_PLOT_POST_TRIGGER_S` seconds (captures the
             settling/regulation phase on the SAME time axis as the tracking
             motion) before the trial is finalized. A vertical dashed grey
             line is drawn at the exact instant of the False edge on every
             time-series subplot -- intentionally unlabeled (no legend entry)
             so it reads as a visual cue, not a data series.

`trajectory_generator.py` drives this topic today (True on WAITING->TRACKING,
False on TRACKING->REGULATION -- see its `update_phase`). Nothing in THIS
file is aware of trajectory_generator, quintic motions, or phases: any other
node may publish on the exact same topic later -- e.g. a future
teleoperation-side trigger such as "the operator is holding the handle and
the clutch is released" -- and this script picks it up with ZERO changes.

Ctrl-C: whatever has been recorded so far (even mid-trial, even before the
post-trigger grace window elapses) is finalized and saved before the process
exits.

Figures produced (mirrors plotter.py's content, minus the two windows that
are meaningless as a static artifact -- the live CBF-active-pairs debug view
and the live joint-position slider GUI):
    fig0_summary                   The 10 headline metrics below, rendered as
                                    a monospace text page (same numbers as
                                    summary_metrics.json / trial_summary.txt,
                                    just viewable as PDF/PNG alongside fig1..6)
    fig1_joint_kinematics          Position / Velocity / QP solution, 3x2
                                    (col 0 = Left arm, col 1 = Right arm)
    fig2_qp_data                   Slacks, CBF/joint shadow prices, loop
                                    frequency, safety margin, min distance
    fig3_task_error_adaptation     Cartesian tracking error + dynamic weights
    fig4_task_authority            Soft-task QP cost decomposition (shares)
    fig5_3d_trajectory              3D commanded vs. executed gripper path
                                    (solid = commanded reference, dashed =
                                    executed EE pose; red = Right, blue =
                                    Left). Saved as PDF + PNG always; ALSO
                                    saved as a browser-navigable HTML
                                    (free rotate/zoom/pan) if the optional
                                    `plotly` package is installed --
                                    `pip install plotly` to enable it, no
                                    other change needed.
    fig6_reference_governor        Commanded ("raw") vs. governed reference
                                    signal (linear/angular velocity, position/
                                    orientation tracking error), per arm, with
                                    a DASHED horizontal line marking the
                                    governor's configured limit on each
                                    quantity (cfg.GOV_V_MAX_LIN, GOV_V_MAX_ANG,
                                    GOV_E_MAX_POS, GOV_E_MAX_ORI) -- makes it
                                    immediately visible when/how much the
                                    governor reshaped the input trajectory.
                                    (only emitted if cfg.ENABLE_REFERENCE_GOVERNOR)

    summary_metrics.json           Up to 10 headline numbers ("was this trial
                                    smooth / accurate / safe") distilled from
                                    the same buffers the figures above are
                                    built from -- meant to be compared side by
                                    side across runs (e.g. a CONTROL_FREQ_DEFAULT
                                    300 vs 150 Hz A/B) without having to eyeball
                                    six figures per trial. Also echoed to the
                                    console at finalize time. See
                                    _compute_summary_metrics for the exact
                                    definitions; NaN means the source topic
                                    never produced enough data this trial, not
                                    an error.

    bag/                        A `ros2 bag record` capture spanning the exact
                                    same t=0..end window as the figures above
                                    (cfg.OFFLINE_BAG_TOPICS -- QP-controller
                                    telemetry only, no shared-autonomy/
                                    teleoperation topics), so any trial can be
                                    replayed offline later. Off via
                                    cfg.OFFLINE_BAG_ENABLE = False.
    trial_summary.txt           Plain-text copy of the console block printed
                                    at finalize time (reason, duration, figure
                                    count, the 10 summary metrics) -- so the
                                    "index" survives after the terminal scrolls
                                    away, without having to re-parse
                                    summary_metrics.json.
"""
import matplotlib
matplotlib.use('Agg')  # headless: this script only ever SAVES figures

import os
import json
import signal
import datetime
import textwrap
import subprocess

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, String, Bool

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

import triago_control.qp_controller.config as cfg

# Optional: a browser-navigable (free rotate/zoom/pan) HTML export of the 3D
# trajectory figure, on top of the always-produced static PDF/PNG. Purely
# additive -- if plotly isn't installed, the HTML export is silently skipped
# and only the static PDF/PNG (from matplotlib) are written.
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# =============================================================================
# PUBLICATION STYLE (applied once, shared by every figure this script saves)
# =============================================================================
def _apply_publication_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 15,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.4,
        'axes.linewidth': 0.9,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


JOINT_COLORS = plt.cm.jet(np.linspace(0, 1, 7))
# The "trajectory finished" marker: a plain, unlabeled vertical dashed grey
# line (per instruction -- no legend entry, no explanation needed on the plot).
TRIGGER_LINE_KW = dict(color='0.45', linestyle='--', linewidth=1.1, zorder=0)

# How long to wait for `ros2 bag record` to finalize after SIGINT before
# forcing (same grace-period pattern as scripts/analysis/study_recorder.py).
_BAG_STOP_GRACE_S = 15.0


def _draw_trigger_line(ax, t_off):
    if t_off is not None:
        ax.axvline(t_off, **TRIGGER_LINE_KW)


def _moving_average(x, win):
    if win < 3 or len(x) < win:
        return np.asarray(x, dtype=float).copy()
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, kernel, mode='valid')[:len(x)]


def _ripple_rms(t, x, detrend_window_s=0.15):
    """Moving-average-detrend residual RMS -- the SAME ripple definition used
    by freq_oscillation_diagnostic.py, duplicated here (not imported -- each
    is an independent `ros2 run` entrypoint) so the two tools' smoothness
    numbers are directly comparable side by side."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return float('nan')
    duration = max(t[-1] - t[0], 1e-6)
    fs_hint = len(x) / duration
    win = max(3, int(round(detrend_window_s * fs_hint)))
    residual = x - _moving_average(x, win)
    return float(np.sqrt(np.mean(residual ** 2)))


class OfflinePlotter(Node):
    """Records exactly one trial at a time (see module docstring) and saves
    a fixed set of static, publication-styled figures when it ends."""

    STATE_WAITING = 'WAITING'       # no trial in progress; ignoring data
    STATE_RECORDING = 'RECORDING'   # trigger is True; actively recording
    STATE_POST_ROLL = 'POST_ROLL'   # trigger went False; recording the tail

    def __init__(self):
        super().__init__('offline_plotter')

        self.left_joints = [f'arm_left_{i}_joint' for i in range(1, 8)]
        self.right_joints = [f'arm_right_{i}_joint' for i in range(1, 8)]
        self.all_joints = self.left_joints + self.right_joints

        self.state = self.STATE_WAITING
        self.t0 = None            # ROS time [s] of the rising edge (t=0 anchor)
        self.t_off = None         # trial-relative time [s] of the falling edge
        self.post_roll_deadline = None  # ROS time [s] at which to finalize
        self.trial_index = 0
        self.out_dir = None       # this trial's output folder (created at t0, not finalize)
        self.bag_proc = None      # subprocess.Popen for `ros2 bag record`, or None

        # Latest real EE pose (position + RPY), kept for the governor figure's
        # tracking-error rows -- deliberately OUTSIDE _reset_buffers: this is
        # "latest known robot state", not per-trial data, and must survive
        # across trials (and be populated from the very first tick of a new
        # trial without waiting on a fresh /qp_debug/ee_real message). See
        # cb_real (always updates it, gated only on has_ref_right, not on
        # whether a trial is in progress) and gov_callback (consumes it).
        self.last_real_pos_r = None
        self.last_real_pos_l = None
        self.last_real_rpy_r = None
        self.last_real_rpy_l = None

        self._reset_buffers()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- The one topic that governs everything (see module docstring) ---
        self.create_subscription(
            Bool, cfg.OFFLINE_RECORD_TRIGGER_TOPIC, self.record_trigger_callback, 10)

        # --- Joint state (position; velocity comes from qdot_measured instead) ---
        self.create_subscription(JointState, '/joint_states', self.listener_callback, qos_profile)

        # --- Generic measured joint velocity (sim-filtered OR real-sensor,
        # already resolved upstream by main_qp_controller/robot_kinematics --
        # see cfg.py section 7 and main_qp_controller's pub_qdot_measured) ---
        self.create_subscription(
            Float64MultiArray, '/qp_debug/qdot_measured', self.qdot_measured_callback, 10)

        # --- QP-solved joint velocity (what was actually commanded) ---
        self.create_subscription(
            Float64MultiArray, '/qp_debug/qdot_cmd', self.qdot_cmd_callback, 10)

        # --- Cartesian references + real EE state (for tracking error) ---
        self.ref_right = np.zeros(13)
        self.ref_left = np.zeros(13)
        self.has_ref_right = False
        self.has_ref_left = False
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference',
                                 self.cb_ref_right, qos_profile)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference',
                                 self.cb_ref_left, qos_profile)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.cb_real, qos_profile)

        # --- QP Data window sources ---
        self.create_subscription(Float64MultiArray, '/qp_debug/slacks', self.slack_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_cbf',
                                 self.lambda_cbf_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_joints',
                                 self.lambda_joints_callback, qos_profile)
        self.create_subscription(Float64, '/qp_debug/loop_freq', self.freq_callback, qos_profile)
        self.create_subscription(Float64, '/qp_debug/safety_margin', self.h_callback, qos_profile)
        self.create_subscription(Float64, '/qp_debug/min_distance', self.min_dist_callback, qos_profile)

        # --- Dynamic weight scheduler sources ---
        self.create_subscription(Float64MultiArray, '/qp_debug/dynamic_weights',
                                 self.dyn_weights_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/qp_debug/d_safe_dynamic',
                                 self.d_safe_callback, qos_profile)

        # --- Task authority (soft-task cost decomposition) ---
        self.create_subscription(Float64MultiArray, '/qp_debug/task_authority',
                                 self.task_authority_callback, qos_profile)

        # --- Reference governor telemetry (only meaningful if enabled) ---
        self.create_subscription(Float64MultiArray, '/qp_debug/governor',
                                 self.gov_callback, qos_profile)

        # Slow watchdog: finalizes a POST_ROLL trial once its grace window
        # elapses. Deliberately NOT tied to any particular data topic's rate.
        self.create_timer(0.2, self._check_post_roll_deadline)

        self.get_logger().info(
            f"[offline_plotter] Ready. Waiting for a rising edge on "
            f"'{cfg.OFFLINE_RECORD_TRIGGER_TOPIC}' to start recording a trial. "
            f"Output root: {os.path.expanduser(cfg.OFFLINE_PLOT_ROOT_DIR)}")

    # =====================================================================
    # RECORDING STATE MACHINE
    # =====================================================================
    def _reset_buffers(self):
        self.time_js = []
        self.q_buffers = {j: [] for j in self.all_joints}

        self.time_qdot_measured = []
        self.qdot_measured_l = []
        self.qdot_measured_r = []

        self.time_qdot_cmd = []
        self.qdot_cmd_l = []
        self.qdot_cmd_r = []

        self.time_err = []
        self.err_pos_r = []
        self.err_pos_l = []
        self.err_vel_r = []
        self.err_vel_l = []

        # Raw EE speed (not tracking error -- the actual executed speed), same
        # tick/timestamps as time_err. Used only by _compute_summary_metrics's
        # ee_speed_ripple_rms_max metric (see cb_real for where it's appended).
        self.ee_speed_r = []
        self.ee_speed_l = []

        # 3D trajectory trace (commanded reference vs. executed EE pose),
        # sampled at the SAME rate/callback as the tracking error above
        # (/qp_debug/ee_real, gated on has_ref_right) -- see cb_real.
        self.traj_time = []
        self.traj_ref_r = []    # commanded [x,y,z], Right
        self.traj_ref_l = []    # commanded [x,y,z], Left
        self.traj_real_r = []   # executed  [x,y,z], Right
        self.traj_real_l = []   # executed  [x,y,z], Left

        self.time_slack = []
        self.slack_buffer = []
        self.slack_mode = None

        self.time_lambda_cbf = []
        self.lambda_cbf_buffer = []
        self.time_lambda_joints = []
        self.lambda_joints_buffer = []
        self.time_freq = []
        self.freq_buffer = []
        self.time_h = []
        self.h_buffer = []
        self.time_min_dist = []
        self.min_dist_buffer = []

        self.time_dyn_weights = []
        self.dyn_weights_buffer = []
        self.time_d_safe = []
        self.d_safe_buffer = []

        self.time_task_auth = []
        self.task_auth_buffer = []

        self.time_gov = []
        self.gov_buffer = []
        # Governor detailed telemetry (2026-07-04): commanded ("raw") vs.
        # governed ABSOLUTE magnitudes, reconstructed from the raw reference
        # (self.ref_right/left, already tracked) and the raw-minus-governed
        # diff already published on /qp_debug/governor -- no new topic
        # needed (governed = raw - diff, algebraically exact). See
        # gov_callback and _build_fig_reference_governor.
        self.gov_lin_vel_raw_r, self.gov_lin_vel_gov_r = [], []
        self.gov_lin_vel_raw_l, self.gov_lin_vel_gov_l = [], []
        self.gov_ang_vel_raw_r, self.gov_ang_vel_gov_r = [], []
        self.gov_ang_vel_raw_l, self.gov_ang_vel_gov_l = [], []
        self.gov_pos_err_raw_r, self.gov_pos_err_gov_r = [], []
        self.gov_pos_err_raw_l, self.gov_pos_err_gov_l = [], []
        self.gov_ori_err_raw_r, self.gov_ori_err_gov_r = [], []
        self.gov_ori_err_raw_l, self.gov_ori_err_gov_l = [], []

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _t(self):
        """Trial-relative time [s], anchored to the rising-edge instant."""
        return self._now() - self.t0

    def _recording_active(self):
        return self.state in (self.STATE_RECORDING, self.STATE_POST_ROLL)

    def record_trigger_callback(self, msg: Bool):
        now = self._now()
        if msg.data:
            if self.state == self.STATE_WAITING:
                self._reset_buffers()
                self.t0 = now
                self.state = self.STATE_RECORDING
                self.trial_index += 1
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                self.out_dir = os.path.join(
                    os.path.expanduser(cfg.OFFLINE_PLOT_ROOT_DIR), f"trial_{timestamp}")
                os.makedirs(self.out_dir, exist_ok=True)
                self._start_bag_recording()
                self.get_logger().info(
                    f"[offline_plotter] Trigger TRUE -- recording started (t=0 anchored here). "
                    f"Output: {self.out_dir}")
            elif self.state == self.STATE_POST_ROLL:
                # Motion resumed before the post-roll window elapsed: extend
                # the SAME trial (and its already-running bag) rather than
                # finalizing a truncated one.
                self.state = self.STATE_RECORDING
                self.t_off = None
                self.post_roll_deadline = None
                self.get_logger().info(
                    "[offline_plotter] Trigger TRUE again during post-roll -- extending the same trial.")
            # else: already RECORDING -- no-op
        else:
            if self.state == self.STATE_RECORDING:
                self.t_off = now - self.t0
                self.state = self.STATE_POST_ROLL
                self.post_roll_deadline = now + cfg.OFFLINE_PLOT_POST_TRIGGER_S
                self.get_logger().info(
                    f"[offline_plotter] Trigger FALSE at t={self.t_off:.2f}s -- "
                    f"recording {cfg.OFFLINE_PLOT_POST_TRIGGER_S:.1f}s more "
                    f"(regulation/settling), then saving.")
            # else: already WAITING/POST_ROLL -- no-op

    def _check_post_roll_deadline(self):
        if self.state == self.STATE_POST_ROLL and self._now() >= self.post_roll_deadline:
            self._finalize_and_save(reason='post_roll_elapsed')

    # =====================================================================
    # ROSBAG CAPTURE (raw replay data, alongside the figures/metrics below)
    # =====================================================================
    def _start_bag_recording(self):
        """Spawn `ros2 bag record` for cfg.OFFLINE_BAG_TOPICS into
        <out_dir>/bag, covering the same t=0..end window as the figures.
        QP-controller telemetry only -- see cfg.OFFLINE_BAG_TOPICS for why
        shared-autonomy/teleoperation topics are excluded here."""
        if not cfg.OFFLINE_BAG_ENABLE:
            self.bag_proc = None
            return
        bag_dir = os.path.join(self.out_dir, 'bag')
        cmd = ["ros2", "bag", "record", "-s", cfg.OFFLINE_BAG_STORAGE_ID,
               "-o", bag_dir, *cfg.OFFLINE_BAG_TOPICS]
        try:
            # New session so we can SIGINT the whole process group cleanly,
            # same as scripts/analysis/study_recorder.py.
            self.bag_proc = subprocess.Popen(cmd, start_new_session=True)
            self.get_logger().info(f"[offline_plotter] bag recording -> {bag_dir}")
        except FileNotFoundError:
            self.get_logger().warn(
                "[offline_plotter] 'ros2' not found on PATH -- skipping bag "
                "recording for this trial (figures/metrics are unaffected).")
            self.bag_proc = None

    def _stop_bag_recording(self):
        """SIGINT the bag process group so rosbag2 finalizes its files
        cleanly, waiting up to _BAG_STOP_GRACE_S before forcing."""
        if self.bag_proc is None:
            return
        try:
            os.killpg(os.getpgid(self.bag_proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self.bag_proc.wait(timeout=_BAG_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self.get_logger().warn(
                "[offline_plotter] bag record did not stop in time -- terminating.")
            self.bag_proc.terminate()
            try:
                self.bag_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.bag_proc.kill()
        self.bag_proc = None

    # =====================================================================
    # DATA CALLBACKS (all short-circuit unless a trial is in progress)
    # =====================================================================
    def listener_callback(self, msg):
        if not self._recording_active():
            return
        if len(msg.name) != len(msg.position):
            return
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if not all(j in name_to_idx for j in self.all_joints):
            return
        self.time_js.append(self._t())
        for j in self.all_joints:
            self.q_buffers[j].append(msg.position[name_to_idx[j]])

    def qdot_measured_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 14:
            return
        self.time_qdot_measured.append(self._t())
        self.qdot_measured_r.append(list(msg.data[:7]))
        self.qdot_measured_l.append(list(msg.data[7:14]))

    def qdot_cmd_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 14:
            return
        self.time_qdot_cmd.append(self._t())
        self.qdot_cmd_r.append(list(msg.data[:7]))
        self.qdot_cmd_l.append(list(msg.data[7:14]))

    def cb_ref_right(self, msg):
        if len(msg.data) >= 12:
            self.ref_right = np.array(msg.data)
            self.has_ref_right = True

    def cb_ref_left(self, msg):
        if len(msg.data) >= 12:
            self.ref_left = np.array(msg.data)
            self.has_ref_left = True

    def cb_real(self, msg):
        if not self.has_ref_right:
            return
        real = np.array(msg.data)
        p_real_r, v_real_r = real[0:3], real[3:6]
        p_real_l, v_real_l = real[6:9], real[9:12]
        # Real orientation (RPY), if present -- see /qp_debug/ee_real's 18-float
        # layout: [p_r(3), v_r(3), p_l(3), v_l(3), rpy_r(3), rpy_l(3)]. Cached
        # unconditionally (even outside a trial) so the governor figure's
        # geodesic-orientation-error reconstruction always has the LATEST real
        # pose available the instant a trial starts -- avoids an empty first
        # sample. Cached regardless of _recording_active() for the same reason
        # ref_right/ref_left already are.
        self.last_real_pos_r, self.last_real_pos_l = p_real_r.copy(), p_real_l.copy()
        if len(real) >= 18:
            self.last_real_rpy_r, self.last_real_rpy_l = real[12:15].copy(), real[15:18].copy()

        if not self._recording_active():
            return

        e_p_r = float(np.linalg.norm(self.ref_right[0:3] - p_real_r))
        e_v_r = float(np.linalg.norm(self.ref_right[6:9] - v_real_r))
        e_p_l = float(np.linalg.norm(self.ref_left[0:3] - p_real_l)) if self.has_ref_left else 0.0
        e_v_l = float(np.linalg.norm(self.ref_left[6:9] - v_real_l)) if self.has_ref_left else 0.0

        self.time_err.append(self._t())
        self.err_pos_r.append(e_p_r)
        self.err_pos_l.append(e_p_l)
        self.err_vel_r.append(e_v_r)
        self.err_vel_l.append(e_v_l)
        self.ee_speed_r.append(float(np.linalg.norm(v_real_r)))
        self.ee_speed_l.append(float(np.linalg.norm(v_real_l)))

        # 3D trajectory trace -- same tick, same anchors as the error above.
        self.traj_time.append(self._t())
        self.traj_ref_r.append(self.ref_right[0:3].copy())
        self.traj_real_r.append(p_real_r.copy())
        if self.has_ref_left:
            self.traj_ref_l.append(self.ref_left[0:3].copy())
        else:
            self.traj_ref_l.append(np.full(3, np.nan))
        self.traj_real_l.append(p_real_l.copy())

    def slack_callback(self, msg):
        if not self._recording_active():
            return
        data = list(msg.data)
        if self.slack_mode is None:
            if len(data) == 2:
                self.slack_mode = 'scalar'
            elif len(data) == 6:
                self.slack_mode = 'vector'
            else:
                return
        self.time_slack.append(self._t())
        self.slack_buffer.append(data)

    def lambda_cbf_callback(self, msg):
        if not self._recording_active():
            return
        data = list(msg.data)
        if len(data) < 2:
            data = (data + [0.0, 0.0])[:2]
        self.time_lambda_cbf.append(self._t())
        self.lambda_cbf_buffer.append(data)

    def lambda_joints_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 2:
            return
        self.time_lambda_joints.append(self._t())
        self.lambda_joints_buffer.append(list(msg.data[:2]))

    def freq_callback(self, msg):
        if not self._recording_active():
            return
        self.time_freq.append(self._t())
        self.freq_buffer.append(msg.data)

    def h_callback(self, msg):
        if not self._recording_active():
            return
        self.time_h.append(self._t())
        self.h_buffer.append(msg.data)

    def min_dist_callback(self, msg):
        if not self._recording_active():
            return
        self.time_min_dist.append(self._t())
        self.min_dist_buffer.append(msg.data)

    def dyn_weights_callback(self, msg):
        if not self._recording_active():
            return
        data = list(msg.data)
        if len(data) < 3:
            data = (data + [0.0, 0.0, 0.0])[:3]
        self.time_dyn_weights.append(self._t())
        self.dyn_weights_buffer.append(data)

    def d_safe_callback(self, msg):
        if not self._recording_active():
            return
        data = list(msg.data)
        if len(data) < 2:
            data = (data + [0.0, 0.0])[:2]
        self.time_d_safe.append(self._t())
        self.d_safe_buffer.append(data)

    def task_authority_callback(self, msg):
        # 3 floats = [damp, posture, slack] (cfg.ENABLE_RATE_DAMPING was False
        # when this trial was recorded); 4 floats also carries [..., rate].
        # Padded to 4 either way so _build_fig_task_authority can index freely.
        if not self._recording_active() or len(msg.data) < 3:
            return
        data = list(msg.data[:4])
        if len(data) < 4:
            data.append(0.0)
        self.time_task_auth.append(self._t())
        self.task_auth_buffer.append(data)

    @staticmethod
    def _geodesic_angle(rpy, R_real):
        """||log3(R_des . R_real^T)|| -- same construction as
        ReferenceGovernor._clamp_orientation_error, used here purely as a
        READ-ONLY diagnostic (never fed back into control)."""
        if rpy is None or R_real is None:
            return 0.0
        R_des = pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        R_error = R_des @ np.asarray(R_real).T
        trace = np.clip(np.trace(R_error), -1.0, 3.0)
        if trace <= -1.0 + 1e-6:
            return float(np.pi)  # near-singularity guard, mirrors the governor's own code
        return float(np.linalg.norm(pin.log3(R_error)))

    def gov_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 24:
            return
        self.time_gov.append(self._t())
        self.gov_buffer.append(list(msg.data[:24]))

        # --- Reconstruct RAW vs. GOVERNED absolute signals (2026-07-04) ---
        # /qp_debug/governor already carries (raw - governed) per DOF; the raw
        # side is exactly self.ref_right/left (already tracked from
        # /arm_*/cartesian_reference), so governed = raw - diff -- no new
        # topic/publisher was needed. Reconstructed HERE (not in
        # main_qp_controller.py) to keep that node's telemetry contract
        # unchanged; this is purely a plotting-side convenience.
        diff = np.array(msg.data[:24], dtype=float)
        pos_diff_r, ori_diff_r, vel_diff_r, wvel_diff_r = diff[0:3], diff[3:6], diff[6:9], diff[9:12]
        pos_diff_l, ori_diff_l, vel_diff_l, wvel_diff_l = diff[12:15], diff[15:18], diff[18:21], diff[21:24]

        raw_pos_r, raw_rpy_r = self.ref_right[0:3], self.ref_right[3:6]
        raw_vel_r, raw_wvel_r = self.ref_right[6:9], self.ref_right[9:12]
        raw_pos_l, raw_rpy_l = self.ref_left[0:3], self.ref_left[3:6]
        raw_vel_l, raw_wvel_l = self.ref_left[6:9], self.ref_left[9:12]

        gov_pos_r, gov_rpy_r = raw_pos_r - pos_diff_r, raw_rpy_r - ori_diff_r
        gov_vel_r, gov_wvel_r = raw_vel_r - vel_diff_r, raw_wvel_r - wvel_diff_r
        gov_pos_l, gov_rpy_l = raw_pos_l - pos_diff_l, raw_rpy_l - ori_diff_l
        gov_vel_l, gov_wvel_l = raw_vel_l - vel_diff_l, raw_wvel_l - wvel_diff_l

        # Velocity magnitudes -- directly comparable to GOV_V_MAX_LIN/ANG (the
        # governor clamps the MAGNITUDE, direction preserved -- see
        # ReferenceGovernor._clamp_velocity).
        self.gov_lin_vel_raw_r.append(float(np.linalg.norm(raw_vel_r)))
        self.gov_lin_vel_gov_r.append(float(np.linalg.norm(gov_vel_r)))
        self.gov_lin_vel_raw_l.append(float(np.linalg.norm(raw_vel_l)))
        self.gov_lin_vel_gov_l.append(float(np.linalg.norm(gov_vel_l)))
        self.gov_ang_vel_raw_r.append(float(np.linalg.norm(raw_wvel_r)))
        self.gov_ang_vel_gov_r.append(float(np.linalg.norm(gov_wvel_r)))
        self.gov_ang_vel_raw_l.append(float(np.linalg.norm(raw_wvel_l)))
        self.gov_ang_vel_gov_l.append(float(np.linalg.norm(gov_wvel_l)))

        # Tracking-error magnitudes -- directly comparable to GOV_E_MAX_POS/ORI
        # (the governor bounds the ERROR the CLF perceives, i.e. ||ref-real||,
        # NOT the reference itself -- see ReferenceGovernor._bound_position_error
        # / _clamp_orientation_error). Uses the latest cached real EE pose
        # (same downsampled tick as this message -- see cb_real).
        real_pos_r = self.last_real_pos_r if self.last_real_pos_r is not None else raw_pos_r
        real_pos_l = self.last_real_pos_l if self.last_real_pos_l is not None else raw_pos_l
        self.gov_pos_err_raw_r.append(float(np.linalg.norm(raw_pos_r - real_pos_r)))
        self.gov_pos_err_gov_r.append(float(np.linalg.norm(gov_pos_r - real_pos_r)))
        self.gov_pos_err_raw_l.append(float(np.linalg.norm(raw_pos_l - real_pos_l)))
        self.gov_pos_err_gov_l.append(float(np.linalg.norm(gov_pos_l - real_pos_l)))

        real_rpy_r = self.last_real_rpy_r
        real_rpy_l = self.last_real_rpy_l
        R_real_r = pin.rpy.rpyToMatrix(*real_rpy_r) if real_rpy_r is not None else None
        R_real_l = pin.rpy.rpyToMatrix(*real_rpy_l) if real_rpy_l is not None else None
        self.gov_ori_err_raw_r.append(self._geodesic_angle(raw_rpy_r, R_real_r))
        self.gov_ori_err_gov_r.append(self._geodesic_angle(gov_rpy_r, R_real_r))
        self.gov_ori_err_raw_l.append(self._geodesic_angle(raw_rpy_l, R_real_l))
        self.gov_ori_err_gov_l.append(self._geodesic_angle(gov_rpy_l, R_real_l))

    # =====================================================================
    # FINALIZATION: build + save every figure, then reset for the next trial
    # =====================================================================
    def _finalize_and_save(self, reason):
        if self.t0 is None or not self.time_js and not self.time_qdot_cmd:
            self.get_logger().warn(
                "[offline_plotter] Finalize requested but no data was recorded -- skipping save.")
            self._stop_bag_recording()
            self.state = self.STATE_WAITING
            self.t0 = None
            self.t_off = None
            self.out_dir = None
            return

        out_dir = self.out_dir
        trial_name = os.path.basename(out_dir)
        bag_was_running = self.bag_proc is not None
        self._stop_bag_recording()

        self.get_logger().info(
            f"[offline_plotter] Finalizing trial #{self.trial_index} ({reason}) -- saving to {out_dir}")

        trial_duration = self.time_qdot_cmd[-1] if self.time_qdot_cmd else \
            (self.time_js[-1] if self.time_js else 0.0)
        metrics = self._compute_summary_metrics(trial_duration)
        bag_note = 'yes -> bag/' if bag_was_running else 'disabled (cfg.OFFLINE_BAG_ENABLE=False)'

        figs = []
        figs.append(('fig0_summary', self._build_fig_summary(
            trial_name, reason, out_dir, trial_duration, bag_note, metrics)))
        figs.append(('fig1_joint_kinematics', self._build_fig_joint_kinematics()))
        figs.append(('fig2_qp_data', self._build_fig_qp_data()))
        figs.append(('fig3_task_error_adaptation', self._build_fig_task_error_adaptation()))
        figs.append(('fig4_task_authority', self._build_fig_task_authority()))
        figs.append(('fig5_3d_trajectory', self._build_fig_3d_trajectory()))
        if cfg.ENABLE_REFERENCE_GOVERNOR:
            fig_gov = self._build_fig_reference_governor()
            if fig_gov is not None:
                figs.append(('fig6_reference_governor', fig_gov))

        for name, fig in figs:
            if fig is None:
                continue
            pdf_path = os.path.join(out_dir, f"{name}.pdf")
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(pdf_path)
            fig.savefig(png_path)
            plt.close(fig)
            self.get_logger().info(f"[offline_plotter]   saved {name}.pdf / {name}.png")

        # Optional interactive/navigable HTML export of the 3D trajectory
        # figure (plotly, opened in any browser -- free rotate/zoom/pan).
        # Purely additive on top of the PDF/PNG saved above.
        if self.traj_time:
            if _HAS_PLOTLY:
                html_path = os.path.join(out_dir, 'fig5_3d_trajectory.html')
                self._save_3d_trajectory_html(html_path)
                self.get_logger().info(f"[offline_plotter]   saved fig5_3d_trajectory.html (interactive)")
            else:
                self.get_logger().info(
                    "[offline_plotter]   skipped interactive 3D HTML export "
                    "('plotly' not installed -- pip install plotly to enable it).")

        # --- Summary metrics (already computed above, for fig0_summary) ---
        metrics_path = os.path.join(out_dir, 'summary_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'trial': trial_name,
                'control_freq_hz': cfg.CONTROL_FREQ_DEFAULT,
                'reason': reason,
                'metrics': metrics,
            }, f, indent=2)

        # --- Console index -- built as lines so the exact text printed below
        # is ALSO saved to disk (trial_summary.txt), not console-only. ---
        lines = [
            "=" * 70,
            " OFFLINE PLOTTER -- TRIAL SAVED",
            "=" * 70,
            f"  reason           : {reason}",
            f"  output directory : {out_dir}",
            f"  bag recorded     : {bag_note}",
            f"  trial duration   : {trial_duration:.2f} s "
            f"(trajectory-finished marker at "
            f"{self.t_off if self.t_off is not None else 'N/A'})",
            f"  figures saved    : {len(figs)}",
            "=" * 70,
            " SUMMARY METRICS  (CONTROL_FREQ_DEFAULT="
            f"{cfg.CONTROL_FREQ_DEFAULT:.0f} Hz, see summary_metrics.json)",
            "-" * 70,
        ]
        for key, m in metrics.items():
            lines.append(f"  {key:<32}: {m['value']}")
        lines.append("=" * 70)

        for line in lines:
            print(line, flush=True)
        with open(os.path.join(out_dir, 'trial_summary.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")

        # Reset for the next trial.
        self._reset_buffers()
        self.state = self.STATE_WAITING
        self.t0 = None
        self.t_off = None
        self.out_dir = None
        self.post_roll_deadline = None

    # =====================================================================
    # SUMMARY METRICS
    # =====================================================================
    def _compute_summary_metrics(self, trial_duration):
        """Up to 10 headline numbers summarizing 'how this trial went', built
        purely from buffers the figures above already populate (no new
        subscriptions). Meant to be compared side by side across runs -- e.g.
        a CONTROL_FREQ_DEFAULT 300 vs 150 Hz A/B -- alongside
        freq_oscillation_diagnostic.py's per-arm bucketed verdict. NaN means
        the source topic never produced enough data this trial, not an error
        (e.g. min_observed_distance_m is NaN whenever nothing came within
        cfg.DISTANCE_FILTER_THRESHOLD of anything all trial).
        """
        def _rms(*arrays):
            vals = [np.asarray(a, dtype=float).ravel() for a in arrays if len(a)]
            if not vals:
                return float('nan')
            allv = np.concatenate(vals)
            return float(np.sqrt(np.mean(allv ** 2)))

        freq_arr = np.asarray(self.freq_buffer, dtype=float)
        mean_freq = float(np.mean(freq_arr)) if freq_arr.size else float('nan')
        freq_jitter_pct = (100.0 * float(np.std(freq_arr)) / mean_freq
                          if freq_arr.size and mean_freq > 1e-9 else float('nan'))

        ripple_r = _ripple_rms(self.time_err, self.ee_speed_r)
        ripple_l = _ripple_rms(self.time_err, self.ee_speed_l)
        ripple_candidates = [v for v in (ripple_r, ripple_l) if not np.isnan(v)]
        ripple_max = max(ripple_candidates) if ripple_candidates else float('nan')

        n_r = min(len(self.qdot_cmd_r), len(self.qdot_measured_r))
        n_l = min(len(self.qdot_cmd_l), len(self.qdot_measured_l))
        mismatch_parts = []
        if n_r:
            mismatch_parts.append(np.array(self.qdot_cmd_r[:n_r]) - np.array(self.qdot_measured_r[:n_r]))
        if n_l:
            mismatch_parts.append(np.array(self.qdot_cmd_l[:n_l]) - np.array(self.qdot_measured_l[:n_l]))
        qdot_mismatch_rms = _rms(*mismatch_parts) if mismatch_parts else float('nan')

        slack_arr = np.asarray(self.slack_buffer, dtype=float)
        mean_abs_slack = float(np.mean(np.abs(slack_arr))) if slack_arr.size else float('nan')

        lambda_arr = np.asarray(self.lambda_cbf_buffer, dtype=float)
        max_lambda_cbf = float(np.max(lambda_arr)) if lambda_arr.size else float('nan')

        md_arr = np.asarray(self.min_dist_buffer, dtype=float)
        finite_md = md_arr[np.isfinite(md_arr)] if md_arr.size else md_arr
        min_observed_distance = float(np.min(finite_md)) if finite_md.size else float('nan')

        gov_activity_frac = float('nan')
        if self.gov_buffer:
            gov_arr = np.asarray(self.gov_buffer, dtype=float)
            gov_activity_frac = float(np.mean(np.linalg.norm(gov_arr, axis=1) > 1e-3))

        def _entry(value, note):
            v = round(value, 5) if isinstance(value, float) and not np.isnan(value) else value
            return {'value': v, 'note': note}

        return {
            'duration_s': _entry(round(trial_duration, 3),
                'trial length (trajectory-finished marker to post-roll end)'),
            'mean_loop_freq_hz': _entry(mean_freq,
                f'measured vs configured CONTROL_FREQ_DEFAULT={cfg.CONTROL_FREQ_DEFAULT:.0f}'),
            'loop_freq_jitter_pct': _entry(freq_jitter_pct,
                'std/mean of loop_freq -- timing regularity, lower is steadier'),
            'ee_pos_tracking_rms_m': _entry(_rms(self.err_pos_r, self.err_pos_l),
                'combined R+L Cartesian position tracking error (accuracy)'),
            'ee_speed_ripple_rms_max_mps': _entry(ripple_max,
                'worse-arm EE-speed ripple (same definition as freq_oscillation_diagnostic.py)'),
            'qdot_cmd_vs_measured_rms_rads': _entry(qdot_mismatch_rms,
                'joint-velocity command/execution mismatch, 14 joints both arms (low-level fidelity)'),
            'mean_abs_slack': _entry(mean_abs_slack,
                'CLF slack usage -- how much tracking relaxed against constraints'),
            'max_lambda_cbf': _entry(max_lambda_cbf,
                'peak CBF shadow price -- how hard the collision barrier engaged'),
            'min_observed_distance_m': _entry(min_observed_distance,
                'closest approach seen (NaN = nothing within DISTANCE_FILTER_THRESHOLD all trial)'),
            'governor_activity_frac': _entry(gov_activity_frac,
                'fraction of ticks the reference governor visibly reshaped the reference'),
        }

    # =====================================================================
    # FIGURE BUILDERS
    # =====================================================================
    def _max_time(self, *time_lists):
        candidates = [tl[-1] for tl in time_lists if tl]
        return max(candidates) if candidates else 1.0

    def _build_fig_summary(self, trial_name, reason, out_dir, trial_duration, bag_note, metrics):
        """A publication-styled TEXT page (fig0_summary) rendering the same
        headline numbers as trial_summary.txt / summary_metrics.json, so
        'how did this trial go' is visible right alongside fig1..fig6 instead
        of requiring a separate text-file open."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        fig.suptitle(f'Trial Summary -- {trial_name}', fontsize=14, fontweight='bold')

        header_lines = [
            f"reason            : {reason}",
            f"control_freq_hz   : {cfg.CONTROL_FREQ_DEFAULT:.0f}",
            f"bag recorded      : {bag_note}",
            f"trial duration    : {trial_duration:.2f} s (trajectory-finished "
            f"marker at {self.t_off if self.t_off is not None else 'N/A'})",
            f"output directory  : {out_dir}",
            "",
            "-" * 78,
            f"{'METRIC':<32}{'VALUE':<14}NOTE",
            "-" * 78,
        ]

        body_lines = []
        for key, m in metrics.items():
            wrapped = textwrap.wrap(m['note'], width=44) or ['']
            body_lines.append(f"{key:<32}{str(m['value']):<14}{wrapped[0]}")
            for cont in wrapped[1:]:
                body_lines.append(f"{'':<46}{cont}")

        text = "\n".join(header_lines + body_lines)
        ax.text(0.01, 0.97, text, transform=ax.transAxes, family='monospace',
               fontsize=8.5, va='top', ha='left')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _build_fig_joint_kinematics(self):
        """3 rows (Position / Velocity / QP solution) x 2 cols (Left / Right)."""
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(11, 9))
        fig.suptitle('Joint Kinematics -- Bimanual QP-CLF-CBF Trial')

        axs[0, 0].set_title('Left Arm -- Joint Position')
        axs[0, 1].set_title('Right Arm -- Joint Position')
        axs[1, 0].set_title('Left Arm -- Joint Velocity')
        axs[1, 1].set_title('Right Arm -- Joint Velocity')
        axs[2, 0].set_title('Left Arm -- QP Solution ($\\dot{q}_{safe}$)')
        axs[2, 1].set_title('Right Arm -- QP Solution ($\\dot{q}_{safe}$)')

        # Row 0: Position (from /joint_states)
        if self.time_js:
            t = self.time_js
            for i, j in enumerate(self.left_joints):
                axs[0, 0].plot(t, self.q_buffers[j], color=JOINT_COLORS[i])
            for i, j in enumerate(self.right_joints):
                axs[0, 1].plot(t, self.q_buffers[j], color=JOINT_COLORS[i])

        # Row 1: Velocity (generic measured signal -- sim-filtered or
        # real-sensor, already resolved upstream; see /qp_debug/qdot_measured)
        if self.time_qdot_measured:
            t = self.time_qdot_measured
            arr_l = np.array(self.qdot_measured_l)
            arr_r = np.array(self.qdot_measured_r)
            for i in range(7):
                axs[1, 0].plot(t, arr_l[:, i], color=JOINT_COLORS[i])
                axs[1, 1].plot(t, arr_r[:, i], color=JOINT_COLORS[i])

        # Row 2: QP solution (commanded velocity)
        if self.time_qdot_cmd:
            t = self.time_qdot_cmd
            arr_l = np.array(self.qdot_cmd_l)
            arr_r = np.array(self.qdot_cmd_r)
            for i in range(7):
                axs[2, 0].plot(t, arr_l[:, i], color=JOINT_COLORS[i])
                axs[2, 1].plot(t, arr_r[:, i], color=JOINT_COLORS[i])

        axs[0, 0].set_ylabel('Position [rad]')
        axs[1, 0].set_ylabel('Velocity [rad/s]')
        axs[2, 0].set_ylabel('Velocity [rad/s]')
        axs[2, 0].set_xlabel('Time [s]')
        axs[2, 1].set_xlabel('Time [s]')

        max_t = self._max_time(self.time_js, self.time_qdot_measured, self.time_qdot_cmd)
        for ax in axs.flatten():
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)

        legend_handles = [Line2D([0], [0], color=JOINT_COLORS[i], label=f'J{i+1}') for i in range(7)]
        fig.legend(handles=legend_handles, loc='upper center', ncol=7,
                  bbox_to_anchor=(0.5, 0.965))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

    def _build_fig_qp_data(self):
        fig, axs = plt.subplots(7, 1, sharex=True, figsize=(8, 12))
        fig.suptitle('QP Data -- Slacks, Shadow Prices, Loop Health')

        # Row 0/1: Slack (right/left)
        if self.slack_mode == 'scalar' and self.time_slack:
            data = np.array(self.slack_buffer)
            axs[0].plot(self.time_slack, data[:, 0], 'r-', label=r'$\delta_{R}$')
            axs[1].plot(self.time_slack, data[:, 1], 'b-', label=r'$\delta_{L}$')
        elif self.slack_mode == 'vector' and self.time_slack:
            data = np.array(self.slack_buffer)
            for i, comp in enumerate(['x', 'y', 'z']):
                axs[0].plot(self.time_slack, data[:, i], label=fr'$\delta_{{R,{comp}}}$')
                axs[1].plot(self.time_slack, data[:, 3 + i], label=fr'$\delta_{{L,{comp}}}$')
        axs[0].set_title('Right-Arm CLF Slack')
        axs[1].set_title('Left-Arm CLF Slack')
        axs[0].set_ylabel(r'$\delta_R$')
        axs[1].set_ylabel(r'$\delta_L$')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')

        # Row 2: CBF shadow prices
        if self.time_lambda_cbf:
            data = np.array(self.lambda_cbf_buffer)
            axs[2].plot(self.time_lambda_cbf, data[:, 0], 'r-', label=r'$\lambda_{CBF,R}$')
            axs[2].plot(self.time_lambda_cbf, data[:, 1], 'b-', label=r'$\lambda_{CBF,L}$')
        axs[2].set_title('CBF Shadow Price (per arm)')
        axs[2].set_ylabel(r'$\lambda_{CBF}$')
        axs[2].legend(loc='upper right')

        # Row 3: Joint-limit shadow prices
        if self.time_lambda_joints:
            data = np.array(self.lambda_joints_buffer)
            axs[3].plot(self.time_lambda_joints, data[:, 0], 'r-', label=r'$\lambda_{Joints,R}$')
            axs[3].plot(self.time_lambda_joints, data[:, 1], 'b-', label=r'$\lambda_{Joints,L}$')
        axs[3].set_title('Joint-Limit Shadow Price (per arm)')
        axs[3].set_ylabel(r'$\lambda_{Joints}$')
        axs[3].legend(loc='upper right')

        # Row 4: Loop frequency
        if self.time_freq:
            axs[4].plot(self.time_freq, self.freq_buffer, 'g-', label='Loop frequency')
        axs[4].set_title('Control Loop Frequency')
        axs[4].set_ylabel('Freq [Hz]')
        axs[4].legend(loc='upper right')

        # Row 5: Safety margin
        if self.time_h:
            axs[5].plot(self.time_h, self.h_buffer, 'm-', label='SoftMin margin $h$')
        axs[5].axhline(0, color='r', linestyle='--', linewidth=1)
        axs[5].set_title('CBF Safety Margin')
        axs[5].set_ylabel('Margin [m]')
        axs[5].legend(loc='upper right')

        # Row 6: Minimum distance
        if self.time_min_dist:
            axs[6].plot(self.time_min_dist, self.min_dist_buffer, 'c-', label='Abs. min. distance')
        axs[6].axhline(0, color='r', linestyle='--', linewidth=1)
        axs[6].set_title('Absolute Minimum Collision Distance')
        axs[6].set_ylabel('Dist [m]')
        axs[6].legend(loc='upper right')
        axs[6].set_xlabel('Time [s]')

        max_t = self._max_time(self.time_slack, self.time_lambda_cbf, self.time_lambda_joints,
                               self.time_freq, self.time_h, self.time_min_dist)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    def _build_fig_task_error_adaptation(self):
        dyn_plots = []
        if cfg.DYNAMIC_CBF:
            dyn_plots.append(('d_safe_dynamic', r'$d_{safe}^{dyn}$ [m]', 'Dynamic Safety Margin'))
        if cfg.DYNAMIC_GAMMA_CLF:
            dyn_plots.append(('gamma_clf', r'$\gamma_{CLF}$', 'CLF Convergence Rate'))
        if cfg.DYNAMIC_SLACK_WEIGHT:
            dyn_plots.append(('weight_slack', r'$w_{\delta}$', 'Slack Weight'))

        n_rows = 2 + len(dyn_plots)
        fig, axs = plt.subplots(n_rows, 1, sharex=True, squeeze=False, figsize=(8, 3 * n_rows))
        axs = axs.flatten()
        fig.suptitle('Task Tracking Error and Adaptive Weighting')

        if self.time_err:
            axs[0].plot(self.time_err, self.err_pos_r, 'r-', label='Position error -- Right')
            axs[0].plot(self.time_err, self.err_pos_l, 'b-', label='Position error -- Left')
        axs[0].set_title('Cartesian Position Tracking Error')
        axs[0].set_ylabel('Error [m]')
        axs[0].legend(loc='upper right')

        if self.time_err:
            axs[1].plot(self.time_err, self.err_vel_r, 'r-', label='Velocity error -- Right')
            axs[1].plot(self.time_err, self.err_vel_l, 'b-', label='Velocity error -- Left')
        axs[1].set_title('Cartesian Velocity Tracking Error')
        axs[1].set_ylabel('Error [m/s]')
        axs[1].legend(loc='upper right')

        for idx, (key, ylabel, title) in enumerate(dyn_plots):
            ax = axs[2 + idx]
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if key == 'weight_slack' and self.time_dyn_weights:
                data = np.array(self.dyn_weights_buffer)
                ax.plot(self.time_dyn_weights, data[:, 0], 'r-', label=r'$w_{\delta,R}$')
                ax.plot(self.time_dyn_weights, data[:, 1], 'b-', label=r'$w_{\delta,L}$')
                ax.legend(loc='upper right')
            elif key == 'gamma_clf' and self.time_dyn_weights:
                data = np.array(self.dyn_weights_buffer)
                ax.plot(self.time_dyn_weights, data[:, 2], 'm-')
            elif key == 'd_safe_dynamic' and self.time_d_safe:
                data = np.array(self.d_safe_buffer)
                ax.plot(self.time_d_safe, data[:, 0], 'r-', label=r'$d_{safe,R}^{dyn}$')
                ax.plot(self.time_d_safe, data[:, 1], 'b-', label=r'$d_{safe,L}^{dyn}$')
                ax.legend(loc='upper right')

        axs[-1].set_xlabel('Time [s]')
        max_t = self._max_time(self.time_err, self.time_dyn_weights, self.time_d_safe)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _build_fig_task_authority(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        fig.suptitle('Soft-Task QP Cost Decomposition')
        ax.set_title('Normalized Objective Share at the QP Solution')

        if self.time_task_auth:
            arr = np.array(self.task_auth_buffer, dtype=float)
            total = arr.sum(axis=1, keepdims=True)
            total[total < 1e-12] = 1.0
            shares = arr / total
            ax.plot(self.time_task_auth, shares[:, 0], color='#888888', label='Damping')
            ax.plot(self.time_task_auth, shares[:, 1], color='#2a9d8f', label='Posture / limit')
            ax.plot(self.time_task_auth, shares[:, 2], color='#e63946', label='Slack (CLF give)')
            # Rate-damping (cfg.ENABLE_RATE_DAMPING, ||dq-dq_prev||^2 -- see
            # qp_formulator.py's build_and_solve §A) -- 0 share whenever it was
            # disabled for this trial, plotted anyway so an A/B is a straight
            # before/after comparison of the same figure.
            ax.plot(self.time_task_auth, shares[:, 3], color='#457b9d', label='Rate damping (smoothness)')

        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel('Authority share [-]')
        ax.set_xlabel('Time [s]')
        ax.legend(loc='upper left', ncol=4, fontsize=8)
        max_t = self._max_time(self.time_task_auth)
        ax.set_xlim(0, max_t)
        _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

    def _build_fig_3d_trajectory(self):
        """3D commanded-vs-executed gripper path for both arms.

        Solid line  = the Cartesian reference commanded by the active
                      trajectory source (trajectory_generator.py today; any
                      future source publishing on /arm_*/cartesian_reference
                      works identically, per the existing reference contract).
        Dashed line = the REAL gripper pose actually achieved (/qp_debug/ee_real).
        Red = Right hand, Blue = Left hand -- same color convention as every
        other per-arm plot in this dashboard and in plotter.py.
        """
        if not self.traj_time:
            return None

        ref_r = np.array(self.traj_ref_r)
        ref_l = np.array(self.traj_ref_l)
        real_r = np.array(self.traj_real_r)
        real_l = np.array(self.traj_real_l)

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle('Commanded vs. Executed Gripper Trajectory (3D)')

        ax.plot(ref_r[:, 0], ref_r[:, 1], ref_r[:, 2], color='r', linestyle='-',
               linewidth=1.8, label='Right -- commanded')
        ax.plot(real_r[:, 0], real_r[:, 1], real_r[:, 2], color='r', linestyle='--',
               linewidth=1.6, label='Right -- executed')

        if not np.all(np.isnan(ref_l)):
            ax.plot(ref_l[:, 0], ref_l[:, 1], ref_l[:, 2], color='b', linestyle='-',
                   linewidth=1.8, label='Left -- commanded')
        ax.plot(real_l[:, 0], real_l[:, 1], real_l[:, 2], color='b', linestyle='--',
               linewidth=1.6, label='Left -- executed')

        # Start/end markers -- helps a static print reader orient the path
        # without needing to rotate the view (which a PDF/PNG cannot do).
        ax.scatter(*real_r[0], color='r', marker='o', s=40, zorder=5)
        ax.scatter(*real_r[-1], color='r', marker='X', s=55, zorder=5)
        ax.scatter(*real_l[0], color='b', marker='o', s=40, zorder=5)
        ax.scatter(*real_l[-1], color='b', marker='X', s=55, zorder=5)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_title('Solid = commanded reference, Dashed = executed EE pose  '
                     '($\\circ$ = start, $\\times$ = end)', fontsize=9)
        fig.tight_layout()
        return fig

    def _save_3d_trajectory_html(self, html_path):
        """Interactive, browser-navigable (free rotate/zoom/pan) counterpart
        to _build_fig_3d_trajectory, written only if plotly is available."""
        ref_r = np.array(self.traj_ref_r)
        ref_l = np.array(self.traj_ref_l)
        real_r = np.array(self.traj_real_r)
        real_l = np.array(self.traj_real_l)

        traces = [
            go.Scatter3d(x=ref_r[:, 0], y=ref_r[:, 1], z=ref_r[:, 2], mode='lines',
                        line=dict(color='red', width=5), name='Right -- commanded'),
            go.Scatter3d(x=real_r[:, 0], y=real_r[:, 1], z=real_r[:, 2], mode='lines',
                        line=dict(color='red', width=4, dash='dash'), name='Right -- executed'),
        ]
        if not np.all(np.isnan(ref_l)):
            traces.append(go.Scatter3d(x=ref_l[:, 0], y=ref_l[:, 1], z=ref_l[:, 2], mode='lines',
                                       line=dict(color='blue', width=5), name='Left -- commanded'))
        traces.append(go.Scatter3d(x=real_l[:, 0], y=real_l[:, 1], z=real_l[:, 2], mode='lines',
                                   line=dict(color='blue', width=4, dash='dash'), name='Left -- executed'))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Commanded vs. Executed Gripper Trajectory (3D, navigable)',
            scene=dict(xaxis_title='X [m]', yaxis_title='Y [m]', zaxis_title='Z [m]',
                      aspectmode='data'),
            legend=dict(x=0.01, y=0.99),
        )
        fig.write_html(html_path)

    def _build_fig_reference_governor(self):
        """Commanded ("raw") vs. governed reference, 4 rows x 1 col, with a
        DASHED horizontal line on every row marking the governor's configured
        limit on that quantity (cfg.GOV_V_MAX_LIN/ANG, GOV_E_MAX_POS/ORI). Per
        arm: DASHED = commanded reference (what trajectory_generator/teleop
        asked for), SOLID = governed reference (what the CLF actually saw)
        -- Red = Right, Blue = Left, matching every other per-arm convention
        in this dashboard. Commanded is dashed (rather than solid+alpha) so
        that when the Right and Left curves overlap they don't alpha-blend
        into an ambiguous purple smear -- the dash gaps keep pure red/blue
        visible even in overlap regions. The dashed GREY limit line is the
        governor's configured ceiling for that row: whenever a "commanded"
        curve pokes above it while the "governed" curve stays clipped at (or
        below) it, that is the governor visibly doing its job.
        """
        if not self.time_gov:
            return None

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 11))
        fig.suptitle('Reference Governor -- Commanded vs. Governed Trajectory')
        t = self.time_gov

        rows = [
            (axs[0], 'Linear Velocity', '[m/s]',
             self.gov_lin_vel_raw_r, self.gov_lin_vel_gov_r,
             self.gov_lin_vel_raw_l, self.gov_lin_vel_gov_l, cfg.GOV_V_MAX_LIN),
            (axs[1], 'Angular Velocity', '[rad/s]',
             self.gov_ang_vel_raw_r, self.gov_ang_vel_gov_r,
             self.gov_ang_vel_raw_l, self.gov_ang_vel_gov_l, cfg.GOV_V_MAX_ANG),
            (axs[2], 'Position Tracking Error', '[m]',
             self.gov_pos_err_raw_r, self.gov_pos_err_gov_r,
             self.gov_pos_err_raw_l, self.gov_pos_err_gov_l, cfg.GOV_E_MAX_POS),
            (axs[3], 'Orientation Tracking Error', '[rad]',
             self.gov_ori_err_raw_r, self.gov_ori_err_gov_r,
             self.gov_ori_err_raw_l, self.gov_ori_err_gov_l, cfg.GOV_E_MAX_ORI),
        ]

        for ax, title, ylabel, raw_r, gov_r, raw_l, gov_l, limit in rows:
            # Commanded ("raw") -- DASHED, so Right/Left overlap stays
            # distinguishable instead of alpha-blending into purple.
            ax.plot(t, raw_r, color='r', linestyle='--', linewidth=1.1,
                   alpha=0.8, label='Right -- commanded')
            ax.plot(t, raw_l, color='b', linestyle='--', linewidth=1.1,
                   alpha=0.8, label='Left -- commanded')
            # Governed (what the CLF actually saw) -- SOLID, drawn on top.
            ax.plot(t, gov_r, color='r', linestyle='-', linewidth=1.7,
                   label='Right -- governed')
            ax.plot(t, gov_l, color='b', linestyle='-', linewidth=1.7,
                   label='Left -- governed')
            # The governor's configured ceiling for THIS quantity -- unlabeled
            # in the legend text but visually distinct (grey dashed), same
            # spirit as the trajectory-finished marker: self-explanatory once
            # you see the governed curve hug it.
            ax.axhline(limit, color='0.35', linestyle='--', linewidth=1.3, zorder=1)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.set_xlim(0, t[-1])
            _draw_trigger_line(ax, self.t_off)

        axs[-1].set_xlabel('Time [s]')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig


def main(args=None):
    _apply_publication_style()
    rclpy.init(args=args)
    node = OfflinePlotter()

    # Auto-detect sim vs. real hardware for the ROS clock, same convention
    # as plotter.py -- otherwise the time axis freezes at 0 on real hardware
    # (no /clock topic) or drifts out of sync with the simulated world.
    topic_names = [name for name, _ in node.get_topic_names_and_types()]
    use_sim = '/clock' in topic_names
    node.set_parameters([
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, use_sim)
    ])
    node.get_logger().info(
        f"[ENV] use_sim_time={'TRUE (sim)' if use_sim else 'FALSE (real hardware)'}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[offline_plotter] Ctrl-C received -- saving whatever was recorded so far...",
              flush=True)
        if node.t0 is not None:
            node._finalize_and_save(reason='keyboard_interrupt')
        else:
            print("[offline_plotter] No trial was in progress -- nothing to save.", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
