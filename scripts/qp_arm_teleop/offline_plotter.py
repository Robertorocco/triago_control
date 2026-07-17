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
    fig0_summary                   The 10+ headline metrics below, rendered as
                                    a monospace text page (same numbers as
                                    summary_metrics.json / trial_summary.txt,
                                    just viewable as PDF/PNG alongside fig1..7)
    fig1_joint_kinematics          Position / Velocity / QP solution, 3x2
                                    (col 0 = Left arm, col 1 = Right arm)
    fig2_qp_data                   Slacks, CBF/joint shadow prices, safety
                                    margin, min distance (loop frequency moved
                                    to fig4_adaptive_parameters below)
    fig3_task_tracking_error       Cartesian position/velocity tracking error
                                    PLUS angular tracking error (geodesic
                                    SO(3) angle) and angular velocity tracking
                                    error (see cb_real for how the latter is
                                    estimated -- ee_real carries no measured
                                    angular velocity directly)
    fig4_adaptive_parameters       Slack weight, CLF convergence rate, dynamic
                                    safety margin (all flat-if-disabled, see
                                    that method's docstring) + loop frequency
    fig5_task_authority            Soft-task QP cost decomposition (shares)
    fig6_3d_trajectory              3D commanded vs. executed gripper path
                                    (solid = commanded reference, dashed =
                                    executed EE pose; red = Right, blue =
                                    Left). Saved as PDF + PNG always; ALSO
                                    saved as a browser-navigable HTML
                                    (free rotate/zoom/pan) if the optional
                                    `plotly` package is installed --
                                    `pip install plotly` to enable it, no
                                    other change needed.
    fig7_reference_governor        Commanded ("raw") vs. governed reference
                                    signal (linear/angular velocity, position/
                                    orientation tracking error), per arm, with
                                    a DASHED horizontal line marking the
                                    governor's configured limit on each
                                    quantity (cfg.GOV_V_MAX_LIN, GOV_V_MAX_ANG,
                                    GOV_E_MAX_POS, GOV_E_MAX_ORI) -- makes it
                                    immediately visible when/how much the
                                    governor reshaped the input trajectory.
                                    (only emitted if cfg.ENABLE_REFERENCE_GOVERNOR)
    fig8_head_perception            Head camera perception: per-object (red/
                                    blue) position/radius/height/confidence
                                    over time, active-vision phase + rate-based
                                    convergence progress + drift rate, cloud/
                                    map size. Only emitted if main_head.py was
                                    running this trial (data on
                                    /head_perception/telemetry|markers). A
                                    vertical marker line is drawn at the exact
                                    instant /perceived_world/snapshot latched,
                                    if it did during this trial.
    fig9_head_kinematics            Head joint position / measured velocity /
                                    QP-commanded velocity, 3x1 (single chain,
                                    no left/right split). Only emitted if head
                                    joints appeared on /joint_states this trial.
    fig10_head_active_tracking      head_active_arm_tracking.py's camera-
                                    framing performance: pixel centering error,
                                    angular errors (pointing/roll/approach),
                                    stand-off distance vs. target. Only emitted
                                    if that node was running this trial (data
                                    on /head_active_tracking/telemetry).
    fig11_haption_device            Haption handle telemetry (position, speed,
                                    clutch/grasp-trigger/deadman buttons,
                                    force/torque wrench) from
                                    haption_teleoperation. Only emitted if
                                    that device was publishing this trial
                                    (data on virtuose/pose or
                                    virtuose/force_cmd).
    fig12_shared_autonomy           main_shared_autonomy.py's belief/blend
                                    telemetry: reference-blending authority
                                    alpha, resulting user/policy action share,
                                    per-goal belief probabilities, active-arm/
                                    autonomous-grasp state. Only emitted if
                                    that node was running this trial.

    fig3/fig11/fig12 all shade, in green, the windows where virtuose/deadman
    was held AND virtuose/button_right (clutch) was released -- i.e. the
    operator was actually driving, as opposed to indexing or having let go
    of the handle. This is a HIGHLIGHT, not a filter: every data series still
    plots for the FULL trial regardless of grip state (robot motion doesn't
    stop just because the operator let go), so nothing is ever silently
    hidden. Shading is skipped entirely (no green anywhere) if
    virtuose/deadman never published this trial -- an older
    haption_teleoperation build, or no teleop device at all.

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
                                    (cfg.OFFLINE_BAG_TOPICS -- QP-controller +
                                    head-perception + haption/teleoperation +
                                    shared-autonomy telemetry, the merged
                                    superset of this script's own topics and
                                    scripts/analysis/study_config.py's
                                    allowlist), so any trial can be replayed
                                    offline later. Off via
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
import threading
import time

# Tkinter GUI (Start/Stop front-end) is optional -- guarded the same way
# scripts/analysis/study_recorder.py does, but here a missing python3-tk
# degrades to the ORIGINAL headless/topic-only behavior (see main()) rather
# than exiting, since this script is meant to always be runnable.
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    _TK_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - environment dependent
    tk = ttk = messagebox = None  # type: ignore
    _TK_IMPORT_ERROR = _exc

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, String, Bool
from geometry_msgs.msg import Pose, Twist, Wrench
from visualization_msgs.msg import MarkerArray
from rosgraph_msgs.msg import Clock

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

# EMA smoothing for the finite-differenced angular-velocity estimate (see
# cb_real) -- same idiom as cfg.STALL_SPEED_FILTER_ALPHA elsewhere in this
# codebase, applied here rather than in cfg since it's an analysis-only knob.
_ANG_VEL_EMA_ALPHA = 0.85


def _draw_trigger_line(ax, t_off):
    if t_off is not None:
        ax.axvline(t_off, **TRIGGER_LINE_KW)


def _draw_world_snapshot_line(ax, t_snap):
    """Vertical marker at the instant /perceived_world/snapshot latched this
    trial (None if it never fired while recording -- see perceived_world_callback).
    Green, dotted, distinct from the grey trajectory-finished trigger line."""
    if t_snap is not None:
        ax.axvline(t_snap, color='green', linestyle=':', linewidth=1.3, zorder=0)


# Shading, not masking: robot/shared-autonomy/haption data stays plotted for
# the FULL trial regardless of grip state -- this only highlights, on top,
# the windows where the operator was actually driving (handle held, clutch
# released), so a trial's data is never silently incomplete.
_CONTROL_SPAN_KW = dict(color='#2ca02c', alpha=0.12, zorder=0, linewidth=0)


def _draw_control_spans(ax, intervals):
    for start, end in intervals:
        ax.axvspan(start, end, **_CONTROL_SPAN_KW)


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
        self.last_saved_dir = None  # out_dir of the last FINALIZED trial (survives the reset below, for the GUI)
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

        # Latest perceived-world snapshot text (latched topic -- survives across
        # trials like last_real_pos_r above; per-trial capture time/text are in
        # self.world_snapshot_t/_text, reset each trial by _reset_buffers).
        self._last_world_snapshot_text = None

        self._reset_buffers()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- The one topic that governs everything (see module docstring) ---
        self.create_subscription(
            Bool, cfg.OFFLINE_RECORD_TRIGGER_TOPIC, self.record_trigger_callback, 10)
        # Manual GUI Start/Stop publishes on this SAME topic (see _RecorderGUI /
        # main()) -- it is just another source of the one signal this node
        # already understands, exactly like trajectory_generator.py. The two
        # can coexist: whichever edge arrives first drives the state machine,
        # and a redundant edge from the other source is already a documented no-op.
        self.pub_manual_trigger = self.create_publisher(
            Bool, cfg.OFFLINE_RECORD_TRIGGER_TOPIC, 10)

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
        self.create_subscription(Float64MultiArray, '/qp_debug/safety_margin', self.h_callback, qos_profile)
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

        # --- Head perception (main_head.py) -- optional; a trial with no head
        # node running simply never populates these buffers, and the head
        # figures are skipped at finalize time (see _finalize_and_save). ---
        self.create_subscription(Float64MultiArray, '/head_perception/telemetry',
                                 self.head_telemetry_callback, 10)
        self.create_subscription(MarkerArray, '/head_perception/markers',
                                 self.head_markers_callback, 10)
        self.create_subscription(Float64MultiArray, '/head_perception/qdot_cmd',
                                 self.head_qdot_cmd_callback, 10)
        self.create_subscription(Float64MultiArray, '/head_perception/qdot_measured',
                                 self.head_qdot_measured_callback, 10)
        # Latched (TRANSIENT_LOCAL) -- fires once per convergence; a subscriber
        # that joins after the fact still gets the last snapshot (see
        # world_convergence.py). Depth-1 QoS matches the publisher's own.
        world_qos = QoSProfile(depth=1)
        world_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        world_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(MarkerArray, '/perceived_world/snapshot',
                                 self.perceived_world_callback, world_qos)

        # --- Head active-arm tracking (head_active_arm_tracking.py) -- optional,
        # same "skip if absent" handling as the head-perception topics above. ---
        self.create_subscription(Float64MultiArray, '/head_active_tracking/telemetry',
                                 self.head_active_telemetry_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/head_active_tracking/qdot',
                                 self.head_active_qdot_callback, qos_profile)

        # --- Haption device (haption_teleoperation) -- optional; a trial with
        # no teleop node running simply never populates these buffers, and
        # fig11_haption_device is skipped at finalize time (see the docstring
        # of triago_control/CLAUDE.md's runtime-interface table for topic
        # ownership: this repo only CONSUMES these, never publishes them). ---
        self.create_subscription(Pose, 'virtuose/pose', self.virt_pose_callback, qos_profile)
        self.create_subscription(Twist, 'virtuose/velocity', self.virt_vel_callback, qos_profile)
        self.create_subscription(Wrench, 'virtuose/force_cmd', self.force_callback, qos_profile)
        self.create_subscription(Bool, 'virtuose/button_right', self.clutch_callback, qos_profile)
        self.create_subscription(Bool, 'virtuose/button_left', self.grasp_trigger_callback, qos_profile)
        self.create_subscription(Bool, 'virtuose/deadman', self.deadman_callback, qos_profile)
        self.create_subscription(Float64MultiArray, 'joystick/home_pose',
                                 self.home_pose_callback, qos_profile)

        # --- Shared autonomy (main_shared_autonomy.py) -- optional, same
        # "skip if absent" handling as the head/haption topics above. ---
        self.create_subscription(Float64MultiArray, '/shared_autonomy/blend_debug',
                                 self.blend_debug_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/shared_autonomy/goal_probabilities',
                                 self.goal_prob_callback, qos_profile)
        self.create_subscription(String, '/shared_autonomy/goal_names',
                                 self.goal_names_callback, qos_profile)
        self.create_subscription(String, '/shared_autonomy/active_arm',
                                 self.active_arm_callback, qos_profile)
        self.create_subscription(Bool, '/shared_autonomy/grasp_active',
                                 self.grasp_active_callback, qos_profile)

        # Slow watchdog: finalizes a POST_ROLL trial once its grace window
        # elapses. Deliberately NOT tied to any particular data topic's rate.
        self.create_timer(0.2, self._check_post_roll_deadline)

        # Clock-stall watchdog: self._t() (every buffer's x-axis) is
        # self.get_clock().now() - t0, which SILENTLY collapses every sample
        # to ~0 if the node's clock stops advancing mid-recording (sim paused,
        # or a stale/non-advancing /clock source) -- no exception, just a
        # ruined trial discovered only after the fact. Checked against WALL
        # time (datetime.now()), never the ROS clock itself, or a frozen ROS
        # clock would hide its own stall.
        self._clock_watch_last_ros_t = None
        self._clock_watch_last_wall_t = None
        self._clock_stall_warned = False
        self.create_timer(1.0, self._clock_watchdog_tick)

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

        # Angular tracking error (geodesic SO(3) angle, same metric as
        # gov_callback's _geodesic_angle) + a finite-differenced angular
        # velocity error -- see cb_real for how both are computed. Reset
        # each trial so the first sample never diffs across a trial boundary.
        self.err_ang_r = []
        self.err_ang_l = []
        self.err_ang_vel_r = []
        self.err_ang_vel_l = []
        self._prev_R_r = None
        self._prev_R_l = None
        self._prev_ang_t = None
        self._w_est_ema_r = None
        self._w_est_ema_l = None

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
        # Governed = raw - diff (from /qp_debug/governor), algebraically exact.
        self.gov_lin_vel_raw_r, self.gov_lin_vel_gov_r = [], []
        self.gov_lin_vel_raw_l, self.gov_lin_vel_gov_l = [], []
        self.gov_ang_vel_raw_r, self.gov_ang_vel_gov_r = [], []
        self.gov_ang_vel_raw_l, self.gov_ang_vel_gov_l = [], []
        self.gov_pos_err_raw_r, self.gov_pos_err_gov_r = [], []
        self.gov_pos_err_raw_l, self.gov_pos_err_gov_l = [], []
        self.gov_ori_err_raw_r, self.gov_ori_err_gov_r = [], []
        self.gov_ori_err_raw_l, self.gov_ori_err_gov_l = [], []

        # --- Head perception (main_head.py) -- all optional; a trial with no
        # head node running simply leaves these empty (see head_* callbacks). ---
        self.time_head_js = []
        self.q_head_buffers = {j: [] for j in cfg.HEAD_JOINTS}

        self.time_head_qdot_measured = []
        self.head_qdot_measured = []
        self.time_head_qdot_cmd = []
        self.head_qdot_cmd = []

        # Raw 15-float telemetry rows (see main_head.py's pub_telemetry layout
        # comment): [n_raw, n_crop, plane_z, look_err_deg, slack, proc_ms,
        # red_conf, blue_conf, map_size, converge_progress, converged, phase,
        # weakest_cov, drift_rate_mps, head_qdot_norm].
        self.time_head_tel = []
        self.head_tel_buffer = []

        # Per-object (red/blue) position + dimensions, parsed from
        # /head_perception/markers -- same fields/classification as
        # head_plotter.py's _markers_cb (ns="objects", CYLINDER, R/B channel).
        self.head_obj = {
            "red": {"t": [], "x": [], "y": [], "z": [], "r": [], "h": []},
            "blue": {"t": [], "x": [], "y": [], "z": [], "r": [], "h": []},
        }
        self.time_head_plane = []
        self.head_plane_z = []

        # Perceived-world snapshot (latched, one-shot): trial-relative time it
        # latched THIS trial (None if it never fired while recording) + a
        # short text summary for fig7's caption. self._last_world_snapshot_text
        # (set outside this reset -- see __init__) is the always-latest text
        # regardless of trial boundaries, mirroring last_real_pos_r's pattern.
        self.world_snapshot_t = None
        self.world_snapshot_text = None

        # --- Head active-arm tracking (head_active_arm_tracking.py) --------
        self.time_head_active_tel = []
        self.head_active_tel_buffer = []
        self.time_head_active_qdot = []
        self.head_active_qdot_buffer = []

        # --- Haption device (haption_teleoperation) -- all optional; a trial
        # with no teleop node running simply leaves these empty. ---
        self.time_virt_pose = []
        self.virt_pose_buffer = []          # [px,py,pz, qx,qy,qz,qw]
        self.time_virt_vel = []
        self.virt_vel_buffer = []           # [lx,ly,lz, ax,ay,az]
        self.time_force = []
        self.force_buffer = []              # [fx,fy,fz, tx,ty,tz]
        self.time_clutch = []
        self.clutch_buffer = []             # virtuose/button_right (1.0/0.0)
        self.time_grasp_trigger = []
        self.grasp_trigger_buffer = []      # virtuose/button_left (1.0/0.0)
        self.time_deadman = []
        self.deadman_buffer = []            # virtuose/deadman (1.0/0.0) -- grip-presence sensor
        self.time_home_pose = []
        self.home_pose_buffer = []          # [px,py,pz, qx,qy,qz,qw] (JOYSTICK mode)

        # --- Shared autonomy (main_shared_autonomy.py) -- all optional. ---
        self.time_blend = []
        self.blend_buffer = []              # [alpha, v_user(6), v_policy(6), v_blend(6)]
        self.time_goal_prob = []
        self.goal_prob_buffer = []
        self.goal_names = None              # latest CSV name list (String, rare updates)
        self.time_active_arm = []
        self.active_arm_buffer = []         # 1.0=right, 0.0=left
        self.time_grasp_active = []
        self.grasp_active_buffer = []       # 1.0/0.0

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

    def _clock_watchdog_tick(self):
        """Loudly warn if self._now() hasn't moved in >1.5s of WALL time while
        recording -- sim paused / stale or non-advancing /clock -- BEFORE the
        trial ends, not after (every buffer's timestamp collapses toward 0,
        which otherwise only shows up as tiny figures nobody notices until
        the trial is already lost)."""
        if not self._recording_active():
            self._clock_watch_last_ros_t = None
            self._clock_watch_last_wall_t = None
            self._clock_stall_warned = False
            return
        now_ros = self._now()
        now_wall = datetime.datetime.now().timestamp()
        if self._clock_watch_last_ros_t is not None:
            ros_dt = now_ros - self._clock_watch_last_ros_t
            wall_dt = now_wall - self._clock_watch_last_wall_t
            if wall_dt > 1.5 and ros_dt < 0.05:
                if not self._clock_stall_warned:
                    self._clock_stall_warned = True
                    self.get_logger().error(
                        f"[offline_plotter] ROS clock has NOT advanced in {wall_dt:.1f}s "
                        "of real time while recording -- every timestamp from here "
                        "collapses toward t=0. Is the simulation PAUSED, or is /clock "
                        "stale/not publishing? (use_sim_time="
                        f"{self.get_parameter('use_sim_time').value})")
            elif ros_dt >= 0.05:
                self._clock_stall_warned = False   # recovered -- re-arm for next stall
        self._clock_watch_last_ros_t = now_ros
        self._clock_watch_last_wall_t = now_wall

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
        if all(j in name_to_idx for j in self.all_joints):
            self.time_js.append(self._t())
            for j in self.all_joints:
                self.q_buffers[j].append(msg.position[name_to_idx[j]])
        # Head joints (independent gate -- main_head.py may not be running this
        # trial, which must not block the arm position buffers above).
        if all(j in name_to_idx for j in cfg.HEAD_JOINTS):
            self.time_head_js.append(self._t())
            for j in cfg.HEAD_JOINTS:
                self.q_head_buffers[j].append(msg.position[name_to_idx[j]])

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

    def head_qdot_measured_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 7:
            return
        self.time_head_qdot_measured.append(self._t())
        self.head_qdot_measured.append(list(msg.data[:7]))

    def head_qdot_cmd_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 7:
            return
        self.time_head_qdot_cmd.append(self._t())
        self.head_qdot_cmd.append(list(msg.data[:7]))

    def head_telemetry_callback(self, msg):
        # See main_head.py's pub_telemetry layout comment (up to 15 floats).
        if not self._recording_active() or len(msg.data) < 6:
            return
        data = list(msg.data)
        if len(data) < 15:
            data = data + [0.0] * (15 - len(data))
        self.time_head_tel.append(self._t())
        self.head_tel_buffer.append(data[:15])

    def head_markers_callback(self, msg):
        """Per-object (red/blue) position + dimensions, parsed the same way as
        head_plotter.py's _markers_cb (ns='objects' CYLINDER, classified by
        R/B color channel; ns='table_top' CUBE for the detected plane height)."""
        if not self._recording_active():
            return
        t = self._t()
        got_plane = False
        for m in msg.markers:
            if m.ns == "objects" and m.type == 3:  # CYLINDER=3
                if m.color.r > 0.5:
                    key = "red"
                elif m.color.b > 0.5:
                    key = "blue"
                else:
                    continue
                d = self.head_obj[key]
                d["t"].append(t)
                d["x"].append(m.pose.position.x)
                d["y"].append(m.pose.position.y)
                d["z"].append(m.pose.position.z)
                d["r"].append(m.scale.x / 2.0)
                d["h"].append(m.scale.z)
            if m.ns == "table_top" and m.type == 1:  # CUBE=1
                self.time_head_plane.append(t)
                self.head_plane_z.append(m.pose.position.z)
                got_plane = True

    def perceived_world_callback(self, msg):
        """Latched, one-shot: the converged perceived-world snapshot handed to
        the CBF stack (see world_convergence.py / main_head.py's
        _publish_world_snapshot). Parses build_world_snapshot_markers' fixed
        layout (ns='table' CUBE, ns='objects' CYLINDER) into a short text
        summary. Text is cached unconditionally (a late-subscribing offline_plotter
        still gets the last snapshot via TRANSIENT_LOCAL, same as any other
        subscriber -- see the topic's QoS in __init__); the trial-relative
        CAPTURE time/text are only recorded while a trial is actually running,
        so fig7 can draw a marker line at the instant it happened THIS trial."""
        lines = []
        for m in msg.markers:
            if m.ns == "table" and m.type == 1:  # CUBE=1
                lines.append(
                    f"table: centre=({m.pose.position.x:.3f},{m.pose.position.y:.3f},"
                    f"{m.pose.position.z:.3f}) size=({m.scale.x:.3f},{m.scale.y:.3f},{m.scale.z:.3f})")
            if m.ns == "objects" and m.type == 3:  # CYLINDER=3
                color = "red" if m.color.r > 0.5 else ("blue" if m.color.b > 0.5 else "obj")
                lines.append(
                    f"{color}: centre=({m.pose.position.x:.3f},{m.pose.position.y:.3f},"
                    f"{m.pose.position.z:.3f}) r={m.scale.x/2.0*100:.1f}cm h={m.scale.z*100:.1f}cm")
        text = "\n".join(lines) if lines else None
        self._last_world_snapshot_text = text
        if self._recording_active() and self.world_snapshot_t is None:
            self.world_snapshot_t = self._t()
            self.world_snapshot_text = text

    def head_active_telemetry_callback(self, msg):
        # See head_active_arm_tracking.py's pub_telemetry layout comment (9 floats).
        if not self._recording_active() or len(msg.data) < 9:
            return
        self.time_head_active_tel.append(self._t())
        self.head_active_tel_buffer.append(list(msg.data[:9]))

    def head_active_qdot_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 7:
            return
        self.time_head_active_qdot.append(self._t())
        self.head_active_qdot_buffer.append(list(msg.data[:7]))

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

        # Angular tracking error -- geodesic SO(3) angle ||log3(R_ref R_real^T)||,
        # the SAME metric already used by gov_callback's _geodesic_angle (0.0,
        # not NaN, when ee_real carries no RPY -- same sentinel convention as
        # that panel, for consistency within this file).
        t_now = self._t()
        R_real_r = pin.rpy.rpyToMatrix(*self.last_real_rpy_r) if len(real) >= 18 else None
        R_real_l = pin.rpy.rpyToMatrix(*self.last_real_rpy_l) if len(real) >= 18 else None
        e_ang_r = self._geodesic_angle(self.ref_right[3:6], R_real_r)
        e_ang_l = self._geodesic_angle(self.ref_left[3:6], R_real_l) if self.has_ref_left else 0.0

        # Angular velocity error: the reference angular velocity vs. a
        # finite-differenced angular velocity estimated from consecutive
        # measured rotations (w_est = log3(R_now R_prev^T) / dt) -- ee_real
        # carries no measured angular velocity directly (only linear), so
        # this is derived the same way analyze_trial.py differentiates EE
        # position to get speed (its own ee_real velocity slots read ~0).
        # Raw finite differences of a real-hardware-sourced signal are noisy
        # (1/dt amplifies FK/encoder noise) -- EMA-smoothed same idiom as
        # cfg.STALL_SPEED_FILTER_ALPHA elsewhere in this codebase. Only fed
        # on ticks where a fresh difference was actually computed, so a
        # missing-orientation or duplicate-timestamp tick can't drag the
        # filter toward a spurious zero.
        w_est_r = self._w_est_ema_r if self._w_est_ema_r is not None else np.zeros(3)
        w_est_l = self._w_est_ema_l if self._w_est_ema_l is not None else np.zeros(3)
        if R_real_r is not None and R_real_l is not None:
            if self._prev_R_r is not None and self._prev_ang_t is not None:
                dt_ang = t_now - self._prev_ang_t
                if dt_ang > 1e-4:
                    raw_w_r = pin.log3(R_real_r @ self._prev_R_r.T) / dt_ang
                    raw_w_l = pin.log3(R_real_l @ self._prev_R_l.T) / dt_ang
                    if self._w_est_ema_r is None:
                        self._w_est_ema_r, self._w_est_ema_l = raw_w_r, raw_w_l
                    else:
                        a = _ANG_VEL_EMA_ALPHA
                        self._w_est_ema_r = a * self._w_est_ema_r + (1.0 - a) * raw_w_r
                        self._w_est_ema_l = a * self._w_est_ema_l + (1.0 - a) * raw_w_l
                    w_est_r, w_est_l = self._w_est_ema_r, self._w_est_ema_l
            self._prev_R_r, self._prev_R_l = R_real_r, R_real_l
            self._prev_ang_t = t_now
        e_ang_vel_r = float(np.linalg.norm(self.ref_right[9:12] - w_est_r))
        e_ang_vel_l = (float(np.linalg.norm(self.ref_left[9:12] - w_est_l))
                       if self.has_ref_left else 0.0)

        self.time_err.append(t_now)
        self.err_pos_r.append(e_p_r)
        self.err_pos_l.append(e_p_l)
        self.err_vel_r.append(e_v_r)
        self.err_vel_l.append(e_v_l)
        self.err_ang_r.append(e_ang_r)
        self.err_ang_l.append(e_ang_l)
        self.err_ang_vel_r.append(e_ang_vel_r)
        self.err_ang_vel_l.append(e_ang_vel_l)
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
        data = list(msg.data)
        if len(data) < 2:
            data = (data + [float('nan'), float('nan')])[:2]
        self.time_h.append(self._t())
        self.h_buffer.append(data)

    def min_dist_callback(self, msg):
        if not self._recording_active():
            return
        self.time_min_dist.append(self._t())
        self.min_dist_buffer.append(msg.data)

    def dyn_weights_callback(self, msg):
        # 5 floats: [weight_slack_r, weight_slack_l, gamma_mean, gamma_r, gamma_l].
        if not self._recording_active():
            return
        data = list(msg.data)
        if len(data) < 5:
            data = (data + [0.0] * 5)[:5]
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
        # 12 floats: [0:4] combined [damp, posture, slack, rate] totals,
        # [4:8] right-arm-only, [8:12] left-arm-only.
        if not self._recording_active() or len(msg.data) < 3:
            return
        data = list(msg.data[:12])
        if len(data) < 12:
            data = data + [0.0] * (12 - len(data))
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

        # governed = raw - diff; raw is self.ref_right/left, already tracked.
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

    # --- Haption device (haption_teleoperation) --------------------------
    def virt_pose_callback(self, msg):
        if not self._recording_active():
            return
        self.time_virt_pose.append(self._t())
        self.virt_pose_buffer.append([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def virt_vel_callback(self, msg):
        if not self._recording_active():
            return
        self.time_virt_vel.append(self._t())
        self.virt_vel_buffer.append([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z])

    def force_callback(self, msg):
        if not self._recording_active():
            return
        self.time_force.append(self._t())
        self.force_buffer.append([
            msg.force.x, msg.force.y, msg.force.z,
            msg.torque.x, msg.torque.y, msg.torque.z])

    def clutch_callback(self, msg):
        if not self._recording_active():
            return
        self.time_clutch.append(self._t())
        self.clutch_buffer.append(1.0 if msg.data else 0.0)

    def grasp_trigger_callback(self, msg):
        if not self._recording_active():
            return
        self.time_grasp_trigger.append(self._t())
        self.grasp_trigger_buffer.append(1.0 if msg.data else 0.0)

    def deadman_callback(self, msg):
        if not self._recording_active():
            return
        self.time_deadman.append(self._t())
        self.deadman_buffer.append(1.0 if msg.data else 0.0)

    def home_pose_callback(self, msg):
        if not self._recording_active() or len(msg.data) < 7:
            return
        self.time_home_pose.append(self._t())
        self.home_pose_buffer.append(list(msg.data[:7]))

    # --- Shared autonomy (main_shared_autonomy.py) ------------------------
    def blend_debug_callback(self, msg):
        # 19 floats: [alpha, v_user(6), v_policy(6), v_blend(6)].
        if not self._recording_active() or len(msg.data) < 19:
            return
        self.time_blend.append(self._t())
        self.blend_buffer.append(list(msg.data[:19]))

    def goal_prob_callback(self, msg):
        if not self._recording_active() or not msg.data:
            return
        self.time_goal_prob.append(self._t())
        self.goal_prob_buffer.append(list(msg.data))

    def goal_names_callback(self, msg):
        # Rare/near-static (only changes when the goal set itself changes) --
        # cached outside the recording gate like last_real_pos_r, so fig11's
        # legend always has a name even if this fired before t=0 this trial.
        self.goal_names = msg.data

    def active_arm_callback(self, msg):
        if not self._recording_active():
            return
        self.time_active_arm.append(self._t())
        self.active_arm_buffer.append(1.0 if msg.data == 'right' else 0.0)

    def grasp_active_callback(self, msg):
        if not self._recording_active():
            return
        self.time_grasp_active.append(self._t())
        self.grasp_active_buffer.append(1.0 if msg.data else 0.0)

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
        figs.append(('fig3_task_tracking_error', self._build_fig_task_tracking_error()))
        figs.append(('fig4_adaptive_parameters', self._build_fig_adaptive_parameters()))
        figs.append(('fig5_task_authority', self._build_fig_task_authority()))
        figs.append(('fig6_3d_trajectory', self._build_fig_3d_trajectory()))
        if cfg.ENABLE_REFERENCE_GOVERNOR:
            fig_gov = self._build_fig_reference_governor()
            if fig_gov is not None:
                figs.append(('fig7_reference_governor', fig_gov))

        # --- Head figures (main_head.py / head_active_arm_tracking.py) --
        # only emitted if the corresponding node was actually running this
        # trial (checked via whether its buffers ever received data), so a
        # trial with no head node produces no empty/blank head figures.
        if self.time_head_tel or self.head_obj["red"]["t"] or self.head_obj["blue"]["t"]:
            figs.append(('fig8_head_perception', self._build_fig_head_perception()))
        if self.time_head_js or self.time_head_qdot_measured or self.time_head_qdot_cmd:
            figs.append(('fig9_head_kinematics', self._build_fig_head_kinematics()))
        if self.time_head_active_tel:
            figs.append(('fig10_head_active_tracking', self._build_fig_head_active_tracking()))

        # --- Haption device + shared-autonomy figures -- only emitted if the
        # corresponding node was actually running this trial, same "skip if
        # absent" gating as the head figures above. ---
        if self.time_virt_pose or self.time_force:
            figs.append(('fig11_haption_device', self._build_fig_haption_device()))
        if self.time_blend or self.time_goal_prob or self.time_active_arm:
            figs.append(('fig12_shared_autonomy', self._build_fig_shared_autonomy()))

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
                html_path = os.path.join(out_dir, 'fig6_3d_trajectory.html')
                self._save_3d_trajectory_html(html_path)
                self.get_logger().info(f"[offline_plotter]   saved fig6_3d_trajectory.html (interactive)")
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
        self.last_saved_dir = out_dir   # survives the reset below, for the GUI's "Last saved" line
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

        force_arr = np.asarray(self.force_buffer, dtype=float)
        peak_device_force = (float(np.max(np.linalg.norm(force_arr[:, 0:3], axis=1)))
                            if force_arr.size else float('nan'))

        clutch_arr = np.asarray(self.clutch_buffer, dtype=float)
        clutch_duty_frac = float(np.mean(clutch_arr > 0.5)) if clutch_arr.size else float('nan')

        control_intervals = self._control_intervals(trial_duration)
        teleop_in_control_frac = (sum(end - start for start, end in control_intervals) / trial_duration
                                  if control_intervals and trial_duration > 1e-9 else float('nan'))

        blend_alpha_mean = float('nan')
        if self.blend_buffer:
            blend_alpha_mean = float(np.mean(np.asarray(self.blend_buffer, dtype=float)[:, 0]))

        autonomy_grasp_frac = float('nan')
        if self.grasp_active_buffer:
            autonomy_grasp_frac = float(np.mean(np.asarray(self.grasp_active_buffer, dtype=float) > 0.5))

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
            'ee_ang_tracking_rms_rad': _entry(_rms(self.err_ang_r, self.err_ang_l),
                'combined R+L angular tracking error, geodesic SO(3) angle (accuracy)'),
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
            'peak_device_force_n': _entry(peak_device_force,
                'peak Haption handle force magnitude (NaN = no teleop device this trial)'),
            'clutch_duty_frac': _entry(clutch_duty_frac,
                'fraction of trial spent with the clutch (right button) engaged'),
            'teleop_in_control_frac': _entry(teleop_in_control_frac,
                'fraction of trial with the deadman held AND clutch released (NaN = no virtuose/deadman this trial)'),
            'blend_alpha_mean': _entry(blend_alpha_mean,
                'mean reference-blending authority, 0=user 1=policy (NaN = ASSIST_BLENDING off/absent)'),
            'autonomy_grasp_frac': _entry(autonomy_grasp_frac,
                'fraction of trial spent in an autonomous (non-teleoperated) grasp'),
        }

    # =====================================================================
    # FIGURE BUILDERS
    # =====================================================================
    def _max_time(self, *time_lists):
        candidates = [tl[-1] for tl in time_lists if tl]
        return max(candidates) if candidates else 1.0

    def _control_intervals(self, end_t):
        """Contiguous [start, end) windows, clipped to [0, end_t], where the
        operator was actively driving: deadman held AND clutch released.
        Both signals are async step functions (button/deadman edges), so
        state at any query time is the last-known value at-or-before it.
        Returns [] if virtuose/deadman never published this trial (older
        haption_teleoperation build without it) -- shading nothing is more
        honest than guessing a state we never actually observed."""
        if not self.time_deadman:
            return []

        def _last_at(times, values, t, default):
            idx = np.searchsorted(times, t, side='right') - 1
            return bool(values[idx]) if idx >= 0 else default

        breakpoints = sorted(set(self.time_deadman) | set(self.time_clutch) | {0.0, end_t})
        intervals = []
        run_start = None
        for t in breakpoints:
            if t > end_t:
                break
            in_control = (_last_at(self.time_deadman, self.deadman_buffer, t, False)
                         and not _last_at(self.time_clutch, self.clutch_buffer, t, False))
            if in_control and run_start is None:
                run_start = t
            elif not in_control and run_start is not None:
                intervals.append((run_start, t))
                run_start = None
        if run_start is not None:
            intervals.append((run_start, end_t))
        return intervals

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
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
        fig.suptitle('QP Data -- Slacks, Shadow Prices, Safety')

        # Row 0: CLF Slack -- both arms on the SAME axis (red=Right, blue=Left),
        # matching the shadow-price rows below instead of two separate axes.
        if self.slack_mode == 'scalar' and self.time_slack:
            data = np.array(self.slack_buffer)
            axs[0].plot(self.time_slack, data[:, 0], 'r-', label=r'$\delta_{R}$')
            axs[0].plot(self.time_slack, data[:, 1], 'b-', label=r'$\delta_{L}$')
        elif self.slack_mode == 'vector' and self.time_slack:
            data = np.array(self.slack_buffer)
            comp_style = ['-', '--', ':']
            for i, comp in enumerate(['x', 'y', 'z']):
                axs[0].plot(self.time_slack, data[:, i], color='r', linestyle=comp_style[i],
                           label=fr'$\delta_{{R,{comp}}}$')
                axs[0].plot(self.time_slack, data[:, 3 + i], color='b', linestyle=comp_style[i],
                           label=fr'$\delta_{{L,{comp}}}$')
        axs[0].set_title('CLF Slack (per arm)')
        axs[0].set_ylabel(r'$\delta$')
        axs[0].legend(loc='upper right', fontsize=8,
                      ncol=2 if self.slack_mode == 'vector' else 1)

        # Row 1: CBF shadow prices
        if self.time_lambda_cbf:
            data = np.array(self.lambda_cbf_buffer)
            axs[1].plot(self.time_lambda_cbf, data[:, 0], 'r-', label=r'$\lambda_{CBF,R}$')
            axs[1].plot(self.time_lambda_cbf, data[:, 1], 'b-', label=r'$\lambda_{CBF,L}$')
        axs[1].set_title('CBF Shadow Price (per arm)')
        axs[1].set_ylabel(r'$\lambda_{CBF}$')
        axs[1].legend(loc='upper right')

        # Row 2: Joint-limit shadow prices
        if self.time_lambda_joints:
            data = np.array(self.lambda_joints_buffer)
            axs[2].plot(self.time_lambda_joints, data[:, 0], 'r-', label=r'$\lambda_{Joints,R}$')
            axs[2].plot(self.time_lambda_joints, data[:, 1], 'b-', label=r'$\lambda_{Joints,L}$')
        axs[2].set_title('Joint-Limit Shadow Price (per arm)')
        axs[2].set_ylabel(r'$\lambda_{Joints}$')
        axs[2].legend(loc='upper right')

        # Safety margin, per arm; NaN when that arm has no active pair
        # (not the h_soft=1.0 QP sentinel, which isn't a real margin).
        if self.time_h:
            data = np.array(self.h_buffer)
            axs[3].plot(self.time_h, data[:, 0], 'r-', marker='.', markersize=3,
                       label=r'$h_R - d_{safe,R}$ (NaN = no active pair)')
            axs[3].plot(self.time_h, data[:, 1], 'b-', marker='.', markersize=3,
                       label=r'$h_L - d_{safe,L}$ (NaN = no active pair)')
        axs[3].axhline(0, color='r', linestyle='--', linewidth=1)
        axs[3].set_title('CBF Safety Margin (per arm)')
        axs[3].set_ylabel('Margin [m]')
        axs[3].legend(loc='upper right', fontsize=8)

        # Row 4: Minimum distance. NaN whenever nothing is within
        # cfg.DISTANCE_FILTER_THRESHOLD -- an honest "nothing nearby", not
        # missing data (collision_manager.compute_softmin_jacobian). The
        # resulting gaps are correct, not a bug; the marker below is the same
        # short-segment-visibility touch as the margin row above.
        if self.time_min_dist:
            axs[4].plot(self.time_min_dist, self.min_dist_buffer, 'c-', marker='.', markersize=3,
                       label='Abs. min. distance (NaN = nothing in range)')
        axs[4].axhline(0, color='r', linestyle='--', linewidth=1)
        axs[4].set_title('Absolute Minimum Collision Distance')
        axs[4].set_ylabel('Dist [m]')
        axs[4].set_xlabel('Time [s]')
        axs[4].legend(loc='upper right', fontsize=8)

        # Loop frequency moved to fig4_adaptive_parameters (see that method).
        max_t = self._max_time(self.time_slack, self.time_lambda_cbf, self.time_lambda_joints,
                               self.time_h, self.time_min_dist)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    def _build_fig_task_tracking_error(self):
        """Cartesian + angular tracking error, position and velocity domain
        (4 rows). Angular error uses the geodesic SO(3) angle
        ||log3(R_ref R_real^T)|| -- the standard robotics metric for
        orientation-tracking error (same one gov_callback's
        _geodesic_angle already used for the reference-governor figure) --
        and angular velocity error compares the commanded angular velocity
        against a finite-differenced estimate of the measured one (see
        cb_real; ee_real carries no measured angular velocity directly)."""
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
        fig.suptitle('Task Tracking Error (Position + Orientation)')

        if self.time_err:
            axs[0].plot(self.time_err, self.err_pos_r, 'r-', label=r'$\|e_{p,R}\|$')
            axs[0].plot(self.time_err, self.err_pos_l, 'b-', label=r'$\|e_{p,L}\|$')
        axs[0].set_title('Cartesian Position Tracking Error')
        axs[0].set_ylabel('Error [m]')
        axs[0].legend(loc='upper right')

        if self.time_err:
            axs[1].plot(self.time_err, self.err_vel_r, 'r-', label=r'$\|e_{v,R}\|$')
            axs[1].plot(self.time_err, self.err_vel_l, 'b-', label=r'$\|e_{v,L}\|$')
        axs[1].set_title('Cartesian Velocity Tracking Error')
        axs[1].set_ylabel('Error [m/s]')
        axs[1].legend(loc='upper right')

        if self.time_err:
            axs[2].plot(self.time_err, self.err_ang_r, 'r-', label=r'$\|e_{\theta,R}\|$')
            axs[2].plot(self.time_err, self.err_ang_l, 'b-', label=r'$\|e_{\theta,L}\|$')
        axs[2].set_title('Angular Tracking Error (geodesic SO(3) angle)')
        axs[2].set_ylabel('Error [rad]')
        axs[2].legend(loc='upper right')

        if self.time_err:
            axs[3].plot(self.time_err, self.err_ang_vel_r, 'r-', label=r'$\|e_{\omega,R}\|$')
            axs[3].plot(self.time_err, self.err_ang_vel_l, 'b-', label=r'$\|e_{\omega,L}\|$')
        axs[3].set_title('Angular Velocity Tracking Error')
        axs[3].set_ylabel('Error [rad/s]')
        axs[3].set_xlabel('Time [s]')
        axs[3].legend(loc='upper right')

        max_t = self._max_time(self.time_err)
        control_intervals = self._control_intervals(max_t)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
            _draw_control_spans(ax, control_intervals)
        if control_intervals:
            fig.text(0.99, 0.005, 'green = operator in control (handle held, clutch released)',
                    ha='right', va='bottom', fontsize=7, color='#2ca02c', alpha=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _build_fig_adaptive_parameters(self):
        # Always plot all three adaptive quantities, regardless of their
        # cfg.DYNAMIC_* flags -- a flat line at the default when a flag is off
        # is itself useful information (confirms nothing is silently
        # adapting), and it makes a flag-on/off A/B a straight before/after
        # comparison of the same figure. Previously d_safe_dynamic was gated
        # on cfg.DYNAMIC_CBF -- the WRONG flag (that one controls pair
        # removal, not the margin formula) -- so this panel had never
        # actually appeared in any trial this project has run;
        # d_safe_dynamic = D_SAFE_BASE + K_V_SAFE*|v| is computed
        # unconditionally every tick (collision_manager.py). Loop frequency
        # (a loop-health diagnostic, not a task-adaptive quantity) lives here
        # rather than in fig2_qp_data purely to keep this figure's theme --
        # "things that change the QP's behavior over the trial" -- coherent.
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
        fig.suptitle('Adaptive QP Parameters + Loop Health')

        ax = axs[0]
        ax.set_title('Slack Weight'); ax.set_ylabel(r'$w_{\delta}$')
        if self.time_dyn_weights:
            data = np.array(self.dyn_weights_buffer)
            # Both solid; differentiated by linewidth + zorder so a
            # coincident overlap still shows a visible rim, not one hidden line.
            ax.plot(self.time_dyn_weights, data[:, 0], color='r', linestyle='-',
                    linewidth=2.6, zorder=2, label=r'$w_{\delta,R}$')
            ax.plot(self.time_dyn_weights, data[:, 1], color='b', linestyle='-',
                    linewidth=1.3, zorder=3, label=r'$w_{\delta,L}$')
            ax.legend(loc='upper right')

        ax = axs[1]
        ax.set_title('CLF Convergence Rate'); ax.set_ylabel(r'$\gamma_{CLF}$')
        if self.time_dyn_weights:
            data = np.array(self.dyn_weights_buffer)
            # gamma_clf_r/l: per-arm, same overlap-safety convention as above.
            ax.plot(self.time_dyn_weights, data[:, 3], color='r', linestyle='-',
                    linewidth=2.6, zorder=2, label=r'$\gamma_{CLF,R}$')
            ax.plot(self.time_dyn_weights, data[:, 4], color='b', linestyle='-',
                    linewidth=1.3, zorder=3, label=r'$\gamma_{CLF,L}$')
            # cfg.GAMMA_CLF_DEFAULT is what BOTH pin to whenever
            # cfg.DYNAMIC_GAMMA_CLF is False -- flat lines exactly on this
            # reference confirm the scheduler is off, not silently broken.
            ax.axhline(cfg.GAMMA_CLF_DEFAULT, color='0.5', linestyle=':',
                      linewidth=1, zorder=1, label=r'$\gamma_{CLF}^{default}$')
            ax.legend(loc='upper right')

        ax = axs[2]
        ax.set_title('Dynamic Safety Margin (base + velocity term)')
        ax.set_ylabel(r'$d_{safe}$ [m]')
        if self.time_d_safe:
            data = np.array(self.d_safe_buffer)
            # Only the TOTAL effective margin per arm + the static floor --
            # the gap between a total curve and the floor line already IS
            # the velocity term (d_safe_dynamic = D_SAFE_BASE + K_V_SAFE*|v|,
            # collision_manager.py), so a separate dotted remainder line
            # was redundant. Names chosen to read standalone in the legend.
            ax.plot(self.time_d_safe, data[:, 0], 'r-', zorder=2,
                   label=r'Effective Margin -- Right ($d_{safe,R}$)')
            ax.plot(self.time_d_safe, data[:, 1], 'b-', zorder=2,
                   label=r'Effective Margin -- Left ($d_{safe,L}$)')
            # Drawn thicker, pure black, and at a high zorder so it is
            # never buried under a curve that sits close to it (e.g. an
            # idle/frozen arm's margin, which stays near d0 the whole
            # trial) -- an explicit y-floor at 0 also keeps it clear of
            # the bottom axis spine instead of hugging it.
            ax.axhline(cfg.D_SAFE_BASE, color='k', linestyle='--',
                      linewidth=1.6, zorder=5, label=r'Static Floor ($d_0$)')
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=8)

        ax = axs[3]
        ax.set_title('Control Loop Frequency'); ax.set_ylabel('Freq [Hz]')
        ax.set_xlabel('Time [s]')
        if self.time_freq:
            ax.plot(self.time_freq, self.freq_buffer, 'g-', label='Loop frequency')
        ax.legend(loc='upper right')

        max_t = self._max_time(self.time_dyn_weights, self.time_d_safe, self.time_freq)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _build_fig_task_authority(self):
        # task_energies: [0:4] combined, [4:8] right only, [8:12] left only.
        # Each panel normalizes against only that arm's own total; stacked
        # vertically to match every other per-arm figure in this dashboard.
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)
        fig.suptitle('Soft-Task QP Cost Decomposition (per arm)')

        cost_terms = [
            ('Damping', '#888888'),
            ('Posture / limit', '#2a9d8f'),
            ('Slack (CLF give)', '#eda100'),
            # Rate-damping (cfg.ENABLE_RATE_DAMPING, ||dq-dq_prev||^2 -- see
            # qp_formulator.py's build_and_solve §A) -- 0 share whenever it was
            # disabled for this trial, plotted anyway so an A/B is a straight
            # before/after comparison of the same figure.
            ('Rate damping (smoothness)', '#4a3aa7'),
        ]

        if self.time_task_auth:
            arr = np.array(self.task_auth_buffer, dtype=float)
            for ax, title, col_offset in ((axs[0], 'Right Arm', 4), (axs[1], 'Left Arm', 8)):
                sub = arr[:, col_offset:col_offset + 4]
                total = sub.sum(axis=1, keepdims=True)
                total[total < 1e-12] = 1.0
                shares = sub / total
                for i, (label, color) in enumerate(cost_terms):
                    ax.plot(self.time_task_auth, shares[:, i], color=color, label=label)
                ax.set_title(title)
        else:
            axs[0].set_title('Right Arm')
            axs[1].set_title('Left Arm')

        max_t = self._max_time(self.time_task_auth)
        for ax in axs:
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel('Authority share [-]')
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        axs[-1].set_xlabel('Time [s]')
        axs[0].legend(loc='upper left', ncol=2, fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    # =====================================================================
    # HEAD FIGURES (main_head.py + head_active_arm_tracking.py -- optional,
    # only emitted when the corresponding node was running this trial)
    # =====================================================================
    def _build_fig_head_perception(self):
        """Head camera perception: per-object (red/blue) pose/radius/height/
        confidence, active-vision phase, rate-based convergence progress +
        drift rate (world_convergence.py), cloud/map size. No ground-truth
        overlay -- unlike head_plotter.py's live dashboard, this must also be
        meaningful on real hardware, where no GT exists."""
        fig, axs = plt.subplots(4, 2, figsize=(11, 13))
        fig.suptitle('Head Perception -- Active Vision + Convergence')

        red, blue = self.head_obj["red"], self.head_obj["blue"]

        # [0,0] XY top-down estimate trajectory (the spiral-then-settle pattern).
        ax = axs[0, 0]
        ax.set_title('Top-Down Position Estimate (XY)')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_aspect('equal')
        if red["x"]:
            ax.plot(red["x"], red["y"], 'r-', alpha=0.5, linewidth=1)
            ax.plot(red["x"][0], red["y"][0], 'ro', markersize=6)
            ax.plot(red["x"][-1], red["y"][-1], 'rX', markersize=9)
        if blue["x"]:
            ax.plot(blue["x"], blue["y"], 'b-', alpha=0.5, linewidth=1)
            ax.plot(blue["x"][0], blue["y"][0], 'bo', markersize=6)
            ax.plot(blue["x"][-1], blue["y"][-1], 'bX', markersize=9)
        ax.legend(handles=[
            Line2D([0], [0], color='r', label='Red ($\\circ$=start, $\\times$=end)'),
            Line2D([0], [0], color='b', label='Blue ($\\circ$=start, $\\times$=end)'),
        ], loc='best', fontsize=8)

        # [0,1] Height (Z) of the object centre over time.
        ax = axs[0, 1]
        ax.set_title('Object Centre Height (Z)')
        ax.set_ylabel('Z [m]')
        if red["t"]:
            ax.plot(red["t"], red["z"], 'r-', label='Red')
        if blue["t"]:
            ax.plot(blue["t"], blue["z"], 'b-', label='Blue')
        ax.legend(loc='upper right', fontsize=8)

        # [1,0] Radius estimate over time.
        ax = axs[1, 0]
        ax.set_title('Radius Estimate')
        ax.set_ylabel('Radius [cm]')
        if red["t"]:
            ax.plot(red["t"], np.asarray(red["r"]) * 100.0, 'r-', label='Red')
        if blue["t"]:
            ax.plot(blue["t"], np.asarray(blue["r"]) * 100.0, 'b-', label='Blue')
        ax.legend(loc='upper right', fontsize=8)

        # [1,1] Cylinder height (the object's own height, not centre Z).
        ax = axs[1, 1]
        ax.set_title('Cylinder Height Estimate')
        ax.set_ylabel('Height [cm]')
        if red["t"]:
            ax.plot(red["t"], np.asarray(red["h"]) * 100.0, 'r-', label='Red')
        if blue["t"]:
            ax.plot(blue["t"], np.asarray(blue["h"]) * 100.0, 'b-', label='Blue')
        ax.legend(loc='upper right', fontsize=8)

        # Unpack the 15-float telemetry buffer once for the remaining panels.
        tel = np.array(self.head_tel_buffer) if self.head_tel_buffer else None
        t_tel = self.time_head_tel

        # [2,0] Confidence + weakest-object coverage.
        ax = axs[2, 0]
        ax.set_title('Estimation Confidence / Weakest Coverage')
        ax.set_ylabel('[%]'); ax.set_ylim(0, 105)
        if tel is not None:
            ax.plot(t_tel, tel[:, 6] * 100.0, 'r-', label='Red conf')
            ax.plot(t_tel, tel[:, 7] * 100.0, 'b-', label='Blue conf')
            ax.plot(t_tel, tel[:, 12] * 100.0, 'm-', label='Weakest coverage')
        ax.legend(loc='lower right', fontsize=7)

        # [2,1] Convergence-window progress (left axis) + drift rate (right,
        # GT-free "has it settled?" signal -- see world_convergence.py).
        ax = axs[2, 1]
        ax.set_title('Convergence Progress + Drift Rate')
        ax.set_ylabel('Window progress [%]'); ax.set_ylim(0, 105)
        ax_drift = ax.twinx()
        ax_drift.set_ylabel('Drift rate [mm/s]', color='darkgreen')
        ax_drift.tick_params(axis='y', labelcolor='darkgreen')
        if tel is not None:
            ax.plot(t_tel, tel[:, 9] * 100.0, 'k--', linewidth=1.2, label='Window progress')
            drift = tel[:, 13]
            drift_mm = np.where(drift >= 0, drift * 1000.0, np.nan)
            ax_drift.plot(t_tel, drift_mm, color='darkgreen', linewidth=1.3, label='Drift rate')
        ax.legend(loc='upper left', fontsize=7)

        # [3,0] Processing time per perception tick.
        ax = axs[3, 0]
        ax.set_title('Perception Processing Time')
        ax.set_ylabel('proc [ms]'); ax.set_xlabel('Time [s]')
        if tel is not None:
            ax.plot(t_tel, tel[:, 5], color='purple', linewidth=1.2)

        # [3,1] Raw cloud size + fused map size.
        ax = axs[3, 1]
        ax.set_title('Cloud / Map Size')
        ax.set_ylabel('Points / voxels'); ax.set_xlabel('Time [s]')
        if tel is not None:
            ax.plot(t_tel, tel[:, 0], color='darkorange', linewidth=1.2, label='Raw cloud (n)')
            ax.plot(t_tel, tel[:, 8], color='teal', linewidth=1.2, label='Fused map (n)')
            ax.legend(loc='upper right', fontsize=7)

        max_t = self._max_time(t_tel, red["t"], blue["t"])
        for ax in (axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1], axs[3, 0], axs[3, 1]):
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
            _draw_world_snapshot_line(ax, self.world_snapshot_t)

        if self.world_snapshot_text:
            fig.text(0.01, 0.005,
                    f"Perceived-world snapshot @ t={self.world_snapshot_t:.2f}s:\n"
                    f"{self.world_snapshot_text}",
                    fontsize=6.5, family='monospace', va='bottom')

        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        return fig

    def _build_fig_head_kinematics(self):
        """Head joint position / measured velocity / QP-commanded velocity,
        3x1 -- same structure as _build_fig_joint_kinematics but a single
        chain (no left/right split)."""
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
        fig.suptitle('Head Joint Kinematics')

        axs[0].set_title('Joint Position')
        axs[2].set_title('QP-Commanded Velocity')

        if self.time_head_js:
            t = self.time_head_js
            for i, j in enumerate(cfg.HEAD_JOINTS):
                axs[0].plot(t, self.q_head_buffers[j], color=JOINT_COLORS[i])
        if self.time_head_qdot_measured:
            # Live signal (main_head.py's own EMA-filtered reconstruction).
            t = self.time_head_qdot_measured
            arr = np.array(self.head_qdot_measured)
            for i in range(7):
                axs[1].plot(t, arr[:, i], color=JOINT_COLORS[i])
            axs[1].set_title('Measured Velocity (reconstructed)')
        elif self.time_head_js:
            # No live qdot: derive it offline. dt is floored at a fraction of
            # the median sample period (clustered timestamps would otherwise
            # divide position noise by ~0, spiking); lightly smoothed after.
            t = self.time_head_js
            t_arr = np.asarray(t, dtype=float)
            dt = np.diff(t_arr)
            pos_dt = dt[dt > 0]
            dt_floor = 0.3 * float(np.median(pos_dt)) if pos_dt.size else 1e-3
            dt_guarded = np.maximum(dt, dt_floor)
            for i, j in enumerate(cfg.HEAD_JOINTS):
                pos = np.asarray(self.q_head_buffers[j], dtype=float)
                if len(pos) > 2:
                    raw = np.empty_like(pos)
                    raw[1:] = np.diff(pos) / dt_guarded
                    raw[0] = raw[1]
                    vel = _moving_average(raw, 5)
                else:
                    vel = np.zeros_like(pos)
                axs[1].plot(t, vel, color=JOINT_COLORS[i], linestyle='--', linewidth=1.0)
            axs[1].set_title('Measured Velocity (derived offline, finite diff., dt-guarded)')
        else:
            axs[1].set_title('Measured Velocity (reconstructed)')
        if self.time_head_qdot_cmd:
            t = self.time_head_qdot_cmd
            arr = np.array(self.head_qdot_cmd)
            for i in range(7):
                axs[2].plot(t, arr[:, i], color=JOINT_COLORS[i])

        axs[0].set_ylabel('Position [rad]')
        axs[1].set_ylabel('Velocity [rad/s]')
        axs[2].set_ylabel('Velocity [rad/s]')
        axs[2].set_xlabel('Time [s]')

        max_t = self._max_time(self.time_head_js, self.time_head_qdot_measured,
                               self.time_head_qdot_cmd)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
            _draw_world_snapshot_line(ax, self.world_snapshot_t)

        legend_handles = [Line2D([0], [0], color=JOINT_COLORS[i], label=f'H{i+1}') for i in range(7)]
        fig.legend(handles=legend_handles, loc='upper center', ncol=7,
                  bbox_to_anchor=(0.5, 0.965))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

    def _build_fig_head_active_tracking(self):
        """head_active_arm_tracking.py's camera-framing performance during
        teleoperation: pixel centering error, angular errors (pointing/roll/
        approach), stand-off distance, and FOV/active-arm state -- reproduces
        that node's own live dashboard as a static figure."""
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle('Head Active-Arm Tracking -- Camera Framing Performance')

        tel = np.array(self.head_active_tel_buffer) if self.head_active_tel_buffer else None
        t = self.time_head_active_tel

        ax = axs[0, 0]
        ax.set_title('Centering -- Pixel Error (target = 0)')
        ax.set_ylabel('Error [px]')
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        if tel is not None:
            ax.plot(t, tel[:, 0], 'b-', linewidth=1.3, label='u error')
            ax.plot(t, tel[:, 1], 'r-', linewidth=1.3, label='v error')
        ax.legend(loc='upper right', fontsize=8)

        ax = axs[0, 1]
        ax.set_title('Angular Errors')
        ax.set_ylabel('Error [deg]')
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        if tel is not None:
            ax.plot(t, tel[:, 3], 'm-', linewidth=1.3, label='pointing')
            ax.plot(t, tel[:, 4], 'orange', linewidth=1.3, label='roll align')
            ax.plot(t, tel[:, 5], 'g-', linewidth=1.3, label='approach align')
        ax.legend(loc='upper right', fontsize=8)

        ax = axs[1, 0]
        ax.set_title('Stand-off Distance')
        ax.set_ylabel('Distance [m]'); ax.set_xlabel('Time [s]')
        if tel is not None:
            ax.plot(t, tel[:, 6], 'g-', linewidth=1.3)

        ax = axs[1, 1]
        ax.set_title('FOV Mode / Active Arm (step)')
        ax.set_ylabel('State'); ax.set_xlabel('Time [s]')
        ax.set_ylim(-0.1, 1.1)
        if tel is not None:
            ax.step(t, tel[:, 7], where='post', color='teal', linewidth=1.3, label='in FOV (IBVS)')
            ax.step(t, tel[:, 8], where='post', color='purple', linewidth=1.3,
                   linestyle='--', label='active arm (1=R, 0=L)')
        ax.legend(loc='upper right', fontsize=7)

        max_t = self._max_time(t)
        for ax in axs.flatten():
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def _build_fig_haption_device(self):
        """Haption handle telemetry during teleoperation: position, speed,
        clutch/grasp-trigger buttons, and the commanded force/torque wrench --
        the device-side counterpart of fig1..6 (robot side), same signals
        analyze_trial.py's Haption dashboard shows for a recorded study bag."""
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(9, 15))
        fig.suptitle('Haption Device -- Teleoperation Telemetry')

        ax = axs[0]
        ax.set_title('Handle Position (Haption base frame)')
        ax.set_ylabel('Position [m]')
        if self.virt_pose_buffer:
            P = np.array(self.virt_pose_buffer)
            for i, lab in enumerate(('x', 'y', 'z')):
                ax.plot(self.time_virt_pose, P[:, i], linewidth=1.2, label=lab)
            ax.legend(loc='upper right', fontsize=8, ncol=3)

        ax = axs[1]
        ax.set_title('Handle Speed')
        ax.set_ylabel('Speed')
        if self.virt_vel_buffer:
            V = np.array(self.virt_vel_buffer)
            lin = np.linalg.norm(V[:, 0:3], axis=1)
            ang = np.linalg.norm(V[:, 3:6], axis=1)
            ax.plot(self.time_virt_vel, lin, color='#1f77b4', linewidth=1.2, label='|v| lin [m/s]')
            ax.plot(self.time_virt_vel, ang, color='#ff7f0e', linewidth=1.2, label='|ω| ang [rad/s]')
            ax.legend(loc='upper right', fontsize=8)

        ax = axs[2]
        ax.set_title('Buttons / Deadman')
        ax.set_ylabel('Pressed / Held'); ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 1])
        if self.clutch_buffer:
            ax.step(self.time_clutch, self.clutch_buffer, where='post',
                   color='#add8e6', linewidth=1.4, label='clutch (right btn)')
        if self.grasp_trigger_buffer:
            ax.step(self.time_grasp_trigger, self.grasp_trigger_buffer, where='post',
                   color='#8c564b', linewidth=1.2, label='grasp trigger (left btn)')
        if self.deadman_buffer:
            ax.step(self.time_deadman, self.deadman_buffer, where='post',
                   color='#2ca02c', linewidth=1.6, label='deadman (handle held)')
        ax.legend(loc='upper right', fontsize=8, ncol=3)

        # Hardcoded device wrench safety clips (the force managers' final
        # MAX_FORCE/MAX_TORQUE clip -- matches analyze_trial.py's constants).
        ax = axs[3]
        ax.set_title('Device Force on the Handle (virtuose/force_cmd)')
        ax.set_ylabel('Force [N]')
        if self.force_buffer:
            F = np.array(self.force_buffer)
            for i, lab in enumerate(('Fx', 'Fy', 'Fz')):
                ax.plot(self.time_force, F[:, i], linewidth=1.2, label=lab)
            ax.axhline(10.0, color='0.2', linestyle='--', linewidth=1.0, alpha=0.7, label='±10 N limit')
            ax.axhline(-10.0, color='0.2', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.legend(loc='upper right', fontsize=7, ncol=4)

        ax = axs[4]
        ax.set_title('Device Torque on the Handle (virtuose/force_cmd)')
        ax.set_ylabel('Torque [Nm]'); ax.set_xlabel('Time [s]')
        if self.force_buffer:
            F = np.array(self.force_buffer)
            for i, lab in enumerate(('Tx', 'Ty', 'Tz')):
                ax.plot(self.time_force, F[:, 3 + i], linewidth=1.2, label=lab)
            ax.axhline(1.0, color='0.2', linestyle='--', linewidth=1.0, alpha=0.7, label='±1 Nm limit')
            ax.axhline(-1.0, color='0.2', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.legend(loc='upper right', fontsize=7, ncol=4)

        max_t = self._max_time(self.time_virt_pose, self.time_virt_vel, self.time_clutch,
                               self.time_grasp_trigger, self.time_force, self.time_deadman)
        control_intervals = self._control_intervals(max_t)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
            _draw_control_spans(ax, control_intervals)
        if control_intervals:
            fig.text(0.99, 0.005, 'green = operator in control (handle held, clutch released)',
                    ha='right', va='bottom', fontsize=7, color='#2ca02c', alpha=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    def _build_fig_shared_autonomy(self):
        """main_shared_autonomy.py's belief/blend telemetry: reference-blending
        authority alpha, the resulting user/policy action share, per-goal
        belief probabilities, and active-arm/autonomous-grasp state over time."""
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 12))
        fig.suptitle('Shared Autonomy -- Belief / Blend Telemetry')

        blend = np.array(self.blend_buffer) if self.blend_buffer else None

        ax = axs[0]
        ax.set_title('Blending Authority (0 = user, 1 = policy)')
        ax.set_ylabel(r'$\alpha$'); ax.set_ylim(-0.05, 1.05)
        if blend is not None:
            ax.plot(self.time_blend, blend[:, 0], color='#ff7f0e', linewidth=1.3)
            ax.axhline(0.5, color='0.5', linestyle=':', linewidth=0.9)

        ax = axs[1]
        ax.set_title(r'Blended-Action Share:  $(1-\alpha)\Vert v_{user}\Vert$  vs  '
                    r'$\alpha\Vert v_{policy}\Vert$')
        ax.set_ylabel('Share [%]'); ax.set_ylim(-5, 105)
        if blend is not None:
            alpha = blend[:, 0]
            v_user = blend[:, 1:7]
            v_policy = blend[:, 7:13]
            u = (1.0 - alpha) * np.linalg.norm(v_user, axis=1)
            p = alpha * np.linalg.norm(v_policy, axis=1)
            tot = u + p
            good = tot > 1e-9
            u_pct = np.full_like(alpha, np.nan)
            p_pct = np.full_like(alpha, np.nan)
            u_pct[good] = 100.0 * u[good] / tot[good]
            p_pct[good] = 100.0 * p[good] / tot[good]
            ax.plot(self.time_blend, u_pct, color='#1f77b4', linewidth=1.3, label='user %')
            ax.plot(self.time_blend, p_pct, color='#ff7f0e', linewidth=1.3, label='policy %')
            ax.legend(loc='upper right', fontsize=8, ncol=2)

        ax = axs[2]
        ax.set_title('Goal Belief Probabilities')
        ax.set_ylabel('P(goal)'); ax.set_ylim(-0.05, 1.05)
        if self.goal_prob_buffer:
            P = np.array(self.goal_prob_buffer)
            names = self.goal_names.split(',') if self.goal_names else []
            for i in range(P.shape[1]):
                lab = names[i] if i < len(names) else f'g{i}'
                ax.plot(self.time_goal_prob, P[:, i], linewidth=1.2, label=lab)
            ax.legend(loc='upper right', fontsize=7, ncol=min(P.shape[1], 4))

        ax = axs[3]
        ax.set_title('Active Arm / Autonomous Grasp State')
        ax.set_ylabel('State'); ax.set_xlabel('Time [s]'); ax.set_ylim(-0.1, 1.1)
        if self.active_arm_buffer:
            ax.step(self.time_active_arm, self.active_arm_buffer, where='post',
                   color='purple', linewidth=1.3, label='active arm (1=R, 0=L)')
        if self.grasp_active_buffer:
            ax.step(self.time_grasp_active, self.grasp_active_buffer, where='post',
                   color='teal', linewidth=1.3, linestyle='--', label='autonomous grasp')
        ax.legend(loc='upper right', fontsize=8, ncol=2)

        max_t = self._max_time(self.time_blend, self.time_goal_prob,
                               self.time_active_arm, self.time_grasp_active)
        control_intervals = self._control_intervals(max_t)
        for ax in axs:
            ax.set_xlim(0, max_t)
            _draw_trigger_line(ax, self.t_off)
            _draw_control_spans(ax, control_intervals)
        if control_intervals:
            fig.text(0.99, 0.005, 'green = operator in control (handle held, clutch released)',
                    ha='right', va='bottom', fontsize=7, color='#2ca02c', alpha=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig


class _RecorderGUI:
    """Thin Start/Stop front-end (Tkinter, same idiom as study_recorder.py)
    for OfflinePlotter. Manual clicks and trajectory_generator.py's automatic
    phase-based triggering are just two publishers of the SAME
    cfg.OFFLINE_RECORD_TRIGGER_TOPIC -- this class contains no recording
    logic of its own, only a thin view over the node's existing state
    machine (see record_trigger_callback)."""

    def __init__(self, root: "tk.Tk", node: OfflinePlotter):
        self.root = root
        self.node = node

        root.title("Offline Plotter -- Trial Recorder")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.resizable(False, False)

        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(root, padding=14)
        frm.grid(row=0, column=0, sticky="nsew")

        self.var_status = tk.StringVar(value="Idle.")
        ttk.Label(frm, textvariable=self.var_status,
                 font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", **pad)

        self.btn_start = ttk.Button(frm, text="●  START", command=self.on_start)
        self.btn_start.grid(row=1, column=0, **pad)
        self.btn_stop = ttk.Button(frm, text="■  STOP", command=self.on_stop)
        self.btn_stop.grid(row=1, column=1, **pad)

        ttk.Label(frm, text="Auto-triggered too (e.g. trajectory_generator.py) -- "
                            "whichever fires first drives the recording.",
                 foreground="#666", wraplength=280).grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)

        self.var_last = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.var_last,
                 foreground="#666", wraplength=280).grid(
            row=3, column=0, columnspan=2, sticky="w", **pad)

        self._tick()

    def on_start(self):
        self.node.pub_manual_trigger.publish(Bool(data=True))

    def on_stop(self):
        self.node.pub_manual_trigger.publish(Bool(data=False))

    def _tick(self):
        # A timer already queued via root.after() can still fire once after
        # the window is torn down (e.g. Ctrl-C destroys root directly,
        # bypassing _on_close) -- bail out rather than touch dead widgets.
        try:
            node = self.node
            state = node.state
            if state == node.STATE_RECORDING:
                elapsed = node._now() - node.t0
                name = os.path.basename(node.out_dir or "")
                self.var_status.set(f"● REC  {elapsed:6.1f}s   -> {name}")
                self.btn_start.configure(state="disabled")
                self.btn_stop.configure(state="normal")
            elif state == node.STATE_POST_ROLL:
                remaining = max(0.0, (node.post_roll_deadline or node._now()) - node._now())
                name = os.path.basename(node.out_dir or "")
                self.var_status.set(
                    f"○ settling ({remaining:0.1f}s left)   -> {name}   "
                    "(press START to extend)")
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")
            else:
                self.var_status.set("Idle -- press START, or wait for an auto-trigger.")
                self.btn_start.configure(state="normal")
                self.btn_stop.configure(state="disabled")

            if node.last_saved_dir:
                self.var_last.set(f"Last saved: {node.last_saved_dir}")

            self.root.after(300, self._tick)
        except tk.TclError:
            pass

    def _on_close(self):
        if self.node.state != self.node.STATE_WAITING:
            if not messagebox.askyesno(
                    "Recording in progress",
                    "A trial is still recording/settling. Quit and save what's "
                    "been recorded so far?"):
                return
        self.root.destroy()


def _clock_is_actually_ticking(node, timeout_s=1.0):
    """`/clock` appearing in the ROS graph only means SOMETHING somewhere
    advertises it -- a stale/leftover sim process elsewhere on the same ROS
    domain (real-hardware sessions run over a shared network) can do that
    without ever publishing, which silently freezes every recorded timestamp
    to ~0. Require an ACTUAL received message before trusting it."""
    received = []
    sub = node.create_subscription(Clock, '/clock', lambda msg: received.append(msg), 10)
    start = time.time()
    while not received and (time.time() - start) < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return bool(received)


def main(args=None):
    _apply_publication_style()
    rclpy.init(args=args)
    node = OfflinePlotter()

    # Sim-vs-real clock source. 'auto' (default) detects it; pass
    # -p sim_time_mode:=false to force wall-clock on real hardware regardless
    # of what else is on the ROS graph (see _clock_is_actually_ticking).
    node.declare_parameter('sim_time_mode', 'auto')
    mode = node.get_parameter('sim_time_mode').get_parameter_value().string_value or 'auto'
    topic_names = [name for name, _ in node.get_topic_names_and_types()]
    if mode == 'true':
        use_sim = True
    elif mode == 'false':
        use_sim = False
    else:
        use_sim = '/clock' in topic_names and _clock_is_actually_ticking(node)
        if '/clock' in topic_names and not use_sim:
            node.get_logger().warn(
                "[offline_plotter] '/clock' is advertised on the ROS graph but never "
                "published a message in 1s -- treating it as a stale/unrelated source "
                "and staying on wall-clock time. Override with -p sim_time_mode:=true "
                "if this IS your real clock source.")
    node.set_parameters([
        rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, use_sim)
    ])
    node.get_logger().info(
        f"[ENV] use_sim_time={'TRUE (sim)' if use_sim else 'FALSE (real hardware)'}")

    if tk is None:
        print(f"[offline_plotter] Tkinter unavailable ({_TK_IMPORT_ERROR}) -- running "
              "headless, auto-trigger only (no Start/Stop GUI). Install with: "
              "sudo apt install python3-tk", flush=True)
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
        return

    # GUI path: Tkinter needs the main thread's event loop, so ROS callbacks
    # (the auto-trigger from trajectory_generator.py included) run on a
    # background thread via the same spin_once-loop pattern already used by
    # every real-time entrypoint in this repo (see main_qp_controller.py) --
    # a single thread calling spin_once, never spin() and spin_once() at once.
    stop_spin = threading.Event()

    def _spin_loop():
        while rclpy.ok() and not stop_spin.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    spin_thread = threading.Thread(target=_spin_loop, daemon=True)
    spin_thread.start()

    root = tk.Tk()
    _RecorderGUI(root, node)

    # Ctrl-C in the terminal should still work like every other entrypoint in
    # this repo -- break Tk's mainloop instead of being silently swallowed by it.
    signal.signal(signal.SIGINT, lambda signum, frame: root.destroy())

    root.mainloop()

    print("\n[offline_plotter] GUI closed -- saving whatever was recorded so far...", flush=True)
    stop_spin.set()
    spin_thread.join(timeout=2.0)   # ROS context still alive here -- logger/finalize below are safe
    if node.t0 is not None:
        node._finalize_and_save(reason='gui_closed')
    else:
        print("[offline_plotter] No trial was in progress -- nothing to save.", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
