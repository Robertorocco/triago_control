#!/usr/bin/env python3
"""
freq_oscillation_diagnostic.py -- Buckets a control-loop oscillation to one of
three subsystems using ONLY existing /qp_debug telemetry, so a control-frequency
retune (e.g. CONTROL_FREQ_DEFAULT 300 -> 150) can be diagnosed without guessing
which config.py parameter to touch first.

Rides the SAME one-shot volatile Bool trigger contract trajectory_generator.py
already drives for offline_plotter.py (cfg.OFFLINE_RECORD_TRIGGER_TOPIC): rising
edge = WAITING->TRACKING (trial start), falling edge = TRACKING->REGULATION
(motion concluded). Each rising/falling pair is one trial (e.g. "home pose",
then "cross-arm") -- run this alongside (or instead of) offline_plotter.py and
it prints one bucketed verdict per trial automatically, no extra workflow.

THE THREE BUCKETS:
  1. CBF-SIDE          Oscillation concentrated where /qp_debug/min_distance is
                        small (the SoftMin collision barrier is actively
                        engaged). Suspects: D_SAFE_BASE, K_V_SAFE, GAMMA_CBF,
                        the slack scheduler's BETA / SLACK_FILTER_TAU.
  2. CLF/POSTURE-SIDE   Oscillation present throughout the motion (not just
                        near obstacles) AND already visible in the QP's OWN
                        command (/qp_debug/qdot_cmd), not only in what the arm
                        actually does. Suspects: DAMP, W_CENTER, K_GRADIENT,
                        GAMMA_CLF_DEFAULT/MIN/MAX, the reference governor's
                        GOV_A_MAX_*.
  3. DOWNSTREAM-OF-QP   /qp_debug/qdot_cmd is smooth but /qp_debug/qdot_measured
                        rings anyway: the QP's math isn't the problem, the
                        ringing happens below it (the TSID joint-velocity
                        controller or Gazebo's response to a coarser ZOH
                        setpoint update). No config.py parameter here would
                        fix that.

This is a HEURISTIC (moving-average detrend + residual RMS / zero-crossing
rate), not a proof -- read the printed numbers, don't just trust the label.
Designed to be run on the SAME trajectory preset at two different
CONTROL_FREQ_DEFAULT values so the two printed reports can be compared
directly.

Usage:
    ros2 run triago_control freq_oscillation_diagnostic.py
"""

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Float64MultiArray

import triago_control.qp_controller.config as cfg

# How long to keep buffering after the falling edge (TRACKING->REGULATION), to
# catch post-motion settling ringing. Independent of the plotter's own
# OFFLINE_PLOT_POST_TRIGGER_S (that one is tuned for figure aesthetics; this is
# just "long enough to see whether it rings out or keeps oscillating").
SETTLE_WINDOW_S = 3.0

# A channel is considered "driven this trial" above this speed -- below it we
# skip bucketing (nothing moved, nothing to diagnose). Same numeric threshold
# for EE linear speed [m/s] and joint speed [rad/s] -- close enough for a gate.
MIN_DRIVEN_SPEED = 0.005

# Moving-average detrend window: long enough to remove genuine oscillation
# ripple, short enough to preserve the intended point-to-point motion profile.
DETREND_WINDOW_S = 0.15

# Ratio thresholds for the bucket classification (see module docstring).
DOWNSTREAM_RATIO_THRESHOLD = 3.0    # measured ripple vs commanded ripple
CBF_NEAR_FAR_RATIO_THRESHOLD = 1.5  # ripple near an obstacle vs far from one
CMD_RIPPLE_FRACTION_THRESHOLD = 0.15  # ripple vs mean |qdot_cmd|


def _moving_average(x, win):
    if win < 3 or len(x) < win:
        return np.asarray(x, dtype=float).copy()
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, kernel, mode='valid')[:len(x)]


def _oscillation_metrics(t, x):
    """Detrend x (moving-average) and return (residual_rms, zero_cross_hz, residual).

    residual_rms   -- amplitude of whatever the smooth trend didn't explain.
    zero_cross_hz  -- how often the residual changes sign per second, a cheap
    stand-in for "dominant ripple frequency" without an FFT/scipy dependency.
    residual       -- the detrended series itself (same length as x), for
                       callers that need to slice it further (near/far split).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return 0.0, 0.0, np.zeros_like(x)
    duration = max(t[-1] - t[0], 1e-6)
    fs_hint = len(x) / duration
    win = max(3, int(round(DETREND_WINDOW_S * fs_hint)))
    residual = x - _moving_average(x, win)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    signs = np.sign(residual)
    signs[signs == 0] = 1
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    zero_cross_hz = crossings / (2.0 * duration)  # 2 crossings per cycle
    return rms, zero_cross_hz, residual


def _near_far_split(sig_t, residual, md_t, md_v):
    """Compare ripple RMS in the near-obstacle quartile of the trial vs the rest.

    Returns (result, reason): result is (near_rms, far_rms, n_near, n_far, n_nan)
    on success, or None with a human-readable reason otherwise. The most common
    "failure" is NOT a bug: collision_manager.compute_softmin_jacobian publishes
    abs_min_distance as NaN -- deliberately -- whenever no pair is within
    cfg.DISTANCE_FILTER_THRESHOLD of anything, which is the normal case for any
    trial that stays in free space. NaN must be filtered out before the
    percentile split (a raw NaN comparison is always False in numpy, which would
    silently collapse the whole "near" bucket to empty).
    """
    md_t = np.asarray(md_t, dtype=float)
    md_v = np.asarray(md_v, dtype=float)
    if len(md_t) < 8 or len(sig_t) < 8:
        return None, "too few min_distance samples this trial (check the topic is publishing)"
    finite = np.isfinite(md_v)
    n_nan = int(np.sum(~finite))
    if np.sum(finite) < 8:
        return None, (
            f"{n_nan}/{len(md_v)} min_distance samples were NaN -- nothing came within "
            f"cfg.DISTANCE_FILTER_THRESHOLD={cfg.DISTANCE_FILTER_THRESHOLD}m of anything this "
            "trial (this is the controller's deliberate 'no pair in range' telemetry value, not "
            "a bug -- see collision_manager.compute_softmin_jacobian). The CBF-side bucket can't "
            "be assessed without a trajectory that actually approaches an obstacle or the other arm.")
    md_t_f, md_v_f = md_t[finite], md_v[finite]
    idx = np.clip(np.searchsorted(md_t_f, sig_t), 0, len(md_t_f) - 1)
    md_at_sig = md_v_f[idx]
    near_mask = md_at_sig <= np.percentile(md_v_f, 25)
    n_near, n_far = int(np.sum(near_mask)), int(np.sum(~near_mask))
    if n_near < 5 or n_far < 5:
        return None, (f"near/far split too lopsided to compare (n_near={n_near}, n_far={n_far}, "
                      f"{n_nan} NaN samples excluded)")
    near_rms = float(np.sqrt(np.mean(residual[near_mask] ** 2)))
    far_rms = float(np.sqrt(np.mean(residual[~near_mask] ** 2)))
    return (near_rms, far_rms, n_near, n_far, n_nan), None


class TrialBuffer:
    """Per-trial time series, cleared on every rising edge of the record trigger."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.ee_t = {'right': [], 'left': []}
        self.ee_v = {'right': [], 'left': []}
        self.qdot_cmd_t = []
        self.qdot_cmd = {'right': [], 'left': []}
        self.qdot_meas_t = []
        self.qdot_meas = {'right': [], 'left': []}
        self.min_dist_t = []
        self.min_dist_v = []
        # Adaptive slack-weight scheduler (weight_slack_r, weight_slack_l,
        # gamma_clf) -- see _arm_report's scheduler-ripple check: is the
        # scheduler itself hunting (oscillating) rather than converging,
        # possibly DRIVING the qdot_cmd ripple rather than just reacting to it.
        self.dyn_weights_t = []
        self.dyn_weights = {'right': [], 'left': []}

    def now(self):
        return time.perf_counter() - self._t0


def _arm_report(side, buf):
    """Build the bucketed report for one arm, or None if it wasn't driven this trial."""
    ee_t, ee_v = buf.ee_t[side], buf.ee_v[side]
    if len(ee_v) < 5 or max(ee_v) < MIN_DRIVEN_SPEED:
        return None

    ee_rms, ee_hz, ee_residual = _oscillation_metrics(ee_t, ee_v)

    # Per-joint ripple on commanded vs measured velocity, aggregated as an
    # RMS-of-RMS "total ripple energy" across the arm's 7 joints.
    cmd_arr = np.asarray(buf.qdot_cmd[side], dtype=float)   # (N, 7)
    meas_arr = np.asarray(buf.qdot_meas[side], dtype=float)  # (M, 7)
    cmd_t = np.asarray(buf.qdot_cmd_t, dtype=float)
    meas_t = np.asarray(buf.qdot_meas_t, dtype=float)

    def _joint_rms(t, arr):
        if arr.ndim != 2 or arr.shape[0] < 5:
            return 0.0, 0.0
        rms_per_joint = []
        for j in range(arr.shape[1]):
            r, _, _ = _oscillation_metrics(t, arr[:, j])
            rms_per_joint.append(r)
        return float(np.sqrt(np.mean(np.square(rms_per_joint)))), float(np.mean(np.abs(arr)))

    cmd_rms, cmd_mean_abs = _joint_rms(cmd_t, cmd_arr)
    meas_rms, _ = _joint_rms(meas_t, meas_arr)

    downstream_ratio = meas_rms / max(cmd_rms, 1e-6)
    cmd_ripple_fraction = cmd_rms / max(cmd_mean_abs, 1e-6)

    near_far, near_far_reason = _near_far_split(np.asarray(ee_t), ee_residual, buf.min_dist_t, buf.min_dist_v)

    # --- Scheduler-side check (supplementary, not yet a priority-ordered bucket):
    # is the adaptive slack-weight scheduler (qp_formulator._schedule_weights)
    # itself hunting/oscillating rather than converging, at roughly the SAME
    # frequency as the EE-speed ripple? If so it's a plausible DRIVER of the
    # ripple (not just a symptom of it) -- the weight directly changes how
    # expensive it is for the QP to relax the CLF row, so an oscillating
    # weight produces an oscillating qdot_cmd even with unchanging task/CBF
    # geometry. Not folded into the label above yet: one A/B isn't enough to
    # pick a validated threshold for it.
    n_weight = len(buf.dyn_weights[side])
    if n_weight < 5:
        scheduler_note = "no weight data this trial (topic never received)"
    else:
        weight_rms, weight_hz, _ = _oscillation_metrics(buf.dyn_weights_t, buf.dyn_weights[side])
        if weight_rms < 1e-6:
            scheduler_note = (f"essentially constant (n={n_weight}) -- scheduler isn't reacting "
                              "to anything this trial (expected in free space: nothing threatens "
                              "the CBF, so the weight sits pinned at MAX_WEIGHT_SLACK throughout)")
        else:
            rel_diff = abs(weight_hz - ee_hz) / max(ee_hz, 1e-6)
            match = rel_diff < 0.3
            scheduler_note = (
                f"rms={weight_rms:.3f}  dominant~{weight_hz:.2f} Hz"
                + ("  <-- MATCHES EE-speed ripple frequency: the scheduler itself "
                   "may be driving the oscillation, not just reacting to it." if match else
                   "  (frequency doesn't match the EE-speed ripple -- less likely the driver)"))

    # --- Classification (priority order; see module docstring for rationale) ---
    label = "INCONCLUSIVE"
    reason = "no single check crossed its threshold -- inspect the raw numbers below."
    if downstream_ratio > DOWNSTREAM_RATIO_THRESHOLD:
        label = "DOWNSTREAM-OF-QP"
        reason = (f"measured joint-velocity ripple is {downstream_ratio:.1f}x the "
                  "QP's own commanded ripple -- the QP output is smooth, the ringing "
                  "happens below it (TSID velocity controller / Gazebo).")
    elif near_far is not None and near_far[1] > 1e-9 and (near_far[0] / near_far[1]) > CBF_NEAR_FAR_RATIO_THRESHOLD:
        label = "CBF-SIDE"
        reason = (f"EE-speed ripple is {near_far[0] / near_far[1]:.1f}x larger in the "
                  "bottom quartile of min_distance (near an obstacle) than elsewhere -- "
                  "the collision barrier is engaging unevenly at this rate.")
    elif cmd_ripple_fraction > CMD_RIPPLE_FRACTION_THRESHOLD:
        label = "CLF/POSTURE-SIDE"
        reason = (f"the QP's own command ripples at {100*cmd_ripple_fraction:.0f}% of its "
                  "mean commanded speed, throughout the motion -- the QP's task/posture "
                  "math is producing the oscillation, not something downstream of it.")

    lines = [
        f"  [{side.upper()} arm] -- driven, peak EE speed {max(ee_v):.3f} m/s",
        f"    EE-speed ripple:      rms={ee_rms:.4f} m/s   dominant~{ee_hz:.2f} Hz",
        f"    qdot_cmd ripple:      rms={cmd_rms:.4f} rad/s "
        f"({100*cmd_ripple_fraction:.1f}% of mean |cmd|)",
        f"    qdot_measured ripple: rms={meas_rms:.4f} rad/s "
        f"(measured/commanded ratio = {downstream_ratio:.2f}x)",
    ]
    if near_far is not None:
        lines.append(
            f"    near/far-obstacle EE ripple: near={near_far[0]:.4f} "
            f"(n={near_far[2]}) vs far={near_far[1]:.4f} (n={near_far[3]}, "
            f"{near_far[4]} NaN excluded)")
    else:
        lines.append(f"    near/far-obstacle split: {near_far_reason}")
    lines.append(f"    slack-weight scheduler ripple: {scheduler_note}")
    lines.append(f"    >> VERDICT: {label} -- {reason}")
    return "\n".join(lines)


def analyze_trial(trial_index, buf):
    header = (
        "\n" + "=" * 90 +
        f"\n[freq_oscillation_diagnostic] TRIAL #{trial_index} "
        f"(CONTROL_FREQ_DEFAULT={cfg.CONTROL_FREQ_DEFAULT:.0f} Hz)\n" + "=" * 90)
    bodies = []
    for side in ('right', 'left'):
        r = _arm_report(side, buf)
        bodies.append(r if r is not None else f"  [{side.upper()} arm] -- not driven this trial (skipped)")
    footer = ("-" * 90 +
              "\nHeuristic thresholds: downstream>%.1fx, cbf near/far>%.1fx, "
              "cmd-ripple-fraction>%.0f%%. Read the raw numbers, not just the label.\n"
              % (DOWNSTREAM_RATIO_THRESHOLD, CBF_NEAR_FAR_RATIO_THRESHOLD,
                 100 * CMD_RIPPLE_FRACTION_THRESHOLD) + "=" * 90)
    return header + "\n" + "\n".join(bodies) + "\n" + footer


class FreqOscillationDiagnostic(Node):

    def __init__(self):
        super().__init__('freq_oscillation_diagnostic')
        self.recording = False
        self.trial_index = 0
        self.buf = TrialBuffer()
        self._settle_timer = None

        self.create_subscription(Bool, cfg.OFFLINE_RECORD_TRIGGER_TOPIC, self._trigger_cb, 10)
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self._ee_cb, 50)
        self.create_subscription(Float64MultiArray, '/qp_debug/qdot_cmd', self._qdot_cmd_cb, 50)
        self.create_subscription(Float64MultiArray, '/qp_debug/qdot_measured', self._qdot_meas_cb, 50)
        self.create_subscription(Float64, '/qp_debug/min_distance', self._min_dist_cb, 50)
        self.create_subscription(Float64MultiArray, '/qp_debug/dynamic_weights', self._dyn_weights_cb, 50)

        self.get_logger().info(
            f"[FreqOscDiag] Listening for trials on '{cfg.OFFLINE_RECORD_TRIGGER_TOPIC}' "
            f"(CONTROL_FREQ_DEFAULT={cfg.CONTROL_FREQ_DEFAULT:.0f} Hz). "
            "Run a trajectory_generator preset to trigger a trial.")

    def _trigger_cb(self, msg: Bool):
        if bool(msg.data):
            if self._settle_timer is not None:
                self._settle_timer.cancel()
                self._settle_timer = None
            self.trial_index += 1
            self.buf = TrialBuffer()
            self.recording = True
            self.get_logger().info(f"[FreqOscDiag] Trial #{self.trial_index} started.")
        elif self.recording:
            self.get_logger().info(
                f"[FreqOscDiag] Trial #{self.trial_index} motion concluded -- "
                f"settling {SETTLE_WINDOW_S:.1f}s before analysis.")
            self._settle_timer = self.create_timer(SETTLE_WINDOW_S, self._finalize)

    def _finalize(self):
        self.recording = False
        if self._settle_timer is not None:
            self._settle_timer.cancel()
            self._settle_timer = None
        print(analyze_trial(self.trial_index, self.buf))

    def _ee_cb(self, msg):
        if not self.recording or len(msg.data) < 18:
            return
        t = self.buf.now()
        v_right = np.array(msg.data[3:6])
        v_left = np.array(msg.data[9:12])
        self.buf.ee_t['right'].append(t)
        self.buf.ee_v['right'].append(float(np.linalg.norm(v_right)))
        self.buf.ee_t['left'].append(t)
        self.buf.ee_v['left'].append(float(np.linalg.norm(v_left)))

    def _qdot_cmd_cb(self, msg):
        if not self.recording or len(msg.data) < 14:
            return
        self.buf.qdot_cmd_t.append(self.buf.now())
        self.buf.qdot_cmd['right'].append(list(msg.data[0:7]))
        self.buf.qdot_cmd['left'].append(list(msg.data[7:14]))

    def _qdot_meas_cb(self, msg):
        if not self.recording or len(msg.data) < 14:
            return
        self.buf.qdot_meas_t.append(self.buf.now())
        self.buf.qdot_meas['right'].append(list(msg.data[0:7]))
        self.buf.qdot_meas['left'].append(list(msg.data[7:14]))

    def _min_dist_cb(self, msg):
        if not self.recording:
            return
        self.buf.min_dist_t.append(self.buf.now())
        self.buf.min_dist_v.append(float(msg.data))

    def _dyn_weights_cb(self, msg):
        if not self.recording or len(msg.data) < 2:
            return
        self.buf.dyn_weights_t.append(self.buf.now())
        self.buf.dyn_weights['right'].append(float(msg.data[0]))
        self.buf.dyn_weights['left'].append(float(msg.data[1]))


def main(args=None):
    rclpy.init(args=args)
    node = FreqOscillationDiagnostic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
