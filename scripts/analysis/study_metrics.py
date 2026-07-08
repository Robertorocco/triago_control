#!/usr/bin/env python3
"""Offline metric engine for recorded study trials (numpy-only, no ROS runtime).

Reads a single trial rosbag (recorded by ``study_recorder.py``) and turns it
into (a) per-topic numpy time-series and (b) a flat dict of publication-ready
summary metrics, computed **per arm** (right / left) since the QP-CLF-CBF
controller decouples the two arms and only one is teleoperated at a time.

It has NO plotting and NO pandas dependency -- matching the repo's numpy/
matplotlib stack (``analyze_trial.py`` does the plotting).

Bag reading uses ``rosbag2_py`` + ``rclpy.serialization`` and therefore needs a
sourced ROS 2 environment. Everything else is pure numpy and unit-testable.

Message-array layouts consumed (confirmed against the publishing nodes):
  /qp_debug/ee_real (18) : [R_pos(3) R_vel(3) L_pos(3) L_vel(3) R_rpy(3) L_rpy(3)]
  /qp_debug/qdot_cmd (14): [R_joint(7) L_joint(7)]  -- QP solution
  /qp_debug/qdot_measured(14): [R_joint(7) L_joint(7)] -- EMA-filtered ground truth
  /qp_debug/slacks (2)   : [|slack_R| |slack_L|]
  /qp_debug/lambda_cbf(2): [lambda_R lambda_L]
  /qp_debug/arm_frozen(2): [right_frozen left_frozen]  (0/1; active arm = NOT frozen)
  /shared_autonomy/blend_debug (19): [alpha v_user(6) v_policy(6) v_blend(6)]
  /shared_autonomy/goal_probabilities (N): aligned with goal_names (CSV string)
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402


# =============================================================================
# Topic name constants
# =============================================================================
T_EE = "/qp_debug/ee_real"
T_QDOT_CMD = "/qp_debug/qdot_cmd"
T_QDOT_MEAS = "/qp_debug/qdot_measured"
T_MINDIST = "/qp_debug/min_distance"
T_SAFETY = "/qp_debug/safety_margin"
T_LAMBDA_CBF = "/qp_debug/lambda_cbf"
T_SLACKS = "/qp_debug/slacks"
T_LOOPFREQ = "/qp_debug/loop_freq"
T_BLEND = "/shared_autonomy/blend_debug"
T_GOALPROB = "/shared_autonomy/goal_probabilities"
T_GOALNAMES = "/shared_autonomy/goal_names"
T_GRASP_ACTIVE = "/shared_autonomy/grasp_active"
T_ACTIVE_ARM = "/shared_autonomy/active_arm"
T_ARM_FROZEN = "/qp_debug/arm_frozen"
T_FORCE = "/virtuose/force_cmd"
T_CLUTCH = "/virtuose/button_right"
T_TRIGGER = "/virtuose/button_left"
T_VIRT_POSE = "/virtuose/pose"
T_VIRT_VEL = "/virtuose/velocity"
T_ARTICULAR = "/virtuose/articular_position"
T_HOME_POSE = "/joystick/home_pose"
T_REF = {"right": "/arm_right/cartesian_reference",
         "left":  "/arm_left/cartesian_reference"}

# Per-arm column indices.
_EE_IDX = {"right": {"pos": (0, 1, 2), "vel": (3, 4, 5)},
           "left":  {"pos": (6, 7, 8), "vel": (9, 10, 11)}}
_JOINT_SLICE = {"right": (0, 7), "left": (7, 14)}   # qdot_cmd / qdot_measured
_SLACK_IDX = {"right": 0, "left": 1}
_LAMBDA_IDX = {"right": 0, "left": 1}
ARMS = ("right", "left")

# Active-arm inference tuning (fallback when /shared_autonomy/active_arm absent)
_ACTIVE_SPEED_EPS = 0.005      # m/s: below this on both arms -> "idle", hold last
_ACTIVE_SMOOTH_S = 0.3         # s: moving-average window on EE speed


# =============================================================================
# Bag reading
# =============================================================================
class Series:
    """A single topic's time-series: t (seconds from trial start) + columns."""

    def __init__(self, t, cols):
        self.t = t              # np.ndarray[float], seconds
        self.cols = cols        # dict: name -> np.ndarray(float) OR list(str)

    def col(self, name):
        return self.cols.get(name)

    def __len__(self):
        return len(self.t)


def _extract_multiarray(msg):
    return {f"d{i}": float(v) for i, v in enumerate(msg.data)}


_EXTRACTORS = {
    "Float64": lambda m: {"value": float(m.data)},
    "Float32": lambda m: {"value": float(m.data)},
    "Int32": lambda m: {"value": float(m.data)},
    "Bool": lambda m: {"value": 1.0 if m.data else 0.0},
    "String": lambda m: {"value": str(m.data)},
    "Float64MultiArray": _extract_multiarray,
    "Float32MultiArray": _extract_multiarray,
    "Pose": lambda m: {"px": m.position.x, "py": m.position.y, "pz": m.position.z,
                       "qx": m.orientation.x, "qy": m.orientation.y,
                       "qz": m.orientation.z, "qw": m.orientation.w},
    "Twist": lambda m: {"lx": m.linear.x, "ly": m.linear.y, "lz": m.linear.z,
                        "ax": m.angular.x, "ay": m.angular.y, "az": m.angular.z},
    "Wrench": lambda m: {"fx": m.force.x, "fy": m.force.y, "fz": m.force.z,
                         "tx": m.torque.x, "ty": m.torque.y, "tz": m.torque.z},
}


def detect_storage_id(bag_dir: str) -> str:
    if glob.glob(os.path.join(bag_dir, "*.mcap")):
        return "mcap"
    return "sqlite3"


def load_bag(bag_dir: str) -> dict:
    """Read a trial bag into ``{topic: Series}``. Requires a sourced ROS 2 env."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=detect_storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls: dict = {}
    raw: dict = {name: [] for name in types}

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        base = types[topic].split("/")[-1]
        extractor = _EXTRACTORS.get(base)
        if extractor is None:
            continue
        if topic not in msg_cls:
            msg_cls[topic] = get_message(types[topic])
        row = extractor(deserialize_message(data, msg_cls[topic]))
        row["_t_ns"] = t_ns
        raw[topic].append(row)

    t0 = min((r[0]["_t_ns"] for r in raw.values() if r), default=0)
    series: dict = {}
    for topic, rows in raw.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r["_t_ns"])
        t = np.array([(r["_t_ns"] - t0) * 1e-9 for r in rows], dtype=float)
        keys = []
        for r in rows:
            for k in r:
                if k != "_t_ns" and k not in keys:
                    keys.append(k)
        cols = {}
        for k in keys:
            vals = [r.get(k) for r in rows]
            if all(isinstance(v, (int, float)) or v is None for v in vals):
                cols[k] = np.array([np.nan if v is None else float(v) for v in vals])
            else:
                cols[k] = [("" if v is None else str(v)) for v in vals]
        series[topic] = Series(t, cols)
    return series


# =============================================================================
# Pure numpy helpers
# =============================================================================
def sparc(speed: np.ndarray, fs: float, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length smoothness (Balasubramanian 2015). More negative =
    less smooth; near 0 = very smooth. NaN if the profile is too short/flat."""
    speed = np.asarray(speed, dtype=float)
    speed = speed[np.isfinite(speed)]
    if speed.size < 4 or fs <= 0 or not np.any(speed):
        return float("nan")
    nfft = int(2 ** (math.ceil(math.log2(speed.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    if mag.max() <= 0:
        return float("nan")
    mag = mag / mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.where(mag >= amp_th)[0]
    if above.size < 2:
        return float("nan")
    i0, i1 = above[0], above[-1]
    f, mag = f[i0:i1 + 1], mag[i0:i1 + 1]
    df = np.diff(f) / (f[-1] - f[0]) if f[-1] != f[0] else np.diff(f)
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(mag) ** 2)))


def _fs_of(t: np.ndarray) -> float:
    if t is None or len(t) < 2:
        return 0.0
    dt = np.median(np.diff(t))
    return 1.0 / dt if dt > 0 else 0.0


def _zoh(t_target: np.ndarray, t_src: np.ndarray, v_src) -> np.ndarray:
    """Zero-order-hold resample of a step signal v_src(t_src) onto t_target."""
    v_src = np.asarray(v_src)
    if len(t_src) == 0:
        return np.zeros(len(t_target), dtype=v_src.dtype)
    idx = np.searchsorted(t_src, t_target, side="right") - 1
    idx = np.clip(idx, 0, len(v_src) - 1)
    return v_src[idx]


def _rising_edges(binary) -> int:
    b = np.asarray(binary, dtype=float) > 0.5
    if b.size < 2:
        return 0
    return int(np.sum((~b[:-1]) & b[1:]))


def _time_true(t: np.ndarray, binary) -> float:
    b = np.asarray(binary, dtype=float) > 0.5
    if len(t) < 2:
        return 0.0
    return float(np.sum(np.diff(t) * b[:-1]))


def _stack(series: Series, idxs) -> np.ndarray | None:
    if series is None:
        return None
    cols = [series.col(f"d{i}") for i in idxs]
    if any(c is None for c in cols):
        return None
    return np.column_stack(cols)


def _stack_named(series: Series, names) -> np.ndarray | None:
    if series is None:
        return None
    cols = [series.col(n) for n in names]
    if any(c is None for c in cols):
        return None
    return np.column_stack(cols)


def _joint_block(series: Series, arm: str) -> np.ndarray | None:
    """The 7 joint columns (d<a>..d<b-1>) for the given arm, shape (T, 7)."""
    a, b = _JOINT_SLICE[arm]
    return _stack(series, range(a, b))


def _mode_str(values):
    vals = [v for v in (values or []) if v]
    if not vals:
        return None
    uniq, counts = np.unique(np.array(vals, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(counts))])


def _goal_names(series: dict):
    gn = series.get(T_GOALNAMES)
    if not gn or gn.col("value") is None:
        return None
    for v in reversed(gn.col("value")):
        if v:
            return v.split(",")
    return None


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


# =============================================================================
# EE speed (from PUBLISHED velocity -- /qp_debug/ee_real slots, ground truth)
# =============================================================================
def ee_speed_series(ee: Series, arm: str):
    """EE linear speed for `arm`, DIFFERENTIATED from the published position.

    NOTE: /qp_debug/ee_real's velocity slots read ~0 -- the controller's
    forwardKinematics is called with position only (no joint velocity), so
    pin.getFrameVelocity() returns a stale/zero twist. The POSITION slots are
    correct, so we differentiate them and apply a light (~100 ms) smoothing to
    tame publish jitter. Returns (t, speed) with speed the same length as t.
    """
    P = _stack(ee, _EE_IDX[arm]["pos"]) if ee else None
    if ee is None or P is None or len(P) < 2:
        return None, None
    t = np.asarray(ee.t, dtype=float)
    dt = np.diff(t)
    dt[dt <= 1e-9] = np.nan                       # guard duplicate timestamps
    v = np.diff(P, axis=0) / dt[:, None]
    speed = np.linalg.norm(v, axis=1)
    speed = np.concatenate([speed[:1], speed])    # pad back to len(t)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    win = max(1, int(0.1 * _fs_of(t)))
    return t, _smooth(speed, win)


# =============================================================================
# Active-arm resolution
# =============================================================================
def resolve_active_arm(series: dict):
    """Return (t, right_active_bool, source).

    Priority chain (all exploit the one-arm-at-a-time study invariant):
      1. /qp_debug/arm_frozen -- controller ground truth [right_frozen, left_frozen]:
         the ACTIVE arm is the one NOT frozen (the QP watchdog freezes the arm with
         no live reference). Preferred: it is the controller's own state and exists
         in every cell.
      2. /shared_autonomy/active_arm -- the shared-autonomy node's declared arm.
      3. EE-speed inference -- whichever arm is actually moving (published v_real),
         so the active hand is still recovered if neither topic is present.
    `right_active_bool[i]` is True when the right arm is the active one.
    """
    af = series.get(T_ARM_FROZEN)
    if (af is not None and af.col("d0") is not None
            and af.col("d1") is not None and len(af)):
        rf = np.asarray(af.col("d0"), dtype=float) > 0.5   # right frozen
        lf = np.asarray(af.col("d1"), dtype=float) > 0.5   # left frozen
        right = np.ones(len(af.t), dtype=bool)
        cur = True   # default to right until evidence says otherwise
        for i in range(len(af.t)):
            if (not rf[i]) and lf[i]:
                cur = True          # right live, left held -> right active
            elif rf[i] and (not lf[i]):
                cur = False         # left live, right held -> left active
            # both-frozen / both-live (idle between motions) -> hold last
            right[i] = cur
        return af.t, right, "arm_frozen_topic"

    aa = series.get(T_ACTIVE_ARM)
    if aa is not None and aa.col("value") is not None and len(aa):
        vals = aa.col("value")
        if any(v in ("right", "left") for v in vals):
            right = np.array([v != "left" for v in vals], dtype=bool)
            return aa.t, right, "active_arm_topic"

    ee = series.get(T_EE)
    if ee is None or len(ee) == 0:
        return None, None, "none"
    _, sr = ee_speed_series(ee, "right")
    _, sl = ee_speed_series(ee, "left")
    if sr is None or sl is None:
        return None, None, "none"
    win = max(1, int(_ACTIVE_SMOOTH_S * _fs_of(ee.t)))
    sr, sl = _smooth(sr, win), _smooth(sl, win)
    right = np.ones(len(ee.t), dtype=bool)
    cur = True   # default to right until evidence says otherwise
    for i in range(len(ee.t)):
        if max(sr[i], sl[i]) >= _ACTIVE_SPEED_EPS:
            cur = sr[i] >= sl[i]
        right[i] = cur
    return ee.t, right, "velocity_inference"


def active_fraction(series: dict, arm: str) -> float:
    t, right, _ = resolve_active_arm(series)
    if t is None or len(t) < 2:
        return float("nan")
    is_arm = right if arm == "right" else ~right
    return float(np.sum(np.diff(t) * is_arm[:-1]) / max(t[-1] - t[0], 1e-9))


# =============================================================================
# Metric computation (PER ARM + shared)
# =============================================================================
def compute_metrics(series: dict, metadata: dict | None = None,
                    arm: str = "right") -> dict:
    """Flat dict of metrics for ONE arm (+ trial-shared quantities)."""
    m: dict = {"arm": arm}
    nan = float("nan")

    spans = [(s.t[0], s.t[-1]) for s in series.values() if len(s)]
    m["duration_s"] = round(max(b for _, b in spans) - min(a for a, _ in spans), 3) \
        if spans else nan

    _, right_active, src = resolve_active_arm(series)
    m["active_arm_source"] = src
    m["this_arm_active_frac"] = round(active_fraction(series, arm), 4)
    if right_active is not None and len(right_active):
        m["primary_active_arm"] = "right" if np.mean(right_active) >= 0.5 else "left"
    else:
        m["primary_active_arm"] = None

    # --- EE motion (this arm) ---
    ee = series.get(T_EE)
    P = _stack(ee, _EE_IDX[arm]["pos"]) if ee else None
    if P is not None and len(P) > 1:
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        path_len = float(np.sum(steps))
        straight = float(np.linalg.norm(P[-1] - P[0]))
        m["ee_path_len_m"] = round(path_len, 4)
        m["ee_straight_len_m"] = round(straight, 4)
        m["ee_path_efficiency"] = round(straight / path_len, 4) if path_len > 1e-6 else nan
    else:
        m["ee_path_len_m"] = m["ee_straight_len_m"] = m["ee_path_efficiency"] = nan

    ts, speed = ee_speed_series(ee, arm)
    if speed is not None and speed.size:
        m["ee_speed_mean_mps"] = round(float(np.nanmean(speed)), 4)
        m["ee_speed_max_mps"] = round(float(np.nanmax(speed)), 4)
        m["ee_sparc"] = round(sparc(speed, _fs_of(ts)), 4)
    else:
        m["ee_speed_mean_mps"] = m["ee_speed_max_mps"] = m["ee_sparc"] = nan

    # --- QP solution + measured joint velocity (this arm's 7 joints) ---
    qc = _joint_block(series.get(T_QDOT_CMD), arm)
    if qc is not None and len(qc):
        m["qdot_cmd_max"] = round(float(np.nanmax(np.abs(qc))), 4)
        m["qdot_cmd_rms"] = round(float(np.sqrt(np.nanmean(qc ** 2))), 4)
    else:
        m["qdot_cmd_max"] = m["qdot_cmd_rms"] = nan
    qm = _joint_block(series.get(T_QDOT_MEAS), arm)
    if qm is not None and len(qm):
        m["qdot_meas_max"] = round(float(np.nanmax(np.abs(qm))), 4)
        m["qdot_meas_rms"] = round(float(np.sqrt(np.nanmean(qm ** 2))), 4)
    else:
        m["qdot_meas_max"] = m["qdot_meas_rms"] = nan

    # --- this arm's CLF slack + CBF shadow price ---
    sl = series.get(T_SLACKS)
    scol = sl.col(f"d{_SLACK_IDX[arm]}") if sl else None
    if scol is not None and len(scol):
        m["slack_mean"] = round(float(np.nanmean(scol)), 5)
        m["slack_peak"] = round(float(np.nanmax(scol)), 5)
    else:
        m["slack_mean"] = m["slack_peak"] = nan
    lam = series.get(T_LAMBDA_CBF)
    lcol = lam.col(f"d{_LAMBDA_IDX[arm]}") if lam else None
    if lcol is not None and len(lcol):
        m["cbf_lambda_peak"] = round(float(np.nanmax(lcol)), 4)
        m["cbf_lambda_mean"] = round(float(np.nanmean(lcol)), 4)
        m["cbf_active_frac"] = round(float(np.nanmean(np.asarray(lcol) > sc.CBF_ACTIVE_LAMBDA)), 4)
    else:
        m["cbf_lambda_peak"] = m["cbf_lambda_mean"] = m["cbf_active_frac"] = nan

    # --- safety (shared: min_distance is a single scalar over all pairs) ---
    md = series.get(T_MINDIST)
    ga = series.get(T_GRASP_ACTIVE)
    if md and md.col("value") is not None and len(md):
        d = np.asarray(md.col("value"), dtype=float)
        thr = sc.NEAR_MISS_DISTANCE_M
        # Exclude the autonomous-grasp window: min_distance is the raw signed
        # distance over all pairs BEFORE the grasp CBF-bypass, and the grasped
        # cylinder is in the collision set, so the intentional gripper<->cylinder
        # overlap otherwise reads as a safety violation.
        teleop = np.ones(len(d), dtype=bool)
        if ga and ga.col("value") is not None and len(ga):
            g = _zoh(md.t, ga.t, np.asarray(ga.col("value")) > 0.5)
            teleop = ~g.astype(bool)
        if not np.any(teleop):
            teleop = np.ones(len(d), dtype=bool)
        below = (d < thr) & teleop
        m["safety_min_dist_m"] = round(float(np.nanmin(d[teleop])), 4)
        m["safety_mean_dist_m"] = round(float(np.nanmean(d[teleop])), 4)
        m["safety_nearmiss_frac"] = round(float(np.sum(below) / max(int(np.sum(teleop)), 1)), 4)
        m["safety_nearmiss_episodes"] = _rising_edges(below)
        m["safety_min_dist_graspincl_m"] = round(float(np.nanmin(d)), 4)
    else:
        m["safety_min_dist_m"] = m["safety_mean_dist_m"] = nan
        m["safety_nearmiss_frac"] = nan
        m["safety_nearmiss_episodes"] = 0
        m["safety_min_dist_graspincl_m"] = nan

    # --- human effort (shared: device-level) ---
    fr = series.get(T_FORCE)
    Fl = _stack_named(fr, ("fx", "fy", "fz"))
    if Fl is not None and len(Fl) > 1:
        fmag = np.linalg.norm(Fl, axis=1)
        m["force_mean_N"] = round(float(np.nanmean(fmag)), 4)
        m["force_peak_N"] = round(float(np.nanmax(fmag)), 4)
        m["force_impulse_Ns"] = round(float(np.trapz(fmag, fr.t)), 4)
    else:
        m["force_mean_N"] = m["force_peak_N"] = m["force_impulse_Ns"] = nan

    cl = series.get(T_CLUTCH)
    if cl and cl.col("value") is not None and len(cl):
        b = cl.col("value")
        m["clutch_presses"] = _rising_edges(b)
        m["clutch_duty_frac"] = round(float(np.nanmean(np.asarray(b) > 0.5)), 4)
    else:
        m["clutch_presses"] = 0
        m["clutch_duty_frac"] = nan

    if ga and ga.col("value") is not None and len(ga):
        m["autonomy_grasp_time_s"] = round(_time_true(ga.t, ga.col("value")), 3)
        m["autonomy_grasp_frac"] = round(float(np.nanmean(np.asarray(ga.col("value")) > 0.5)), 4)
    else:
        m["autonomy_grasp_time_s"] = 0.0
        m["autonomy_grasp_frac"] = nan

    # --- assistance / blending (shared) ---
    bd = series.get(T_BLEND)
    if bd and bd.col("d0") is not None and len(bd):
        alpha = bd.col("d0")
        m["alpha_mean"] = round(float(np.nanmean(alpha)), 4)
        m["alpha_autonomy_frac"] = round(float(np.nanmean(np.asarray(alpha) > 0.5)), 4)
        vu = _stack(bd, range(1, 7))
        vp = _stack(bd, range(7, 13))
        if vu is not None and vp is not None:
            nu = np.linalg.norm(vu, axis=1)
            act = nu > 1e-4
            if np.any(act):
                a, b2 = vu[act], vp[act]
                denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b2, axis=1)
                good = denom > 1e-9
                cos = np.sum(a[good] * b2[good], axis=1) / denom[good]
                m["agreement_mean_cos"] = round(float(np.nanmean(cos)), 4) if cos.size else nan
                m["user_active_frac"] = round(float(np.mean(act)), 4)
            else:
                m["agreement_mean_cos"] = nan
                m["user_active_frac"] = 0.0
        else:
            m["agreement_mean_cos"] = m["user_active_frac"] = nan
    else:
        m["alpha_mean"] = m["alpha_autonomy_frac"] = nan
        m["agreement_mean_cos"] = m["user_active_frac"] = nan

    # --- intent inference (shared) ---
    gp = series.get(T_GOALPROB)
    width = sum(1 for k in gp.cols if k.startswith("d")) if gp else 0
    Pb = _stack(gp, range(width)) if width else None
    if Pb is not None and Pb.size:
        row_max = np.nanmax(Pb, axis=1)
        m["belief_max_prob"] = round(float(np.nanmax(row_max)), 4)
        hits = np.where(row_max >= sc.BELIEF_CONFIDENCE)[0]
        m["belief_time_to_conf_s"] = round(float(gp.t[hits[0]]), 3) if hits.size else nan
        names = _goal_names(series)
        widx = int(np.nanargmax(Pb[-1]))
        m["belief_winner"] = names[widx] if names and widx < len(names) else f"goal_{widx}"
    else:
        m["belief_max_prob"] = m["belief_time_to_conf_s"] = nan
        m["belief_winner"] = None

    lf = series.get(T_LOOPFREQ)
    m["loop_freq_mean_hz"] = round(float(np.nanmean(lf.col("value"))), 2) \
        if (lf and lf.col("value") is not None and len(lf)) else nan
    return m


# =============================================================================
# Human-readable summary
# =============================================================================
_SUMMARY_LAYOUT = [
    ("Trial", [("duration_s", "duration", "s"),
               ("this_arm_active_frac", "this arm active", "frac"),
               ("primary_active_arm", "primary active arm", ""),
               ("active_arm_source", "active-arm source", ""),
               ("loop_freq_mean_hz", "loop freq", "Hz")]),
    ("Motion (this arm)", [("ee_path_len_m", "EE path length", "m"),
                           ("ee_path_efficiency", "path efficiency", ""),
                           ("ee_speed_mean_mps", "mean EE speed", "m/s"),
                           ("ee_speed_max_mps", "max EE speed", "m/s"),
                           ("ee_sparc", "smoothness (SPARC)", "")]),
    ("QP solution (this arm)", [("qdot_cmd_max", "qdot_cmd peak", "rad/s"),
                                ("qdot_cmd_rms", "qdot_cmd rms", "rad/s"),
                                ("qdot_meas_max", "qdot_meas peak", "rad/s"),
                                ("qdot_meas_rms", "qdot_meas rms", "rad/s")]),
    ("Safety", [("safety_min_dist_m", "min obst dist (excl grasp)", "m"),
                ("safety_min_dist_graspincl_m", "min dist (incl grasp)", "m"),
                ("safety_nearmiss_frac", "time in near-miss", "frac"),
                ("safety_nearmiss_episodes", "near-miss episodes", ""),
                ("cbf_lambda_peak", "CBF lambda peak (this arm)", ""),
                ("cbf_active_frac", "CBF active (this arm)", "frac"),
                ("slack_peak", "CLF slack peak (this arm)", "")]),
    ("Human effort", [("force_impulse_Ns", "force impulse", "N.s"),
                      ("force_mean_N", "mean force", "N"),
                      ("clutch_presses", "clutch presses", ""),
                      ("clutch_duty_frac", "clutch duty", "frac")]),
    ("Assistance", [("alpha_mean", "mean authority alpha", ""),
                    ("alpha_autonomy_frac", "time autonomy-led", "frac"),
                    ("agreement_mean_cos", "user-policy agreement", "cos"),
                    ("autonomy_grasp_time_s", "autonomous grasp time", "s")]),
    ("Intent inference", [("belief_max_prob", "max belief prob", ""),
                          ("belief_time_to_conf_s", "time to confident", "s"),
                          ("belief_winner", "inferred goal", "")]),
]


def format_summary(metrics: dict, metadata: dict | None = None) -> str:
    md = metadata or {}
    arm = metrics.get("arm", "?")
    lines = ["=" * 58, f" TRIAGo trial metrics -- {arm.upper()} arm", "=" * 58]
    if md:
        lines += [
            f" participant : {md.get('participant', '?')}",
            f" condition   : {md.get('condition', '?')}  ({md.get('condition_short', '?')})",
            f" world       : {md.get('world_shortcut', '?')}",
            f" success     : {md.get('success', '?')}",
        ]
        if md.get("notes"):
            lines.append(f" notes       : {md['notes']}")
        lines.append("-" * 58)
    for section, rows in _SUMMARY_LAYOUT:
        lines.append(f"[{section}]")
        for key, label, unit in rows:
            val = metrics.get(key)
            if isinstance(val, float):
                sval = "n/a" if (val != val) else f"{val:.4g}"
            else:
                sval = "n/a" if val is None else str(val)
            unit = f" {unit}" if unit else ""
            lines.append(f"   {label:<26}: {sval}{unit}")
        lines.append("")
    return "\n".join(lines)


def load_metadata(bag_dir: str) -> dict:
    path = os.path.join(bag_dir, sc.METADATA_NAME)
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {}
