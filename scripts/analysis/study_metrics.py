#!/usr/bin/env python3
"""Offline metric engine for recorded study trials (numpy-only, no ROS runtime).

Reads a single trial rosbag (recorded by ``study_recorder.py``) and turns it
into (a) per-topic numpy time-series and (b) a flat dict of publication-ready
summary metrics. It has NO plotting and NO pandas dependency -- matching the
repo's numpy/matplotlib stack (``analyze_trial.py`` does the plotting).

Bag reading uses ``rosbag2_py`` + ``rclpy.serialization`` and therefore needs a
sourced ROS 2 environment (run via ``ros2 run`` or after ``source install/
setup.bash``). Everything else (SPARC, path length, integrals, ``compute_metrics``,
``format_summary``) is pure numpy and unit-testable without ROS.

Message-array layouts consumed (confirmed against the publishing nodes):
  /qp_debug/ee_real (18) : [R_pos(3) R_vel(3) L_pos(3) L_vel(3) R_rpy(3) L_rpy(3)]
  /qp_debug/slacks  (2)  : [|slack_R| |slack_L|]
  /qp_debug/lambda_cbf(2): [lambda_R lambda_L]
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
# Topic name constants (single place to change if a topic is renamed)
# =============================================================================
T_EE = "/qp_debug/ee_real"
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
T_FORCE = "/virtuose/force_cmd"
T_CLUTCH = "/virtuose/button_right"


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
    """Infer the rosbag2 storage backend from the files present."""
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
        if extractor is None:            # tf / joint_states etc. -> not needed
            continue
        if topic not in msg_cls:
            msg_cls[topic] = get_message(types[topic])
        row = extractor(deserialize_message(data, msg_cls[topic]))
        row["_t_ns"] = t_ns
        raw[topic].append(row)

    # Common t0 across all recorded topics.
    t0 = min((r[0]["_t_ns"] for r in raw.values() if r), default=0)

    series: dict = {}
    for topic, rows in raw.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r["_t_ns"])
        t = np.array([(r["_t_ns"] - t0) * 1e-9 for r in rows], dtype=float)
        keys = [k for k in rows[0].keys() if k != "_t_ns"]
        # union of keys (multiarray widths can differ in principle)
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
# Pure numpy metric helpers
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


def _mode_str(values) -> str | None:
    vals = [v for v in (values or []) if v]
    if not vals:
        return None
    uniq, counts = np.unique(np.array(vals, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(counts))])


# EE column indices (per the /qp_debug/ee_real layout)
_EE_IDX = {"right": {"pos": (0, 1, 2), "vel": (3, 4, 5)},
           "left":  {"pos": (6, 7, 8), "vel": (9, 10, 11)}}


# =============================================================================
# Metric computation
# =============================================================================
def compute_metrics(series: dict, metadata: dict | None = None) -> dict:
    """Return a FLAT dict of summary metrics for one trial (NaN where absent)."""
    m: dict = {}
    nan = float("nan")

    # --- trial duration / active arm ---
    all_spans = [(s.t[0], s.t[-1]) for s in series.values() if len(s)]
    if all_spans:
        m["duration_s"] = round(max(b for _, b in all_spans)
                                - min(a for a, _ in all_spans), 3)
    else:
        m["duration_s"] = nan

    arm_series = series.get(T_ACTIVE_ARM)
    active = _mode_str(arm_series.col("value")) if arm_series else None
    active = active if active in ("right", "left") else "right"
    m["active_arm"] = active

    # --- EE motion (active arm) ---
    ee = series.get(T_EE)
    P = _stack(ee, _EE_IDX[active]["pos"]) if ee else None
    V = _stack(ee, _EE_IDX[active]["vel"]) if ee else None
    if P is not None and len(P) > 1:
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        path_len = float(np.sum(steps))
        straight = float(np.linalg.norm(P[-1] - P[0]))
        m["ee_path_len_m"] = round(path_len, 4)
        m["ee_straight_len_m"] = round(straight, 4)
        m["ee_path_efficiency"] = round(straight / path_len, 4) if path_len > 1e-6 else nan
    else:
        m["ee_path_len_m"] = m["ee_straight_len_m"] = m["ee_path_efficiency"] = nan

    if V is not None and len(V):
        speed = np.linalg.norm(V, axis=1)
        m["ee_speed_mean_mps"] = round(float(np.nanmean(speed)), 4)
        m["ee_speed_max_mps"] = round(float(np.nanmax(speed)), 4)
        m["ee_sparc"] = round(sparc(speed, _fs_of(ee.t)), 4)
    else:
        m["ee_speed_mean_mps"] = m["ee_speed_max_mps"] = m["ee_sparc"] = nan

    # --- safety ---
    md = series.get(T_MINDIST)
    if md and md.col("value") is not None and len(md):
        d = md.col("value")
        thr = sc.NEAR_MISS_DISTANCE_M
        m["safety_min_dist_m"] = round(float(np.nanmin(d)), 4)
        m["safety_mean_dist_m"] = round(float(np.nanmean(d)), 4)
        m["safety_nearmiss_frac"] = round(float(np.nanmean(d < thr)), 4)
        m["safety_nearmiss_episodes"] = _rising_edges(d < thr)
    else:
        m["safety_min_dist_m"] = m["safety_mean_dist_m"] = nan
        m["safety_nearmiss_frac"] = nan
        m["safety_nearmiss_episodes"] = 0

    lam = series.get(T_LAMBDA_CBF)
    L = _stack(lam, (0, 1)) if lam else None
    if L is not None and len(L):
        lmax = np.nanmax(L, axis=1)
        m["cbf_lambda_peak"] = round(float(np.nanmax(lmax)), 4)
        m["cbf_lambda_mean"] = round(float(np.nanmean(lmax)), 4)
        m["cbf_active_frac"] = round(float(np.nanmean(lmax > sc.CBF_ACTIVE_LAMBDA)), 4)
    else:
        m["cbf_lambda_peak"] = m["cbf_lambda_mean"] = m["cbf_active_frac"] = nan

    sl = series.get(T_SLACKS)
    S = _stack(sl, (0, 1)) if sl else None
    if S is not None and len(S):
        smax = np.nanmax(S, axis=1)
        m["slack_mean"] = round(float(np.nanmean(smax)), 5)
        m["slack_peak"] = round(float(np.nanmax(smax)), 5)
    else:
        m["slack_mean"] = m["slack_peak"] = nan

    # --- human effort ---
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

    ga = series.get(T_GRASP_ACTIVE)
    if ga and ga.col("value") is not None and len(ga):
        m["autonomy_grasp_time_s"] = round(_time_true(ga.t, ga.col("value")), 3)
        m["autonomy_grasp_frac"] = round(float(np.nanmean(np.asarray(ga.col("value")) > 0.5)), 4)
    else:
        m["autonomy_grasp_time_s"] = 0.0
        m["autonomy_grasp_frac"] = nan

    # --- assistance / blending ---
    bd = series.get(T_BLEND)
    if bd and bd.col("d0") is not None and len(bd):
        alpha = bd.col("d0")
        m["alpha_mean"] = round(float(np.nanmean(alpha)), 4)
        m["alpha_autonomy_frac"] = round(float(np.nanmean(np.asarray(alpha) > 0.5)), 4)
        vu = _stack(bd, range(1, 7))
        vp = _stack(bd, range(7, 13))
        if vu is not None and vp is not None:
            nu = np.linalg.norm(vu, axis=1)
            active_rows = nu > 1e-4
            if np.any(active_rows):
                a, b2 = vu[active_rows], vp[active_rows]
                denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b2, axis=1)
                good = denom > 1e-9
                cos = np.sum(a[good] * b2[good], axis=1) / denom[good]
                m["agreement_mean_cos"] = round(float(np.nanmean(cos)), 4) if cos.size else nan
                m["user_active_frac"] = round(float(np.mean(active_rows)), 4)
            else:
                m["agreement_mean_cos"] = nan
                m["user_active_frac"] = 0.0
        else:
            m["agreement_mean_cos"] = nan
            m["user_active_frac"] = nan
    else:
        m["alpha_mean"] = m["alpha_autonomy_frac"] = nan
        m["agreement_mean_cos"] = m["user_active_frac"] = nan

    # --- intent inference (belief) ---
    gp = series.get(T_GOALPROB)
    if gp and len(gp):
        width = sum(1 for k in gp.cols if k.startswith("d"))
        Pb = _stack(gp, range(width)) if width else None
        if Pb is not None and Pb.size:
            row_max = np.nanmax(Pb, axis=1)
            m["belief_max_prob"] = round(float(np.nanmax(row_max)), 4)
            conf_hits = np.where(row_max >= sc.BELIEF_CONFIDENCE)[0]
            m["belief_time_to_conf_s"] = round(float(gp.t[conf_hits[0]]), 3) \
                if conf_hits.size else nan
            names = _goal_names(series)
            widx = int(np.nanargmax(Pb[-1]))
            m["belief_winner"] = names[widx] if names and widx < len(names) else f"goal_{widx}"
        else:
            m["belief_max_prob"] = m["belief_time_to_conf_s"] = nan
            m["belief_winner"] = None
    else:
        m["belief_max_prob"] = m["belief_time_to_conf_s"] = nan
        m["belief_winner"] = None

    # --- controller health ---
    lf = series.get(T_LOOPFREQ)
    if lf and lf.col("value") is not None and len(lf):
        m["loop_freq_mean_hz"] = round(float(np.nanmean(lf.col("value"))), 2)
    else:
        m["loop_freq_mean_hz"] = nan

    return m


def _stack_named(series: Series, names) -> np.ndarray | None:
    if series is None:
        return None
    cols = [series.col(n) for n in names]
    if any(c is None for c in cols):
        return None
    return np.column_stack(cols)


def _goal_names(series: dict):
    gn = series.get(T_GOALNAMES)
    if not gn or gn.col("value") is None:
        return None
    for v in reversed(gn.col("value")):
        if v:
            return v.split(",")
    return None


# =============================================================================
# Human-readable summary
# =============================================================================
_SUMMARY_LAYOUT = [
    ("Trial", [("duration_s", "duration", "s"), ("active_arm", "active arm", ""),
               ("loop_freq_mean_hz", "loop freq", "Hz")]),
    ("Task / motion", [("ee_path_len_m", "EE path length", "m"),
                       ("ee_path_efficiency", "path efficiency", ""),
                       ("ee_speed_mean_mps", "mean EE speed", "m/s"),
                       ("ee_speed_max_mps", "max EE speed", "m/s"),
                       ("ee_sparc", "smoothness (SPARC)", "")]),
    ("Safety", [("safety_min_dist_m", "min obstacle dist", "m"),
                ("safety_mean_dist_m", "mean obstacle dist", "m"),
                ("safety_nearmiss_frac", "time in near-miss", "frac"),
                ("safety_nearmiss_episodes", "near-miss episodes", ""),
                ("cbf_lambda_peak", "CBF lambda peak", ""),
                ("cbf_active_frac", "CBF active time", "frac"),
                ("slack_peak", "CLF slack peak", "")]),
    ("Human effort", [("force_impulse_Ns", "force impulse", "N.s"),
                      ("force_mean_N", "mean force", "N"),
                      ("clutch_presses", "clutch presses", ""),
                      ("clutch_duty_frac", "clutch duty", "frac")]),
    ("Assistance", [("alpha_mean", "mean authority alpha", ""),
                    ("alpha_autonomy_frac", "time autonomy-led", "frac"),
                    ("agreement_mean_cos", "user-policy agreement", "cos"),
                    ("user_active_frac", "time user driving", "frac"),
                    ("autonomy_grasp_time_s", "autonomous grasp time", "s")]),
    ("Intent inference", [("belief_max_prob", "max belief prob", ""),
                          ("belief_time_to_conf_s", "time to confident", "s"),
                          ("belief_winner", "inferred goal", "")]),
]


def format_summary(metrics: dict, metadata: dict | None = None) -> str:
    md = metadata or {}
    lines = ["=" * 56, " TRIAGo user-study trial metrics", "=" * 56]
    if md:
        lines += [
            f" participant : {md.get('participant', '?')}",
            f" condition   : {md.get('condition', '?')}  ({md.get('condition_short', '?')})",
            f" world       : {md.get('world_shortcut', '?')}",
            f" success     : {md.get('success', '?')}",
        ]
        if md.get("notes"):
            lines.append(f" notes       : {md['notes']}")
        lines.append("-" * 56)
    for section, rows in _SUMMARY_LAYOUT:
        lines.append(f"[{section}]")
        for key, label, unit in rows:
            val = metrics.get(key)
            if isinstance(val, float):
                sval = "n/a" if (val != val) else f"{val:.4g}"
            else:
                sval = "n/a" if val is None else str(val)
            unit = f" {unit}" if unit else ""
            lines.append(f"   {label:<24}: {sval}{unit}")
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
