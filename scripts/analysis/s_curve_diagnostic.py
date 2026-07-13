#!/usr/bin/env python3
"""Time-aligned diagnostic for the s_curve_right/s_curve_left oscillation
investigation: splits a bag into the TRACKING (sweep) window vs the
REGULATION (hold) window using the record-trigger edge, then for each arm
cross-references the worst CBF moments against distance-to-red/blue-cylinder
(recomputed from /qp_debug/ee_real, not the aggregate scalar) AND joint
angles vs their limits AND /qp_debug/lambda_joints -- so "the arm was fully
stretched near an obstacle" can be checked against numbers instead of eyeballed
from plots.

Also prints the raw commanded reference at the end of TRACKING, so a stale
config (preset edited but not actually reloaded) shows up immediately instead
of being mistaken for "the fix didn't help."

Usage (sourced ROS 2 environment -- bag reading needs rosbag2_py):
  ros2 run triago_control s_curve_diagnostic.py <bag_dir>
  ros2 run triago_control s_curve_diagnostic.py <bag_dir> --top-k 8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import triago_control.qp_controller.config as cfg
except Exception:  # pragma: no cover - environment dependent
    cfg = None

_TRIGGER_TOPIC = getattr(cfg, "OFFLINE_RECORD_TRIGGER_TOPIC",
                          "/offline_plotter/record_trigger")
_TOPICS = [
    _TRIGGER_TOPIC,
    "/arm_right/cartesian_reference",
    "/arm_left/cartesian_reference",
    "/qp_debug/ee_real",
    "/qp_debug/min_distance",
    "/qp_debug/lambda_cbf",
    "/qp_debug/lambda_joints",
    "/qp_debug/qdot_cmd",
    "/qp_debug/qdot_measured",
    "/joint_states",
    "/qp_debug/joint_limits",
]

RIGHT_JOINTS = getattr(cfg, "RIGHT_JOINTS", [f"arm_right_{i}_joint" for i in range(1, 8)])
LEFT_JOINTS = getattr(cfg, "LEFT_JOINTS", [f"arm_left_{i}_joint" for i in range(1, 8)])
RED_POS = np.array(getattr(cfg, "RED_CYLINDER_POS", [0.800, -0.20, 0.775]))
BLUE_POS = np.array(getattr(cfg, "BLUE_CYLINDER_POS", [0.800, 0.20, 0.775]))
CYL_R, CYL_HALF_H = getattr(cfg, "CYLINDER_SIZE", [0.02, 0.15])[0], getattr(cfg, "CYLINDER_SIZE", [0.02, 0.15])[1] / 2.0
CAPSULE_R = float(getattr(cfg, "CAPSULE_RADIUS", 0.06))
D_SAFE = float(getattr(cfg, "D_SAFE_BASE", 0.015))


def _detect_storage_id(bag_dir: str) -> str:
    import glob, os
    if glob.glob(os.path.join(bag_dir, "*.mcap")):
        return "mcap"
    return "sqlite3"


def _load_topics(bag_dir: str, topics_wanted):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id=_detect_storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics_wanted if t not in types]
    if missing:
        print(f"[s_curve_diagnostic] WARNING: topic(s) not in bag: {missing}")

    msg_cls = {}
    out = {t: ([], []) for t in topics_wanted if t in types}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in out:
            continue
        if topic not in msg_cls:
            msg_cls[topic] = get_message(types[topic])
        msg = deserialize_message(data, msg_cls[topic])
        out[topic][0].append(t_ns)
        out[topic][1].append(msg)
    return out


def _series(data, topic, extractor):
    """Sorted (t_s, values) for a topic, t_s relative to that stream's own
    first sample -- caller re-zeroes against the global t0 itself."""
    if topic not in data or not data[topic][0]:
        return np.array([]), None
    t_ns = np.array(data[topic][0], dtype=np.int64)
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    rows = [extractor(data[topic][1][i]) for i in order]
    return t_ns, np.array(rows, dtype=float)


def _nearest_row(t_ns_query, t_ns_stream, rows_stream):
    if rows_stream is None or len(t_ns_stream) == 0:
        return None
    idx = np.searchsorted(t_ns_stream, t_ns_query)
    idx = np.clip(idx, 0, len(t_ns_stream) - 1)
    # Check the neighbor too -- searchsorted gives the insertion point, not
    # necessarily the closest sample.
    if idx > 0 and abs(t_ns_stream[idx - 1] - t_ns_query) < abs(t_ns_stream[idx] - t_ns_query):
        idx -= 1
    return rows_stream[idx]


def _h_cylinder(p, center):
    dz = max(0.0, abs(p[2] - center[2]) - CYL_HALF_H)
    dh = max(0.0, float(np.hypot(p[0] - center[0], p[1] - center[1])) - CYL_R)
    dist = float(np.hypot(dz, dh))
    return dist, dist - CAPSULE_R - D_SAFE


def _sweep_window_ns(trig_t_ns, trig_vals, fallback_lo, fallback_hi):
    """(t_start, t_end) of the TRACKING window from the record-trigger edges;
    falls back to the whole span when no falling edge was recorded."""
    lo, hi = fallback_lo, fallback_hi
    if trig_vals is not None and len(trig_t_ns) >= 2:
        rising = np.where((trig_vals[:-1] < 0.5) & (trig_vals[1:] >= 0.5))[0]
        falling = np.where((trig_vals[:-1] >= 0.5) & (trig_vals[1:] < 0.5))[0]
        if len(rising):
            lo = trig_t_ns[rising[0] + 1]
        if len(falling):
            hi = trig_t_ns[falling[0] + 1]
    return lo, hi


def _per_joint_ripple(t_ns, rows, lo_ns, hi_ns, window_s=0.3):
    """Per-joint ripple over [lo_ns, hi_ns): each column is detrended with a
    centered moving average (window_s wide) and the residual's RMS is the
    ripple. Returns (ripple_rms[14], mean_abs[14]) or None if too few samples.
    Same idea as freq_oscillation_diagnostic's aggregate ripple, but split per
    joint so the oscillation can be attributed to specific DOFs."""
    mask = (t_ns >= lo_ns) & (t_ns < hi_ns)
    if int(mask.sum()) < 30:
        return None
    t = t_ns[mask].astype(float) * 1e-9
    X = rows[mask]
    dt = float(np.median(np.diff(t)))
    win = max(3, int(round(window_s / max(dt, 1e-6))))
    if win % 2 == 0:
        win += 1
    kern = np.ones(win) / win
    resid = np.empty_like(X)
    for j in range(X.shape[1]):
        resid[:, j] = X[:, j] - np.convolve(X[:, j], kern, mode='same')
    h = win // 2  # trim the convolution's edge artifacts
    if resid.shape[0] > 2 * h + 10:
        resid = resid[h:-h]
        X = X[h:-h]
    return np.sqrt(np.mean(resid ** 2, axis=0)), np.mean(np.abs(X), axis=0)


def _print_per_joint_ripple(joint_names, cmd_t_ns, cmd_rows, meas_t_ns, meas_rows,
                            js_t_ns, js_rows, limits, lo_ns, hi_ns):
    """Table: per-joint cmd/measured ripple over the SWEEP window + where the
    joint sat in its range (mean normalized position p in [-1,1], |p|->1 =
    near a limit, where the posture field's gradient explodes). This is the
    check that separates 'posture-field ringing on stretched joints' from a
    uniform/task-side source."""
    cmd = _per_joint_ripple(cmd_t_ns, cmd_rows, lo_ns, hi_ns) if cmd_rows is not None else None
    meas = _per_joint_ripple(meas_t_ns, meas_rows, lo_ns, hi_ns) if meas_rows is not None else None
    if cmd is None:
        print("\n  (per-joint ripple: not enough /qp_debug/qdot_cmd samples in the sweep)")
        return
    cmd_rms, cmd_mean = cmd

    # Mean normalized joint position p = (q - mid)/half_range over the sweep.
    p_mean = [float('nan')] * len(joint_names)
    if js_rows is not None and limits:
        jmask = (js_t_ns >= lo_ns) & (js_t_ns < hi_ns)
        if jmask.sum() > 5:
            q_mean = np.nanmean(js_rows[jmask], axis=0)
            for j, jn in enumerate(joint_names):
                if jn in limits:
                    lo, hi = limits[jn]
                    if hi - lo > 1e-6:
                        p_mean[j] = (q_mean[j] - 0.5 * (hi + lo)) / (0.5 * (hi - lo))

    print(f"\n  PER-JOINT ripple over the SWEEP window (sorted by cmd ripple):")
    print(f"  {'joint':<22}{'cmd rms':>10}{'cmd %':>8}{'meas rms':>10}{'mean p':>8}  note")
    order = np.argsort(-cmd_rms)
    for j in order:
        frac = 100.0 * cmd_rms[j] / cmd_mean[j] if cmd_mean[j] > 1e-6 else float('nan')
        m_rms = meas[0][j] if meas is not None else float('nan')
        p = p_mean[j]
        note = ""
        if np.isfinite(p) and abs(p) > 0.7:
            note = "<-- deep in posture-field territory (|p|>0.7)"
        print(f"  {joint_names[j]:<22}{cmd_rms[j]:>10.5f}{frac:>7.1f}%{m_rms:>10.5f}"
              f"{p if np.isfinite(p) else float('nan'):>8.2f}  {note}")


def _parse_joint_limits(msg):
    """'name:lower:upper;...' -> {name: (lower, upper)}"""
    out = {}
    if msg is None:
        return out
    for chunk in msg.data.split(";"):
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            continue
        name, lo, hi = parts
        out[name] = (float(lo), float(hi))
    return out


def _report_arm(side, t0_ns, trig_t_ns, trig_vals, ee_t_ns, ee_rows,
                lam_cbf_t_ns, lam_cbf_rows, lam_j_t_ns, lam_j_rows,
                qdot_t_ns, qdot_rows, js_t_ns, js_rows, joint_names, limits,
                ref_t_ns, ref_rows, min_dist_t_ns, min_dist_rows, top_k):
    print(f"\n{'='*78}\n  ARM: {side.upper()}\n{'='*78}")

    ee_off = 0 if side == "right" else 9  # ee_real layout: [p_r,v_r,p_l,v_l,...]
    lam_idx = 0 if side == "right" else 1
    qdot_off = 0 if side == "right" else 7
    joint_off = 0 if side == "right" else 7  # joint_names/js_row are [R7, L7]

    if ee_rows is None:
        print("  (no /qp_debug/ee_real data -- skipping)")
        return
    ee_pos = ee_rows[:, ee_off:ee_off + 3]

    # --- Phase split from the record-trigger edge -------------------------
    t_start_ns = t0_ns
    t_end_tracking_ns = None
    if trig_vals is not None and len(trig_t_ns) >= 2:
        rising = np.where((trig_vals[:-1] < 0.5) & (trig_vals[1:] >= 0.5))[0]
        falling = np.where((trig_vals[:-1] >= 0.5) & (trig_vals[1:] < 0.5))[0]
        if len(rising):
            t_start_ns = trig_t_ns[rising[0] + 1]
        if len(falling):
            t_end_tracking_ns = trig_t_ns[falling[0] + 1]
    if t_end_tracking_ns is None:
        t_end_tracking_ns = ee_t_ns[-1]
        print("  WARNING: no falling edge found on the trigger topic -- "
              "treating the whole bag as SWEEP (no HOLD split).")

    sweep_mask = (ee_t_ns >= t_start_ns) & (ee_t_ns < t_end_tracking_ns)
    hold_mask = ee_t_ns >= t_end_tracking_ns
    print(f"  SWEEP window : {(t_end_tracking_ns - t_start_ns) / 1e9:.2f}s "
          f"({sweep_mask.sum()} ee samples)")
    print(f"  HOLD  window : {(ee_t_ns[-1] - t_end_tracking_ns) / 1e9:.2f}s "
          f"({hold_mask.sum()} ee samples)")

    # --- Staleness check: reference actually commanded at the end of TRACKING
    if ref_rows is not None:
        last_track_idx = np.searchsorted(ref_t_ns, t_end_tracking_ns) - 1
        last_track_idx = max(0, min(last_track_idx, len(ref_t_ns) - 1))
        print(f"  Commanded reference position at end of TRACKING: "
              f"{np.round(ref_rows[last_track_idx][:3], 4).tolist()}  "
              f"(compare against the waypoint list's last entry for this arm "
              f"in trajectory_endpoints.yaml -- mismatch means a stale config)")

    # --- Per-phase stats ----------------------------------------------------
    def _phase_stats(mask, label):
        if mask.sum() == 0:
            print(f"  [{label}] no samples")
            return
        ts = ee_t_ns[mask]
        lam_cbf_seg = np.array([_nearest_row(t, lam_cbf_t_ns, lam_cbf_rows)[lam_idx]
                                for t in ts]) if lam_cbf_rows is not None else np.array([np.nan])
        lam_j_seg = np.array([_nearest_row(t, lam_j_t_ns, lam_j_rows)[lam_idx]
                              for t in ts]) if lam_j_rows is not None else np.array([np.nan])
        h_red = np.array([_h_cylinder(p, RED_POS) for p in ee_pos[mask]])
        h_blue = np.array([_h_cylinder(p, BLUE_POS) for p in ee_pos[mask]])
        if qdot_rows is not None:
            qdot_seg = np.array([_nearest_row(t, qdot_t_ns, qdot_rows) for t in ts])[:, qdot_off:qdot_off + 7]
            qdot_rms = float(np.sqrt(np.mean(np.var(qdot_seg, axis=0))))
        else:
            qdot_rms = float('nan')
        print(f"  [{label}] qdot_measured RMS(this arm)={qdot_rms:.5f} rad/s  "
              f"lambda_cbf: mean={np.nanmean(lam_cbf_seg):.3f} max={np.nanmax(lam_cbf_seg):.3f}  "
              f"lambda_joints: mean={np.nanmean(lam_j_seg):.3f} max={np.nanmax(lam_j_seg):.3f}")
        print(f"           min raw dist to RED ={h_red[:,0].min():.4f}m (h={h_red[:,1].min():+.4f})   "
              f"min raw dist to BLUE={h_blue[:,0].min():.4f}m (h={h_blue[:,1].min():+.4f})")
        if min_dist_rows is not None:
            gt_seg = np.array([_nearest_row(t, min_dist_t_ns, min_dist_rows)[0] for t in ts])
            n_nan = int(np.isnan(gt_seg).sum())
            print(f"           CONTROLLER's own /qp_debug/min_distance (aggregate, ALL pairs, "
                  f"both arms): min={np.nanmin(gt_seg):.4f}m mean={np.nanmean(gt_seg):.4f}m  "
                  f"({n_nan}/{len(gt_seg)} samples were 'no pair in range' NaN)  "
                  f"-- compare against the RED/BLUE numbers above: if this is much smaller, "
                  f"the tightest pair is NOT this arm's fingertip vs. a cylinder.")

    _phase_stats(sweep_mask, "SWEEP  ")
    _phase_stats(hold_mask, "HOLD   ")

    # --- Peak events (worst lambda_cbf) during SWEEP only -------------------
    if lam_cbf_rows is None or sweep_mask.sum() == 0:
        return
    lam_cbf_on_ee = np.array([_nearest_row(t, lam_cbf_t_ns, lam_cbf_rows)[lam_idx]
                              for t in ee_t_ns[sweep_mask]])
    sweep_ee_t = ee_t_ns[sweep_mask]
    sweep_ee_pos = ee_pos[sweep_mask]
    order = np.argsort(-lam_cbf_on_ee)[:top_k]

    print(f"\n  Top-{top_k} lambda_cbf events during SWEEP:")
    for rank, i in enumerate(order):
        t_ns = sweep_ee_t[i]
        t_s = (t_ns - t0_ns) / 1e9
        p = sweep_ee_pos[i]
        dist_red, h_red_i = _h_cylinder(p, RED_POS)
        dist_blue, h_blue_i = _h_cylinder(p, BLUE_POS)
        lam_j = _nearest_row(t_ns, lam_j_t_ns, lam_j_rows)[lam_idx] if lam_j_rows is not None else float('nan')
        min_d = (_nearest_row(t_ns, min_dist_t_ns, min_dist_rows)[0]
                 if min_dist_rows is not None else float('nan'))
        js_row = _nearest_row(t_ns, js_t_ns, js_rows) if js_rows is not None else None
        near_limit = []
        if js_row is not None and limits:
            # Only THIS arm's own 7 joints -- the other arm holds a frozen
            # pose and isn't relevant to "is this arm fully stretched".
            for jn, val in zip(joint_names[joint_off:joint_off + 7],
                               js_row[joint_off:joint_off + 7]):
                if jn not in limits:
                    continue
                lo, hi = limits[jn]
                rng = hi - lo
                if rng <= 1e-6:
                    continue
                frac_to_nearest = min(val - lo, hi - val) / rng
                if frac_to_nearest < 0.08:  # within 8% of range from a bound
                    near_limit.append(f"{jn}={val:.3f} (limits [{lo:.3f},{hi:.3f}], "
                                      f"{frac_to_nearest*100:.1f}% from bound)")
        print(f"   #{rank+1} t={t_s:6.2f}s  lambda_cbf={lam_cbf_on_ee[i]:6.2f}  "
              f"lambda_joints={lam_j:6.2f}  ee_pos={np.round(p,3).tolist()}  "
              f"dist_red={dist_red:.4f}(h={h_red_i:+.4f})  dist_blue={dist_blue:.4f}(h={h_blue_i:+.4f})  "
              f"ctrl_min_distance={min_d:.4f}")
        if near_limit:
            print(f"        NEAR JOINT LIMIT: {'; '.join(near_limit)}")
        else:
            print(f"        (no joint within 8% of a limit at this instant)")

    # --- Correlation: does lambda_joints track lambda_cbf during the sweep?
    lam_j_on_ee = np.array([_nearest_row(t, lam_j_t_ns, lam_j_rows)[lam_idx]
                            for t in sweep_ee_t]) if lam_j_rows is not None else None
    if lam_j_on_ee is not None and np.std(lam_j_on_ee) > 1e-9 and np.std(lam_cbf_on_ee) > 1e-9:
        corr = float(np.corrcoef(lam_cbf_on_ee, lam_j_on_ee)[0, 1])
        print(f"\n  corr(lambda_cbf, lambda_joints) over SWEEP = {corr:+.3f}  "
              f"(near +1 => joint limits and the CBF are binding together, "
              f"supporting a 'fully stretched near the obstacle' compound cause)")
    else:
        print(f"\n  lambda_joints was ~0 throughout the sweep -- joint limits were "
              f"NOT a factor for this arm.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir", help="Path to a `ros2 bag record` directory "
                   "(e.g. <trial_dir>/bag from offline_plotter.py)")
    ap.add_argument("--top-k", type=int, default=5,
                   help="How many worst lambda_cbf events to detail per arm (default: 5)")
    args = ap.parse_args()

    data = _load_topics(args.bag_dir, _TOPICS)
    if "/qp_debug/ee_real" not in data or not data["/qp_debug/ee_real"][0]:
        print("[s_curve_diagnostic] No /qp_debug/ee_real in this bag -- nothing to analyze.",
              file=sys.stderr)
        sys.exit(1)

    trig_t_ns, trig_vals = _series(data, _TRIGGER_TOPIC, lambda m: [1.0 if m.data else 0.0])
    trig_vals = trig_vals[:, 0] if trig_vals is not None else None
    ee_t_ns, ee_rows = _series(data, "/qp_debug/ee_real", lambda m: list(m.data))
    lam_cbf_t_ns, lam_cbf_rows = _series(data, "/qp_debug/lambda_cbf", lambda m: list(m.data))
    lam_j_t_ns, lam_j_rows = _series(data, "/qp_debug/lambda_joints", lambda m: list(m.data))
    qdot_t_ns, qdot_rows = _series(data, "/qp_debug/qdot_measured", lambda m: list(m.data))
    cmd_t_ns, cmd_rows = _series(data, "/qp_debug/qdot_cmd", lambda m: list(m.data))
    ref_r_t_ns, ref_r_rows = _series(data, "/arm_right/cartesian_reference", lambda m: list(m.data))
    ref_l_t_ns, ref_l_rows = _series(data, "/arm_left/cartesian_reference", lambda m: list(m.data))
    min_dist_t_ns, min_dist_rows = _series(data, "/qp_debug/min_distance", lambda m: [m.data])

    def _js_extract(m):
        d = dict(zip(m.name, m.position))
        return [d.get(jn, np.nan) for jn in (RIGHT_JOINTS + LEFT_JOINTS)]
    js_t_ns, js_rows = _series(data, "/joint_states", _js_extract)

    limits = {}
    if "/qp_debug/joint_limits" in data and data["/qp_debug/joint_limits"][1]:
        limits = _parse_joint_limits(data["/qp_debug/joint_limits"][1][0])
        print(f"[s_curve_diagnostic] Loaded joint limits for {len(limits)} joints.")
    else:
        print("[s_curve_diagnostic] WARNING: no /qp_debug/joint_limits in bag -- "
              "near-limit checks will be skipped.")

    t0_ns = ee_t_ns[0]

    _report_arm("right", t0_ns, trig_t_ns, trig_vals, ee_t_ns, ee_rows,
                lam_cbf_t_ns, lam_cbf_rows, lam_j_t_ns, lam_j_rows,
                qdot_t_ns, qdot_rows, js_t_ns, js_rows, RIGHT_JOINTS + LEFT_JOINTS,
                limits, ref_r_t_ns, ref_r_rows, min_dist_t_ns, min_dist_rows, args.top_k)
    _report_arm("left", t0_ns, trig_t_ns, trig_vals, ee_t_ns, ee_rows,
                lam_cbf_t_ns, lam_cbf_rows, lam_j_t_ns, lam_j_rows,
                qdot_t_ns, qdot_rows, js_t_ns, js_rows, RIGHT_JOINTS + LEFT_JOINTS,
                limits, ref_l_t_ns, ref_l_rows, min_dist_t_ns, min_dist_rows, args.top_k)

    # --- Per-joint ripple attribution (both arms in one table) --------------
    lo_ns, hi_ns = _sweep_window_ns(trig_t_ns, trig_vals, ee_t_ns[0], ee_t_ns[-1])
    _print_per_joint_ripple(RIGHT_JOINTS + LEFT_JOINTS, cmd_t_ns, cmd_rows,
                            qdot_t_ns, qdot_rows, js_t_ns, js_rows, limits,
                            lo_ns, hi_ns)


if __name__ == "__main__":
    main()
