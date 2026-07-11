#!/usr/bin/env python3
"""Data-driven calibration helper for `cfg.ALPHA_FILTER` (robot_kinematics.py's
EMA on the sim's differentiated joint velocity).

Rather than guessing a new alpha, this measures the CURRENT filter's noise
floor from an already-recorded trial bag (a `ros2 bag record` capture written
by offline_plotter.py, e.g. `<trial_dir>/bag`) and predicts what noise level
every other candidate alpha would produce, using the standard EMA noise-
transfer relation for white input:

    Var(filtered) = Var(raw) * alpha / (2 - alpha)

MEASUREMENT WINDOW: the last `--hold-window-s` seconds of the bag (minus a
small `--end-margin-s` trim at the very tail). Every offline_plotter trial
ends in a POST_ROLL/REGULATION hold at zero commanded velocity (the quintic's
own zero-velocity endpoint), so any spread in `/qp_debug/qdot_measured` there
is PURE differentiation + EMA noise, not real motion -- no separate
instrumentation or a special "hold still" recording is needed.

From that noise floor at the CURRENT alpha, Var(raw) is solved for by
inverting the same relation, then re-applied to a range of candidate alphas
so the noise/responsiveness trade-off can be read off directly instead of
tuned by trial and error. Also reports each candidate's equivalent time
constant tau = -dt/ln(1-alpha) (dt = the bag's OWN measured /joint_states
sample interval, not assumed), so it can be weighed against a responsiveness
budget (e.g. the reference governor's own velocity-ramp time,
GOV_V_MAX_LIN/GOV_A_MAX_LIN, printed for reference).

Usage (sourced ROS 2 environment -- bag reading needs rosbag2_py):
  ros2 run triago_control velocity_filter_calibration.py <bag_dir>
  ros2 run triago_control velocity_filter_calibration.py <bag_dir> --hold-window-s 4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import triago_control.qp_controller.config as cfg
except Exception:  # pragma: no cover - environment dependent
    cfg = None

# Candidate alphas swept for the trade-off table (current alpha is inserted
# into this list too, if not already present, so it's shown in context).
_CANDIDATE_ALPHAS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

_QDOT_TOPIC = "/qp_debug/qdot_measured"
_JOINT_STATES_TOPIC = "/joint_states"


def _detect_storage_id(bag_dir: str) -> str:
    import glob
    import os
    if glob.glob(os.path.join(bag_dir, "*.mcap")):
        return "mcap"
    return "sqlite3"


def _load_topics(bag_dir: str, topics_wanted):
    """Read the given topics into {topic: (t_ns[], rows[])}. rows[] holds the
    raw deserialized message for each sample (caller extracts what it needs).
    Same rosbag2_py idiom as scripts/analysis/study_metrics.load_bag."""
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
        print(f"[velocity_filter_calibration] WARNING: topic(s) not in bag: {missing}")

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


def _joint_states_dt_s(joint_state_msgs) -> float:
    """Median inter-message dt [s] from the messages' OWN header stamps --
    this is exactly the `dt` robot_kinematics.update_from_joint_state uses,
    not an assumed 50 Hz."""
    stamps = np.array([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                       for m in joint_state_msgs], dtype=float)
    stamps.sort()
    diffs = np.diff(stamps)
    diffs = diffs[diffs > 1e-5]
    if diffs.size == 0:
        raise RuntimeError("Could not determine /joint_states sample interval "
                          "(fewer than 2 valid timestamps in the bag).")
    return float(np.median(diffs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag_dir", help="Path to a `ros2 bag record` directory "
                   "(e.g. <trial_dir>/bag from offline_plotter.py)")
    ap.add_argument("--hold-window-s", type=float, default=5.0,
                   help="Length of the trailing hold/regulation window to sample "
                        "for the noise floor (default: 5.0 s)")
    ap.add_argument("--end-margin-s", type=float, default=0.3,
                   help="Trim this much off the very end of the bag before the "
                        "hold window, in case of shutdown artifacts (default: 0.3 s)")
    ap.add_argument("--current-alpha", type=float, default=None,
                   help="Override cfg.ALPHA_FILTER (default: read from "
                        "triago_control.qp_controller.config)")
    args = ap.parse_args()

    current_alpha = args.current_alpha
    if current_alpha is None:
        if cfg is not None and hasattr(cfg, "ALPHA_FILTER"):
            current_alpha = float(cfg.ALPHA_FILTER)
        else:
            current_alpha = 0.5
            print("[velocity_filter_calibration] WARNING: cfg.ALPHA_FILTER not "
                  "importable -- assuming 0.5. Pass --current-alpha to override.")

    data = _load_topics(args.bag_dir, [_QDOT_TOPIC, _JOINT_STATES_TOPIC])
    if _QDOT_TOPIC not in data or not data[_QDOT_TOPIC][0]:
        print(f"[velocity_filter_calibration] No '{_QDOT_TOPIC}' messages found "
              "in this bag -- nothing to measure.", file=sys.stderr)
        sys.exit(1)
    if _JOINT_STATES_TOPIC not in data or len(data[_JOINT_STATES_TOPIC][0]) < 2:
        print(f"[velocity_filter_calibration] No usable '{_JOINT_STATES_TOPIC}' "
              "messages found -- cannot measure the true sample interval.",
              file=sys.stderr)
        sys.exit(1)

    dt_s = _joint_states_dt_s(data[_JOINT_STATES_TOPIC][1])

    qdot_t_ns = np.array(data[_QDOT_TOPIC][0], dtype=np.int64)
    order = np.argsort(qdot_t_ns)
    qdot_t_ns = qdot_t_ns[order]
    qdot_rows = np.array([list(m.data) for m in data[_QDOT_TOPIC][1]], dtype=float)[order]

    t_s = (qdot_t_ns - qdot_t_ns[0]) * 1e-9
    t_end = t_s[-1] - args.end_margin_s
    t_start = t_end - args.hold_window_s
    mask = (t_s >= t_start) & (t_s <= t_end)
    n_hold = int(mask.sum())
    if n_hold < 10:
        print(f"[velocity_filter_calibration] Only {n_hold} samples in the hold "
              f"window [{t_start:.2f}, {t_end:.2f}]s -- bag may be shorter than "
              "--hold-window-s + --end-margin-s. Try a smaller --hold-window-s.",
              file=sys.stderr)
        sys.exit(1)

    hold = qdot_rows[mask]                       # (n_hold, 14)
    per_dof_mean = hold.mean(axis=0)
    per_dof_std = hold.std(axis=0)
    mean_abs_bias = float(np.mean(np.abs(per_dof_mean)))
    rms_std_measured = float(np.sqrt(np.mean(per_dof_std ** 2)))  # aggregate over 14 DOF

    print("=" * 78)
    print(" ALPHA_FILTER CALIBRATION -- noise floor from a held-still trial tail")
    print("=" * 78)
    print(f"  bag                        : {args.bag_dir}")
    print(f"  /joint_states dt (measured): {dt_s * 1000:.2f} ms "
          f"({1.0 / dt_s:.1f} Hz)")
    print(f"  hold window                : [{t_start:.2f}, {t_end:.2f}] s "
          f"({n_hold} samples of /qp_debug/qdot_measured)")
    print(f"  mean |bias| over 14 DOF    : {mean_abs_bias:.6f} rad/s "
          "(should be ~0 -- large values mean this window wasn't truly at rest)")
    print(f"  measured noise RMS (alpha={current_alpha:.2f}): {rms_std_measured:.6f} rad/s")
    print("-" * 78)

    if mean_abs_bias > 0.3 * rms_std_measured and mean_abs_bias > 1e-3:
        print("  WARNING: bias is a substantial fraction of the noise RMS -- this "
              "window may still be settling. Consider a larger --end-margin-s or "
              "checking the trial's t_off in trial_summary.txt.")
        print("-" * 78)

    # Invert the EMA noise-transfer relation at the CURRENT alpha to recover
    # the underlying raw (pre-filter) noise variance, then re-apply it to
    # every candidate alpha.
    var_filtered_measured = rms_std_measured ** 2
    var_raw = var_filtered_measured * (2.0 - current_alpha) / current_alpha

    alphas = sorted(set(_CANDIDATE_ALPHAS) | {current_alpha})
    print(f"  {'alpha':>7} {'tau_ms':>9} {'cutoff_hz':>10} {'predicted_noise_rms':>20}")
    print("-" * 78)
    for a in alphas:
        tau_s = -dt_s / np.log(1.0 - a)
        cutoff_hz = 1.0 / (2.0 * np.pi * tau_s)
        var_pred = var_raw * a / (2.0 - a)
        marker = "  <-- current" if abs(a - current_alpha) < 1e-9 else ""
        print(f"  {a:7.2f} {tau_s * 1000:9.2f} {cutoff_hz:10.2f} "
              f"{np.sqrt(var_pred):20.6f}{marker}")
    print("=" * 78)

    # Responsiveness context: the reference governor's own velocity-ramp time
    # (GOV_V_MAX/GOV_A_MAX) is a natural upper bound for tau -- the filter
    # should settle well within one ramp, not lag behind it.
    if cfg is not None and all(hasattr(cfg, k) for k in
                               ("GOV_V_MAX_LIN", "GOV_A_MAX_LIN",
                                "GOV_V_MAX_ANG", "GOV_A_MAX_ANG")):
        ramp_lin_s = cfg.GOV_V_MAX_LIN / cfg.GOV_A_MAX_LIN
        ramp_ang_s = cfg.GOV_V_MAX_ANG / cfg.GOV_A_MAX_ANG
        ramp_min_s = min(ramp_lin_s, ramp_ang_s)
        print(f"  For reference: governor ramp times are {ramp_lin_s * 1000:.0f} ms "
              f"(linear) / {ramp_ang_s * 1000:.0f} ms (angular) at the current "
              "GOV_V_MAX/GOV_A_MAX -- a tau comfortably below "
              f"~{ramp_min_s * 1000 / 5:.0f}-{ramp_min_s * 1000 / 3:.0f} ms "
              "(1/5 to 1/3 of the shorter ramp) keeps the filter from lagging it.")
        print("=" * 78)


if __name__ == "__main__":
    main()
