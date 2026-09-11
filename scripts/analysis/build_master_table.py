#!/usr/bin/env python3
"""Aggregates every trial's metrics.json into one master CSV for statistics.

Walks DATA_ROOT/<participant>/<world_shortcut>_<cell_code>/ (study_config's
storage layout), reads each trial's metrics.json (written by analyze_trial.py:
{metadata, right:{...}, left:{...}}), and writes ONE row per trial to
trials_summary.csv (MASTER_TABLE_NAME) at DATA_ROOT. Always overwritten, never
appended -- the table is fully rebuilt from whatever is on disk each run, so
it can never drift from the bags themselves.

Only one arm is actually teleoperated per trial; the other sits idle. Per-arm
fields are collapsed to whichever arm metrics.json marks as
`primary_active_arm` and kept UNPREFIXED, since that's the quantity that's
actually comparable across conditions. The idle arm's same fields are kept
too, `passive_`-prefixed, for sanity-checking only (should read near-zero).
Fields identical on both arms' dicts (single Haption device, shared bag
topics) are read once, un-prefixed.

No ROS import (same principle as study_schedule.py): usable with plain
python3, no workspace sourcing required -- this only reads JSON already
written by analyze_trial.py, never touches a bag directly.

Usage:
  python3 build_master_table.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc  # noqa: E402

# Must match analyze_trial.METRICS_NAME -- not imported from there directly
# so this script stays free of analyze_trial's rosbag2/rclpy dependencies.
_METRICS_FILENAME = "metrics.json"

_META_FIELDS = [
    "participant", "world_shortcut", "cell_code", "condition",
    "control_mode", "assist_feedback", "assist_blending",
    "success", "notes", "bag_name", "start_time",
]

# (B) motion quality + (C) safety's per-arm half -- differ meaningfully
# between the controlled and idle arm, so each is carried in both forms.
_PER_ARM_FIELDS = [
    "ee_path_len_m", "ee_straight_len_m", "ee_path_efficiency",
    "ee_speed_mean_mps", "ee_speed_max_mps", "ee_sparc",
    "qdot_cmd_max", "qdot_cmd_rms", "qdot_meas_max", "qdot_meas_rms",
    "slack_mean", "slack_peak",
    "cbf_lambda_peak", "cbf_lambda_mean", "cbf_active_frac",
    "this_arm_active_frac",
]

# (A) effectiveness, (C) safety's shared half, (D) human effort, (E)
# assistance/intent -- identical on both arms' dicts, read once.
_SHARED_FIELDS = [
    "duration_s", "active_arm_source", "primary_active_arm",
    "safety_min_dist_m", "safety_min_dist_graspincl_m", "safety_mean_dist_m",
    "safety_nearmiss_frac", "safety_nearmiss_episodes",
    "force_mean_N", "force_peak_N", "force_impulse_Ns",
    "clutch_presses", "clutch_duty_frac",
    "autonomy_grasp_time_s", "autonomy_grasp_frac",
    "alpha_mean", "alpha_autonomy_frac", "agreement_mean_cos", "user_active_frac",
    "belief_max_prob", "belief_time_to_conf_s", "belief_winner",
    "loop_freq_mean_hz",
]

FIELDNAMES = (["trial_folder"] + _META_FIELDS + _SHARED_FIELDS
              + _PER_ARM_FIELDS + [f"passive_{f}" for f in _PER_ARM_FIELDS])


def build_row(metrics_path: str) -> dict | None:
    """One flat dict for a single trial, or None if metrics.json is unusable."""
    try:
        with open(metrics_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! skipping {metrics_path}: {exc}")
        return None

    meta = data.get("metadata") or {}
    right = data.get("right") or {}
    left = data.get("left") or {}
    # Fall back to "right" if the active arm couldn't be resolved offline
    # (e.g. an empty/very short trial) rather than dropping the row.
    active = right.get("primary_active_arm") or "right"
    active_m, passive_m = (right, left) if active == "right" else (left, right)

    row = {"trial_folder": os.path.basename(os.path.dirname(metrics_path))}
    row.update({f: meta.get(f) for f in _META_FIELDS})
    for f in _SHARED_FIELDS:
        row[f] = active_m.get(f, passive_m.get(f))
    for f in _PER_ARM_FIELDS:
        row[f] = active_m.get(f)
        row[f"passive_{f}"] = passive_m.get(f)
    return row


def find_metrics_files(data_root: str) -> list[str]:
    """Every metrics.json under DATA_ROOT, sorted for a stable row order."""
    pattern = os.path.join(data_root, "*", "*", _METRICS_FILENAME)
    return sorted(glob.glob(pattern))


def write_master_table(rows: list[dict], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    metrics_files = find_metrics_files(sc.DATA_ROOT)
    if not metrics_files:
        print(f"No {_METRICS_FILENAME} found under {sc.DATA_ROOT} -- "
              f"run analyze_trial.py on the recorded bags first.")
        return

    rows, skipped = [], []
    for path in metrics_files:
        row = build_row(path)
        (rows if row else skipped).append(row or path)

    out_path = os.path.join(sc.DATA_ROOT, sc.MASTER_TABLE_NAME)
    write_master_table(rows, out_path)
    print(f"Wrote {len(rows)} trial(s) -> {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable metrics.json (see warnings above).")

    # Trials recorded but never analyzed have no metrics.json at all, so they
    # are otherwise invisible here -- surface them explicitly.
    all_trial_dirs = {os.path.dirname(p) for p in
                      glob.glob(os.path.join(sc.DATA_ROOT, "*", "*", sc.METADATA_NAME))}
    missing = sorted(all_trial_dirs - {os.path.dirname(p) for p in metrics_files})
    if missing:
        print(f"\n{len(missing)} recorded trial(s) have no {_METRICS_FILENAME} yet "
              f"(analyze_trial.py hasn't run on them):")
        for d in missing:
            print(f"  - {d}")


if __name__ == "__main__":
    main()
