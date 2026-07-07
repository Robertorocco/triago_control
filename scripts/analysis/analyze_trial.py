#!/usr/bin/env python3
"""Analyze recorded study trial(s): write a PNG dashboard + a metrics summary.

For each trial folder (a rosbag directory recorded by ``study_recorder.py``) it
produces, IN THAT SAME FOLDER:
  * ``plot_dashboard.png``   -- a visual summary (EE path, speed, safety, CBF,
                                force/clutch, blending authority, goal beliefs)
  * ``metrics_summary.txt``  -- the human-readable metric table
  * ``metrics.json``         -- the flat metric dict (for later aggregation)

Usage (in a sourced ROS 2 environment):
  ros2 run triago_control analyze_trial.py                 # all trials in DATA_ROOT
  ros2 run triago_control analyze_trial.py <trial_folder>  # one trial
  ros2 run triago_control analyze_trial.py <participant_or_root_dir>
  # or plain:  python3 analyze_trial.py [path ...]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")               # headless: write PNGs, never open a window
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc           # noqa: E402
import study_metrics as sm          # noqa: E402

DASHBOARD_NAME = "plot_dashboard.png"
SUMMARY_NAME = "metrics_summary.txt"
METRICS_NAME = "metrics.json"


# ---------------------------------------------------------------- discovery
def find_trials(path: str) -> list[str]:
    """Return trial-bag folders under `path` (a dir holding rosbag2 metadata.yaml)."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(os.path.join(path, "metadata.yaml")):
        return [path]
    trials = []
    for root, dirs, files in os.walk(path):
        if "metadata.yaml" in files:
            trials.append(root)
            dirs[:] = []            # don't descend into a bag folder
    return sorted(trials)


# ---------------------------------------------------------------- plotting
def _series_xy(series, topic, col):
    s = series.get(topic)
    if s is None or s.col(col) is None or len(s) == 0:
        return None, None
    return s.t, np.asarray(s.col(col), dtype=float)


def _no_data(ax, title):
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, color="#999")
    ax.set_title(title)


def make_dashboard(series: dict, metrics: dict, metadata: dict, out_path: str):
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.22)
    active = metrics.get("active_arm", "right")
    title = (f"{metadata.get('participant', '?')} | "
             f"{metadata.get('condition', '?')} | "
             f"{metadata.get('world_shortcut', '?')} | "
             f"success={metadata.get('success', '?')}")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) EE path (top view)
    ax = fig.add_subplot(gs[0, 0])
    ee = series.get(sm.T_EE)
    P = sm._stack(ee, sm._EE_IDX[active]["pos"]) if ee else None
    if P is not None and len(P) > 1:
        ax.plot(P[:, 0], P[:, 1], "-", color="#1f77b4", lw=1.2)
        ax.plot(P[0, 0], P[0, 1], "o", color="green", label="start")
        ax.plot(P[-1, 0], P[-1, 1], "s", color="red", label="end")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"EE path (top view) - {active} arm")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "EE path")

    # 2) EE speed
    ax = fig.add_subplot(gs[0, 1])
    V = sm._stack(ee, sm._EE_IDX[active]["vel"]) if ee else None
    if V is not None and len(V):
        ax.plot(ee.t, np.linalg.norm(V, axis=1), color="#1f77b4", lw=1.0)
        ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]")
        ax.set_title(f"EE speed (SPARC={metrics.get('ee_sparc')})")
        ax.grid(alpha=0.3)
    else:
        _no_data(ax, "EE speed")

    # 3) Safety: min distance + margin
    ax = fig.add_subplot(gs[1, 0])
    t, d = _series_xy(series, sm.T_MINDIST, "value")
    if t is not None:
        ax.plot(t, d, color="#2ca02c", lw=1.0, label="min distance")
        ax.axhline(sc.NEAR_MISS_DISTANCE_M, color="red", ls="--", lw=1,
                   label=f"near-miss {sc.NEAR_MISS_DISTANCE_M} m")
        ts, ds = _series_xy(series, sm.T_SAFETY, "value")
        if ts is not None:
            ax.plot(ts, ds, color="#888", lw=0.8, alpha=0.7, label="safety margin")
        ax.set_xlabel("t [s]"); ax.set_ylabel("distance [m]")
        ax.set_title("Safety: obstacle clearance")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Safety")

    # 4) CBF shadow prices
    ax = fig.add_subplot(gs[1, 1])
    lam = series.get(sm.T_LAMBDA_CBF)
    L = sm._stack(lam, (0, 1)) if lam else None
    if L is not None and len(L):
        ax.plot(lam.t, L[:, 0], lw=1.0, label="lambda R")
        ax.plot(lam.t, L[:, 1], lw=1.0, label="lambda L")
        ax.axhline(sc.CBF_ACTIVE_LAMBDA, color="red", ls="--", lw=1, label="active")
        ax.set_xlabel("t [s]"); ax.set_ylabel("lambda_cbf")
        ax.set_title("CBF activity (shadow prices)")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "CBF activity")

    # 5) Haptic force + clutch shading
    ax = fig.add_subplot(gs[2, 0])
    fr = series.get(sm.T_FORCE)
    F = sm._stack_named(fr, ("fx", "fy", "fz")) if fr else None
    if F is not None and len(F):
        ax.plot(fr.t, np.linalg.norm(F, axis=1), color="#9467bd", lw=1.0,
                label="|force|")
        cl = series.get(sm.T_CLUTCH)
        if cl and cl.col("value") is not None and len(cl):
            _shade_true(ax, cl.t, np.asarray(cl.col("value")) > 0.5, "#ffcc00", "clutch")
        ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]")
        ax.set_title("Haptic force to handle (+ clutch)")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Haptic force")

    # 6) Blending authority alpha
    ax = fig.add_subplot(gs[2, 1])
    t, alpha = _series_xy(series, sm.T_BLEND, "d0")
    if t is not None:
        ax.plot(t, alpha, color="#ff7f0e", lw=1.0, label="alpha (policy authority)")
        ax.axhline(0.5, color="#888", ls="--", lw=1)
        ax.set_ylim(-0.05, 1.05)
        ga = series.get(sm.T_GRASP_ACTIVE)
        if ga and ga.col("value") is not None and len(ga):
            _shade_true(ax, ga.t, np.asarray(ga.col("value")) > 0.5, "#cce5ff",
                        "autonomous grasp")
        ax.set_xlabel("t [s]"); ax.set_ylabel("alpha")
        ax.set_title("Shared-autonomy authority")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Authority (alpha)")

    # 7) Goal beliefs
    ax = fig.add_subplot(gs[3, 0])
    gp = series.get(sm.T_GOALPROB)
    width = sum(1 for k in gp.cols if k.startswith("d")) if gp else 0
    Pb = sm._stack(gp, range(width)) if width else None
    if Pb is not None and len(Pb):
        names = sm._goal_names(series) or [f"goal_{i}" for i in range(width)]
        for i in range(width):
            lbl = names[i] if i < len(names) else f"goal_{i}"
            ax.plot(gp.t, Pb[:, i], lw=1.0, label=lbl)
        ax.axhline(sc.BELIEF_CONFIDENCE, color="red", ls="--", lw=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("t [s]"); ax.set_ylabel("P(goal)")
        ax.set_title("Intent inference (belief)")
        ax.legend(loc="best", fontsize=7); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Belief")

    # 8) Metrics text panel
    ax = fig.add_subplot(gs[3, 1]); ax.axis("off")
    ax.text(0.0, 1.0, sm.format_summary(metrics, metadata),
            family="monospace", fontsize=7.5, va="top", ha="left",
            transform=ax.transAxes)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _shade_true(ax, t, mask, color, label):
    """Shade contiguous intervals where mask is True."""
    t = np.asarray(t); mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    first = True
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=color, alpha=0.3,
                   label=label if first else None)
        first = False


# ---------------------------------------------------------------- driver
def analyze_one(bag_dir: str) -> bool:
    name = os.path.basename(bag_dir.rstrip("/"))
    print(f"\n[analyze_trial] {name}")
    try:
        series = sm.load_bag(bag_dir)
    except Exception as exc:                         # noqa: BLE001
        print(f"  ! failed to read bag: {type(exc).__name__}: {exc}")
        return False
    if not series:
        print("  ! bag contained no usable topics")
        return False
    metadata = sm.load_metadata(bag_dir)
    metrics = sm.compute_metrics(series, metadata)

    with open(os.path.join(bag_dir, METRICS_NAME), "w") as fh:
        json.dump({"metadata": metadata, "metrics": metrics}, fh, indent=2)
    summary = sm.format_summary(metrics, metadata)
    with open(os.path.join(bag_dir, SUMMARY_NAME), "w") as fh:
        fh.write(summary + "\n")
    try:
        make_dashboard(series, metrics, metadata, os.path.join(bag_dir, DASHBOARD_NAME))
    except Exception as exc:                         # noqa: BLE001
        print(f"  ! plotting failed: {type(exc).__name__}: {exc}")
    print(summary)
    print(f"  -> wrote {DASHBOARD_NAME}, {SUMMARY_NAME}, {METRICS_NAME}")
    return True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    targets = argv or [sc.DATA_ROOT]
    trials = []
    for tgt in targets:
        found = find_trials(tgt)
        if not found:
            print(f"[analyze_trial] no trial bags found under: {tgt}")
        trials.extend(found)
    if not trials:
        print("[analyze_trial] nothing to analyze.")
        return
    ok = sum(analyze_one(t) for t in trials)
    print(f"\n[analyze_trial] done: {ok}/{len(trials)} trial(s) analyzed.")


if __name__ == "__main__":
    main()
