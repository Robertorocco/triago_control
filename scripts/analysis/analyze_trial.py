#!/usr/bin/env python3
"""Analyze recorded study trial(s): per-hand PNG dashboards + metric summaries.

The QP-CLF-CBF controller decouples the two arms and only one is teleoperated at
a time, so each trial is analyzed and plotted SEPARATELY per arm. For each trial
folder (a rosbag directory recorded by ``study_recorder.py``) it writes, IN THAT
FOLDER:
  * plot_dashboard_right.png / plot_dashboard_left.png
  * metrics_summary_right.txt / metrics_summary_left.txt
  * metrics.json  ({metadata, right:{...}, left:{...}})

Each dashboard uses that arm's DECOUPLED data: its EE pose/speed (from the
PUBLISHED /qp_debug/ee_real velocity slots -- ground truth, not differentiated),
its 7 joints of measured velocity (ground truth) and QP-solution velocity
(qdot_cmd), its own CLF slack + CBF shadow price, plus which hand is active over
time and the shared device/assistance signals.

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as sc        # noqa: E402
import study_metrics as sm       # noqa: E402

METRICS_NAME = "metrics.json"


# ---------------------------------------------------------------- discovery
def find_trials(path: str) -> list[str]:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(os.path.join(path, "metadata.yaml")):
        return [path]
    trials = []
    for root, dirs, files in os.walk(path):
        if "metadata.yaml" in files:
            trials.append(root)
            dirs[:] = []
    return sorted(trials)


# ---------------------------------------------------------------- plot utils
def _no_data(ax, title):
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, color="#999")
    ax.set_title(title)


def _shade_true(ax, t, mask, color, label):
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
        ax.axvspan(t[s], t[e], color=color, alpha=0.25,
                   label=label if first else None)
        first = False


def _shade_grasp(ax, series):
    ga = series.get(sm.T_GRASP_ACTIVE)
    if ga and ga.col("value") is not None and len(ga):
        _shade_true(ax, ga.t, np.asarray(ga.col("value")) > 0.5,
                    "#cce5ff", "autonomous grasp")


def _shade_active(ax, t_act, right_active, arm):
    """Shade the spans where THIS arm is the active (teleoperated) one."""
    if t_act is None:
        return
    is_this = right_active if arm == "right" else ~right_active
    _shade_true(ax, t_act, is_this, "#ffe0b3", f"{arm} active")


# ---------------------------------------------------------------- dashboard
def make_dashboard(series, metrics, metadata, arm, t_act, right_active, out_path):
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.26)
    ee = series.get(sm.T_EE)
    title = (f"{metadata.get('participant', '?')} | {metadata.get('condition', '?')} | "
             f"{metadata.get('world_shortcut', '?')} | success={metadata.get('success', '?')} "
             f"|  {arm.upper()} arm")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # (0,0) EE path (top view)
    ax = fig.add_subplot(gs[0, 0])
    P = sm._stack(ee, sm._EE_IDX[arm]["pos"]) if ee else None
    if P is not None and len(P) > 1:
        ax.plot(P[:, 0], P[:, 1], "-", color="#1f77b4", lw=1.2)
        ax.plot(P[0, 0], P[0, 1], "o", color="green", label="start")
        ax.plot(P[-1, 0], P[-1, 1], "s", color="red", label="end")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"EE path (top view) - {arm}")
        ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "EE path")

    # (0,1) EE speed (from PUBLISHED velocity) + this-arm-active shading
    ax = fig.add_subplot(gs[0, 1])
    ts, speed = sm.ee_speed_series(ee, arm) if ee else (None, None)
    if speed is not None and speed.size:
        ax.plot(ts, speed, color="#1f77b4", lw=1.0)
        _shade_active(ax, t_act, right_active, arm)
        ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]")
        ax.set_title(f"EE speed (published v_real)   SPARC={metrics.get('ee_sparc')}")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.3)
    else:
        _no_data(ax, "EE speed")

    # (0,2) Active-arm timeline
    ax = fig.add_subplot(gs[0, 2])
    if t_act is not None:
        ax.plot(t_act, right_active.astype(float), drawstyle="steps-post",
                color="#8c564b", lw=1.2)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["left", "right"])
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("t [s]"); ax.set_ylabel("active arm")
        ax.set_title(f"Active arm  (src: {metrics.get('active_arm_source')})")
        ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Active arm")

    # (1,0) Measured joint velocity (ground truth) - this arm
    ax = fig.add_subplot(gs[1, 0])
    qm = sm._joint_block(series.get(sm.T_QDOT_MEAS), arm)
    tqm = series.get(sm.T_QDOT_MEAS).t if series.get(sm.T_QDOT_MEAS) else None
    if qm is not None and len(qm):
        for j in range(qm.shape[1]):
            ax.plot(tqm, qm[:, j], lw=0.8, label=f"J{j+1}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("qdot_meas [rad/s]")
        ax.set_title("Measured joint velocity (ground truth)")
        ax.legend(loc="best", fontsize=6, ncol=2); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Measured joint velocity")

    # (1,1) QP-solution joint velocity - this arm
    ax = fig.add_subplot(gs[1, 1])
    qc = sm._joint_block(series.get(sm.T_QDOT_CMD), arm)
    tqc = series.get(sm.T_QDOT_CMD).t if series.get(sm.T_QDOT_CMD) else None
    if qc is not None and len(qc):
        for j in range(qc.shape[1]):
            ax.plot(tqc, qc[:, j], lw=0.8, label=f"J{j+1}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("qdot_cmd [rad/s]")
        ax.set_title("QP solution (commanded joint velocity)")
        ax.legend(loc="best", fontsize=6, ncol=2); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "QP solution")

    # (1,2) CLF slack + CBF lambda (this arm)
    ax = fig.add_subplot(gs[1, 2])
    sl = series.get(sm.T_SLACKS)
    scol = sl.col(f"d{sm._SLACK_IDX[arm]}") if sl else None
    lam = series.get(sm.T_LAMBDA_CBF)
    lcol = lam.col(f"d{sm._LAMBDA_IDX[arm]}") if lam else None
    if scol is not None or lcol is not None:
        if scol is not None:
            ax.plot(sl.t, scol, color="#d62728", lw=1.0, label="CLF slack")
        ax.set_xlabel("t [s]"); ax.set_ylabel("slack")
        if lcol is not None:
            ax2 = ax.twinx()
            ax2.plot(lam.t, lcol, color="#2ca02c", lw=1.0, label="lambda_cbf")
            ax2.axhline(sc.CBF_ACTIVE_LAMBDA, color="#2ca02c", ls="--", lw=0.8, alpha=0.6)
            ax2.set_ylabel("lambda_cbf", color="#2ca02c")
        ax.set_title(f"CLF slack + CBF lambda ({arm})")
        ax.grid(alpha=0.3)
    else:
        _no_data(ax, "slack / lambda")

    # (2,0) Safety (grasp shaded)
    ax = fig.add_subplot(gs[2, 0])
    md = series.get(sm.T_MINDIST)
    if md and md.col("value") is not None and len(md):
        ax.plot(md.t, md.col("value"), color="#2ca02c", lw=1.0,
                label="min pair dist (signed)")
        sf = series.get(sm.T_SAFETY)
        if sf and sf.col("value") is not None:
            ax.plot(sf.t, sf.col("value"), color="#888", lw=0.8, alpha=0.8,
                    label="CBF margin (d - d_safe)")
        ax.axhline(sc.NEAR_MISS_DISTANCE_M, color="red", ls="--", lw=1,
                   label=f"near-miss {sc.NEAR_MISS_DISTANCE_M} m")
        ax.axhline(0.0, color="#333", ls=":", lw=0.8)
        _shade_grasp(ax, series)
        ax.set_xlabel("t [s]"); ax.set_ylabel("distance [m]")
        ax.set_title("Safety: obstacle clearance (grasp shaded)")
        ax.legend(loc="best", fontsize=7); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Safety")

    # (2,1) Haptic force + clutch + alpha
    ax = fig.add_subplot(gs[2, 1])
    fr = series.get(sm.T_FORCE)
    F = sm._stack_named(fr, ("fx", "fy", "fz")) if fr else None
    if F is not None and len(F):
        ax.plot(fr.t, np.linalg.norm(F, axis=1), color="#9467bd", lw=1.0, label="|force|")
        cl = series.get(sm.T_CLUTCH)
        if cl and cl.col("value") is not None and len(cl):
            _shade_true(ax, cl.t, np.asarray(cl.col("value")) > 0.5, "#ffcc00", "clutch")
        ax.set_xlabel("t [s]"); ax.set_ylabel("force [N]")
        bd = series.get(sm.T_BLEND)
        if bd and bd.col("d0") is not None:
            ax2 = ax.twinx()
            ax2.plot(bd.t, bd.col("d0"), color="#ff7f0e", lw=0.9, alpha=0.8, label="alpha")
            ax2.set_ylim(-0.05, 1.05); ax2.set_ylabel("alpha", color="#ff7f0e")
        ax.set_title("Haptic force + clutch (+ authority alpha)")
        ax.legend(loc="upper left", fontsize=7); ax.grid(alpha=0.3)
    else:
        _no_data(ax, "Haptic force")

    # (2,2) Metrics text
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    ax.text(0.0, 1.0, sm.format_summary(metrics, metadata),
            family="monospace", fontsize=7, va="top", ha="left",
            transform=ax.transAxes)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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
    t_act, right_active, src = sm.resolve_active_arm(series)

    out = {"metadata": metadata}
    for arm in sm.ARMS:
        metrics = sm.compute_metrics(series, metadata, arm)
        out[arm] = metrics
        with open(os.path.join(bag_dir, f"metrics_summary_{arm}.txt"), "w") as fh:
            fh.write(sm.format_summary(metrics, metadata) + "\n")
        try:
            make_dashboard(series, metrics, metadata, arm, t_act, right_active,
                           os.path.join(bag_dir, f"plot_dashboard_{arm}.png"))
        except Exception as exc:                     # noqa: BLE001
            print(f"  ! plotting {arm} failed: {type(exc).__name__}: {exc}")
    with open(os.path.join(bag_dir, METRICS_NAME), "w") as fh:
        json.dump(out, fh, indent=2)

    prim = out["right"].get("primary_active_arm")
    print(f"  active-arm source: {src} | primary active: {prim}")
    for arm in sm.ARMS:
        mm = out[arm]
        print(f"  [{arm}] active={mm.get('this_arm_active_frac')} "
              f"path={mm.get('ee_path_len_m')}m speed_mean={mm.get('ee_speed_mean_mps')} "
              f"sparc={mm.get('ee_sparc')} qdot_cmd_peak={mm.get('qdot_cmd_max')} "
              f"min_dist={mm.get('safety_min_dist_m')}")
    print(f"  -> wrote plot_dashboard_(right|left).png, "
          f"metrics_summary_(right|left).txt, {METRICS_NAME}")
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
