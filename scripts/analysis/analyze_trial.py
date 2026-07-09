#!/usr/bin/env python3
"""Analyze recorded study trial(s): per-hand, per-side PNG dashboards + metrics.

The QP-CLF-CBF controller decouples the two arms and only one is teleoperated at
a time, so each trial is analyzed SEPARATELY per arm AND per side of the loop.
For each trial folder (a rosbag directory recorded by ``study_recorder.py``) it
writes, IN THAT FOLDER, FOUR figures:

    plot_triago_right.png    robot-side telemetry, RIGHT arm
    plot_triago_left.png     robot-side telemetry, LEFT  arm
    plot_haption_right.png   device-side telemetry (single Haption handle; the
                             per-arm split is redundant, so only this one)

plus the numeric summaries (unchanged):

    metrics_summary_right.txt / metrics_summary_left.txt
    metrics.json  ({metadata, right:{...}, left:{...}})

The Haption figure is CASE-ADAPTIVE from the trial's metadata flags
(control_mode / assist_feedback / assist_blending):
  * device force + torque (virtuose/force_cmd, 3 components each, with the
    hardcoded ±limit dashed) are ALWAYS shown -- the handle always renders at
    least the F_sync tether / centering spring,
  * authority alpha AND the blended-action share (user vs policy) shown only
    when ASSIST_BLENDING is True,
  * handle xyz + a light-blue clutch-engaged band shown in CLUTCH mode,
  * handle displacement-from-home (the velocity command) shown in JOYSTICK mode.

EE speed is DIFFERENTIATED from the published EE position (the /qp_debug/ee_real
velocity slots read ~0 -- see study_metrics.ee_speed_series).

Usage (in a sourced ROS 2 environment -- bag reading needs rosbag2):
  ros2 run triago_control analyze_trial.py                 # all trials in DATA_ROOT
  ros2 run triago_control analyze_trial.py <trial_folder>  # one trial
  ros2 run triago_control analyze_trial.py <participant_or_root_dir>
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

# Per-arm colour convention (matches offline_plotter.py / plotter.py).
ARM_COLOR = {"right": "#d62728", "left": "#1f77b4"}   # red / blue
JOINT_COLORS = plt.cm.jet(np.linspace(0, 1, 7))
XYZ_COLORS = ("#d62728", "#2ca02c", "#1f77b4")        # x, y, z
# Hardcoded Haption device wrench safety clips (the force managers' final
# MAX_FORCE / MAX_TORQUE clip). Drawn as dashed reference lines on the wrench plots.
MAX_DEVICE_FORCE = 10.0     # N
MAX_DEVICE_TORQUE = 1.0     # Nm


def _apply_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.titlesize": 14,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "axes.linewidth": 0.8,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


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
def _no_data(ax, title=""):
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, color="#999")
    if title:
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
    if t_act is None:
        return
    is_this = right_active if arm == "right" else ~right_active
    _shade_true(ax, t_act, is_this, "#ffe0b3", f"{arm} arm active")


def _zoh_cols(t_target, t_src, M):
    """Zero-order-hold resample of an (N, k) step signal onto t_target."""
    if t_src is None or len(t_src) == 0:
        return None
    idx = np.searchsorted(np.asarray(t_src), np.asarray(t_target), side="right") - 1
    idx = np.clip(idx, 0, len(M) - 1)
    return np.asarray(M)[idx]


def _legend(ax, **kw):
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", **{"fontsize": 7, **kw})


def _active_mask_on(t_target, t_act, right_active, arm):
    """Boolean mask on t_target that is True while THIS arm is the active one.
    Returns None if no active-arm signal is available."""
    if t_act is None or right_active is None:
        return None
    is_this = right_active if arm == "right" else ~right_active
    m = _zoh_cols(t_target, t_act, np.asarray(is_this).reshape(-1, 1))
    if m is None:
        return None
    return m[:, 0].astype(bool)


# ---------------------------------------------------------------- TRIAGO (robot) dashboard
def make_triago_dashboard(series, metrics, metadata, arm, t_act, right_active, out_path):
    """Robot-side telemetry for ONE arm, 3x3 (offline_plotter style)."""
    col = ARM_COLOR[arm]
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)
    fig.suptitle(_title(metadata, arm, "TRIAGo (robot)"), fontsize=14, fontweight="bold")

    ee = series.get(sm.T_EE)
    P = sm._stack(ee, sm._EE_IDX[arm]["pos"]) if ee else None

    # (0,0) EE path -- top view (x-y)
    ax = fig.add_subplot(gs[0, 0])
    if P is not None and len(P) > 1:
        ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.2)
        ax.plot(P[0, 0], P[0, 1], "o", color="green", ms=6, label="start")
        ax.plot(P[-1, 0], P[-1, 1], "X", color="black", ms=7, label="end")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("EE path (top)")
        _legend(ax)
    else:
        _no_data(ax, "EE path (top)")

    # (0,1) EE position x/y/z vs time (+ active shading)
    ax = fig.add_subplot(gs[0, 1])
    if P is not None and len(P) > 1:
        for k, lab in enumerate(("x", "y", "z")):
            ax.plot(ee.t, P[:, k], lw=1.0, color=XYZ_COLORS[k], label=lab)
        _shade_active(ax, t_act, right_active, arm)
        ax.set_xlabel("t [s]"); ax.set_ylabel("pos [m]"); ax.set_title("EE position")
        _legend(ax, ncol=3)
    else:
        _no_data(ax, "EE position")

    # (0,2) EE speed (differentiated) + active shading
    ax = fig.add_subplot(gs[0, 2])
    ts, speed = sm.ee_speed_series(ee, arm) if ee else (None, None)
    if speed is not None and speed.size:
        ax.plot(ts, speed, color=col, lw=1.0)
        _shade_active(ax, t_act, right_active, arm)
        ax.set_xlabel("t [s]"); ax.set_ylabel("speed [m/s]")
        ax.set_title(f"EE speed   (SPARC {metrics.get('ee_sparc')})")
        _legend(ax)
    else:
        _no_data(ax, "EE speed")

    # (1,0) Cartesian position tracking error ||ref - real||, ACTIVE-ARM ONLY.
    # When an arm is inactive the controller freezes it internally at its
    # current EE (see main_qp_controller._freeze_arm) but the PUBLISHED
    # reference goes stale, so ||stale_ref - ee|| there is meaningless (and is
    # what produced the huge inactive error + the handover jump). Mask it to the
    # spans where this arm is actually the teleoperated one.
    ax = fig.add_subplot(gs[1, 0])
    # Use the truthful effective reference (/qp_debug/reference_effective); fall
    # back to /arm_*/cartesian_reference for bags recorded before it existed.
    ref = series.get(sm.T_REF_EFF)
    Rp = sm._stack(ref, sm._REF_EFF_POS_IDX[arm]) if ref else None
    if Rp is None:
        ref = series.get(sm.T_REF[arm])
        Rp = sm._stack(ref, range(0, 3)) if ref else None
    if P is not None and Rp is not None and len(P) > 1:
        ref_z = _zoh_cols(ee.t, ref.t, Rp)
        err = np.linalg.norm(ref_z - P, axis=1)
        amask = _active_mask_on(ee.t, t_act, right_active, arm)
        if amask is not None:
            err = np.where(amask, err, np.nan)
        ax.plot(ee.t, err, color=col, lw=1.0)
        _shade_active(ax, t_act, right_active, arm)
        _shade_grasp(ax, series)
        ax.set_xlabel("t [s]"); ax.set_ylabel("error [m]")
        ax.set_title("Cartesian tracking error (active arm)")
        _legend(ax)
    else:
        _no_data(ax, "Cartesian tracking error")

    # (1,1) Measured joint velocity (7 joints)
    ax = fig.add_subplot(gs[1, 1])
    qm = sm._joint_block(series.get(sm.T_QDOT_MEAS), arm)
    tqm = series.get(sm.T_QDOT_MEAS).t if series.get(sm.T_QDOT_MEAS) else None
    if qm is not None and len(qm):
        for j in range(qm.shape[1]):
            ax.plot(tqm, qm[:, j], lw=0.8, color=JOINT_COLORS[j], label=f"J{j+1}")
        ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\dot{q}$ [rad/s]")
        ax.set_title("Joint velocity (measured)")
        ax.legend(loc="best", fontsize=6, ncol=2)
    else:
        _no_data(ax, "Joint velocity (measured)")

    # (1,2) QP solution joint velocity (7 joints)
    ax = fig.add_subplot(gs[1, 2])
    qc = sm._joint_block(series.get(sm.T_QDOT_CMD), arm)
    tqc = series.get(sm.T_QDOT_CMD).t if series.get(sm.T_QDOT_CMD) else None
    if qc is not None and len(qc):
        for j in range(qc.shape[1]):
            ax.plot(tqc, qc[:, j], lw=0.8, color=JOINT_COLORS[j], label=f"J{j+1}")
        ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\dot{q}$ [rad/s]")
        ax.set_title(r"QP solution (commanded $\dot{q}$)")
        ax.legend(loc="best", fontsize=6, ncol=2)
    else:
        _no_data(ax, "QP solution")

    # (2,0) CLF slack (this arm) -- ALONE (never mixed with CBF)
    ax = fig.add_subplot(gs[2, 0])
    sl = series.get(sm.T_SLACKS)
    scol = sl.col(f"d{sm._SLACK_IDX[arm]}") if sl else None
    if scol is not None and len(scol):
        ax.plot(sl.t, scol, color="#d62728", lw=1.0)
        ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\delta$")
        ax.set_title("CLF slack")
    else:
        _no_data(ax, "CLF slack")

    # (2,1) Barrier shadow prices (this arm): CBF + joint-limit
    ax = fig.add_subplot(gs[2, 1])
    lam = series.get(sm.T_LAMBDA_CBF)
    lcol = lam.col(f"d{sm._LAMBDA_IDX[arm]}") if lam else None
    lj = series.get("/qp_debug/lambda_joints")
    ljcol = lj.col(f"d{sm._LAMBDA_IDX[arm]}") if lj else None
    plotted = False
    if lcol is not None and len(lcol):
        ax.plot(lam.t, lcol, color="#2ca02c", lw=1.0, label=r"$\lambda_{CBF}$")
        plotted = True
    if ljcol is not None and len(ljcol):
        ax.plot(lj.t, ljcol, color="#9467bd", lw=1.0, label=r"$\lambda_{joints}$")
        plotted = True
    if plotted:
        ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\lambda$")
        ax.set_title("Barrier shadow prices (CBF / joint-limit)")
        _legend(ax)
    else:
        _no_data(ax, "Barrier shadow prices")

    # (2,2) Obstacle clearance. /qp_debug/min_distance is the true closest-pair
    # distance ONLY while a pair is within cfg.DISTANCE_FILTER_THRESHOLD; when
    # nothing is that close the controller publishes a 1.0 sentinel (and the CBF
    # margin ~ 1 - d_safe) -- that sentinel is exactly what makes the raw trace
    # jump to 1. Mask samples at/above the threshold so the trace shows only
    # genuine proximity, with no discontinuous jump.
    ax = fig.add_subplot(gs[2, 2])
    md = series.get(sm.T_MINDIST)
    thr = float((metadata.get("cfg_snapshot", {}) or {}).get(
        "DISTANCE_FILTER_THRESHOLD", 0.15))
    if md and md.col("value") is not None and len(md):
        d = np.asarray(md.col("value"), dtype=float)
        d = np.where(d >= thr, np.nan, d)
        drew = bool(np.any(np.isfinite(d)))
        if drew:
            ax.plot(md.t, d, color="#2ca02c", lw=1.1, label="min pair distance")
        sf = series.get(sm.T_SAFETY)
        if sf and sf.col("value") is not None:
            m = np.asarray(sf.col("value"), dtype=float)
            m = np.where(m >= thr, np.nan, m)
            if np.any(np.isfinite(m)):
                ax.plot(sf.t, m, color="#888", lw=0.8, alpha=0.8, label="CBF margin")
                drew = True
        ax.axhline(0.0, color="#333", ls=":", lw=0.8)   # 0 = contact
        _shade_grasp(ax, series)
        ax.set_xlabel("t [s]"); ax.set_ylabel("distance [m]")
        ax.set_title(f"Obstacle clearance (< {thr:g} m)")
        if drew:
            _legend(ax)
        else:
            ax.text(0.5, 0.5, f"no pair within {thr:g} m", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
    else:
        _no_data(ax, "Obstacle clearance")

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------- HAPTION (device) dashboard
def make_haption_dashboard(series, metrics, metadata, arm, t_act, right_active, out_path):
    """Device-side telemetry, CASE-ADAPTIVE. Rows are built per the trial's
    control_mode / assist_feedback / assist_blending flags."""
    mode = (metadata.get("control_mode") or "").upper()
    fb = bool(metadata.get("assist_feedback"))
    bl = bool(metadata.get("assist_blending"))
    is_clutch = mode == "CLUTCH"
    is_joystick = mode == "JOYSTICK"
    snap = metadata.get("cfg_snapshot", {}) or {}

    # Build the row plan (label -> builder key), adaptive to the cell.
    rows = ["handle_pos", "handle_speed", "buttons"]
    if is_joystick:
        rows.append("joystick_disp")
    # The device wrench (/virtuose/force_cmd) is ALWAYS rendered -- at minimum the
    # F_sync tether (clutch) or centering spring (joystick) -- so always plot the
    # raw force + torque (3 components each) with the hardcoded ±limit dashed.
    rows += ["force", "torque"]
    if bl:
        rows += ["alpha", "blend_share"]

    n = len(rows)
    fig, axs = plt.subplots(n, 1, figsize=(13, 2.7 * n), squeeze=False)
    axs = axs[:, 0]
    fig.suptitle(_title(metadata, arm, "Haption (device)"), fontsize=14, fontweight="bold")

    pose = series.get(sm.T_VIRT_POSE)
    clutch = series.get(sm.T_CLUTCH)

    def _clutch_band(ax):
        if is_clutch and clutch and clutch.col("value") is not None and len(clutch):
            _shade_true(ax, clutch.t, np.asarray(clutch.col("value")) > 0.5,
                        "#add8e6", "clutch engaged")

    for ax, key in zip(axs, rows):
        if key == "handle_pos":
            if pose and pose.col("px") is not None and len(pose):
                for c, lab in zip(("px", "py", "pz"), ("x", "y", "z")):
                    ax.plot(pose.t, pose.col(c), lw=1.0,
                            color=XYZ_COLORS[("x", "y", "z").index(lab)], label=lab)
                _shade_active(ax, t_act, right_active, arm)
                _clutch_band(ax)
                ax.set_ylabel("pos [m]"); ax.set_title("Handle position (Haption base frame)")
                _legend(ax, ncol=3)
            else:
                _no_data(ax, "Handle position")

        elif key == "handle_speed":
            vel = series.get(sm.T_VIRT_VEL)
            if vel and vel.col("lx") is not None and len(vel):
                lin = np.linalg.norm(sm._stack_named(vel, ("lx", "ly", "lz")), axis=1)
                ang = np.linalg.norm(sm._stack_named(vel, ("ax", "ay", "az")), axis=1)
                ax.plot(vel.t, lin, color="#1f77b4", lw=1.0, label="|v| lin [m/s]")
                ax.plot(vel.t, ang, color="#ff7f0e", lw=1.0, label="|ω| ang [rad/s]")
                _clutch_band(ax)
                ax.set_ylabel("speed"); ax.set_title("Handle speed")
                _legend(ax)
            else:
                _no_data(ax, "Handle speed")

        elif key == "buttons":
            drew = False
            if clutch and clutch.col("value") is not None and len(clutch):
                ax.plot(clutch.t, np.asarray(clutch.col("value")) > 0.5,
                        drawstyle="steps-post", color="#add8e6", lw=1.4,
                        label="clutch (right btn)")
                drew = True
            trig = series.get(sm.T_TRIGGER)
            if trig and trig.col("value") is not None and len(trig):
                ax.plot(trig.t, (np.asarray(trig.col("value")) > 0.5).astype(float) * 0.9,
                        drawstyle="steps-post", color="#8c564b", lw=1.2,
                        label="grasp trigger (left btn)")
                drew = True
            if drew:
                ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 1])
                ax.set_ylabel("pressed"); ax.set_title("Buttons")
                _legend(ax, ncol=2)
            else:
                _no_data(ax, "Buttons")

        elif key == "joystick_disp":
            # The joystick COMMAND: handle displacement from the (dynamic) home.
            home = series.get(sm.T_HOME_POSE)
            if pose and pose.col("px") is not None and home and home.col("d0") is not None:
                Ph = sm._stack_named(pose, ("px", "py", "pz"))
                Hh = _zoh_cols(pose.t, home.t, sm._stack(home, range(0, 3)))
                lin_disp = np.linalg.norm(Ph - Hh, axis=1)
                ax.plot(pose.t, lin_disp, color="#1f77b4", lw=1.1, label="||pos - home|| [m]")
                # angular gap from quaternions (2*acos|dot|)
                Q = sm._stack_named(pose, ("qx", "qy", "qz", "qw"))
                Hq = _zoh_cols(pose.t, home.t, sm._stack(home, range(3, 7)))
                if Q is not None and Hq is not None:
                    dot = np.abs(np.sum(Q * Hq, axis=1))
                    ang = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
                    ax.plot(pose.t, ang, color="#9b59b6", lw=1.1, label="ang gap [rad]")
                db = snap.get("JOYSTICK_DEADBAND_LIN")
                if db is not None:
                    ax.axhline(float(db), color="#1f77b4", ls=":", lw=1.0, alpha=0.7,
                               label=f"lin deadband {db} m")
                dba = snap.get("JOYSTICK_DEADBAND_ANG")
                if dba is not None:
                    ax.axhline(float(dba), color="#9b59b6", ls=":", lw=1.0, alpha=0.7,
                               label=f"ang deadband {dba:.3f} rad")
                ax.set_ylabel("displacement")
                ax.set_title("Joystick command: handle displacement from home (→ commanded twist)")
                _legend(ax, ncol=2)
            else:
                _no_data(ax, "Joystick displacement from home")

        elif key == "force":
            fr = series.get(sm.T_FORCE)
            F = sm._stack_named(fr, ("fx", "fy", "fz")) if fr else None
            if F is not None and len(F):
                for k, lab in enumerate(("Fx", "Fy", "Fz")):
                    ax.plot(fr.t, F[:, k], lw=1.0, color=XYZ_COLORS[k], label=lab)
                ax.axhline(MAX_DEVICE_FORCE, color="#333", ls="--", lw=1.0, alpha=0.7,
                           label=f"\u00b1{MAX_DEVICE_FORCE:g} N limit")
                ax.axhline(-MAX_DEVICE_FORCE, color="#333", ls="--", lw=1.0, alpha=0.7)
                _clutch_band(ax)
                ax.set_ylabel("force [N]")
                ax.set_title("Device force on the handle (virtuose/force_cmd)")
                _legend(ax, ncol=4)
            else:
                _no_data(ax, "Device force")

        elif key == "torque":
            fr = series.get(sm.T_FORCE)
            Tq = sm._stack_named(fr, ("tx", "ty", "tz")) if fr else None
            if Tq is not None and len(Tq):
                for k, lab in enumerate(("Tx", "Ty", "Tz")):
                    ax.plot(fr.t, Tq[:, k], lw=1.0, color=XYZ_COLORS[k], label=lab)
                ax.axhline(MAX_DEVICE_TORQUE, color="#333", ls="--", lw=1.0, alpha=0.7,
                           label=f"\u00b1{MAX_DEVICE_TORQUE:g} Nm limit")
                ax.axhline(-MAX_DEVICE_TORQUE, color="#333", ls="--", lw=1.0, alpha=0.7)
                _clutch_band(ax)
                ax.set_ylabel("torque [Nm]")
                ax.set_title("Device torque on the handle (virtuose/force_cmd)")
                _legend(ax, ncol=4)
            else:
                _no_data(ax, "Device torque")

        elif key == "blend_share":
            # Share of the final blended ACTION contributed by user vs policy:
            # v_blend = (1-alpha)*v_user + alpha*v_policy, so the weighted
            # contributions are ||(1-alpha)*v_user|| and ||alpha*v_policy||.
            bd = series.get(sm.T_BLEND)
            a = (np.asarray(bd.col("d0"), dtype=float)
                 if (bd and bd.col("d0") is not None) else None)
            vu = sm._stack(bd, range(1, 7)) if bd else None
            vp = sm._stack(bd, range(7, 13)) if bd else None
            if a is not None and vu is not None and vp is not None and len(bd):
                u = (1.0 - a) * np.linalg.norm(vu, axis=1)
                p = a * np.linalg.norm(vp, axis=1)
                tot = u + p
                good = tot > 1e-9
                u_pct = np.full_like(a, np.nan)
                p_pct = np.full_like(a, np.nan)
                u_pct[good] = 100.0 * u[good] / tot[good]
                p_pct[good] = 100.0 * p[good] / tot[good]
                ax.plot(bd.t, u_pct, color="#1f77b4", lw=1.2, label="user %")
                ax.plot(bd.t, p_pct, color="#ff7f0e", lw=1.2, label="policy %")
                ax.set_ylim(-5, 105); ax.set_ylabel("share [%]")
                ax.set_title("Blended-action share: (1-\u03b1)\u00b7v_user  vs  \u03b1\u00b7v_policy")
                _legend(ax, ncol=2)
            else:
                _no_data(ax, "Blended-action share")

        elif key == "alpha":
            bd = series.get(sm.T_BLEND)
            if bd and bd.col("d0") is not None and len(bd):
                ax.plot(bd.t, bd.col("d0"), color="#ff7f0e", lw=1.2, label="authority α")
                ax.axhline(0.5, color="#888", ls=":", lw=0.8)
                ax.set_ylim(-0.05, 1.05); ax.set_ylabel("α")
                ax.set_title("Blending authority (0 = user, 1 = policy)")
                _legend(ax)
            else:
                _no_data(ax, "Blending authority")

        ax.set_xlabel("t [s]")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def _title(metadata, arm, side):
    return (f"{metadata.get('participant', '?')} | "
            f"{metadata.get('condition', metadata.get('cell_code', '?'))} | "
            f"{metadata.get('world_shortcut', '?')} | "
            f"success={metadata.get('success', '?')}  —  {side} · {arm.upper()} arm")


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
            make_triago_dashboard(series, metrics, metadata, arm, t_act, right_active,
                                  os.path.join(bag_dir, f"plot_triago_{arm}.png"))
        except Exception as exc:                     # noqa: BLE001
            print(f"  ! triago {arm} plot failed: {type(exc).__name__}: {exc}")

    # The Haption device is single (one operator, one handle), so a per-arm
    # split is redundant -- emit ONE device figure keyed to the right arm.
    try:
        make_haption_dashboard(series, out["right"], metadata, "right", t_act,
                               right_active, os.path.join(bag_dir, "plot_haption_right.png"))
    except Exception as exc:                         # noqa: BLE001
        print(f"  ! haption plot failed: {type(exc).__name__}: {exc}")

    with open(os.path.join(bag_dir, METRICS_NAME), "w") as fh:
        json.dump(out, fh, indent=2)

    prim = out["right"].get("primary_active_arm")
    print(f"  active-arm source: {src} | primary active: {prim}")
    for arm in sm.ARMS:
        mm = out[arm]
        print(f"  [{arm}] active={mm.get('this_arm_active_frac')} "
              f"path={mm.get('ee_path_len_m')}m speed_mean={mm.get('ee_speed_mean_mps')} "
              f"sparc={mm.get('ee_sparc')} min_dist={mm.get('safety_min_dist_m')}")
    print(f"  -> wrote plot_triago_(right|left).png, plot_haption_right.png, "
          f"metrics_summary_(right|left).txt, {METRICS_NAME}")
    return True


def main(argv=None):
    _apply_style()
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
