#!/usr/bin/env python3
"""
head_estimation_report.py -- ONE-SHOT estimation-performance report for main_head.py.

Where head_debug_plotter.py is a LIVE spatial view ("what does the detector see
right now"), this is a TEMPORAL performance report: it silently buffers the
whole session's per-object estimate + head motion, and on Ctrl-C renders ONE
figure answering "how well/steadily did the estimate converge, and did head
MOTION corrupt it?".

It subscribes only (no main_head.py change needed):
    cfg.DEBUG_JSON_TOPIC        per-object height/radius/vertical_coverage/
                                confidence/fit_rms + plane_z + convergence
                                (published only with real_hardware_head:=true)
    /head_perception/telemetry  index 14 = head joint-velocity norm

Figure (saved to /tmp/head_estimation_report.png, shown if a display exists):
    1. HEIGHT estimate per object over time (+ optional GT reference line, see
       below). With two EQUAL cylinders the two traces should coincide -- their
       spread is a direct, ground-truth-free repeatability measure.
    2. RADIUS estimate per object over time (+ optional GT line).
    3. vertical_coverage per object (left axis) vs head-velocity norm (right
       axis) + the INTEGRATE_VEL_THRESH gate line -- THE panel for "estimation
       is bad during head movement": watch whether the estimate jumps whenever
       head_vel crosses the gate.
    4. plane_z + convergence progress + drift rate over time.

Ground truth is NEVER assumed from config.py's world-specific GT_* constants
(those describe one specific sim world, not whatever is actually on the table
this run) -- pass the ACTUAL measured dimensions of this session's cylinders
explicitly, or omit them for a GT-free repeatability-only report:
    ros2 run triago_control head_estimation_report.py --ros-args \\
        -p gt_height_m:=0.115 -p gt_radius_m:=0.04

Run on a machine with a display (subscribes over the network to main_head.py):
    ros2 run triago_control head_estimation_report.py
    # ... let the head do its Phase 1 -> Phase 2 run, then Ctrl-C to render.
"""

import json
import time

import numpy as np
import matplotlib
_HAS_DISPLAY = True
try:
    matplotlib.use("TkAgg")
except ImportError:
    _HAS_DISPLAY = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray

import triago_control.head_control.config as cfg

_OUT_PNG = "/tmp/head_estimation_report.png"
_TELE_HEAD_VEL_IDX = 14        # /head_perception/telemetry layout (main_head.py)


class HeadEstimationReport(Node):
    def __init__(self):
        super().__init__("head_estimation_report")
        # Ground truth is whatever cylinders are ACTUALLY on the table this run --
        # never assumed from cfg.GT_RED_HEIGHT/etc, which describe one specific
        # sim world, not this session's objects. 0.0 (default) means "unknown",
        # so no reference line/error is drawn -- pass the real measured values:
        #   -p gt_height_m:=0.115 -p gt_radius_m:=0.04
        self.declare_parameter("gt_height_m", 0.0)
        self.declare_parameter("gt_radius_m", 0.0)
        self.gt_height = float(self.get_parameter("gt_height_m").value) or None
        self.gt_radius = float(self.get_parameter("gt_radius_m").value) or None

        self._t0 = None
        self.objs = {}            # id -> {label, t, height, radius, vcov, conf, fit_rms}
        self.t_plane, self.plane_z = [], []
        self.t_conv, self.converge, self.drift = [], [], []
        self.t_vel, self.head_vel = [], []

        self.create_subscription(String, cfg.DEBUG_JSON_TOPIC, self._debug_cb, 10)
        self.create_subscription(
            Float64MultiArray, "/head_perception/telemetry", self._tele_cb, 10)
        self.get_logger().info(
            "head_estimation_report buffering -- run the head's estimation, then "
            "Ctrl-C to render the report (needs main_head.py with "
            "real_hardware_head:=true so it publishes the debug JSON).")

    def _elapsed(self):
        # NOT named _clock: rclpy's Node base stores its ROSClock in self._clock,
        # which would shadow a method of that name (calling it -> TypeError).
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    def _debug_cb(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        t = self._elapsed()
        pz = d.get("plane_z")
        if pz is not None:
            self.t_plane.append(t)
            self.plane_z.append(float(pz))
        cp = d.get("converge_progress")
        if cp is not None:
            self.t_conv.append(t)
            self.converge.append(float(cp))
            dr = d.get("drift_rate")
            self.drift.append(float(dr) if dr is not None else np.nan)
        for o in d.get("objects", []):
            oid = o.get("id")
            rec = self.objs.setdefault(oid, {
                "label": o.get("label", str(oid)),
                "t": [], "height": [], "radius": [], "vcov": [], "conf": [], "fit_rms": []})
            rec["t"].append(t)
            rec["height"].append(float(o.get("height", np.nan)))
            rec["radius"].append(float(o.get("radius", np.nan)))
            rec["vcov"].append(float(o.get("vertical_coverage", np.nan)))
            rec["conf"].append(float(o.get("confidence", np.nan)))
            rec["fit_rms"].append(float(o.get("fit_rms", np.nan)))

    def _tele_cb(self, msg):
        if len(msg.data) <= _TELE_HEAD_VEL_IDX:
            return
        self.t_vel.append(self._elapsed())
        self.head_vel.append(float(msg.data[_TELE_HEAD_VEL_IDX]))

    # ------------------------------------------------------------------ #
    # One-shot render                                                    #
    # ------------------------------------------------------------------ #
    def _gt_values(self, kind):
        """The user-supplied ground truth for THIS session's actual cylinders
        (gt_height_m/gt_radius_m params), or [] if not supplied -- never a
        world-specific constant assumed from config.py."""
        val = self.gt_height if kind == "height" else self.gt_radius
        return [val] if val is not None else []

    def render(self):
        if not self.objs and not self.t_plane:
            self.get_logger().warn(
                "No estimation data received -- nothing to report. Was main_head.py "
                "running with real_hardware_head:=true (it gates the debug JSON)?")
            return

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 12),
                                num="Head estimation report")
        fig.suptitle("Head Estimation Performance -- session report")
        palette = plt.cm.tab10(np.linspace(0, 1, 10))
        ordered = sorted(self.objs.items())

        # 1. HEIGHT ----------------------------------------------------
        ax = axs[0]
        ax.set_title("Cylinder HEIGHT estimate  (equal cylinders => traces should coincide)")
        ax.set_ylabel("height [m]")
        for i, (oid, rec) in enumerate(ordered):
            ax.plot(rec["t"], rec["height"], color=palette[i % 10], lw=1.4,
                    label=f"{rec['label']} (#{oid})")
        for gv in self._gt_values("height"):
            ax.axhline(gv, ls="--", color="0.35", lw=1.0, alpha=0.8)
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, len(ordered)))

        # 2. RADIUS ----------------------------------------------------
        ax = axs[1]
        ax.set_title("Cylinder RADIUS estimate")
        ax.set_ylabel("radius [m]")
        for i, (oid, rec) in enumerate(ordered):
            ax.plot(rec["t"], rec["radius"], color=palette[i % 10], lw=1.4)
        for gv in self._gt_values("radius"):
            ax.axhline(gv, ls="--", color="0.35", lw=1.0, alpha=0.8)

        # 3. vertical_coverage vs head motion --------------------------
        ax = axs[2]
        ax.set_title("Column coverage vs HEAD MOTION  (does movement corrupt the estimate?)")
        ax.set_ylabel("vertical_coverage"); ax.set_ylim(-0.05, 1.05)
        for i, (oid, rec) in enumerate(ordered):
            ax.plot(rec["t"], rec["vcov"], color=palette[i % 10], lw=1.2)
        ax.axhline(cfg.REAL_VERT_COVERAGE_MIN, ls=":", color="0.5", lw=1.0,
                   label=f"accept floor ({cfg.REAL_VERT_COVERAGE_MIN:.2f})")
        ax.legend(loc="upper left", fontsize=8)
        axv = ax.twinx()
        axv.set_ylabel("head |q̇| [rad/s]", color="#d62728")
        if self.t_vel:
            axv.plot(self.t_vel, self.head_vel, color="#d62728", lw=1.0, alpha=0.8)
        axv.axhline(cfg.INTEGRATE_VEL_THRESH, ls="--", color="#d62728", lw=1.0, alpha=0.7)
        axv.tick_params(axis="y", labelcolor="#d62728")

        # 4. plane + convergence ---------------------------------------
        ax = axs[3]
        ax.set_title("Table plane height, convergence progress, drift")
        ax.set_ylabel("plane z [m]"); ax.set_xlabel("time [s]")
        if self.t_plane:
            ax.plot(self.t_plane, self.plane_z, color="saddlebrown", lw=1.3, label="plane_z")
        ax.legend(loc="upper left", fontsize=8)
        axc = ax.twinx()
        axc.set_ylabel("progress / drift"); axc.set_ylim(-0.05, 1.05)
        if self.t_conv:
            axc.plot(self.t_conv, self.converge, color="#2ca02c", lw=1.1, label="converge_progress")
            axc.plot(self.t_conv, self.drift, color="#9467bd", lw=1.0, alpha=0.8, label="drift [m/s]")
            axc.legend(loc="upper right", fontsize=8, ncol=2)

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(_OUT_PNG, dpi=150)
        self.get_logger().info(f"[report] saved {_OUT_PNG}")
        self._print_summary()
        if _HAS_DISPLAY:
            plt.show()

    def _print_summary(self):
        lines = ["", "=" * 64, " HEAD ESTIMATION REPORT -- final per-object estimate", "-" * 64]
        if self.gt_height is None and self.gt_radius is None:
            lines.append("  (no ground truth supplied -- pass -p gt_height_m:=.. "
                         "-p gt_radius_m:=.. for a signed error readout)")
        gt_h = self._gt_values("height")
        gt_r = self._gt_values("radius")
        finals = []
        for oid, rec in sorted(self.objs.items()):
            h = next((v for v in reversed(rec["height"]) if np.isfinite(v)), float("nan"))
            r = next((v for v in reversed(rec["radius"]) if np.isfinite(v)), float("nan"))
            finals.append(h)
            # SIGNED error (direction, not just magnitude) -- that's what tells
            # you which way to move CYL_TOP_SLICE/CYL_TOP_PERCENTILE_REAL etc.
            gt_txt = f"  | h_err {(h - gt_h[0]) * 100:+.1f} cm" if gt_h else ""
            gt_txt += f"  r_err {(r - gt_r[0]) * 100:+.1f} cm" if gt_r else ""
            lines.append(f"  {rec['label']:>18s} (#{oid}): h={h * 100:5.1f} cm  "
                         f"r={r * 100:4.1f} cm{gt_txt}")
        finite = [v for v in finals if np.isfinite(v)]
        if len(finite) >= 2:
            lines.append("-" * 64)
            lines.append(f"  inter-object HEIGHT spread (should be ~0 for equal cyls): "
                         f"{(max(finite) - min(finite)) * 100:.1f} cm")
        lines.append("=" * 64)
        for ln in lines:
            print(ln, flush=True)


def main():
    rclpy.init()
    node = HeadEstimationReport()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Render ONCE, here on the main thread after the spin loop exits --
        # never mid-callback (matplotlib is not thread-safe).
        node.render()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
