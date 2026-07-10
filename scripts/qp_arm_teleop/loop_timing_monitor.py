#!/usr/bin/env python3
"""
loop_timing_monitor.py -- Real-time diagnostic for main_qp_controller's control
loop timing (subscribes to /qp_debug/loop_timing, published every tick as
[tick_dt_ms, solve_ms]).

tick_dt_ms is wall-clock time since the PREVIOUS tick STARTED: it captures
everything between "the timer should have fired" and "solve_and_publish
actually got to run" -- i.e. OS/container scheduling jitter, on top of the
solve itself. solve_ms is just the QP build+solve compute time. Comparing the
two separates "the OS isn't scheduling us on time" (tick_dt spikes, solve_ms
stays flat) from "the solve itself is too slow for this CPU" (both rise
together) -- the two have different fixes (RT scheduling/CPU isolation vs.
lowering the control frequency and retuning).

Prints a rolling summary every REPORT_INTERVAL_S seconds, and a final full-run
summary (exact min/max/mean/overrun-count, plus percentile estimates from a
bounded rolling window) on Ctrl-C.

For an OS-level cross-check independent of this node entirely, run
`cyclictest` (rt-tests package) in the same container:
    sudo cyclictest -p 80 -i 1000 -d 0 -m -q -l 20000

Usage:
    ros2 run triago_control loop_timing_monitor.py
    ros2 run triago_control loop_timing_monitor.py --ros-args -p overrun_factor:=1.5
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import triago_control.qp_controller.config as cfg

REPORT_INTERVAL_S = 2.0
PERCENTILE_WINDOW = 100_000   # ~5.5 min at 300 Hz -- bounded memory for p50/p95/p99


class _Stream:
    """Exact running min/max/mean/overrun-count over an unbounded stream, plus
    a bounded deque for percentile estimates (memory-safe on a long-running trial)."""

    def __init__(self, maxlen=PERCENTILE_WINDOW):
        self.count = 0
        self.total = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.overruns = 0
        self._window = deque(maxlen=maxlen)

    def add(self, value, overrun_threshold=None):
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if overrun_threshold is not None and value > overrun_threshold:
            self.overruns += 1
        self._window.append(value)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    def percentiles(self, ps=(50, 95, 99)):
        if not self._window:
            return {p: 0.0 for p in ps}
        arr = np.fromiter(self._window, dtype=float)
        return {p: float(np.percentile(arr, p)) for p in ps}


class LoopTimingMonitor(Node):

    def __init__(self):
        super().__init__('loop_timing_monitor')
        self.declare_parameter('overrun_factor', 1.5)
        overrun_factor = self.get_parameter('overrun_factor').value
        nominal_ms = 1000.0 / cfg.CONTROL_FREQ_DEFAULT
        self.overrun_threshold_ms = overrun_factor * nominal_ms

        self.tick = _Stream()
        self.solve = _Stream()

        self.create_subscription(
            Float64MultiArray, '/qp_debug/loop_timing', self._cb, 10)
        self.create_timer(REPORT_INTERVAL_S, self._report)

        self.get_logger().info(
            f"Listening on /qp_debug/loop_timing. Nominal tick = {nominal_ms:.2f}ms "
            f"(CONTROL_FREQ_DEFAULT={cfg.CONTROL_FREQ_DEFAULT:.0f}Hz); flagging "
            f"overruns > {self.overrun_threshold_ms:.2f}ms ({overrun_factor:.1f}x nominal).")

    def _cb(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            return
        tick_dt_ms, solve_ms = msg.data[0], msg.data[1]
        if tick_dt_ms > 0.0:   # the very first sample after startup has no valid dt
            self.tick.add(tick_dt_ms, overrun_threshold=self.overrun_threshold_ms)
        self.solve.add(solve_ms)

    def _report(self):
        if self.tick.count == 0:
            self.get_logger().info("Waiting for data on /qp_debug/loop_timing...")
            return
        tp = self.tick.percentiles()
        sp = self.solve.percentiles()
        overrun_pct = 100.0 * self.tick.overruns / self.tick.count
        self.get_logger().info(
            f"[TICK dt ms] n={self.tick.count} mean={self.tick.mean:.3f} "
            f"min={self.tick.min:.3f} p50={tp[50]:.3f} p95={tp[95]:.3f} "
            f"p99={tp[99]:.3f} max={self.tick.max:.3f} | "
            f"overruns(>{self.overrun_threshold_ms:.2f}ms)={self.tick.overruns} "
            f"({overrun_pct:.2f}%)")
        self.get_logger().info(
            f"[SOLVE ms]   n={self.solve.count} mean={self.solve.mean:.3f} "
            f"min={self.solve.min:.3f} p50={sp[50]:.3f} p95={sp[95]:.3f} "
            f"p99={sp[99]:.3f} max={self.solve.max:.3f}")

    def final_report(self):
        if self.tick.count == 0:
            print("[loop_timing_monitor] No data received -- nothing to report.")
            return
        tp = self.tick.percentiles()
        sp = self.solve.percentiles()
        overrun_pct = 100.0 * self.tick.overruns / self.tick.count
        nominal_ms = 1000.0 / cfg.CONTROL_FREQ_DEFAULT
        print("\n" + "=" * 70)
        print("[loop_timing_monitor] FINAL SUMMARY")
        print("=" * 70)
        print(f"Samples: {self.tick.count}")
        print(f"Nominal tick: {nominal_ms:.3f} ms ({cfg.CONTROL_FREQ_DEFAULT:.0f} Hz)")
        print("\nTICK dt (scheduling + full solve_and_publish, ms):")
        print(f"  mean={self.tick.mean:.3f}  min={self.tick.min:.3f}  "
              f"p50={tp[50]:.3f}  p95={tp[95]:.3f}  p99={tp[99]:.3f}  "
              f"max={self.tick.max:.3f}")
        print(f"  overruns (> {self.overrun_threshold_ms:.2f}ms): "
              f"{self.tick.overruns} ({overrun_pct:.2f}%)")
        print("\nSOLVE time (QP build+solve only, ms):")
        print(f"  mean={self.solve.mean:.3f}  min={self.solve.min:.3f}  "
              f"p50={sp[50]:.3f}  p95={sp[95]:.3f}  p99={sp[99]:.3f}  "
              f"max={self.solve.max:.3f}")
        print("\nInterpretation:")
        if self.solve.max > nominal_ms:
            print("  - SOLVE time alone exceeds the nominal tick budget at times:")
            print("    the compute itself doesn't fit in one tick -> likely a real")
            print("    compute-budget problem, independent of OS scheduling.")
        if overrun_pct > 1.0:
            print(f"  - {overrun_pct:.1f}% of ticks overran. If SOLVE time stayed")
            print("    small while TICK dt spiked, that points at OS/container")
            print("    scheduling jitter (missed deadlines), not raw compute load.")
            print("    Cross-check with `cyclictest` in the same container.")
        else:
            print("  - Overrun rate is low: this run shows no evidence of the loop")
            print("    missing its deadline.")
        print("=" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = LoopTimingMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.final_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
