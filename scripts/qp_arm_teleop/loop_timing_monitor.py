#!/usr/bin/env python3
"""
loop_timing_monitor.py -- Real-time diagnostic for main_qp_controller's control
loop timing (subscribes to /qp_debug/loop_timing, published every tick as
[tick_dt_ms, kin_ms, cbf_ms, gov_ms, solve_ms, telemetry_ms, misc_ms]).

tick_dt_ms is wall-clock time since the PREVIOUS tick STARTED -- everything
between "the timer should have fired" and "this tick finished publishing",
i.e. OS/container scheduling jitter PLUS all of solve_and_publish's own compute.
The remaining fields break that total down by phase:
  kin_ms       kinematics/FK refresh (RobotKinematics.update_kinematics + debug
               interrogate + CollisionManager.update_geometry)
  cbf_ms       SoftMin CBF Jacobian aggregation (compute_softmin_jacobian, scales
               with collision-pair count)
  gov_ms       reference governor + CLF task-error computation
  solve_ms     QP build_and_solve only
  telemetry_ms downstream telemetry/command/viz publishing
  misc_ms      whatever this breakdown didn't explicitly bracket (arm-freeze
               checks, attach/detach handling, contact-distance telemetry,
               tracking-error compute) plus any residual OS scheduling wait

If tick_dt is high but every phase above is small AND misc_ms is large, that
points at OS/container scheduling (this process isn't being given the CPU on
time). If one phase dominates instead, that phase IS the compute bottleneck --
a genuine "too much work per tick" problem (fix: optimize that phase, or lower
the control frequency and retune), not a scheduling one.

Prints a rolling summary every REPORT_INTERVAL_S seconds, and a final full-run
summary (exact min/max/mean/overrun-count, plus percentile estimates from a
bounded rolling window) on Ctrl-C.

For an OS-level cross-check independent of this node entirely, run
`cyclictest` (rt-tests package) on the same machine:
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

# Order must match main_qp_controller.py's pub_loop_timing payload.
PHASES = ['tick_dt', 'kin', 'cbf', 'gov', 'solve', 'telemetry', 'misc']
PHASE_LABELS = {
    'tick_dt': 'TICK dt   (full tick, scheduling + everything below)',
    'kin': 'kin_ms    (kinematics/FK + geometry refresh)',
    'cbf': 'cbf_ms    (SoftMin CBF aggregation)',
    'gov': 'gov_ms    (reference governor + task error)',
    'solve': 'solve_ms  (QP build+solve)',
    'telemetry': 'telemetry_ms (downstream publish/viz)',
    'misc': 'misc_ms   (unbracketed code + residual sched wait)',
}


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

        self.streams = {name: _Stream() for name in PHASES}

        self.create_subscription(
            Float64MultiArray, '/qp_debug/loop_timing', self._cb, 10)
        self.create_timer(REPORT_INTERVAL_S, self._report)

        self.get_logger().info(
            f"Listening on /qp_debug/loop_timing. Nominal tick = {nominal_ms:.2f}ms "
            f"(CONTROL_FREQ_DEFAULT={cfg.CONTROL_FREQ_DEFAULT:.0f}Hz); flagging "
            f"tick_dt overruns > {self.overrun_threshold_ms:.2f}ms ({overrun_factor:.1f}x nominal).")

    def _cb(self, msg: Float64MultiArray):
        if len(msg.data) < len(PHASES):
            return
        values = dict(zip(PHASES, msg.data))
        if values['tick_dt'] <= 0.0:
            return   # the very first sample after startup has no valid dt
        self.streams['tick_dt'].add(values['tick_dt'], overrun_threshold=self.overrun_threshold_ms)
        for name in PHASES[1:]:
            self.streams[name].add(values[name])

    def _line(self, name):
        s = self.streams[name]
        p = s.percentiles()
        return (f"{PHASE_LABELS[name]:<45} n={s.count} mean={s.mean:6.3f} "
                f"min={s.min:6.3f} p50={p[50]:6.3f} p95={p[95]:6.3f} "
                f"p99={p[99]:6.3f} max={s.max:6.3f}")

    def _report(self):
        if self.streams['tick_dt'].count == 0:
            self.get_logger().info("Waiting for data on /qp_debug/loop_timing...")
            return
        tick = self.streams['tick_dt']
        overrun_pct = 100.0 * tick.overruns / tick.count
        for name in PHASES:
            self.get_logger().info(self._line(name))
        self.get_logger().info(
            f"overruns(tick_dt>{self.overrun_threshold_ms:.2f}ms)={tick.overruns} ({overrun_pct:.2f}%)")

    def final_report(self):
        tick = self.streams['tick_dt']
        if tick.count == 0:
            print("[loop_timing_monitor] No data received -- nothing to report.")
            return
        overrun_pct = 100.0 * tick.overruns / tick.count
        nominal_ms = 1000.0 / cfg.CONTROL_FREQ_DEFAULT
        print("\n" + "=" * 90)
        print("[loop_timing_monitor] FINAL SUMMARY")
        print("=" * 90)
        print(f"Samples: {tick.count}")
        print(f"Nominal tick: {nominal_ms:.3f} ms ({cfg.CONTROL_FREQ_DEFAULT:.0f} Hz)")
        print()
        for name in PHASES:
            print(self._line(name))
        print(f"\novverruns (tick_dt > {self.overrun_threshold_ms:.2f}ms): "
              f"{tick.overruns} ({overrun_pct:.2f}%)")

        print("\nInterpretation:")
        phase_means = {name: self.streams[name].mean for name in PHASES[1:]}
        worst_phase = max(phase_means, key=phase_means.get)
        worst_mean = phase_means[worst_phase]
        accounted_mean = sum(phase_means.values())
        if worst_phase == 'misc' and worst_mean > 0.5 * tick.mean:
            print(f"  - Most of the tick ({worst_mean:.3f}ms mean, "
                  f"{100*worst_mean/tick.mean:.0f}% of tick_dt) is UNACCOUNTED for by any")
            print("    bracketed phase -- consistent with OS/container scheduling wait")
            print("    (the process is idle, waiting to be scheduled), not our own compute.")
            print("    Cross-check with `cyclictest` and check the process's scheduling")
            print("    class/CPU affinity (chrt -p PID / taskset -pc PID).")
        else:
            print(f"  - Dominant phase: {PHASE_LABELS[worst_phase].split('(')[0].strip()} "
                  f"({worst_mean:.3f}ms mean, {100*worst_mean/tick.mean:.0f}% of tick_dt).")
            print("    This is a genuine compute-budget cost in that phase, not OS")
            print("    scheduling -- optimizing/reducing that phase's work is the fix")
            print("    (or lowering the control frequency + retuning, as a last resort).")
        if accounted_mean < 0.5 * tick.mean and worst_phase != 'misc':
            print(f"  - Note: bracketed phases only account for "
                  f"{100*accounted_mean/tick.mean:.0f}% of tick_dt on average --")
            print("    the rest is misc/unbracketed code or scheduling wait; see misc_ms above.")
        print("=" * 90)


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
