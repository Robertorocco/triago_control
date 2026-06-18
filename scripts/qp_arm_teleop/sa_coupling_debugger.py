#!/usr/bin/env python3
"""
sa_coupling_debugger.py — Policy-QP Coupling Diagnostic Node

Subscribes to both the shared autonomy output and the QP controller telemetry
to expose the instantaneous mismatch that causes oscillation when tracking
policy-generated twists.

Published Topics (all Float64MultiArray unless noted):
    /debug/sa_coupling  [18 floats]:
        [0:3]   e_cursor     : position error between SA virtual cursor and real EE
        [3:6]   v_ref        : the twist (xdot_ref) that SA commanded
        [6:9]   v_achieved   : the actual EE velocity (from ee_real, filtered)
        [9:12]  v_mismatch   : v_ref - v_achieved (the "velocity tracking gap")
        [12:15] cursor_pos   : the SA's virtual cursor position
        [15:18] ee_pos       : the real EE position

    /debug/sa_coupling_scalar  Float64MultiArray [5 floats]:
        [0] |e_cursor|       : cursor-to-EE distance (should be ~const if well-tuned)
        [1] |v_ref|          : commanded speed magnitude
        [2] |v_achieved|     : actual speed magnitude
        [3] |v_mismatch|     : velocity mismatch magnitude (oscillation indicator)
        [4] dt_since_ref     : seconds since last SA reference arrived (staleness)

    /debug/sa_ref_age  Float64:
        Seconds since the last reference message arrived (measures staleness).
        If this is >10ms often, the 100Hz→300Hz mismatch is contributing.

Usage:
    ros2 run triago_control sa_coupling_debugger.py

Then plot with:
    ros2 topic echo /debug/sa_coupling_scalar
    or use rqt_plot on /debug/sa_coupling_scalar

What to look for:
    - |v_mismatch| oscillating at a fixed frequency → servo-level chatter
    - dt_since_ref showing periodic 10ms spikes → frequency mismatch
    - |e_cursor| shrinking near the goal → cursor-collapse causing chatter
    - |v_ref| and |v_achieved| crossing each other → overshoot cycles
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64


class SACouplingDebugger(Node):

    def __init__(self):
        super().__init__('sa_coupling_debugger')

        # State from subscriptions
        self.ee_pos = np.zeros(3)       # real EE position
        self.ee_vel = np.zeros(3)       # real EE velocity (filtered)
        self.cursor_pos = np.zeros(3)   # SA virtual cursor position
        self.v_ref = np.zeros(3)        # SA commanded twist (linear part)
        self.last_ref_time = time.time()
        self.has_ee = False
        self.has_ref = False

        # Subscribers
        self.create_subscription(
            Float64MultiArray, '/qp_debug/ee_real', self.ee_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/arm_right/cartesian_reference', self.ref_cb, 10)

        # Publishers
        self.pub_coupling = self.create_publisher(
            Float64MultiArray, '/debug/sa_coupling', 10)
        self.pub_scalar = self.create_publisher(
            Float64MultiArray, '/debug/sa_coupling_scalar', 10)
        self.pub_age = self.create_publisher(
            Float64, '/debug/sa_ref_age', 10)

        # 300 Hz timer to match the QP rate
        self.create_timer(1.0 / 300.0, self.timer_cb)
        self.last_print = time.time()

        print("=" * 60, flush=True)
        print(" SA-QP COUPLING DEBUGGER", flush=True)
        print("=" * 60, flush=True)
        print("  Subscribes to: /qp_debug/ee_real, /arm_right/cartesian_reference",
              flush=True)
        print("  Publishes to:  /debug/sa_coupling (18), /debug/sa_coupling_scalar (5)",
              flush=True)
        print("                 /debug/sa_ref_age (1)", flush=True)
        print("=" * 60, flush=True)

    def ee_cb(self, msg):
        """Extract position + velocity from ee_real [18 floats]."""
        if len(msg.data) >= 6:
            self.ee_pos = np.array(msg.data[0:3])
            self.ee_vel = np.array(msg.data[3:6])
            self.has_ee = True

    def ref_cb(self, msg):
        """Extract cursor position + twist from the SA reference [13 floats]."""
        if len(msg.data) >= 12:
            self.cursor_pos = np.array(msg.data[0:3])
            self.v_ref = np.array(msg.data[6:9])   # linear velocity part
            self.last_ref_time = time.time()
            self.has_ref = True

    def timer_cb(self):
        if not (self.has_ee and self.has_ref):
            return

        # Compute diagnostics
        e_cursor = self.cursor_pos - self.ee_pos
        v_mismatch = self.v_ref - self.ee_vel
        dt_since_ref = time.time() - self.last_ref_time

        # Full vector topic
        msg = Float64MultiArray()
        msg.data = (e_cursor.tolist() + self.v_ref.tolist() +
                    self.ee_vel.tolist() + v_mismatch.tolist() +
                    self.cursor_pos.tolist() + self.ee_pos.tolist())
        self.pub_coupling.publish(msg)

        # Scalar summary topic (easy to plot)
        msg_s = Float64MultiArray()
        msg_s.data = [
            float(np.linalg.norm(e_cursor)),
            float(np.linalg.norm(self.v_ref)),
            float(np.linalg.norm(self.ee_vel)),
            float(np.linalg.norm(v_mismatch)),
            float(dt_since_ref),
        ]
        self.pub_scalar.publish(msg_s)

        # Staleness topic
        self.pub_age.publish(Float64(data=dt_since_ref))

        # Console heartbeat (2 Hz)
        now = time.time()
        if now - self.last_print > 0.5:
            print(f"[SA-QP] |e_cursor|={np.linalg.norm(e_cursor):.4f}m "
                  f"|v_ref|={np.linalg.norm(self.v_ref):.3f} "
                  f"|v_real|={np.linalg.norm(self.ee_vel):.3f} "
                  f"|mismatch|={np.linalg.norm(v_mismatch):.4f} "
                  f"ref_age={dt_since_ref*1000:.1f}ms", flush=True)
            self.last_print = now


def main():
    rclpy.init()
    node = SACouplingDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
