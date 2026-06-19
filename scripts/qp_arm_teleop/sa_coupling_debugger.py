#!/usr/bin/env python3
"""
sa_coupling_debugger.py — Policy Visualization & Oscillation Diagnostics

Provides:
  1. A 3D matplotlib live plot showing:
     - The EE position trajectory (breadcrumb trail)
     - The SA virtual cursor (carrot) position
     - Twist arrows (commanded velocity direction + magnitude)
     - Goal direction indicator

  2. A low-Hz console summary (every 5s) with oscillation metrics:
     - Twist jitter: std of twist magnitude over the last window
     - Direction consistency: mean cosine similarity between consecutive twists
     - Cursor-to-EE distance stats (mean, std)
     - Reference staleness stats

No console spam. Only summary metrics printed at low frequency.

Usage:
    ros2 run triago_control sa_coupling_debugger.py
"""

import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, Float64


class PolicyVisualizer(Node):

    def __init__(self):
        super().__init__('sa_coupling_debugger')

        # --- Configuration ---
        self.TRAIL_LEN = 500         # breadcrumb history
        self.ARROW_SCALE = 0.3       # visual scaling of twist arrows
        self.METRICS_INTERVAL = 5.0  # seconds between console summaries
        self.WINDOW_SEC = 2.0        # rolling window for metrics computation

        # --- Data buffers ---
        self.ee_trail = deque(maxlen=self.TRAIL_LEN)
        self.cursor_trail = deque(maxlen=self.TRAIL_LEN)
        self.twist_history = deque(maxlen=500)
        self.twist_time = deque(maxlen=500)
        self.cursor_dist_history = deque(maxlen=500)
        self.ref_age_history = deque(maxlen=500)

        # --- Current state ---
        self.ee_pos = np.zeros(3)
        self.cursor_pos = np.zeros(3)
        self.v_ref = np.zeros(3)  # linear twist
        self.w_ref = np.zeros(3)  # angular twist
        self.last_ref_time = time.time()
        self.has_ee = False
        self.has_ref = False

        # --- QoS ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # --- Subscribers ---
        self.create_subscription(
            Float64MultiArray, '/qp_debug/ee_real', self.ee_cb, qos)
        self.create_subscription(
            Float64MultiArray, '/arm_right/cartesian_reference', self.ref_cb, qos)

        # --- Metrics timer (low Hz console) ---
        self.create_timer(self.METRICS_INTERVAL, self.print_metrics)
        self.last_metrics_time = time.time()

        print("[PolicyViz] Node started. Waiting for data...", flush=True)

    # ==================================================================
    # CALLBACKS
    # ==================================================================
    def ee_cb(self, msg):
        if len(msg.data) >= 3:
            self.ee_pos = np.array(msg.data[0:3])
            self.ee_trail.append(self.ee_pos.copy())
            self.has_ee = True

    def ref_cb(self, msg):
        if len(msg.data) >= 12:
            self.cursor_pos = np.array(msg.data[0:3])
            self.v_ref = np.array(msg.data[6:9])
            self.w_ref = np.array(msg.data[9:12]) if len(msg.data) >= 12 else np.zeros(3)
            self.cursor_trail.append(self.cursor_pos.copy())

            now = time.time()
            dt_ref = now - self.last_ref_time
            self.last_ref_time = now
            self.has_ref = True

            # Store for metrics
            self.twist_history.append(self.v_ref.copy())
            self.twist_time.append(now)
            self.cursor_dist_history.append(np.linalg.norm(self.cursor_pos - self.ee_pos))
            self.ref_age_history.append(dt_ref)

    # ==================================================================
    # METRICS (low Hz console — no spam)
    # ==================================================================
    def print_metrics(self):
        if not self.has_ref or len(self.twist_history) < 10:
            print("[PolicyViz] Waiting for sufficient data...", flush=True)
            return

        now = time.time()
        # Filter to recent window
        recent_idx = [i for i, t in enumerate(self.twist_time)
                      if now - t < self.WINDOW_SEC]
        if len(recent_idx) < 5:
            return

        twists = np.array([self.twist_history[i] for i in recent_idx])
        dists = np.array([self.cursor_dist_history[i] for i in recent_idx])
        ages = np.array([self.ref_age_history[i] for i in recent_idx])

        # 1. Twist magnitude stats
        twist_mags = np.linalg.norm(twists, axis=1)
        twist_mean = np.mean(twist_mags)
        twist_std = np.std(twist_mags)

        # 2. Direction consistency (cosine similarity between consecutive twists)
        cos_sims = []
        for i in range(1, len(twists)):
            n1 = np.linalg.norm(twists[i - 1])
            n2 = np.linalg.norm(twists[i])
            if n1 > 1e-4 and n2 > 1e-4:
                cos_sims.append(np.dot(twists[i - 1], twists[i]) / (n1 * n2))
        direction_consistency = np.mean(cos_sims) if cos_sims else 1.0

        # 3. Twist jerk (rate of change of twist)
        if len(twists) >= 3:
            twist_diffs = np.diff(twists, axis=0)
            jerk_magnitudes = np.linalg.norm(twist_diffs, axis=1)
            jerk_mean = np.mean(jerk_magnitudes)
        else:
            jerk_mean = 0.0

        # 4. Cursor distance stats
        dist_mean = np.mean(dists)
        dist_std = np.std(dists)

        # 5. Reference staleness
        age_mean = np.mean(ages) * 1000  # ms
        age_max = np.max(ages) * 1000

        # Print compact summary
        print(f"\n{'─'*60}", flush=True)
        print(f" POLICY HEALTH  (last {self.WINDOW_SEC:.0f}s, "
              f"N={len(recent_idx)} samples)", flush=True)
        print(f"{'─'*60}", flush=True)
        print(f"  |twist| mean={twist_mean:.4f} m/s  "
              f"std={twist_std:.5f} (jitter)", flush=True)
        print(f"  direction consistency = {direction_consistency:.4f} "
              f"(1.0=straight, <0.9=oscillating)", flush=True)
        print(f"  twist jerk (Δv/step) = {jerk_mean:.5f} m/s "
              f"({'smooth' if jerk_mean < 0.005 else 'ROUGH'})", flush=True)
        print(f"  cursor-EE dist: mean={dist_mean*1000:.2f}mm  "
              f"std={dist_std*1000:.3f}mm", flush=True)
        print(f"  ref period: mean={age_mean:.1f}ms  "
              f"max={age_max:.1f}ms", flush=True)
        print(f"{'─'*60}\n", flush=True)


def ros_thread(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def update_plot(frame, node, ax, trail_line, cursor_line, arrow_quiver, ee_dot, cursor_dot):
    """Update the 3D visualization each frame."""
    if not node.has_ee or not node.has_ref:
        return []

    # --- EE trail (blue) ---
    if len(node.ee_trail) > 1:
        trail = np.array(list(node.ee_trail))
        trail_line.set_data(trail[:, 0], trail[:, 1])
        trail_line.set_3d_properties(trail[:, 2])

    # --- Cursor trail (green, dashed) ---
    if len(node.cursor_trail) > 1:
        ct = np.array(list(node.cursor_trail))
        cursor_line.set_data(ct[:, 0], ct[:, 1])
        cursor_line.set_3d_properties(ct[:, 2])

    # --- Current EE dot ---
    ee_dot.set_data([node.ee_pos[0]], [node.ee_pos[1]])
    ee_dot.set_3d_properties([node.ee_pos[2]])

    # --- Current cursor dot ---
    cursor_dot.set_data([node.cursor_pos[0]], [node.cursor_pos[1]])
    cursor_dot.set_3d_properties([node.cursor_pos[2]])

    # --- Twist arrow (from EE position) ---
    # Remove old quiver and redraw (matplotlib 3D quiver can't be updated in-place)
    for coll in list(ax.collections):
        if coll is not arrow_quiver[0]:
            pass  # Keep reference but we'll replace below
    # Clear previous arrows
    while ax.collections:
        ax.collections[0].remove()

    # Draw twist arrow
    v = node.v_ref
    if np.linalg.norm(v) > 1e-4:
        ax.quiver(node.ee_pos[0], node.ee_pos[1], node.ee_pos[2],
                  v[0], v[1], v[2],
                  color='red', arrow_length_ratio=0.2,
                  length=node.ARROW_SCALE, linewidth=2, label='twist')

    # Draw cursor-to-EE line (yellow, shows the "carrot leash")
    ax.plot([node.ee_pos[0], node.cursor_pos[0]],
            [node.ee_pos[1], node.cursor_pos[1]],
            [node.ee_pos[2], node.cursor_pos[2]],
            'y-', linewidth=1, alpha=0.6)

    # --- Auto-scale around the current EE position ---
    margin = 0.15
    ax.set_xlim(node.ee_pos[0] - margin, node.ee_pos[0] + margin)
    ax.set_ylim(node.ee_pos[1] - margin, node.ee_pos[1] + margin)
    ax.set_zlim(node.ee_pos[2] - margin, node.ee_pos[2] + margin)

    return []


def main():
    rclpy.init()
    node = PolicyVisualizer()

    # Start ROS spin in background
    spinner = threading.Thread(target=ros_thread, args=(node,), daemon=True)
    spinner.start()

    # --- Set up 3D plot ---
    fig = plt.figure(figsize=(8, 7))
    fig.suptitle('Policy Command Visualization (SA → QP)')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title('Live EE + Virtual Cursor + Twist')

    # Initialize plot elements
    trail_line, = ax.plot([], [], [], 'b-', linewidth=1.5, alpha=0.7, label='EE path')
    cursor_line, = ax.plot([], [], [], 'g--', linewidth=1, alpha=0.5, label='Cursor path')
    ee_dot, = ax.plot([], [], [], 'bo', markersize=8, label='EE now')
    cursor_dot, = ax.plot([], [], [], 'g^', markersize=8, label='Cursor now')
    arrow_quiver = [None]  # placeholder for quiver reference

    ax.legend(loc='upper left', fontsize='small')

    ani = FuncAnimation(fig, update_plot,
                        fargs=(node, ax, trail_line, cursor_line,
                               arrow_quiver, ee_dot, cursor_dot),
                        interval=100, blit=False)
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
