#!/usr/bin/env python3
"""
head_debug_plotter.py — real-hardware debug dashboard for main_head.py.

main_head.py computes control/perception only (no plotting). This node
subscribes to its telemetry and draws two live windows on whatever machine
it runs on (intended: your dev PC, which has a real display):

    Figure 1 — RECOGNITION: "what does the detector think is a cylinder?"
        The cropped cloud in faint true colour (context) + the ABOVE-PLANE
        candidate points (black, the exact input to clustering) + each fitted
        cylinder drawn as a RING of its fitted radius at its centre. If a ring
        lands on a sparse/ragged black blob, that is a bad fit you can see.

    Figure 2 — RECONSTRUCTION: the world model in 3 orthographic views.
        (a) TOP  (X-Y): table rectangle with its measured X,Y dimensions
            labelled; each cylinder a filled circle of its radius.
        (b) SIDE (X-Z): table cross-section; each cylinder a bar of its
            height standing on the table top.
        (c) FRONT(Y-Z): same, viewed along +X.

Run on a machine with a display (subscribes over the network to main_head.py):
    ros2 run triago_control head_debug_plotter.py
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
import matplotlib.patches as patches
if _HAS_DISPLAY:
    plt.ion()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

import triago_control.head_control.config as cfg

REDRAW_PERIOD_S = 0.5

_CLOUD_DTYPE = np.dtype(
    [("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)]
)
_LABEL_COLOR = {"red": "#d62728", "blue": "#1f77b4", "unknown": "#7f7f7f"}


def _decode_cloud(msg: PointCloud2):
    """Inverse of visualization.py::make_pointcloud2 (XYZ f32 + packed rgb f32)."""
    arr = np.frombuffer(bytes(msg.data), dtype=_CLOUD_DTYPE)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
    packed = arr["rgb"].copy().view(np.uint32)
    r = ((packed >> 16) & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = (packed & 0xFF).astype(np.uint8)
    colors = np.stack([r, g, b], axis=1)
    return xyz, colors


class HeadDebugPlotter(Node):
    def __init__(self):
        super().__init__("head_debug_plotter")
        self.cloud_xyz = None
        self.cloud_colors = None
        self.above_xyz = None
        self.debug = None                # latest parsed JSON payload
        self._last_redraw = 0.0

        self.create_subscription(PointCloud2, "/head_perception/cloud", self._cloud_cb, 1)
        self.create_subscription(PointCloud2, "/head_perception/above_cloud", self._above_cb, 1)
        self.create_subscription(String, cfg.DEBUG_JSON_TOPIC, self._debug_cb, 10)

        self.fig1, self.ax1 = plt.subplots(
            figsize=(7, 7), num="Head debug 1 -- recognition")
        self.fig2, self.ax2 = plt.subplots(
            1, 3, figsize=(15, 5.5), num="Head debug 2 -- reconstruction")
        if _HAS_DISPLAY:
            plt.show(block=False)
        self.get_logger().info(
            "head_debug_plotter ready -- waiting for /head_perception/cloud + "
            f"{cfg.DEBUG_JSON_TOPIC} (needs main_head.py with real_hardware_head:=true).")

    def _cloud_cb(self, msg):
        if msg.width == 0:
            return
        self.cloud_xyz, self.cloud_colors = _decode_cloud(msg)

    def _above_cb(self, msg):
        if msg.width == 0:
            self.above_xyz = None
            return
        self.above_xyz, _ = _decode_cloud(msg)

    def _debug_cb(self, msg):
        try:
            self.debug = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    # ------------------------------------------------------------------ #
    # Redraw                                                              #
    # ------------------------------------------------------------------ #
    def maybe_redraw(self):
        now = time.time()
        if now - self._last_redraw < REDRAW_PERIOD_S or self.debug is None:
            return
        self._last_redraw = now
        self._draw_recognition()
        self._draw_reconstruction()
        if _HAS_DISPLAY:
            self.fig1.canvas.draw_idle()
            self.fig2.canvas.draw_idle()
            plt.pause(0.001)
        else:
            self.fig1.savefig("/tmp/head_debug_fig1.png")
            self.fig2.savefig("/tmp/head_debug_fig2.png")

    # --- Fig1: what does the detector fit as a cylinder? ----------------
    def _draw_recognition(self):
        ax = self.ax1
        ax.clear()
        d = self.debug
        # Context: cropped cloud (table + everything) in faint true colour.
        if self.cloud_xyz is not None and len(self.cloud_xyz) > 0:
            pts = self.cloud_xyz[::4]
            cols = self.cloud_colors[::4].astype(np.float64) / 255.0
            ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=3, marker='o', alpha=0.35)
        # The exact clustering input: above-plane candidate points, in black.
        if self.above_xyz is not None and len(self.above_xyz) > 0:
            ax.scatter(self.above_xyz[:, 0], self.above_xyz[:, 1],
                       c='black', s=6, marker='.', label='above-plane candidates')
        # Every fitted cylinder as a RING of its fitted radius.
        for o in d["objects"]:
            cx, cy = o["center"][0], o["center"][1]
            color = _LABEL_COLOR.get(o["color_name"], "#7f7f7f")
            ax.add_patch(patches.Circle(
                (cx, cy), o["radius"], fill=False, edgecolor=color, linewidth=2))
            ax.plot(cx, cy, '+', color=color, markersize=10, mew=2)
            ax.annotate(
                f"#{o['id']} {o['label']}\nr={o['radius']*100:.1f}cm h={o['height']*100:.1f}cm\n"
                f"arc={o['arc_coverage']:.0%} vert={o.get('vertical_coverage', 0):.0%}\n"
                f"conf={o['confidence']:.0%} rms={o['fit_rms']*1000:.1f}mm",
                (cx, cy), fontsize=7, xytext=(6, 6), textcoords='offset points',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
        ax.set_xlabel('X [m] (base_footprint)')
        ax.set_ylabel('Y [m]')
        ax.set_title(f"Recognition: {len(d['objects'])} fitted cylinder(s) over "
                     "the candidate points\n(ring = fitted radius; is it really a cylinder?)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        self.fig1.tight_layout()

    # --- Fig2: 3 orthographic reconstruction views ----------------------
    def _table(self):
        """(center, size) of the table to draw -- measured if available, else
        the static prior."""
        d = self.debug
        if d["table_center"] is not None and d["table_size"] is not None:
            return np.array(d["table_center"]), np.array(d["table_size"]), "measured"
        return np.array(d["static_prior_center"]), np.array(d["static_prior_size"]), "prior"

    def _draw_reconstruction(self):
        tc, ts, src = self._table()
        objs = self.debug["objects"]
        self._recon_top(self.ax2[0], tc, ts, src, objs)
        self._recon_side(self.ax2[1], tc, ts, objs, axis_h=0, name="X")   # X-Z
        self._recon_side(self.ax2[2], tc, ts, objs, axis_h=1, name="Y")   # Y-Z
        self.fig2.tight_layout()

    def _recon_top(self, ax, tc, ts, src, objs):
        """Top-down X-Y: table rectangle (dimensions labelled) + cylinder discs."""
        ax.clear()
        top_z = tc[2] + ts[2] / 2.0
        ax.add_patch(patches.Rectangle(
            (tc[0] - ts[0] / 2, tc[1] - ts[1] / 2), ts[0], ts[1],
            fill=True, facecolor='#d2b48c', edgecolor='saddlebrown', alpha=0.4))
        ax.annotate(f"X = {ts[0]*100:.0f} cm", (tc[0], tc[1] - ts[1] / 2),
                    ha='center', va='top', fontsize=9, color='saddlebrown')
        ax.annotate(f"Y = {ts[1]*100:.0f} cm", (tc[0] - ts[0] / 2, tc[1]),
                    ha='right', va='center', fontsize=9, rotation=90, color='saddlebrown')
        for o in objs:
            c = _LABEL_COLOR.get(o["color_name"], "#7f7f7f")
            ax.add_patch(patches.Circle((o["center"][0], o["center"][1]),
                                        max(o["radius"], 0.005), color=c, alpha=0.8))
            ax.annotate(f"#{o['id']}", (o["center"][0], o["center"][1]),
                        ha='center', va='center', fontsize=7, color='white')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.set_title(f"TOP view (X-Y)  table top z={top_z:.2f}m  [{src}]")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.autoscale()

    def _recon_side(self, ax, tc, ts, objs, axis_h, name):
        """Side view: horizontal = table axis `axis_h` (0=X, 1=Y), vertical = Z.
        Table drawn as its full box cross-section; each cylinder a bar of its
        height standing on the table top."""
        ax.clear()
        top_z = tc[2] + ts[2] / 2.0
        h0, hs = tc[axis_h], ts[axis_h]
        ax.add_patch(patches.Rectangle(
            (h0 - hs / 2, tc[2] - ts[2] / 2), hs, ts[2],
            fill=True, facecolor='#d2b48c', edgecolor='saddlebrown', alpha=0.4))
        ax.axhline(top_z, color='saddlebrown', linestyle='--', linewidth=1, alpha=0.7)
        for o in objs:
            c = _LABEL_COLOR.get(o["color_name"], "#7f7f7f")
            hc = o["center"][axis_h]
            ax.add_patch(patches.Rectangle(
                (hc - o["radius"], top_z), 2 * o["radius"], o["height"],
                fill=True, facecolor=c, edgecolor='black', alpha=0.8))
            ax.annotate(f"#{o['id']}\nh={o['height']*100:.0f}cm",
                        (hc, top_z + o["height"]), ha='center', va='bottom', fontsize=7)
        ax.set_xlabel(f'{name} [m]'); ax.set_ylabel('Z [m]')
        ax.set_title(f"{'SIDE' if name == 'X' else 'FRONT'} view ({name}-Z)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.autoscale()


def main():
    rclpy.init()
    node = HeadDebugPlotter()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.maybe_redraw()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
