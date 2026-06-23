#!/usr/bin/env python3
"""
head_plotter.py — Live Matplotlib dashboard for the head perception subsystem.

Subscribes to the same topics as main_head.py publishes telemetry on, and
plots real-time perception quality vs ground truth.

RUN:  ros2 run triago_control head_plotter.py

PLOTS (5 subplots):
    1. XY Object Positions (top-down) — detected vs ground truth
    2. Position Error [m] over time — Euclidean distance to nearest GT
    3. Radius estimate [cm] vs ground truth over time
    4. Height estimate [cm] vs ground truth over time
    5. Processing time [ms] + cloud size over time

GROUND TRUTH is hard-coded from the Gazebo SDF (for plotting only — main_head
does NOT know the truth). This lets us visually assess estimation quality.
"""

import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import MarkerArray

# =============================================================================
# GROUND TRUTH (from SDF — ONLY used here for visual comparison)
# =============================================================================
GT_RED = {"label": "red_cylinder", "center": np.array([0.800, -0.20, 0.775]),
           "radius": 0.02, "height": 0.15}
GT_BLUE = {"label": "blue_cylinder", "center": np.array([0.800, 0.20, 0.775]),
            "radius": 0.02, "height": 0.15}
GT_TABLE_TOP_Z = 0.70


class HeadPlotterNode(Node):
    def __init__(self):
        super().__init__("head_plotter")

        # --- Data buffers (thread-safe access via lock) ----------------
        self.lock = threading.Lock()
        self.WINDOW = 60.0  # seconds of history to display
        self.MAXLEN = int(self.WINDOW * 5)  # 5 Hz perception rate

        self.t_data = deque(maxlen=self.MAXLEN)
        self.start_time = time.time()

        # Per-object time series (keyed by color_name: "red" / "blue")
        self.obj_data = {
            "red": {"x": deque(maxlen=self.MAXLEN), "y": deque(maxlen=self.MAXLEN),
                    "z": deque(maxlen=self.MAXLEN), "r": deque(maxlen=self.MAXLEN),
                    "h": deque(maxlen=self.MAXLEN), "err": deque(maxlen=self.MAXLEN),
                    "t": deque(maxlen=self.MAXLEN)},
            "blue": {"x": deque(maxlen=self.MAXLEN), "y": deque(maxlen=self.MAXLEN),
                     "z": deque(maxlen=self.MAXLEN), "r": deque(maxlen=self.MAXLEN),
                     "h": deque(maxlen=self.MAXLEN), "err": deque(maxlen=self.MAXLEN),
                     "t": deque(maxlen=self.MAXLEN)},
        }
        self.proc_ms = deque(maxlen=self.MAXLEN)
        self.cloud_size = deque(maxlen=self.MAXLEN)
        self.plane_z = deque(maxlen=self.MAXLEN)
        self.look_err = deque(maxlen=self.MAXLEN)
        self.slack = deque(maxlen=self.MAXLEN)
        self.conf_red = deque(maxlen=self.MAXLEN)
        self.conf_blue = deque(maxlen=self.MAXLEN)
        self.map_size = deque(maxlen=self.MAXLEN)

        # --- ROS subscriptions -----------------------------------------
        # MarkerArray -> detected cylinder poses; telemetry -> scalar quality.
        self.create_subscription(MarkerArray, "/head_perception/markers", self._markers_cb, 1)
        self.create_subscription(Float64MultiArray, "/head_perception/telemetry",
                                 self._telemetry_cb, 10)

        self.get_logger().info("Head Plotter started. Waiting for /head_perception/markers...")

    def _telemetry_cb(self, msg: Float64MultiArray):
        # [n_raw, n_crop, plane_z, look_err_deg, slack, proc_ms,
        #  red_conf, blue_conf, map_size]
        if len(msg.data) < 6:
            return
        t = time.time() - self.start_time
        with self.lock:
            self.cloud_size.append((t, msg.data[0]))
            self.proc_ms.append((t, msg.data[5]))
            self.look_err.append((t, msg.data[3]))
            self.slack.append((t, msg.data[4]))
            if len(msg.data) >= 9:
                self.conf_red.append((t, msg.data[6] * 100.0))
                self.conf_blue.append((t, msg.data[7] * 100.0))
                self.map_size.append((t, msg.data[8]))

    def _markers_cb(self, msg: MarkerArray):
        t = time.time() - self.start_time

        with self.lock:
            for m in msg.markers:
                # Cylinders have ns="objects"
                if m.ns == "objects" and m.type == 3:  # CYLINDER=3
                    cx = m.pose.position.x
                    cy = m.pose.position.y
                    cz = m.pose.position.z
                    radius = m.scale.x / 2.0
                    height = m.scale.z

                    # Classify by colour (R channel > 0.5 -> red, B > 0.5 -> blue)
                    if m.color.r > 0.5:
                        key = "red"
                        gt = GT_RED
                    elif m.color.b > 0.5:
                        key = "blue"
                        gt = GT_BLUE
                    else:
                        continue

                    pos_err = np.linalg.norm(
                        np.array([cx, cy, cz]) - gt["center"]
                    )

                    d = self.obj_data[key]
                    d["x"].append(cx)
                    d["y"].append(cy)
                    d["z"].append(cz)
                    d["r"].append(radius * 100)  # cm
                    d["h"].append(height * 100)  # cm
                    d["err"].append(pos_err * 100)  # cm
                    d["t"].append(t)

                # Table top plane (ns="table_top", CUBE, scale.z ~ 0.005)
                if m.ns == "table_top" and m.type == 1:  # CUBE=1
                    self.plane_z.append(m.pose.position.z)

            self.t_data.append(t)


def main():
    rclpy.init()
    node = HeadPlotterNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # --- Setup plots ---------------------------------------------------
    plt.ion()
    fig, axs = plt.subplots(4, 2, figsize=(14, 12))
    fig.canvas.manager.set_window_title("Head Perception — Live Dashboard")

    # Subplot layout:
    # [0,0] XY top-down positions     [0,1] Position error over time
    # [1,0] Radius over time          [1,1] Height over time
    # [2,0] Plane Z over time         [2,1] Estimation confidence over time
    # [3,0] Fused map size over time  [3,1] Raw cloud size over time

    ax_xy = axs[0, 0]
    ax_err = axs[0, 1]
    ax_rad = axs[1, 0]
    ax_hgt = axs[1, 1]
    ax_plane = axs[2, 0]
    ax_conf = axs[2, 1]
    ax_map = axs[3, 0]
    ax_cloud = axs[3, 1]

    # --- Static elements -----------------------------------------------
    # XY plot: table footprint + GT positions
    ax_xy.set_title("Top-Down View (XY in base_footprint)")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal")
    ax_xy.set_xlim(0.5, 1.3)
    ax_xy.set_ylim(-0.5, 0.5)
    # Table footprint rectangle
    table_rect = plt.Rectangle((0.7, -0.25), 0.6, 0.5, fill=True,
                                facecolor="wheat", edgecolor="brown", linewidth=2, alpha=0.4)
    ax_xy.add_patch(table_rect)
    # GT circles
    ax_xy.add_patch(Circle((GT_RED["center"][0], GT_RED["center"][1]),
                           GT_RED["radius"], fill=False, edgecolor="red",
                           linestyle="--", linewidth=2, label="GT Red"))
    ax_xy.add_patch(Circle((GT_BLUE["center"][0], GT_BLUE["center"][1]),
                           GT_BLUE["radius"], fill=False, edgecolor="blue",
                           linestyle="--", linewidth=2, label="GT Blue"))
    # Live markers (will be updated)
    scat_red, = ax_xy.plot([], [], "ro", markersize=10, label="Det Red")
    scat_blue, = ax_xy.plot([], [], "bs", markersize=10, label="Det Blue")
    ax_xy.legend(loc="upper left", fontsize=8)

    # Error plot
    ax_err.set_title("Position Error vs Ground Truth")
    ax_err.set_ylabel("Error [cm]")
    ax_err.set_xlabel("Time [s]")
    ax_err.grid(True, alpha=0.3)
    line_err_r, = ax_err.plot([], [], "r-", linewidth=1.5, label="Red")
    line_err_b, = ax_err.plot([], [], "b-", linewidth=1.5, label="Blue")
    ax_err.legend(loc="upper right", fontsize=8)

    # Radius plot
    ax_rad.set_title("Radius Estimate vs GT (2.0 cm)")
    ax_rad.set_ylabel("Radius [cm]")
    ax_rad.set_xlabel("Time [s]")
    ax_rad.grid(True, alpha=0.3)
    ax_rad.axhline(GT_RED["radius"] * 100, color="gray", linestyle="--", label="GT (2.0cm)")
    line_rad_r, = ax_rad.plot([], [], "r-", linewidth=1.5, label="Red")
    line_rad_b, = ax_rad.plot([], [], "b-", linewidth=1.5, label="Blue")
    ax_rad.legend(loc="upper right", fontsize=8)

    # Height plot
    ax_hgt.set_title("Height Estimate vs GT (15.0 cm)")
    ax_hgt.set_ylabel("Height [cm]")
    ax_hgt.set_xlabel("Time [s]")
    ax_hgt.grid(True, alpha=0.3)
    ax_hgt.axhline(GT_RED["height"] * 100, color="gray", linestyle="--", label="GT (15.0cm)")
    line_hgt_r, = ax_hgt.plot([], [], "r-", linewidth=1.5, label="Red")
    line_hgt_b, = ax_hgt.plot([], [], "b-", linewidth=1.5, label="Blue")
    ax_hgt.legend(loc="upper right", fontsize=8)

    # Plane Z
    ax_plane.set_title("Detected Table Top Z vs GT (0.70 m)")
    ax_plane.set_ylabel("Z [m]")
    ax_plane.set_xlabel("Time [s]")
    ax_plane.grid(True, alpha=0.3)
    ax_plane.axhline(GT_TABLE_TOP_Z, color="gray", linestyle="--", label="GT (0.70m)")
    line_plane, = ax_plane.plot([], [], "g-", linewidth=2, label="Detected")
    ax_plane.legend(loc="upper right", fontsize=8)

    # Confidence over time (estimation quality = arc coverage x fit quality)
    ax_conf.set_title("Estimation Confidence (arc coverage x fit quality)")
    ax_conf.set_ylabel("Confidence [%]")
    ax_conf.set_xlabel("Time [s]")
    ax_conf.grid(True, alpha=0.3)
    ax_conf.set_ylim(0, 105)
    line_conf_r, = ax_conf.plot([], [], "r-", linewidth=1.5, label="Red")
    line_conf_b, = ax_conf.plot([], [], "b-", linewidth=1.5, label="Blue")
    ax_conf.legend(loc="upper right", fontsize=8)

    # Fused voxel-map size (grows as accumulation builds the model)
    ax_map.set_title("Fused Map Size (accumulated voxels)")
    ax_map.set_ylabel("Voxels")
    ax_map.set_xlabel("Time [s]")
    ax_map.grid(True, alpha=0.3)
    line_map, = ax_map.plot([], [], "darkorange", linewidth=1.5)

    # Raw cloud size (per-frame perception input)
    ax_cloud.set_title("Raw Cloud Size (per-frame input)")
    ax_cloud.set_ylabel("Points")
    ax_cloud.set_xlabel("Time [s]")
    ax_cloud.grid(True, alpha=0.3)
    line_cloud, = ax_cloud.plot([], [], "purple", linewidth=1.5)

    fig.tight_layout()
    plt.show(block=False)

    # --- Animation loop ------------------------------------------------
    try:
        while rclpy.ok():
            with node.lock:
                if len(node.t_data) == 0:
                    plt.pause(0.2)
                    continue

                t_list = list(node.t_data)
                current_t = t_list[-1] if t_list else 0

                # XY positions (latest only)
                rd = node.obj_data["red"]
                bd = node.obj_data["blue"]

                if rd["x"]:
                    scat_red.set_data([rd["x"][-1]], [rd["y"][-1]])
                if bd["x"]:
                    scat_blue.set_data([bd["x"][-1]], [bd["y"][-1]])

                # Time series
                rt = list(rd["t"])
                bt = list(bd["t"])

                r_err = list(rd["err"])
                b_err = list(bd["err"])

                r_rad = list(rd["r"])
                b_rad = list(bd["r"])

                r_hgt = list(rd["h"])
                b_hgt = list(bd["h"])

                plane_list = list(node.plane_z)
                cloud_list = list(node.cloud_size)
                conf_r_list = list(node.conf_red)
                conf_b_list = list(node.conf_blue)
                map_list = list(node.map_size)

            # Update line data
            line_err_r.set_data(rt, r_err)
            line_err_b.set_data(bt, b_err)
            line_rad_r.set_data(rt, r_rad)
            line_rad_b.set_data(bt, b_rad)
            line_hgt_r.set_data(rt, r_hgt)
            line_hgt_b.set_data(bt, b_hgt)
            line_plane.set_data(t_list[:len(plane_list)], plane_list)
            if cloud_list:
                line_cloud.set_data([p[0] for p in cloud_list], [p[1] for p in cloud_list])
            if conf_r_list:
                line_conf_r.set_data([p[0] for p in conf_r_list], [p[1] for p in conf_r_list])
            if conf_b_list:
                line_conf_b.set_data([p[0] for p in conf_b_list], [p[1] for p in conf_b_list])
            if map_list:
                line_map.set_data([p[0] for p in map_list], [p[1] for p in map_list])

            # Auto-scale time axes
            window_lo = max(0, current_t - node.WINDOW)
            for ax in (ax_err, ax_rad, ax_hgt, ax_plane, ax_conf, ax_map, ax_cloud):
                ax.set_xlim(window_lo, current_t + 1)

            # Auto-scale Y axes (confidence has a fixed 0-105 range)
            for ax in (ax_err, ax_rad, ax_hgt, ax_plane, ax_map, ax_cloud):
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
