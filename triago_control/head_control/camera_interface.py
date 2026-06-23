"""
Camera interface: RealSense RGB-D -> numpy -> 3D point cloud.

WHY this exists and WHY it avoids cv_bridge:
    ``cv_bridge`` was NOT in the robot's package list, and pulling it in is an
    avoidable risk. A ``sensor_msgs/Image`` is just a flat byte buffer plus a
    width/height/encoding header, so we decode it with ``np.frombuffer`` +
    reshape. Zero extra dependencies, fully under our control.

WHY SensorDataQoS:
    RealSense (and most camera drivers) publish with BEST_EFFORT reliability and
    a small queue. A subscriber created with the *default* (RELIABLE) QoS will
    silently receive NOTHING. This is the single most common "my camera node
    gets no data" bug. We therefore subscribe with ``qos_profile_sensor_data``.

DEPROJECTION (pinhole model, optical frame):
    Given pixel (u, v) and metric depth Z:
        X = (u - cx) * Z / fx        (+X right)
        Y = (v - cy) * Z / fy        (+Y down)
        Z =  Z                        (+Z forward, out of the lens)
    All vectorised over a strided pixel grid for CPU speed.
"""

import numpy as np
import threading

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo

import triago_control.head_control.config as cfg


class CameraInterface:
    """Owns the RGB-D subscriptions and produces point clouds on demand.

    Thread-safety: ROS callbacks write the latest frame under a lock; the
    perception loop snapshots it under the same lock. The heavy deprojection
    then runs on the snapshot, outside the lock, so callbacks never stall.
    """

    def __init__(self, node):
        self._node = node
        self._lock = threading.Lock()

        # Latest raw frames (numpy) + intrinsics. None until first message.
        self._color = None          # (H, W, 3) uint8, RGB
        self._depth = None          # (H, W)   float32, metres
        self._depth_stamp = None    # builtin_interfaces/Time of the depth frame
        self._depth_frame_id = None # frame the depth pixels live in (from header)
        self._K = None              # (fx, fy, cx, cy)
        self._info_wh = None        # (width, height) the intrinsics were calibrated at

        # Resolve topic names from ROS params (fall back to config defaults).
        color_topic = node.declare_parameter("color_topic", cfg.COLOR_TOPIC).value
        depth_topic = node.declare_parameter("depth_topic", cfg.DEPTH_TOPIC).value
        info_topic = node.declare_parameter("camera_info_topic", cfg.CAMERA_INFO_TOPIC).value

        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.info_topic = info_topic

        # BEST_EFFORT sensor QoS — see module docstring.
        node.create_subscription(Image, color_topic, self._color_cb, qos_profile_sensor_data)
        node.create_subscription(Image, depth_topic, self._depth_cb, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)

        # Frame-arrival counters (for the startup diagnostic in main_head.py).
        self.n_color = 0
        self.n_depth = 0
        self.n_info = 0

    # ------------------------------------------------------------------ #
    # ROS callbacks                                                       #
    # ------------------------------------------------------------------ #
    def _color_cb(self, msg: Image):
        img = self._image_to_numpy(msg)
        if img is None:
            return
        with self._lock:
            self._color = img
        self.n_color += 1

    def _depth_cb(self, msg: Image):
        depth = self._depth_to_metres(msg)
        if depth is None:
            return
        with self._lock:
            self._depth = depth
            self._depth_stamp = msg.header.stamp
            self._depth_frame_id = msg.header.frame_id
        self.n_depth += 1

    def _info_cb(self, msg: CameraInfo):
        # Pinhole intrinsics live in the 3x3 K matrix: [fx 0 cx; 0 fy cy; 0 0 1]
        K = msg.k
        with self._lock:
            self._K = (K[0], K[4], K[2], K[5])   # fx, fy, cx, cy
            self._info_wh = (msg.width, msg.height)
        self.n_info += 1

    # ------------------------------------------------------------------ #
    # Decoders (no cv_bridge)                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_to_numpy(msg: Image):
        """Decode an RGB/BGR colour Image into an (H, W, 3) uint8 RGB array."""
        enc = msg.encoding.lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            if enc in ("rgb8", "bgr8"):
                img = buf.reshape(msg.height, msg.width, 3)
                if enc == "bgr8":
                    img = img[:, :, ::-1]            # BGR -> RGB
                return np.ascontiguousarray(img)
            if enc in ("rgba8", "bgra8"):
                img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
                if enc == "bgra8":
                    img = img[:, :, ::-1]
                return np.ascontiguousarray(img)
            if enc in ("mono8",):
                g = buf.reshape(msg.height, msg.width)
                return np.repeat(g[:, :, None], 3, axis=2)
        except ValueError:
            return None
        return None     # unsupported encoding

    @staticmethod
    def _depth_to_metres(msg: Image):
        """Decode a depth Image into an (H, W) float32 array in METRES.

        RealSense publishes either 16UC1 (millimetres, real hardware / aligned)
        or 32FC1 (metres, common in the Gazebo plugin). Handle both; invalid /
        zero pixels become NaN so downstream filters drop them cleanly.
        """
        enc = msg.encoding.lower()
        try:
            if enc in ("16uc1", "mono16"):
                d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                d = d.astype(np.float32) / 1000.0           # mm -> m
            elif enc == "32fc1":
                d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
            else:
                return None
        except ValueError:
            return None
        d[d <= 0.0] = np.nan                                 # 0 == "no return"
        return d

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def has_data(self) -> bool:
        with self._lock:
            return self._color is not None and self._depth is not None and self._K is not None

    def get_point_cloud(self):
        """Deproject the latest RGB-D frame into a coloured point cloud.

        Returns
        -------
        points    : (N, 3) float32   XYZ in the camera OPTICAL frame
        colors    : (N, 3) uint8     matching RGB
        stamp     : ROS time of the depth frame  (or None)
        frame_id  : str, the frame the depth pixels live in (for TF lookup)
        None if no complete frame is available yet.
        """
        with self._lock:
            if self._color is None or self._depth is None or self._K is None:
                return None
            color = self._color
            depth = self._depth
            stamp = self._depth_stamp
            frame_id = self._depth_frame_id
            fx, fy, cx, cy = self._K
            info_wh = self._info_wh

        H, W = depth.shape

        # SAFETY: if the camera_info was calibrated at a different resolution
        # than the depth image (common when depth is downscaled), scale the
        # intrinsics to the actual image size. Wrong intrinsics shift points
        # laterally in proportion to depth -> large position errors.
        if info_wh is not None and info_wh != (0, 0):
            iw, ih = info_wh
            if iw and ih and (iw != W or ih != H):
                sx = W / float(iw)
                sy = H / float(ih)
                fx *= sx; cx *= sx
                fy *= sy; cy *= sy

        s = cfg.PIXEL_STRIDE
        us = np.arange(0, W, s)
        vs = np.arange(0, H, s)
        uu, vv = np.meshgrid(us, vs)                 # (h', w')

        z = depth[vv, uu]
        valid = np.isfinite(z) & (z > cfg.DEPTH_MIN) & (z < cfg.DEPTH_MAX)

        z = z[valid]
        uu = uu[valid]
        vv = vv[valid]

        # Pinhole back-projection (optical frame).
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).astype(np.float32)

        # Colour association. If colour and depth share the pixel grid, sample
        # directly; otherwise rescale the depth pixel coords into the colour
        # image (coarse, but colour is only used for red/blue classification).
        ch, cw = color.shape[:2]
        if (ch, cw) == (H, W):
            colors = color[vv, uu]
        else:
            cu = np.clip((uu * cw / W).astype(np.int64), 0, cw - 1)
            cv = np.clip((vv * ch / H).astype(np.int64), 0, ch - 1)
            colors = color[cv, cu]

        return points, colors, stamp, frame_id

    def get_depth_frame_id(self):
        with self._lock:
            return self._depth_frame_id
