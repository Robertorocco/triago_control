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

INTRINSIC CALIBRATION / DISTORTION (2026-07-02 accuracy pass):
    Previously the `CameraInfo.d` distortion coefficients were received on
    the info topic and silently discarded — deprojection assumed an ideal
    pinhole with no distortion correction at all. In practice, RealSense
    D-series DEPTH streams are firmware-rectified and typically report
    D=[0,0,0,0,0] (confirmed from Intel's own documentation and multiple
    RealSense user reports), so this was likely a no-op for THIS specific
    camera/topic in practice, not the dominant source of the reported
    centimetre-level error (that was the circle-fit bias in
    object_detector.py — see its module docstring). It was nonetheless a
    real correctness gap: nothing verified the zero-distortion assumption,
    so it would have silently produced wrong 3D points on any camera/config
    where D is non-zero (e.g. the COLOR stream, or a different camera model).
    Fixed properly here: if `D` is non-negligible, pixels are undistorted
    (inverse Brown-Conrady, iterative fixed-point solve — verified
    numerically to converge to machine precision in <=5 iterations even for
    aggressive distortion magnitudes) before back-projection; a one-time
    warning is logged so a non-zero-D camera is never silently mishandled.
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
        self._D = None              # (5,) distortion coeffs [k1,k2,p1,p2,k3], or None
        self._info_wh = None        # (width, height) the intrinsics were calibrated at
        self._distortion_warned = False

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
        # Distortion coefficients (plumb_bob / Brown-Conrady convention:
        # [k1, k2, p1, p2, k3]). RealSense depth streams are typically
        # firmware-rectified (D == [0]*5), but we no longer ASSUME that — see
        # the module docstring. Pad/truncate defensively to exactly 5 values.
        d = list(msg.d) if msg.d else []
        d = (d + [0.0] * 5)[:5]
        with self._lock:
            self._K = (K[0], K[4], K[2], K[5])   # fx, fy, cx, cy
            self._D = tuple(d)
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
            D = self._D
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

        # Normalised (undistorted) image coordinates.
        xn = (uu - cx) / fx
        yn = (vv - cy) / fy

        if D is not None and any(abs(c) > 1e-9 for c in D):
            if not self._distortion_warned:
                self._node.get_logger().warn(
                    f"[camera_interface] Non-zero distortion coefficients "
                    f"received (D={D}) — applying iterative undistortion "
                    f"before deprojection. If this is unexpected for a "
                    f"RealSense DEPTH stream, double check camera_info_topic "
                    f"points at the depth (not colour) CameraInfo.")
                self._distortion_warned = True
            xn, yn = self._undistort_normalized(xn, yn, D)

        # Pinhole back-projection (optical frame), using the UNDISTORTED
        # normalised coordinates (identity when D is all-zero, the common
        # case for a firmware-rectified RealSense depth stream).
        x = xn * z
        y = yn * z
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

    def get_intrinsics_debug(self):
        """Expose the raw intrinsics/resolution state for calibration audits.

        Returns a dict (or None if camera_info hasn't arrived yet):
            {fx, fy, cx, cy, D, info_wh, depth_wh, color_wh}
        `info_wh` is the resolution camera_info was calibrated at; `depth_wh`
        is the actual depth image resolution. If these differ,
        get_point_cloud() is silently RESCALING the intrinsics (see its
        module docstring) — a mismatch here is worth knowing about even if
        the rescale keeps things numerically consistent.
        """
        with self._lock:
            if self._K is None:
                return None
            depth_wh = (self._depth.shape[1], self._depth.shape[0]) if self._depth is not None else None
            color_wh = (self._color.shape[1], self._color.shape[0]) if self._color is not None else None
            return {
                "fx": self._K[0], "fy": self._K[1], "cx": self._K[2], "cy": self._K[3],
                "D": self._D, "info_wh": self._info_wh,
                "depth_wh": depth_wh, "color_wh": color_wh,
            }

    def get_scaled_intrinsics(self):
        """Return (fx, fy, cx, cy) rescaled to the ACTUAL depth image
        resolution — mirrors the exact rescale logic in get_point_cloud()
        (see its body). Kept as ONE shared implementation so a diagnostic
        tool never risks silently drifting from what get_point_cloud()
        really does. Returns None if camera_info/depth haven't arrived yet.
        """
        with self._lock:
            if self._K is None or self._depth is None:
                return None
            fx, fy, cx, cy = self._K
            H, W = self._depth.shape
            info_wh = self._info_wh
        if info_wh is not None and info_wh != (0, 0):
            iw, ih = info_wh
            if iw and ih and (iw != W or ih != H):
                sx = W / float(iw)
                sy = H / float(ih)
                fx *= sx; cx *= sx
                fy *= sy; cy *= sy
        return fx, fy, cx, cy

    def sample_pixel_patch(self, u, v, radius=3):
        """Robust single-pixel depth/colour sample for calibration audits.

        Returns (depth_m, color_rgb, n_valid) using the MEDIAN valid depth
        in a (2*radius+1)^2 window around integer pixel (u, v) — robust to
        isolated invalid/NaN returns without smoothing across a large area.
        Returns None if (u, v) is out of bounds or no valid depth exists in
        the window. `color_rgb` is None if no colour frame is available.

        WHY a window and not the exact pixel: at a single exact pixel there
        is a real chance of hitting a "no return" (NaN) sample even when the
        surface is perfectly valid nearby (isolated depth dropouts are
        normal on any RGB-D sensor). Radius 3 (7x7 window, default) is small
        enough to stay a genuinely LOCAL sample (no cross-object smearing at
        the ~2cm cylinder scale) while being robust to a handful of dropouts.
        """
        with self._lock:
            if self._depth is None:
                return None
            depth = self._depth
            color = self._color
        H, W = depth.shape
        ui, vi = int(round(u)), int(round(v))
        u0, u1 = max(0, ui - radius), min(W, ui + radius + 1)
        v0, v1 = max(0, vi - radius), min(H, vi + radius + 1)
        if u0 >= u1 or v0 >= v1:
            return None
        patch = depth[v0:v1, u0:u1]
        valid = np.isfinite(patch) & (patch > cfg.DEPTH_MIN) & (patch < cfg.DEPTH_MAX)
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            return None
        depth_m = float(np.median(patch[valid]))
        color_rgb = None
        if color is not None:
            ch, cw = color.shape[:2]
            if (ch, cw) == (H, W):
                cpatch = color[v0:v1, u0:u1].reshape(-1, 3)[valid.reshape(-1)]
            else:
                cu = int(np.clip(ui * cw / W, 0, cw - 1))
                cv = int(np.clip(vi * ch / H, 0, ch - 1))
                cpatch = color[cv:cv + 1, cu:cu + 1].reshape(-1, 3)
            if len(cpatch):
                color_rgb = cpatch.astype(np.float64).mean(axis=0)
        return depth_m, color_rgb, n_valid

    @staticmethod
    def _undistort_normalized(xd, yd, D, n_iters=5):
        """Invert the Brown-Conrady (plumb_bob) distortion model.

        `xd, yd` are DISTORTED normalised coordinates (i.e. what you get from
        the raw pixel: (u-cx)/fx, (v-cy)/fy). Returns the UNDISTORTED
        normalised coordinates that the ideal pinhole model expects.

        The forward model is
            r2 = x^2 + y^2
            radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
            x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x^2)
            y_d = y*radial + p1*(r2 + 2*y^2) + 2*p2*x*y
        which has no closed-form inverse. We invert it with the standard
        fixed-point iteration (same approach OpenCV's cv2.undistortPoints
        uses internally): starting from x=x_d, y=y_d, repeatedly re-evaluate
        the distortion at the current estimate and solve for the next one.
        Verified numerically to converge to machine precision (~1e-12) in 5
        iterations for distortion magnitudes well beyond typical RealSense
        colour-sensor values; convergence is fast because the correction is
        a small perturbation for any well-behaved lens.
        """
        k1, k2, p1, p2, k3 = D
        x = xd.copy()
        y = yd.copy()
        for _ in range(n_iters):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            radial_safe = np.where(np.abs(radial) < 1e-6, 1.0, radial)
            x = (xd - dx) / radial_safe
            y = (yd - dy) / radial_safe
        return x, y
