"""
Object detector: above-plane points -> labelled upright cylinders.

PIPELINE
    1. Voxel downsample the above-plane points (caps the count -> CPU-friendly
       clustering, and evens out the density).
    2. Euclidean clustering (region growing on a KD-tree): points within
       CLUSTER_TOLERANCE of each other belong to the same object.
    3. Per cluster, fit an UPRIGHT cylinder:
         - axis assumed vertical (objects stand on the table) -> robust
         - radius  = high percentile of the radial spread about the centroid XY
         - height  = z-extent of the cluster
         - centre  = (mean XY, mid-height Z)
    4. Classify colour (red / blue / unknown) from the mean hue of the cluster's
       RGB, using HSV thresholds (matplotlib.colors, already a project dep).

WHY upright-cylinder instead of full 6-DOF RANSAC cylinder:
    A generic cylinder RANSAC needs surface normals and is fiddly/slow on a
    noisy partial view. Our objects are known to stand vertically on the table,
    so fixing the axis to +Z turns the fit into two trivial, robust 1-D
    estimates (radial spread + height). This is the right amount of prior.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from matplotlib.colors import rgb_to_hsv

import triago_control.head_control.config as cfg


@dataclass
class DetectedObject:
    label: str                              # "red_cylinder" | "blue_cylinder" | "unknown_object"
    color_name: str                         # "red" | "blue" | "unknown"
    center: np.ndarray                      # (3,) base_footprint
    radius: float                           # [m]
    height: float                           # [m]
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    mean_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_points: int = 0


class ObjectDetector:
    # ------------------------------------------------------------------ #
    # Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def detect(self, points, colors, plane):
        """Cluster + fit + classify.

        Parameters
        ----------
        points : (M, 3) above-plane points in base_footprint
        colors : (M, 3) uint8 matching RGB
        plane  : PlaneModel (used for height reference / axis)

        Returns
        -------
        list[DetectedObject]
        """
        if len(points) < cfg.CLUSTER_MIN_POINTS:
            return []

        pts_ds, cols_ds = self._voxel_downsample(points, colors, cfg.VOXEL_SIZE)
        clusters = self._euclidean_cluster(pts_ds)

        detections = []
        for idx in clusters:
            cluster_pts = pts_ds[idx]
            cluster_cols = cols_ds[idx]
            obj = self._fit_cylinder(cluster_pts, cluster_cols, plane)
            if obj is not None:
                detections.append(obj)
        return detections

    # ------------------------------------------------------------------ #
    # 1. Voxel downsample                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _voxel_downsample(points, colors, leaf):
        """Keep one (averaged) point per occupied voxel. Vectorised."""
        keys = np.floor(points / leaf).astype(np.int64)
        # Unique voxel -> inverse mapping to average members.
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        n_vox = counts.shape[0]
        sum_pts = np.zeros((n_vox, 3))
        sum_cols = np.zeros((n_vox, 3))
        np.add.at(sum_pts, inverse, points)
        np.add.at(sum_cols, inverse, colors.astype(np.float64))
        pts_ds = sum_pts / counts[:, None]
        cols_ds = (sum_cols / counts[:, None]).astype(np.uint8)
        return pts_ds, cols_ds

    # ------------------------------------------------------------------ #
    # 2. Euclidean clustering (KD-tree region growing)                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _euclidean_cluster(points):
        """Return a list of index-arrays, one per cluster."""
        n = len(points)
        if n == 0:
            return []
        tree = cKDTree(points)
        visited = np.zeros(n, dtype=bool)
        clusters = []

        for seed in range(n):
            if visited[seed]:
                continue
            # Breadth-first region growing from the seed.
            queue = [seed]
            visited[seed] = True
            comp = [seed]
            while queue:
                j = queue.pop()
                neighbours = tree.query_ball_point(points[j], cfg.CLUSTER_TOLERANCE)
                for k in neighbours:
                    if not visited[k]:
                        visited[k] = True
                        queue.append(k)
                        comp.append(k)
            if cfg.CLUSTER_MIN_POINTS <= len(comp) <= cfg.CLUSTER_MAX_POINTS:
                clusters.append(np.array(comp))
        return clusters

    # ------------------------------------------------------------------ #
    # 3. Upright cylinder fit                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_cylinder(pts, cols, plane):
        # --- XY centre + radius via ALGEBRAIC CIRCLE FIT (Kasa) ---------
        # The camera only sees the near-facing ARC of the cylinder, so a plain
        # XY centroid is biased toward the camera and the radial spread
        # over-estimates the radius. A circle fit recovers the TRUE axis centre
        # and radius from a partial arc, removing both biases.
        xy = pts[:, :2]
        center_xy, radius, fit_ok = ObjectDetector._fit_circle(xy)
        if not fit_ok:
            # Fallback: centroid + percentile (robust if the arc is too small).
            center_xy = xy.mean(axis=0)
            radial = np.linalg.norm(xy - center_xy, axis=1)
            radius = float(np.percentile(radial, cfg.CYL_RADIUS_PERCENTILE))

        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        # Anchor the base at the detected table top for a meaningful height.
        base_z = plane.height
        height = float(z_max - base_z)
        center_z = base_z + height / 2.0

        # Plausibility gates (reject walls, specks, the robot's own gripper...).
        if not (cfg.CYL_MIN_RADIUS <= radius <= cfg.CYL_MAX_RADIUS):
            return None
        if not (cfg.CYL_MIN_HEIGHT <= height <= cfg.CYL_MAX_HEIGHT):
            return None

        color_name, mean_rgb = ObjectDetector._classify_color(cols)
        label = f"{color_name}_cylinder" if color_name != "unknown" else "unknown_object"

        return DetectedObject(
            label=label,
            color_name=color_name,
            center=np.array([center_xy[0], center_xy[1], center_z]),
            radius=radius,
            height=height,
            axis=plane.normal.copy(),
            mean_rgb=mean_rgb,
            n_points=len(pts),
        )

    @staticmethod
    def _fit_circle(xy):
        """Algebraic (Kasa) circle fit. Returns (center_xy, radius, ok).

        Minimises sum_i ((x_i-a)^2 + (y_i-b)^2 - R^2)^2 in closed form by
        solving the linear system  [2x 2y 1] [a b c]^T = [x^2+y^2],  with
        R = sqrt(c + a^2 + b^2). Robust for arcs >~ 90 deg; flagged not-ok if
        the system is ill-conditioned (near-collinear points).
        """
        if len(xy) < 5:
            return None, 0.0, False
        x = xy[:, 0]
        y = xy[:, 1]
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, 0.0, False
        a, c_, c = sol
        val = c + a * a + c_ * c_
        if val <= 0.0 or not np.isfinite(val):
            return None, 0.0, False
        return np.array([a, c_]), float(np.sqrt(val)), True

    # ------------------------------------------------------------------ #
    # 4. Colour classification (HSV)                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify_color(cols):
        mean_rgb = cols.astype(np.float64).mean(axis=0)         # 0..255
        hsv = rgb_to_hsv(mean_rgb / 255.0)                      # h,s,v in 0..1
        h, s, v = hsv

        if s < cfg.COLOR_SAT_MIN or v < cfg.COLOR_VAL_MIN:
            return "unknown", mean_rgb
        # Red wraps around the hue circle (near 0 and near 1).
        if h >= cfg.RED_HUE_LOW or h <= cfg.RED_HUE_HIGH:
            return "red", mean_rgb
        if cfg.BLUE_HUE_LOW <= h <= cfg.BLUE_HUE_HIGH:
            return "blue", mean_rgb
        return "unknown", mean_rgb
