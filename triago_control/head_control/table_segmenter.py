"""
Table segmenter: find the table TOP surface in a point cloud via RANSAC.

INPUT  : points already expressed in base_footprint (so "up" is simply +Z) and
         already cropped to the table region (see perception_pipeline).
OUTPUT : a PlaneModel (point + normal + height) and a boolean mask splitting the
         cloud into {plane inliers, everything else}.

WHY RANSAC and not "just take points near z = 0.70":
    The whole point of this project is to STOP hard-coding the world. We are
    told *roughly* where the table is (to aim the head and to crop), but the
    exact top height, tilt, and extent are DERIVED from the data. RANSAC fits
    the dominant planar surface robustly even with sensor noise and outliers.

ALGORITHM (vectorised plane RANSAC, biased to horizontal planes):
    repeat N times:
        sample 3 random points -> candidate plane (normal n, offset d)
        reject if the plane is not roughly horizontal (|n.z| < threshold)
        count inliers = points whose |n.p + d| < band
        keep the best
    then refit the normal to all inliers via SVD (least-squares plane).
    Finally gate the plane height against the known table top so we never lock
    onto the floor or a wall ledge that happens to be horizontal.
"""

import numpy as np

import triago_control.head_control.config as cfg


class PlaneModel:
    """A plane n·x + d = 0 with helper queries. Normal points roughly +Z."""

    def __init__(self, normal, d):
        self.normal = normal / np.linalg.norm(normal)
        if self.normal[2] < 0:                  # force the normal to point up
            self.normal = -self.normal
            d = -d
        self.d = d

    @property
    def height(self) -> float:
        """Z of the plane at (x=0, y=0): z = -d / n_z  (n ~ vertical)."""
        return -self.d / self.normal[2]

    def signed_distance(self, pts):
        """Signed distance of points to the plane (positive == above)."""
        return pts @ self.normal + self.d


class TableSegmenter:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def segment(self, points):
        """Return (plane_model, inlier_mask) or (None, None) if no table found."""
        n_pts = len(points)
        if n_pts < cfg.PLANE_MIN_INLIERS:
            return None, None

        up = np.array([0.0, 0.0, 1.0])
        best_count = 0
        best_normal = None
        best_d = 0.0

        for _ in range(cfg.PLANE_RANSAC_ITERS):
            idx = self.rng.choice(n_pts, size=3, replace=False)
            p0, p1, p2 = points[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(normal)
            if nn < 1e-9:
                continue
            normal = normal / nn

            # Reject non-horizontal candidate planes early.
            if abs(normal[2]) < cfg.PLANE_MIN_VERTICAL_DOT:
                continue

            d = -normal @ p0
            dist = np.abs(points @ normal + d)
            count = int(np.count_nonzero(dist < cfg.PLANE_DIST_THRESH))
            if count > best_count:
                best_count = count
                best_normal = normal
                best_d = d

        if best_normal is None or best_count < cfg.PLANE_MIN_INLIERS:
            return None, None

        # Refit the plane to all inliers (SVD least-squares) for a clean normal.
        inlier_mask = np.abs(points @ best_normal + best_d) < cfg.PLANE_DIST_THRESH
        inliers = points[inlier_mask]
        centroid = inliers.mean(axis=0)
        # Smallest singular vector of the centred inliers == plane normal.
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        refined_normal = vh[-1]
        refined_d = -refined_normal @ centroid
        plane = PlaneModel(refined_normal, refined_d)

        # Height gate: the detected top must be near the known table top.
        if abs(plane.height - cfg.TABLE_TOP_Z_WORLD) > cfg.PLANE_Z_TOLERANCE:
            return None, None

        # Recompute the inlier mask with the refined plane.
        final_mask = np.abs(plane.signed_distance(points)) < cfg.PLANE_DIST_THRESH
        return plane, final_mask


def table_box_from_inliers(inlier_pts, plane_height,
                           xy_margin=None, bottom_z=None, pct=None):
    """Derive a full table BOX (centre, size) from the RANSAC plane inliers.

    This is what makes the perceived-world table fully camera-derived (rather than
    reusing a prior XY footprint): the table's XY extent comes from the horizontal
    spread of the plane inliers, its top-Z from the fitted plane, and its bottom
    from a ground reference. Used to build the /perceived_world/snapshot the QP-
    CLF-CBF stack turns into a collision box (see world_convergence.py).

    A PERCENTILE trim (not raw min/max) rejects RGB-D flying-pixel outliers at the
    table edges the same way object_detector's rim extraction does — a couple of
    stray points must not blow up the footprint. Occlusion can only make the
    perceived footprint SMALLER than the true table; PERCEIVED_TABLE_XY_MARGIN
    lets a caller inflate it outward for a conservative collision volume.

    Parameters
    ----------
    inlier_pts   : (K, 3) plane-inlier points in base_footprint.
    plane_height : float, the fitted plane's Z (table top surface).

    Returns
    -------
    (centre (3,), size (3,))  box centre + full extents, or (None, None) if the
    inlier set is too small to bound.
    """
    if inlier_pts is None or len(inlier_pts) < 3:
        return None, None
    xy_margin = cfg.PERCEIVED_TABLE_XY_MARGIN if xy_margin is None else xy_margin
    bottom_z = cfg.PERCEIVED_TABLE_BOTTOM_Z if bottom_z is None else bottom_z
    pct = cfg.PERCEIVED_TABLE_BBOX_PCT if pct is None else pct

    x_lo, x_hi = np.percentile(inlier_pts[:, 0], [pct, 100.0 - pct])
    y_lo, y_hi = np.percentile(inlier_pts[:, 1], [pct, 100.0 - pct])
    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    sx = float((x_hi - x_lo) + 2.0 * xy_margin)
    sy = float((y_hi - y_lo) + 2.0 * xy_margin)

    z_top = float(plane_height)
    sz = max(z_top - float(bottom_z), 1e-3)      # solid column from ground to top
    cz = 0.5 * (z_top + float(bottom_z))
    return (np.array([cx, cy, cz], dtype=float),
            np.array([sx, sy, sz], dtype=float))
