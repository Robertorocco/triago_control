"""
Perception pipeline: ties the geometric stages into one call.

    raw cloud (optical frame)
        -> transform to base_footprint        (using T_cam_base from FK)
        -> crop to the table region           (kill floor / walls / robot body)
        -> RANSAC table plane                 (TableSegmenter)
        -> keep the slab just above the plane  (candidate object points)
        -> cluster + cylinder fit + colour     (ObjectDetector)
        -> temporal EMA association            (stabilise poses across frames)

Everything downstream of the transform works in base_footprint, where "up" is
simply +Z — which is what the plane RANSAC and the upright-cylinder fit assume.

The result is a PerceptionResult that carries both the OUTPUT (plane + objects)
and intermediate clouds for visualisation/debugging.
"""

from dataclasses import dataclass, field

import numpy as np

import triago_control.head_control.config as cfg
from triago_control.head_control.table_segmenter import TableSegmenter, table_box_from_inliers
from triago_control.head_control.object_detector import ObjectDetector, DetectedObject
from triago_control.head_control.voxel_map import VoxelMap
from triago_control.head_control.object_tracker import ObjectTracker


@dataclass
class PerceptionResult:
    plane: object = None                    # PlaneModel or None
    objects: list = field(default_factory=list)     # list[DetectedObject]
    cropped_points: np.ndarray = None       # (N,3) base frame  (for viz)
    cropped_colors: np.ndarray = None       # (N,3) uint8
    above_points: np.ndarray = None         # (M,3) above-plane points (for viz)
    above_colors: np.ndarray = None          # (M,3) uint8 matching above_points
    plane_centroid: np.ndarray = None       # (3,) centroid of plane inliers (debug)
    # Fully camera-derived table BOX (centre + full extents, base frame): XY from
    # the plane inliers' horizontal spread, top-Z from the plane. Feeds the
    # perceived-world snapshot -> QP-CLF-CBF collision box. None until a plane fit.
    table_center: np.ndarray = None         # (3,) box centre
    table_size: np.ndarray = None           # (3,) box full extents [sx, sy, sz]
    n_raw: int = 0
    map_size: int = 0                       # voxels in the fused map (0 if off)
    proc_ms: float = 0.0


class PerceptionPipeline:
    def __init__(self):
        self.segmenter = TableSegmenter()
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()     # object-level temporal fusion
        self._tracked = []                  # (legacy, unused)
        self.voxel_map = VoxelMap() if cfg.ENABLE_ACCUMULATION else None
        # Real hardware only: median-filtered table XY footprint, used to
        # reject hand/arm points reaching in from outside the table (config §11).
        self._table_xy_samples = []         # rolling per-frame (lo, hi) during exploration
        self._table_xy_bounds = None        # (2,2): [[x_lo,y_lo],[x_hi,y_hi]], frozen at HOLD
        # Real hardware only: 1-shot EE-position scene cut (config §11), set
        # once by main_head.py before perception starts.
        self.ee_x_cutoff = None

    def set_ee_x_cutoff(self, x_cutoff):
        """Real hardware only: exclude everything nearer than max(EE_right.x,
        EE_left.x) + margin -- assumed to be the robot's own arm/hand, since
        the table is further forward. x_cutoff=None disables the cut."""
        self.ee_x_cutoff = x_cutoff

    @property
    def table_xy_bounds(self):
        """Current measured table XY gate: (2,2) [[x_lo,y_lo],[x_hi,y_hi]],
        or None before the first valid plane fit. For debug visualisation --
        this is the REAL (median-filtered) boundary, which can differ from
        the static cfg.TABLE_CENTER_BASE/TABLE_SIZE box drawn as a prior."""
        return self._table_xy_bounds

    def reset_table_footprint(self):
        """Re-arm: forget the frozen footprint so the next exploration phase
        re-derives it fresh (called alongside world_monitor.reset())."""
        self._table_xy_samples = []
        self._table_xy_bounds = None

    # ------------------------------------------------------------------ #
    # Frame transform                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _transform_to_base(points, R, t):
        """Apply the camera->base transform (R, t) to an (N,3) cloud."""
        return points @ R.T + t

    # ------------------------------------------------------------------ #
    # Crop                                                                #
    # ------------------------------------------------------------------ #
    def _crop(self, points, colors):
        """Keep only points inside the padded table box (base frame), plus
        (real hardware) forward of the 1-shot EE-position cutoff (config §11)
        -- applied here, before RANSAC even sees the data, so an arm/hand
        can't skew the plane fit either."""
        c = cfg.TABLE_CENTER_BASE
        half = cfg.TABLE_SIZE[:2] / 2.0 + cfg.CROP_MARGIN_XY
        x_min = c[0] - half[0]
        z_min = cfg.CROP_Z_MIN
        if cfg.REAL_HARDWARE_HEAD:
            z_min = cfg.REAL_CROP_Z_MIN     # table >0.5m tall -> drop ground early
            if self.ee_x_cutoff is not None:
                x_min = max(x_min, self.ee_x_cutoff)
        m = (
            (points[:, 0] > x_min) & (points[:, 0] < c[0] + half[0])
            & (points[:, 1] > c[1] - half[1]) & (points[:, 1] < c[1] + half[1])
            & (points[:, 2] > z_min) & (points[:, 2] < cfg.CROP_Z_MAX)
        )
        return points[m], colors[m]

    # ------------------------------------------------------------------ #
    # Main                                                                #
    # ------------------------------------------------------------------ #
    def process(self, points_optical, colors, R_cam_base, t_cam_base,
                allow_integrate=True, allow_track_update=True, explore_phase=0):
        """Run the full pipeline. Returns a PerceptionResult.

        R_cam_base, t_cam_base : the camera-optical -> base_footprint transform,
        looked up from TF at the depth frame's timestamp (correct frame + time).
        allow_integrate : (voxel-map only) fuse this frame's points — kept for
        the optional VoxelMap path; off by default.
        allow_track_update : fuse this frame's DETECTIONS into the object tracker
        — the caller passes False while the head is moving so only clean,
        settled-frame detections update the grow-only object estimates.
        explore_phase : LookAtController.phase (0 SWEEP/1 REFINE/2 HOLD). Real
        hardware only: the table XY footprint keeps growing while phase != 2
        (exploring, hands assumed absent/still per the caller's hypothesis),
        then freezes at HOLD so a hand returning later can't expand it.
        """
        import time
        t0 = time.perf_counter()
        res = PerceptionResult(n_raw=len(points_optical))

        # 1. Optical -> base, then crop to the table region.
        pts_base = self._transform_to_base(points_optical, R_cam_base, t_cam_base)
        pts_c, cols_c = self._crop(pts_base, colors)

        # 1b. MULTI-VIEW FUSION. Integrate this frame's cropped points into the
        # persistent voxel map ONLY when the head is settled (allow_integrate),
        # then run detection on the FUSED cloud. Fusing while moving would smear
        # the map; when not integrating we keep the map untouched (no decay) so
        # it stays crisp and stable during head motion.
        if self.voxel_map is not None:
            if allow_integrate:
                self.voxel_map.integrate(pts_c, cols_c)
            work_pts, work_cols = self.voxel_map.get_cloud()
            res.map_size = self.voxel_map.size()
        else:
            work_pts, work_cols = pts_c, cols_c

        res.cropped_points = work_pts          # what RViz shows = the live model
        res.cropped_colors = work_cols
        if len(work_pts) < cfg.PLANE_MIN_INLIERS:
            res.proc_ms = (time.perf_counter() - t0) * 1e3
            return res

        # 2. Table plane.
        plane, inlier_mask = self.segmenter.segment(work_pts)
        res.plane = plane
        if plane is None:
            res.proc_ms = (time.perf_counter() - t0) * 1e3
            return res

        # Debug: centroid of the plane inliers. If the cloud is correctly
        # placed this should sit near the known table centre (x~1.0, y~0.0).
        # Also derive the full table BOX (XY footprint from the inliers' spread,
        # top-Z from the plane) for the perceived-world snapshot / CBF.
        if inlier_mask is not None and inlier_mask.any():
            inliers = work_pts[inlier_mask]
            res.plane_centroid = inliers.mean(axis=0)
            res.table_center, res.table_size = table_box_from_inliers(
                inliers, plane.height)

        # 3. Above-plane slab = candidate objects.
        sd = plane.signed_distance(work_pts)
        above = (
            (sd > cfg.OBJECT_MIN_HEIGHT_ABOVE_PLANE)
            & (sd < cfg.OBJECT_MAX_HEIGHT_ABOVE_PLANE)
        )
        above_pts = work_pts[above]
        above_cols = work_cols[above]

        # 3b. Real hardware only: reject hand/arm points reaching in from
        # outside the table by gating XY to a median-filtered measured
        # footprint (see __init__/reset_table_footprint and config §11).
        if cfg.REAL_HARDWARE_HEAD and res.table_center is not None:
            half = res.table_size[:2] / 2.0
            lo, hi = res.table_center[:2] - half, res.table_center[:2] + half
            if explore_phase != 2:
                self._table_xy_samples.append((lo, hi))
                if len(self._table_xy_samples) > cfg.REAL_TABLE_FOOTPRINT_HISTORY:
                    self._table_xy_samples.pop(0)
            if self._table_xy_bounds is None or explore_phase != 2:
                los = np.array([s[0] for s in self._table_xy_samples])
                his = np.array([s[1] for s in self._table_xy_samples])
                self._table_xy_bounds = np.array(
                    [np.median(los, axis=0), np.median(his, axis=0)])
            m = cfg.REAL_TABLE_XY_GATE_MARGIN
            b_lo, b_hi = self._table_xy_bounds[0] - m, self._table_xy_bounds[1] + m
            in_xy = (
                (above_pts[:, 0] >= b_lo[0]) & (above_pts[:, 0] <= b_hi[0])
                & (above_pts[:, 1] >= b_lo[1]) & (above_pts[:, 1] <= b_hi[1])
            )
            above_pts = above_pts[in_xy]
            above_cols = above_cols[in_xy]

        res.above_points = above_pts
        res.above_colors = above_cols

        # 4. Cluster + fit + classify.
        detections = self.detector.detect(above_pts, above_cols, plane)

        # 5. Object-level temporal fusion (grow-only dims + persistence). Only
        # fuse when the head is settled so motion never corrupts the estimate.
        objects = self.tracker.update(detections, allow_update=allow_track_update)

        # 5b. Real hardware: force exactly the expected object count -- keep the
        # N highest-confidence tracks, drop the rest as spurious (config §11).
        if cfg.REAL_HARDWARE_HEAD and len(objects) > cfg.WORLD_EXPECTED_CYLINDERS:
            objects = sorted(objects, key=lambda o: o.confidence, reverse=True)
            objects = objects[:cfg.WORLD_EXPECTED_CYLINDERS]
        res.objects = objects

        res.proc_ms = (time.perf_counter() - t0) * 1e3
        return res
