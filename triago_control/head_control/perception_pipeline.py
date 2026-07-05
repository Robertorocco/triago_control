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
from triago_control.head_control.table_segmenter import TableSegmenter
from triago_control.head_control.object_detector import ObjectDetector, DetectedObject
from triago_control.head_control.voxel_map import VoxelMap
from triago_control.head_control.object_tracker import ObjectTracker
from triago_control.head_control.obb_detector import fit_oriented_box
from triago_control.head_control.obb_tracker import ObbTracker


@dataclass
class PerceptionResult:
    plane: object = None                    # PlaneModel or None
    objects: list = field(default_factory=list)     # list[DetectedObject]
    raw_detections: list = field(default_factory=list)  # list[DetectedObject], PRE-TRACKER
                                             # (this frame's fresh, unfiltered fit —
                                             # needed for any bias-vs-range diagnostic,
                                             # since the EMA-tracked `objects` above is
                                             # autocorrelated across frames and would
                                             # hide/smear a range-dependent trend)
    cropped_points: np.ndarray = None       # (N,3) base frame  (for viz)
    cropped_colors: np.ndarray = None       # (N,3) uint8
    above_points: np.ndarray = None         # (M,3) above-plane points (for viz)
    plane_centroid: np.ndarray = None       # (3,) centroid of plane inliers (debug)
    plane_bbox_center: np.ndarray = None    # (3,) bounding-box center of inliers (better estimate)
    table_extent_x: float = 0.0            # [m] observed X extent of plane inliers
    table_extent_y: float = 0.0            # [m] observed Y extent of plane inliers
    n_raw: int = 0
    map_size: int = 0                       # voxels in the fused map (0 if off)
    proc_ms: float = 0.0
    # --- OBB / memory-tracking estimator (config.py §14, OFF by default) ---
    # Populated only when cfg.ENABLE_OBB_ESTIMATOR is True; always empty/None
    # otherwise, so the default pipeline output is completely unaffected.
    obb_tracks: list = field(default_factory=list)     # list[OBBTrack]
    obb_primary_target: object = None                  # OBBTrack or None


class PerceptionPipeline:
    def __init__(self):
        self.segmenter = TableSegmenter()
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()     # object-level temporal fusion
        self._tracked = []                  # (legacy, unused)
        self.voxel_map = VoxelMap() if cfg.ENABLE_ACCUMULATION else None
        # OBB / memory-tracking estimator (config.py §14) — independent,
        # OFF-by-default, ported estimator run alongside the cylinder pipeline.
        self.obb_tracker = ObbTracker() if cfg.ENABLE_OBB_ESTIMATOR else None

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
    @staticmethod
    def _crop(points, colors):
        """Keep only points inside the padded table box (base frame)."""
        c = cfg.TABLE_CENTER_BASE
        half = cfg.TABLE_SIZE[:2] / 2.0 + cfg.CROP_MARGIN_XY
        m = (
            (points[:, 0] > c[0] - half[0]) & (points[:, 0] < c[0] + half[0])
            & (points[:, 1] > c[1] - half[1]) & (points[:, 1] < c[1] + half[1])
            & (points[:, 2] > cfg.CROP_Z_MIN) & (points[:, 2] < cfg.CROP_Z_MAX)
        )
        return points[m], colors[m]

    # ------------------------------------------------------------------ #
    # Main                                                                #
    # ------------------------------------------------------------------ #
    def process(self, points_optical, colors, R_cam_base, t_cam_base,
                allow_integrate=True, allow_track_update=True):
        """Run the full pipeline. Returns a PerceptionResult.

        R_cam_base, t_cam_base : the camera-optical -> base_footprint transform,
        looked up from TF at the depth frame's timestamp (correct frame + time).
        allow_integrate : (voxel-map only) fuse this frame's points — kept for
        the optional VoxelMap path; off by default.
        allow_track_update : fuse this frame's DETECTIONS into the object tracker
        — the caller passes False while the head is moving so only clean,
        settled-frame detections update the grow-only object estimates.
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
        # ALSO compute the BOUNDING-BOX CENTER (less biased than the mean when
        # the camera sees one side of the table more than the other — the mean
        # is pulled toward the dense/near side, while the bbox mid-point only
        # depends on the extreme points which exist on both sides if ANY
        # return reaches there). AND report the X/Y extent of the plane inliers
        # for intrinsics validation (should match TABLE_SIZE within the visible
        # portion).
        if inlier_mask is not None and inlier_mask.any():
            inlier_pts = work_pts[inlier_mask]
            res.plane_centroid = inlier_pts.mean(axis=0)
            # Bounding-box center: robust to asymmetric point density
            xyz_min = inlier_pts.min(axis=0)
            xyz_max = inlier_pts.max(axis=0)
            res.plane_bbox_center = (xyz_min + xyz_max) / 2.0
            res.table_extent_x = float(xyz_max[0] - xyz_min[0])
            res.table_extent_y = float(xyz_max[1] - xyz_min[1])

        # 3. Above-plane slab = candidate objects.
        sd = plane.signed_distance(work_pts)
        above = (
            (sd > cfg.OBJECT_MIN_HEIGHT_ABOVE_PLANE)
            & (sd < cfg.OBJECT_MAX_HEIGHT_ABOVE_PLANE)
        )
        above_pts = work_pts[above]
        above_cols = work_cols[above]
        res.above_points = above_pts

        # 4. Cluster + fit + classify.
        detections = self.detector.detect(above_pts, above_cols, plane)
        res.raw_detections = detections    # pre-tracker, this frame only (diagnostics)

        # 5. Object-level temporal fusion (grow-only dims + persistence). Only
        # fuse when the head is settled so motion never corrupts the estimate.
        res.objects = self.tracker.update(detections, allow_update=allow_track_update)

        # 6. OBB / memory-tracking estimator (config.py §14) — an independent,
        # shape-free estimator run on the SAME above-plane points/clusters,
        # gated OFF by default so the primary cylinder pipeline's output
        # (res.objects above) is completely unaffected either way. Reuses
        # ObjectDetector's own voxel-downsample + Euclidean clustering (single
        # source of truth for clustering, per this project's convention)
        # rather than re-implementing PCL's VoxelGrid + EuclideanClusterExtraction
        # a second time.
        if self.obb_tracker is not None:
            pts_ds, cols_ds = ObjectDetector._voxel_downsample(above_pts, above_cols, cfg.VOXEL_SIZE)
            clusters = ObjectDetector._euclidean_cluster(pts_ds)
            boxes = []
            for idx in clusters:
                box = fit_oriented_box(pts_ds[idx], cols_ds[idx])
                if box is not None:
                    boxes.append(box)
            table_surface_z = plane.height
            res.obb_tracks, res.obb_primary_target = self.obb_tracker.update(boxes, table_surface_z)

        res.proc_ms = (time.perf_counter() - t0) * 1e3
        return res
