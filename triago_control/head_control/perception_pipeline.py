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


@dataclass
class PerceptionResult:
    plane: object = None                    # PlaneModel or None
    objects: list = field(default_factory=list)     # list[DetectedObject]
    cropped_points: np.ndarray = None       # (N,3) base frame  (for viz)
    cropped_colors: np.ndarray = None       # (N,3) uint8
    above_points: np.ndarray = None         # (M,3) above-plane points (for viz)
    plane_centroid: np.ndarray = None       # (3,) centroid of plane inliers (debug)
    n_raw: int = 0
    map_size: int = 0                       # voxels in the fused map (0 if off)
    proc_ms: float = 0.0


class PerceptionPipeline:
    def __init__(self):
        self.segmenter = TableSegmenter()
        self.detector = ObjectDetector()
        self._tracked = []                  # persistent EMA-smoothed objects
        self.voxel_map = VoxelMap() if cfg.ENABLE_ACCUMULATION else None

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
    def process(self, points_optical, colors, R_cam_base, t_cam_base, allow_integrate=True):
        """Run the full pipeline. Returns a PerceptionResult.

        R_cam_base, t_cam_base : the camera-optical -> base_footprint transform,
        looked up from TF at the depth frame's timestamp (correct frame + time).
        allow_integrate : only fuse this frame into the voxel map when True
        (the caller passes False while the head is moving, to avoid smearing).
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
        if inlier_mask is not None and inlier_mask.any():
            res.plane_centroid = work_pts[inlier_mask].mean(axis=0)

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

        # 5. Temporal smoothing.
        res.objects = self._smooth(detections)

        res.proc_ms = (time.perf_counter() - t0) * 1e3
        return res

    # ------------------------------------------------------------------ #
    # Temporal EMA association                                            #
    # ------------------------------------------------------------------ #
    def _smooth(self, detections):
        """Associate each new detection to the nearest tracked object and EMA
        its centre/radius/height. Keeps poses steady despite per-frame noise.
        New objects are added; stale ones (unmatched this frame) are dropped.
        """
        a = cfg.DETECTION_EMA_ALPHA
        updated = []
        used_tracks = set()

        for det in detections:
            best_i, best_d = -1, cfg.DETECTION_MATCH_DIST
            for i, tr in enumerate(self._tracked):
                if i in used_tracks:
                    continue
                d = np.linalg.norm(tr.center - det.center)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                tr = self._tracked[best_i]
                used_tracks.add(best_i)
                det.center = a * det.center + (1 - a) * tr.center
                det.radius = a * det.radius + (1 - a) * tr.radius
                det.height = a * det.height + (1 - a) * tr.height
            updated.append(det)

        self._tracked = updated
        return updated
