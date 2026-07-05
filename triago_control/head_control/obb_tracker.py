"""
obb_tracker.py — position-matched, grow-only-dimension memory tracker.

PROVENANCE
    Ported from the same colleague's C++ `tabletop_perception_node` as
    `obb_detector.py` — see config.py §14 for the full rationale/caveats.
    This module reproduces the node's `TrackedObject` struct and its
    per-tick matching/fusion/persistence loop as faithfully as possible in
    Python, INCLUDING the specific fusion policy choices the C++ node makes
    (position/orientation are REPLACED each match, not EMA'd; dimensions are
    fused with an elementwise MAX, i.e. "can only grow, never shrink").

    This is a DIFFERENT fusion philosophy than `object_tracker.ObjectTracker`
    (which explicitly moved AWAY from grow-only to EMA for radius/height, see
    that module's 2026-07-02 docstring note — grow-only was quietly
    compensating for an since-fixed under-estimation bug in the OLD circle
    fit, and drifts upward on an unbiased signal). It is preserved here
    UNCHANGED because the goal of this port is to faithfully reproduce the
    colleague's own architecture for direct comparison, not to "fix" it with
    this project's own conventions.

MATCHING / FUSION LOOP  (mirrors `cloudCallback`'s tracking block exactly):
    1. Clear `matched_this_frame` on every existing track.
    2. For each new per-cluster `OrientedBox` this frame:
         - find the nearest EXISTING track (by 3D position distance) within
           `cfg.OBB_MATCH_DIST`;
         - if found: REPLACE its position/rotation with this frame's fit,
           grow its dimensions elementwise (`max(old, new)` per axis), reset
           `frames_unseen=0`, mark `matched_this_frame=True`;
         - else: create a brand-new track.
    3. For every track NOT matched this frame: `frames_unseen += 1`.
    4. Drop any track whose `frames_unseen >= cfg.OBB_MAX_UNSEEN_FRAMES`.
    5. Classify each surviving track `is_obstacle = position.z < table_surface_z`
       (mirrors the C++ node's `it->is_obstacle = (it->position.z() <
       table_surface_z)` — table_surface_z is the detected plane height, so
       an object sitting ON TOP of the table has `position.z` at roughly its
       own half-height ABOVE the table, i.e. normally `>= table_surface_z`;
       true obstacles in the original node are points BELOW the segmented
       table top, e.g. objects on the floor/a lower shelf).
    6. The "primary target" is the first non-obstacle track with
       `frames_unseen == 0` encountered in track order (mirrors the C++
       node's `!primary_target_published && it->frames_unseen == 0` gate,
       which publishes at most one `target_pose` per tick).
"""

from dataclasses import dataclass, field

import numpy as np

import triago_control.head_control.config as cfg
from triago_control.head_control.obb_detector import OrientedBox


@dataclass
class OBBTrack:
    """Persistent track state — mirrors the C++ `TrackedObject` struct."""
    id: int
    position: np.ndarray
    rotation: np.ndarray                     # (3,3)
    dimensions: np.ndarray                   # (3,) grows monotonically
    frames_unseen: int = 0
    matched_this_frame: bool = False
    is_obstacle: bool = False
    mean_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_points: int = 0


class ObbTracker:
    """Stateful, faithful port of the C++ node's tracking block.

    One instance persists for the lifetime of the perception node (mirrors
    `tracked_objects_` + `next_marker_id_` as member state on the C++ node).
    """

    def __init__(self):
        self._tracks = []
        self._next_id = 0

    def active(self):
        """All currently-alive tracks (including briefly-unseen ones)."""
        return list(self._tracks)

    def update(self, boxes, table_surface_z: float):
        """Match this frame's OrientedBox fits to tracks, fuse, age, prune,
        and classify. Returns (tracks, primary_target) where `primary_target`
        is an `OBBTrack` or `None` (mirrors the C++ node's single
        `target_pose` publish-per-tick gate).

        Parameters
        ----------
        boxes            : list[OrientedBox], this frame's fresh per-cluster fits
        table_surface_z  : float, the detected table-plane height (base_footprint)
        """
        for t in self._tracks:
            t.matched_this_frame = False

        for box in boxes:
            best_track = None
            best_dist = cfg.OBB_MATCH_DIST
            for t in self._tracks:
                d = float(np.linalg.norm(t.position - box.position))
                if d < best_dist:
                    best_dist = d
                    best_track = t

            if best_track is not None:
                # REPLACE position/orientation (not EMA'd — see module
                # docstring: this is a deliberate faithful-port choice).
                best_track.position = box.position
                best_track.rotation = box.rotation
                # GROW-ONLY dimensions: elementwise max, never shrinks from
                # occlusion (mirrors the C++ node's std::max per axis).
                best_track.dimensions = np.maximum(best_track.dimensions, box.dimensions)
                best_track.frames_unseen = 0
                best_track.matched_this_frame = True
                best_track.mean_rgb = box.mean_rgb
                best_track.n_points = box.n_points
            else:
                new_track = OBBTrack(
                    id=self._next_id,
                    position=box.position,
                    rotation=box.rotation,
                    dimensions=box.dimensions.copy(),
                    frames_unseen=0,
                    matched_this_frame=True,
                    mean_rgb=box.mean_rgb,
                    n_points=box.n_points,
                )
                self._tracks.append(new_track)
                self._next_id += 1

        # --- Age + prune -----------------------------------------------
        alive = []
        for t in self._tracks:
            if not t.matched_this_frame:
                t.frames_unseen += 1
            if t.frames_unseen < cfg.OBB_MAX_UNSEEN_FRAMES:
                alive.append(t)
        self._tracks = alive

        # --- Classify + find the primary target -------------------------
        primary_target = None
        for t in self._tracks:
            t.is_obstacle = bool(t.position[2] < table_surface_z)
            if primary_target is None and not t.is_obstacle and t.frames_unseen == 0:
                primary_target = t

        return self.active(), primary_target
