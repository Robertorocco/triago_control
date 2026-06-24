"""
ObjectTracker — object-level temporal fusion (inspired by a colleague's PCL
tabletop node) adapted for our cylinder + colour scenario.

WHY object-level (not point-level) fusion:
    Our earlier voxel-map fusion failed: stacking raw points from different head
    poses accumulated depth noise + sub-degree extrinsic error into a smeared
    blob ("layered heights", NO TABLE). Fusing at the OBJECT level avoids this
    entirely — each frame yields an independent, clean detection and we only
    combine the DERIVED quantities. No point registration => no error stacking
    => works even while the head moves.

KEY MECHANISMS (borrowed + adapted):
    * Nearest-neighbour matching (2D, within TRACK_MATCH_DIST) gives each object
      a stable identity across frames.
    * GROW-ONLY dimensions (radius, height): a partial view can only ever make
      the estimate BIGGER, never smaller. This (a) fixes the circle-fit radius
      under-estimate on partial arcs and (b) is conservative for collision
      safety. A slow decay lets an over-grown estimate recover from a bad frame.
    * Cumulative arc coverage: we OR the observed angular sectors across frames,
      so coverage (and hence confidence) climbs toward 100% as more of the
      object is seen from different viewpoints — the honest multi-view gain.
    * Persistence: an unmatched object survives TRACK_MAX_UNSEEN frames before
      deletion, so brief occlusions/dropouts don't make it flicker.

    Position is EMA-smoothed (stable) rather than grow-only — averaging
    viewpoints with opposite-sign partial-view bias actually reduces net bias.
"""

import numpy as np

import triago_control.head_control.config as cfg


class TrackedObject:
    _N_BINS = 36

    def __init__(self, tid, det):
        self.id = tid
        self.color_name = det.color_name
        self.center = det.center.astype(float).copy()
        self.radius = float(det.radius)
        self.height = float(det.height)
        self.axis = det.axis.astype(float).copy()
        self.mean_rgb = det.mean_rgb.astype(float).copy()
        self.n_points = int(det.n_points)
        self.arc_bins = (
            det.arc_bins.copy() if det.arc_bins is not None
            else np.zeros(self._N_BINS, dtype=bool)
        )
        self.best_fit_rms = float(det.fit_rms)
        self.frames_unseen = 0
        self.matched = False

    # --- Derived, tracker-level quality ------------------------------- #
    @property
    def arc_coverage(self) -> float:
        return float(self.arc_bins.mean())

    @property
    def confidence(self) -> float:
        cov = self.arc_coverage
        q = float(np.exp(-self.best_fit_rms / 0.005))   # 1 at 0mm, ~0.14 at 1cm
        return float(np.clip(cov * q, 0.0, 1.0))

    @property
    def label(self) -> str:
        return (f"{self.color_name}_cylinder"
                if self.color_name != "unknown" else "unknown_object")

    # --- Fuse a fresh detection into this track ----------------------- #
    def fuse(self, det):
        self.matched = True
        self.frames_unseen = 0

        a = cfg.TRACK_POS_ALPHA
        self.center = a * det.center + (1.0 - a) * self.center
        self.axis = det.axis.astype(float)

        # GROW-ONLY with slow decay (so a transient over-grow can recover).
        dec = cfg.TRACK_DIM_DECAY
        self.radius = max(float(det.radius), self.radius * (1.0 - dec))
        self.height = max(float(det.height), self.height * (1.0 - dec))

        # Cumulative angular coverage across viewpoints.
        if det.arc_bins is not None:
            self.arc_bins = self.arc_bins | det.arc_bins
        # Keep the best (lowest) fit residual ever seen for this object.
        self.best_fit_rms = min(self.best_fit_rms, float(det.fit_rms))

        if det.color_name != "unknown":
            self.color_name = det.color_name
            self.mean_rgb = det.mean_rgb.astype(float)
        self.n_points = int(det.n_points)


class ObjectTracker:
    def __init__(self):
        self._objs = []
        self._next_id = 0

    def active(self):
        """All currently-alive tracks (including briefly-unseen ones)."""
        return list(self._objs)

    def update(self, detections, allow_update=True):
        """Match detections to tracks, fuse, age, and prune.

        allow_update=False (e.g. while the head is moving) returns the current
        tracks untouched — we only fuse clean, settled-frame detections.
        """
        if not allow_update:
            return self.active()

        for o in self._objs:
            o.matched = False

        # --- Match each detection to the nearest unused track (2D) -----
        for det in detections:
            best, best_d = None, cfg.TRACK_MATCH_DIST
            for o in self._objs:
                if o.matched:
                    continue
                d = float(np.linalg.norm(o.center[:2] - det.center[:2]))
                if d < best_d:
                    best_d, best = d, o
            if best is not None:
                best.fuse(det)
            else:
                self._objs.append(TrackedObject(self._next_id, det))
                self._next_id += 1

        # --- Age unmatched tracks; prune the long-unseen ----------------
        alive = []
        for o in self._objs:
            if not o.matched:
                o.frames_unseen += 1
            if o.frames_unseen <= cfg.TRACK_MAX_UNSEEN:
                alive.append(o)
        self._objs = alive
        return self.active()
