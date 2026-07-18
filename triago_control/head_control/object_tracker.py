"""Object-level temporal fusion for the tabletop scene: each frame yields an independent detection
and only DERIVED quantities are fused (no point registration, so no cross-view error stacking).
Mechanisms: nearest-neighbour identity matching, EMA-smoothed position AND dimensions (EMA is the
correct fusion for an unbiased noisy per-frame estimate; a running max drifts upward), cumulative
arc coverage OR'd across frames, and persistence across brief dropouts."""

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
        self.vertical_coverage = float(det.vertical_coverage)
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

        # EMA on dimensions too: a grow-only rule on an unbiased noisy signal drifts upward.
        self.radius = a * float(det.radius) + (1.0 - a) * self.radius

        # Height is QUALITY-GATED, not blind EMA (see cfg.ENABLE_HEIGHT_QUALITY_GATE):
        # only a frame whose vertical coverage is within HEIGHT_FUSE_VCOV_MARGIN of
        # the best seen so far may move height, so a poor (oblique/foreshortened)
        # POV can't drag a good close-view height back toward its bias. Gate reads
        # self.vertical_coverage BEFORE its max-update below (= best-before-this-frame).
        det_vcov = float(det.vertical_coverage)
        if (not cfg.ENABLE_HEIGHT_QUALITY_GATE
                or det_vcov >= self.vertical_coverage - cfg.HEIGHT_FUSE_VCOV_MARGIN):
            self.height = a * float(det.height) + (1.0 - a) * self.height

        # Cumulative angular coverage across viewpoints.
        if det.arc_bins is not None:
            self.arc_bins = self.arc_bins | det.arc_bins
        # Keep the best (lowest) fit residual ever seen for this object.
        self.best_fit_rms = min(self.best_fit_rms, float(det.fit_rms))
        # Best (highest) vertical coverage seen -- more of the column observed.
        self.vertical_coverage = max(self.vertical_coverage, det_vcov)

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
