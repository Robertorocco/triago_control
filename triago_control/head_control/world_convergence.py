"""
WorldConvergenceMonitor — decide WHEN the fused perception estimate is
"confident" enough to freeze into a static QP-CLF-CBF collision world.

WHY THIS EXISTS (the "confident concept")
    The perception pipeline reports a soft, monotonically-climbing per-object
    confidence (arc_coverage x fit-quality, see object_tracker.py) but NO event
    that says "the estimate has settled — hand it to the safety controller".
    Thresholding that soft confidence alone is not enough: confidence can be high
    while the pose is still drifting over the first few fused frames. What we
    actually want, before we build a STATIC collision barrier the arms must trust,
    is that the geometry has STOPPED MOVING.

    So convergence here is a two-part test, evaluated only on SETTLED frames
    (the same head-still gate the tracker uses):
        (a) every expected cylinder is above WORLD_CONF_MIN, AND
        (b) the fused geometry's per-frame drift stays below the position/dimension
            tolerances for WORLD_STABLE_FRAMES consecutive settled frames.
    Only then do we FREEZE the current estimate and emit it once. The QP side then
    builds its collision model from that frozen snapshot exactly once — no dynamic
    per-tick CBF updates (a deliberate design choice; see qp_controller/
    perceived_world_builder.py).

    Cylinders arrive already EMA-fused (object_tracker.py), so their per-frame
    drift after settling is small. The table BOX is computed fresh each frame from
    raw plane inliers (table_segmenter.table_box_from_inliers), so it is noisier —
    we EMA-smooth it HERE (same TRACK_POS_ALPHA) so the frozen footprint is
    denoised the same way the cylinders are, and so its drift is a meaningful
    stability signal.

    A converged monitor stays converged (emits nothing further) until reset() —
    so main_head can re-arm it (e.g. on a /perceived_world/rescan request) to
    re-observe and re-publish.
"""

from dataclasses import dataclass, field

import numpy as np

import triago_control.head_control.config as cfg


@dataclass
class PerceivedCylinder:
    """One frozen, confident cylinder estimate (base_footprint)."""
    color_name: str                                     # "red" | "blue"
    center: np.ndarray                                  # (3,) mid-height centre
    radius: float                                       # [m]
    height: float                                       # [m]
    confidence: float = 0.0                             # [0..1] at freeze time


@dataclass
class PerceivedWorld:
    """A frozen, confident snapshot of the tabletop scene for the CBF."""
    table_center: np.ndarray                            # (3,) box centre
    table_size: np.ndarray                              # (3,) box full extents
    cylinders: list = field(default_factory=list)       # [PerceivedCylinder]


class WorldConvergenceMonitor:
    """Track fused perception across settled frames; fire ONCE on convergence."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    # State                                                               #
    # ------------------------------------------------------------------ #
    def reset(self):
        """Re-arm: forget all progress so the next stable run re-converges."""
        self._stable_frames = 0
        self._prev_cyl = None            # {color: [center, radius, height, conf]}
        self._table_ema = None           # [center(3), size(3)] EMA of the table box
        self._prev_table_ema = None      # previous-frame copy, for the drift check
        self._converged = False

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def stable_frames(self) -> int:
        """How many consecutive stable settled frames so far (for telemetry)."""
        return self._stable_frames

    # ------------------------------------------------------------------ #
    # Main                                                                #
    # ------------------------------------------------------------------ #
    def update(self, result, allow_update=True):
        """Feed one PerceptionResult. Return a PerceivedWorld the FIRST tick it
        converges, else None (and None on every tick after convergence, until
        reset()). `allow_update` is the head-still gate — stability is only ever
        evaluated on settled frames, so head motion cannot spuriously reset or
        advance the counter.
        """
        if self._converged or not allow_update:
            return None

        # Gate: need a table plane + a derived table box this frame.
        if (result.plane is None or result.table_center is None
                or result.table_size is None):
            self._stable_frames = 0
            return None

        # EMA-smooth the (per-frame, noisy) table box so the frozen footprint is
        # denoised like the cylinders and its drift is a meaningful signal.
        tc = np.asarray(result.table_center, dtype=float)
        ts = np.asarray(result.table_size, dtype=float)
        a = cfg.TRACK_POS_ALPHA
        if self._table_ema is None:
            self._table_ema = [tc.copy(), ts.copy()]
        else:
            self._table_ema[0] = a * tc + (1.0 - a) * self._table_ema[0]
            self._table_ema[1] = a * ts + (1.0 - a) * self._table_ema[1]

        # Qualifying coloured cylinders (keep the most-confident per colour).
        current = {}
        for o in result.objects:
            cname = getattr(o, "color_name", "unknown")
            if cname not in ("red", "blue"):
                continue
            conf = float(getattr(o, "confidence", 0.0))
            if cname not in current or conf > current[cname][3]:
                current[cname] = [np.asarray(o.center, dtype=float).copy(),
                                  float(o.radius), float(o.height), conf]

        qualified = (len(current) >= cfg.WORLD_EXPECTED_CYLINDERS
                     and all(v[3] >= cfg.WORLD_CONF_MIN for v in current.values()))

        if not qualified:
            # Not enough confident cylinders yet — hold at zero but keep the
            # table EMA warming up so it is ready once they appear.
            self._stable_frames = 0
            self._prev_cyl = current
            self._prev_table_ema = [self._table_ema[0].copy(), self._table_ema[1].copy()]
            return None

        if self._is_stable(current):
            self._stable_frames += 1
        else:
            self._stable_frames = 1          # first frame of a fresh stable run

        self._prev_cyl = current
        self._prev_table_ema = [self._table_ema[0].copy(), self._table_ema[1].copy()]

        if self._stable_frames >= cfg.WORLD_STABLE_FRAMES:
            self._converged = True
            return self._freeze(current)
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _is_stable(self, current) -> bool:
        """True if every cylinder AND the smoothed table moved less than the
        tolerances since the previous settled frame."""
        if self._prev_cyl is None or self._prev_table_ema is None:
            return False
        if set(current.keys()) != set(self._prev_cyl.keys()):
            return False
        for c, (center, radius, height, _conf) in current.items():
            pcenter, pradius, pheight, _ = self._prev_cyl[c]
            if np.linalg.norm(center - pcenter) > cfg.WORLD_STABLE_POS_TOL:
                return False
            if abs(radius - pradius) > cfg.WORLD_STABLE_DIM_TOL:
                return False
            if abs(height - pheight) > cfg.WORLD_STABLE_DIM_TOL:
                return False
        if np.linalg.norm(self._table_ema[0] - self._prev_table_ema[0]) > cfg.WORLD_STABLE_POS_TOL:
            return False
        if np.linalg.norm(self._table_ema[1] - self._prev_table_ema[1]) > cfg.WORLD_STABLE_DIM_TOL:
            return False
        return True

    def _freeze(self, current) -> PerceivedWorld:
        """Snapshot the current estimate into an immutable PerceivedWorld."""
        cyls = [
            PerceivedCylinder(color_name=cname, center=center.copy(),
                              radius=radius, height=height, confidence=conf)
            for cname, (center, radius, height, conf) in current.items()
        ]
        # Deterministic order (red, then blue) so marker IDs are stable.
        order = {"red": 0, "blue": 1}
        cyls.sort(key=lambda c: order.get(c.color_name, 99))
        return PerceivedWorld(
            table_center=self._table_ema[0].copy(),
            table_size=self._table_ema[1].copy(),
            cylinders=cyls,
        )
