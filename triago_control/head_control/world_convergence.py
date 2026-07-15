"""
WorldConvergenceMonitor -- freezes the fused perception estimate into a
static QP-CLF-CBF collision world once it has stopped moving.

Convergence requires, on settled (head-still) frames only:
  (a) every object above WORLD_CONF_MIN and past the arc-coverage floor
      WORLD_MIN_ARC_COVERAGE, and
  (b) low oscillation: over a sliding WORLD_CONVERGE_WINDOW_S window, each
      object's center/radius/height stays within WORLD_CONVERGE_POS_AMP /
      WORLD_CONVERGE_DIM_AMP of the window mean.

An amplitude band over time is frame-rate independent and needs no ground
truth. The window only accumulates during a continuous settled hold; any
moving frame re-seeds it. The table box is recomputed each frame and is
EMA-smoothed here so its drift is a meaningful stability signal too.

A converged monitor stays converged until reset() re-arms it.
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
        self._buf = []                   # rolling window of settled samples (see _append)
        self._prev = None                # last settled {color: (center, radius, height)} (rate telemetry)
        self._prev_time = None           # timestamp [s] of the previous SETTLED frame
        self._table_ema = None           # [center(3), size(3)] EMA of the table box
        self._max_drift_rate = float("nan")   # last measured centre drift rate [m/s] (telemetry)
        self._progress = 0.0             # fraction [0..1] of the amplitude window accrued
        self._converged = False

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def max_drift_rate(self) -> float:
        """Most recent object-centre drift rate [m/s] (NaN before the first pair
        of settled frames). Kept as a GT-free telemetry signal — the FREEZE gate is
        the oscillation-amplitude lock below, not this rate."""
        return self._max_drift_rate

    def converge_progress(self, now_sec=None) -> float:
        """Fraction [0..1] of the amplitude window currently held. 1.0 once frozen;
        0.0 whenever the coverage floor or the <1mm amplitude band is not met. The
        stored value is clock-independent (computed in update), so callers on any
        clock read the same fraction."""
        return 1.0 if self._converged else self._progress

    # ------------------------------------------------------------------ #
    # Main                                                                #
    # ------------------------------------------------------------------ #
    def update(self, result, allow_update=True, stamp_sec=None):
        """Feed one PerceptionResult. Return a PerceivedWorld the FIRST tick it
        converges, else None (and None on every tick after convergence, until
        reset()).

        allow_update : head-still gate. A moving frame (False) or a missing
        timestamp PAUSES and clears the amplitude window, so head-motion time is
        never credited as stability — the window must accrue during one continuous
        settled hold.

        Freeze criterion: every expected object is confident + above the arc-
        coverage floor, AND its centre stays inside a WORLD_CONVERGE_POS_AMP band
        (peak deviation from the window mean) with radius/height inside
        WORLD_CONVERGE_DIM_AMP, for a full WORLD_CONVERGE_WINDOW_S. Amplitude — not
        a per-frame rate — is what "the oscillation shrank below 1mm" actually means.
        """
        if self._converged:
            return None

        if not allow_update or stamp_sec is None:
            # Head moving (or no clock): pause + clear the window.
            self._buf = []
            self._prev = None
            self._prev_time = None
            self._progress = 0.0
            return None

        # Gate: need a table plane + a derived table box this frame.
        if (result.plane is None or result.table_center is None
                or result.table_size is None):
            self._buf = []
            self._progress = 0.0
            return None

        # EMA-smooth the (per-frame, noisy) table box so the frozen footprint is
        # denoised like the cylinders.
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
            cov = float(getattr(o, "arc_coverage", 0.0))
            if cname not in current or conf > current[cname][3]:
                current[cname] = [np.asarray(o.center, dtype=float).copy(),
                                  float(o.radius), float(o.height), conf, cov]

        # Drift-rate telemetry (last two settled frames) — informational only.
        self._update_drift_rate(current, stamp_sec)
        self._prev = {c: (v[0], v[1], v[2]) for c, v in current.items()}
        self._prev_time = stamp_sec

        # (a) Coverage/confidence floor: every expected object present, confident,
        # AND geometrically well-observed before the amplitude lock can arm.
        qualified = (
            len(current) >= cfg.WORLD_EXPECTED_CYLINDERS
            and all(v[3] >= cfg.WORLD_CONF_MIN for v in current.values())
            and all(v[4] >= cfg.WORLD_MIN_ARC_COVERAGE for v in current.values()))
        if not qualified:
            self._buf = []
            self._progress = 0.0
            return None

        # (b) Push into the rolling window and test the amplitude lock.
        self._buf.append((stamp_sec, {c: (v[0], v[1], v[2]) for c, v in current.items()}))
        while self._buf and (stamp_sec - self._buf[0][0]) > cfg.WORLD_CONVERGE_WINDOW_S:
            self._buf.pop(0)

        stable = self._amplitude_ok()
        timespan = self._buf[-1][0] - self._buf[0][0] if len(self._buf) >= 2 else 0.0
        self._progress = min(1.0, timespan / cfg.WORLD_CONVERGE_WINDOW_S) if stable else 0.0

        if stable and timespan >= cfg.WORLD_CONVERGE_WINDOW_S:
            self._converged = True
            return self._freeze(current)
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _update_drift_rate(self, current, stamp_sec):
        """Max object-centre drift rate [m/s] vs the previous settled frame, stored
        for telemetry. NaN when there is no comparable prior."""
        if self._prev is None or self._prev_time is None \
                or set(current.keys()) != set(self._prev.keys()):
            self._max_drift_rate = float("nan")
            return
        dt = stamp_sec - self._prev_time
        if dt <= 1e-4:
            return
        rate = 0.0
        for c, (center, _r, _h, _conf, _cov) in current.items():
            pcenter = self._prev[c][0]
            rate = max(rate, float(np.linalg.norm(center - pcenter)) / dt)
        self._max_drift_rate = rate

    def _amplitude_ok(self) -> bool:
        """True if, over the whole current window, every object's centre stays
        within WORLD_CONVERGE_POS_AMP (peak deviation from the window mean) and its
        radius/height within WORLD_CONVERGE_DIM_AMP. Requires a consistent object
        set across the window (a mid-window identity change is not 'stable')."""
        if len(self._buf) < 2:
            return False
        keys = set(self._buf[-1][1].keys())
        for _t, cyl in self._buf:
            if set(cyl.keys()) != keys:
                return False
        for c in keys:
            centers = np.array([cyl[c][0] for _t, cyl in self._buf])
            if np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) > cfg.WORLD_CONVERGE_POS_AMP:
                return False
            radii = np.array([cyl[c][1] for _t, cyl in self._buf])
            heights = np.array([cyl[c][2] for _t, cyl in self._buf])
            if np.max(np.abs(radii - radii.mean())) > cfg.WORLD_CONVERGE_DIM_AMP:
                return False
            if np.max(np.abs(heights - heights.mean())) > cfg.WORLD_CONVERGE_DIM_AMP:
                return False
        return True

    def _freeze(self, current) -> PerceivedWorld:
        """Snapshot the current estimate into an immutable PerceivedWorld."""
        cyls = [
            PerceivedCylinder(color_name=cname, center=center.copy(),
                              radius=radius, height=height, confidence=conf)
            for cname, (center, radius, height, conf, _cov) in current.items()
        ]
        # Deterministic order (red, then blue) so marker IDs are stable.
        order = {"red": 0, "blue": 1}
        cyls.sort(key=lambda c: order.get(c.color_name, 99))
        return PerceivedWorld(
            table_center=self._table_ema[0].copy(),
            table_size=self._table_ema[1].copy(),
            cylinders=cyls,
        )
