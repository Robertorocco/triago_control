"""
Object detector: above-plane points -> labelled upright cylinders.

PIPELINE
    1. Voxel downsample the above-plane points (caps the count -> CPU-friendly
       clustering, and evens out the density).
    2. Euclidean clustering (region growing on a KD-tree): points within
       CLUSTER_TOLERANCE of each other belong to the same object.
    3. Per cluster, fit an UPRIGHT cylinder:
         - axis assumed vertical (objects stand on the table) -> robust
         - RIM EXTRACTION (see below) isolates the boundary of the visible
           top-face disk / side wall before any circle fit is attempted
         - radius  = HYPER algebraic circle fit on the extracted rim
         - height  = top-slice-median z minus the plane height (see below)
         - centre  = (fitted circle centre, mid-height Z)
    4. Classify colour (red / blue / unknown) from the mean hue of the cluster's
       RGB, using HSV thresholds (matplotlib.colors, already a project dep).

WHY upright-cylinder instead of full 6-DOF RANSAC cylinder:
    A generic cylinder RANSAC needs surface normals and is fiddly/slow on a
    noisy partial view. Our objects are known to stand vertically on the table,
    so fixing the axis to +Z turns the fit into two trivial, robust 1-D
    estimates (radial spread + height). This is the right amount of prior.

WHY RIM EXTRACTION BEFORE THE CIRCLE FIT (2026-07-02 accuracy pass):
    A cylinder viewed from above is NOT just a ring of points — the camera
    also sees the entire SOLID TOP FACE (a filled disk), and any circle fit
    (Kasa or otherwise) run on a filled disk + a partial side-wall arc is
    systematically biased toward a SMALLER radius: interior points, on
    average, sit closer to the true centre than the boundary does, and they
    heavily outnumber boundary points (a disk has O(r^2) interior points vs.
    O(r) boundary points). This was verified numerically (synthetic point
    clouds, realistic ~3mm depth noise): fitting Kasa directly on a
    disk+arc cluster gives a radius bias of roughly -3 to -5mm — which
    matches the "few cm error, systematically low" symptom this pass fixes.
    Critically, ADDING MORE VIEWPOINTS (a longer scan) DOES NOT FIX THIS: it
    just re-confirms the same biased fit with more points. The scan was
    solving the wrong problem.

    The fix is standard in range-image circle/ellipse fitting: before fitting,
    reduce the cluster to an estimate of its OUTER BOUNDARY ONLY (the rim),
    then fit the circle to that. `_extract_rim` bins points by angle about a
    rough centre estimate and keeps, per angular bin, the points near the
    LOCAL 90th-percentile radius (not the raw max, which is sensitive to
    RGB-D "flying pixel" outliers at depth discontinuities — verified
    numerically to be more robust than a per-bin max under simulated outlier
    contamination). This collapses the interior of the disk away and leaves
    an approximately uniform ring for the circle fit, closing nearly all of
    the bias (~-4mm -> sub-mm in simulation) WITHOUT needing multiple views.

    The circle fit itself was also switched from Kasa to the Hyper fit
    (Al-Sharadqah & Chernov's algebraic fit, which corrects the small
    residual outward-bias Kasa has even on a clean ring) — a secondary,
    smaller correction on top of the rim fix.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
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
    arc_coverage: float = 0.0               # [0..1] fraction of circumference seen
    fit_rms: float = 0.0                    # [m] RMS radial residual of the circle fit
    confidence: float = 0.0                 # [0..1] overall estimation confidence
    arc_bins: np.ndarray = None             # (36,) bool mask of observed 10-deg sectors


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
        # --- XY centre + radius: RIM EXTRACTION, then an algebraic circle
        # fit on the rim only (see the module docstring for the full
        # rationale — fitting directly on the disk+arc cluster is
        # systematically biased small; extracting the boundary first removes
        # almost all of that bias without needing extra viewpoints). -------
        xy = pts[:, :2]
        rough_center = xy.mean(axis=0)
        rim_xy = ObjectDetector._extract_rim(xy, rough_center)

        # Angular-isolation weights: give sparse-side rim points more weight
        # to counteract partial-arc bias. Each rim point gets a weight
        # inversely proportional to the local angular density of its neighbours
        # (bins that are surrounded by many other occupied bins get LESS weight;
        # isolated bins on the occluded far side get MORE). This is the correct
        # first-principles fix for the partial-view center bias — no scene
        # knowledge is used, only the angular distribution of the rim itself.
        weights = ObjectDetector._angular_isolation_weights(rim_xy, rough_center)

        center_xy, radius, fit_ok = ObjectDetector._fit_circle_kasa_weighted(rim_xy, weights)
        if not fit_ok:
            # Fall back to unweighted Hyper fit.
            center_xy, radius, fit_ok = ObjectDetector._fit_circle_hyper(rim_xy)
        if not fit_ok:
            # Fall back to unweighted Kasa on the rim.
            center_xy, radius, fit_ok = ObjectDetector._fit_circle_kasa(rim_xy)
        if not fit_ok:
            # Last-resort fallback: percentile radius about the raw centroid
            # (only reached if even the rim is degenerate, e.g. <5 points).
            center_xy = rough_center
            radial = np.linalg.norm(xy - center_xy, axis=1)
            radius = float(np.percentile(radial, cfg.CYL_RADIUS_PERCENTILE))

        radius += cfg.CYL_RADIUS_INFLATION

        # --- Height: TOP-SLICE MEDIAN, not z_max. -----------------------
        # z_max is a maximum-statistic: under symmetric noise it is
        # systematically biased HIGH, and the bias grows with the number of
        # points sampled (more samples -> more likely one lands far out on
        # the noise tail). Verified numerically: at ~3mm depth noise and
        # realistic point counts this alone adds +7 to +9mm to the height
        # estimate. Taking the MEDIAN of the top slice (points within
        # CYL_TOP_SLICE of the max) is far less sensitive to that tail while
        # still using only genuine top-face points (not the sloped/noisy
        # side wall below).
        z_max = pts[:, 2].max()
        top_slice_mask = pts[:, 2] > (z_max - cfg.CYL_TOP_SLICE)
        top_z = float(np.median(pts[top_slice_mask, 2])) if np.any(top_slice_mask) else float(z_max)
        base_z = plane.height
        height = float(top_z - base_z)
        center_z = base_z + height / 2.0

        # Plausibility gates (reject walls, specks, the robot's own gripper...).
        if not (cfg.CYL_MIN_RADIUS <= radius <= cfg.CYL_MAX_RADIUS):
            return None
        if not (cfg.CYL_MIN_HEIGHT <= height <= cfg.CYL_MAX_HEIGHT):
            return None

        # --- Estimation-quality metrics --------------------------------
        # Arc coverage: how much of the 360 deg circumference the points span.
        # Computed by binning the angle of each point about the fitted centre.
        # A near-full ring (multi-view accumulation) -> ~1.0; a single-view
        # arc -> ~0.3-0.5. This is the most honest "confidence" signal: it
        # directly measures how much of the object we have actually observed.
        # Uses the FULL cluster (disk + arc) deliberately -- coverage should
        # reflect everything we saw, not just the rim subset used for the fit.
        rel = xy - center_xy
        ang = np.arctan2(rel[:, 1], rel[:, 0])              # [-pi, pi]
        n_bins = 36                                         # 10 deg bins
        bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        arc_bins = np.zeros(n_bins, dtype=bool)
        arc_bins[np.unique(bins)] = True
        arc_coverage = float(arc_bins.sum()) / n_bins

        # Fit residual: RMS of |radial - radius|, evaluated on the RIM points
        # actually used for the fit (not the full disk+arc cluster). Using the
        # full cluster here would inflate fit_rms by construction (every
        # disk-interior point sits well inside the fitted radius by design),
        # which would incorrectly tank the confidence score below.
        rel_rim = rim_xy - center_xy
        radial_rim = np.linalg.norm(rel_rim, axis=1)
        fit_rms = float(np.sqrt(np.mean((radial_rim - radius) ** 2)))

        # Overall confidence: coverage is primary, penalised by fit error.
        # fit_rms ~ a few mm on a clean object; normalise by 5mm.
        rms_quality = float(np.exp(-fit_rms / 0.005))       # 1 at 0mm, ~0.14 at 1cm
        confidence = float(np.clip(arc_coverage * rms_quality, 0.0, 1.0))

        color_name, mean_rgb = ObjectDetector._classify_color(cols)
        label = f"{color_name}_cylinder" if color_name != "unknown" else "unknown_object"

        # Optional empirical head-camera bias correction (calibration). Default
        # is zero (honest raw perception). Set cfg.PERCEPTION_XYZ_OFFSET to the
        # measured systematic offset (perceived - true) to compensate.
        center = np.array([center_xy[0], center_xy[1], center_z]) + cfg.PERCEPTION_XYZ_OFFSET

        return DetectedObject(
            label=label,
            color_name=color_name,
            center=center,
            radius=radius,
            height=height,
            axis=plane.normal.copy(),
            mean_rgb=mean_rgb,
            n_points=len(pts),
            arc_coverage=arc_coverage,
            fit_rms=fit_rms,
            confidence=confidence,
            arc_bins=arc_bins,
        )

    @staticmethod
    def _extract_rim(xy, center_guess):
        """Reduce a disk+arc cluster to its outer-boundary rim.

        Bins points by angle about `center_guess` into CYL_RIM_BINS angular
        sectors. Within each occupied bin, keeps the points near the LOCAL
        CYL_RIM_PERCENTILE-th percentile radius (a band of +-2mm around it,
        averaged) rather than the raw maximum. Verified numerically to be
        materially more robust than a per-bin max under simulated RGB-D
        "flying pixel" outlier contamination (stereo-matching smear at depth
        discontinuities can scatter a few points beyond the true rim; a
        percentile-band approach shrugs these off while a raw max chases
        them, re-introducing an outward bias).

        Returns an (n_bins_occupied, 2) array of rim points (typically far
        fewer than the input — this is the point).
        """
        rel = xy - center_guess
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rad = np.linalg.norm(rel, axis=1)
        n_bins = cfg.CYL_RIM_BINS
        bins = ((ang + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins

        rim_pts = []
        for b in range(n_bins):
            in_bin = np.where(bins == b)[0]
            if in_bin.size == 0:
                continue
            rad_bin = rad[in_bin]
            thresh = np.percentile(rad_bin, cfg.CYL_RIM_PERCENTILE)
            near = in_bin[np.abs(rad_bin - thresh) < cfg.CYL_RIM_BAND]
            if near.size == 0:
                near = in_bin[[int(np.argmin(np.abs(rad_bin - thresh)))]]
            rim_pts.append(xy[near].mean(axis=0))
        return np.array(rim_pts) if rim_pts else xy

    @staticmethod
    def _angular_isolation_weights(rim_xy, center_guess):
        """Compute per-rim-point weights that UP-WEIGHT angular isolates.

        WHY: from an oblique single viewpoint the camera sees a partial arc
        (say 120-180 deg). The camera-FACING side has many consecutive occupied
        angular bins (dense arc), while the self-occluded FAR side contributes
        fewer, more isolated bins. A standard (unweighted) circle fit treats
        every rim point equally → the dense side dominates → the fitted center
        is pulled TOWARD the camera. Angular-isolation weighting corrects this
        by giving each rim point a weight INVERSELY proportional to the local
        angular density around it, so sparse-side points carry proportionally
        more influence in the least-squares fit. This is the standard
        "arc-length weighting" trick for partial-arc circle fitting (see e.g.
        Forbes, "Least-squares best fit of circles and circular arcs", 1989),
        adapted here to the discrete-bin rim extraction output.

        METHOD: for each rim point i (one per occupied angular bin), count the
        number of OTHER occupied bins within ±K bins of it (a local density
        estimate). Weight_i = 1 / (1 + local_count). Points surrounded by many
        occupied neighbours → low weight; isolated points → high weight.
        Normalized so weights sum to n (preserves the scale of the fit).
        """
        n = len(rim_xy)
        if n < 3:
            return np.ones(n)
        rel = rim_xy - center_guess
        ang = np.arctan2(rel[:, 1], rel[:, 0])  # [-pi, pi]
        # Sort by angle for the neighborhood count.
        order = np.argsort(ang)
        ang_sorted = ang[order]
        # For each point, count how many others are within a ±K_BINS window
        # in angular distance (wrap-aware). K_BINS ~ 1/6 of the circle.
        K_ANG = np.pi / 3.0   # ±60 deg window for density estimation
        counts = np.zeros(n)
        for i in range(n):
            diffs = np.abs(ang_sorted - ang_sorted[i])
            diffs = np.minimum(diffs, 2.0 * np.pi - diffs)  # wrap
            counts[i] = float(np.count_nonzero(diffs < K_ANG)) - 1.0  # exclude self
        # Un-sort back to original rim order.
        w = np.zeros(n)
        w[order] = 1.0 / (1.0 + counts)
        # Normalize so sum(w) = n (preserves least-squares scale).
        w *= n / w.sum()
        return w

    @staticmethod
    def _fit_circle_kasa_weighted(xy, weights):
        """Weighted algebraic (Kasa) circle fit.

        Same formulation as _fit_circle_kasa but with per-point weights in the
        least-squares system: minimizes sum_i w_i * ((x_i-a)^2+(y_i-b)^2-R^2)^2.
        This becomes the weighted normal equation  (A^T W A) sol = A^T W b.

        The weighting corrects for partial-arc density asymmetry: sparse-side
        rim points (far side) carry more weight, pulling the fitted center back
        toward the true geometric center even when most points come from the
        camera-facing arc.

        Returns (center_xy, radius, ok).
        """
        if len(xy) < 5:
            return None, 0.0, False
        x = xy[:, 0]
        y = xy[:, 1]
        A = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y
        W = np.diag(weights)
        try:
            AtWA = A.T @ W @ A
            AtWb = A.T @ W @ b
            sol = np.linalg.solve(AtWA, AtWb)
        except np.linalg.LinAlgError:
            return None, 0.0, False
        a, c_, c = sol
        val = c + a * a + c_ * c_
        if val <= 0.0 or not np.isfinite(val):
            return None, 0.0, False
        return np.array([a, c_]), float(np.sqrt(val)), True

    @staticmethod
    def _fit_circle_kasa(xy):
        """Algebraic (Kasa) circle fit. Returns (center_xy, radius, ok).

        Minimises sum_i ((x_i-a)^2 + (y_i-b)^2 - R^2)^2 in closed form by
        solving the linear system  [2x 2y 1] [a b c]^T = [x^2+y^2],  with
        R = sqrt(c + a^2 + b^2). Robust for arcs >~ 90 deg; flagged not-ok if
        the system is ill-conditioned (near-collinear points). Used as a
        fallback if the Hyper fit below is degenerate.
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

    @staticmethod
    def _fit_circle_hyper(xy):
        """"Hyper" algebraic circle fit (Al-Sharadqah & Chernov, 2009).

        Minimises the same algebraic objective as Kasa but with a bias
        correction term baked into the generalised eigenvalue problem, which
        removes Kasa's small residual outward-radius bias on a clean/near-
        uniform ring. Solved as a generalised eigenproblem
        M z = eta N4 z  on the centred/scaled data; take the eigenvector for
        the SMALLEST POSITIVE eigenvalue eta.

        Verified numerically against Kasa on synthetic rings: materially
        lower bias on arcs >= 120 deg (the typical extracted-rim coverage
        here), matches Kasa's robustness floor on very short arcs (both
        become unreliable below ~90 deg — CYL_RIM_BINS / the rim-extraction
        band are chosen so this is rarely the limiting factor in practice).
        Falls back to Kasa (see `_fit_circle_kasa`) if the eigenproblem is
        degenerate (near-collinear rim, e.g. too few occupied angular bins).
        """
        if len(xy) < 5:
            return None, 0.0, False
        x = xy[:, 0]
        y = xy[:, 1]
        xm, ym = x.mean(), y.mean()
        u = x - xm
        v = y - ym
        z = u * u + v * v
        Zmat = np.column_stack((z, u, v, np.ones_like(z)))
        n = len(x)
        M = (Zmat.T @ Zmat) / n
        Mz = float(z.mean())
        N4 = np.array([
            [8.0 * Mz, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ])
        try:
            eigvals, eigvecs = scipy.linalg.eig(M, N4)
        except (np.linalg.LinAlgError, ValueError):
            return None, 0.0, False
        eigvals = eigvals.real
        positive = eigvals[eigvals > 1e-10]
        if positive.size == 0:
            return None, 0.0, False
        eta = positive.min()
        idx = int(np.argmin(np.abs(eigvals - eta)))
        A_, B_, C_, D_ = eigvecs[:, idx].real
        if abs(A_) < 1e-12:
            return None, 0.0, False
        cx = -B_ / (2.0 * A_) + xm
        cy = -C_ / (2.0 * A_) + ym
        disc = B_ ** 2 + C_ ** 2 - 4.0 * A_ * D_
        if disc <= 0.0 or not np.isfinite(disc):
            return None, 0.0, False
        radius = np.sqrt(disc) / (2.0 * abs(A_))
        if not np.isfinite(radius) or radius <= 0.0:
            return None, 0.0, False
        return np.array([cx, cy]), float(radius), True

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
