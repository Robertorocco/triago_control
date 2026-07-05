"""
view_planner.py — active-perception (next-best-view) standoff controller.

PURPOSE
    Decide, ONLINE and WITHOUT any hard-coded knowledge of the scene content,
    how far the head camera should stand from the table so that:
        * the whole region of interest (the table surface) stays inside the
          field of view — nothing important clips off the frame edges, AND
        * the objects on it are resolved by enough pixels for a tight geometric
          fit (radius/height estimation).
    These two objectives pull in opposite directions (closer = more detail but
    smaller FOV footprint; farther = whole scene visible but coarser detail),
    so the "right" distance is a scene-dependent trade-off. Rather than baking
    it into a fixed joint posture at design time, we close a control loop around
    it and let the head find it.

DESIGN (paper framing: information-driven active perception)
    Two signals, both computed PURELY from what the camera actually observes
    (the sensed point cloud + the live intrinsics) — never from a known object
    pose or size:

    (1) FRAMING / CONTAINMENT SIGNAL.
        We reproject the observed table-region cloud (``cropped_points``, which
        by construction is everything the camera captured inside the table crop
        box) back into the image plane using the live extrinsics + intrinsics,
        and measure the fraction of those pixels that land in the OUTER BORDER
        band of the frame, PER EDGE (left / right / top / bottom).

        Intuition: the captured cloud is, by definition, inside the image. If it
        reaches all the way to an edge (high border occupancy there), the true
        surface almost certainly continues beyond the frame on that side — i.e.
        we are CLIPPING the region of interest and must back away / re-aim. If
        every edge band is nearly empty, the whole ROI sits comfortably inside
        the frame and we have framing slack to spend on getting closer.

        This is completely object-agnostic: it only asks "does what I see reach
        the limit of what I can see?".

    (2) RESOLUTION-SUFFICIENCY SIGNAL.
        For each currently tracked object we compute its APPARENT RADIUS IN
        PIXELS, ``r_px = fx * r / range`` (r = observed metric radius, range =
        camera-to-object distance), and read back its rim-fit RMS. Small r_px or
        large RMS => the object is under-resolved => moving closer would help.
        This uses the object's OWN observed radius (a live estimate, not a known
        constant) and the live focal length, so it stays honest and scene-free.

    UPDATE LAW (priority: framing > resolution).
        d* is the desired standoff distance along the viewing ray to the look-at
        target. Every perception tick:
            if any edge is clipping        -> d* += STEP_OUT      (retreat)
            elif ROI contained AND under-resolved -> d* -= STEP_IN (approach)
            else                            -> hold
        d* is clamped to [D_STAR_MIN, D_STAR_MAX] and low-pass filtered so the
        setpoint handed to the QP is always smooth (C0). d* is lazily
        initialised to the range observed on the first valid tick, so enabling
        the controller never causes a step.

    The resulting d* is regulated by a dedicated soft "range" task inside the
    look-at QP (see look_at_controller.py), which slides the camera along the
    ray toward/away from the target while the higher-priority pointing task
    keeps the table centred.

OUTPUT
    ``update()`` returns the (filtered) desired standoff d* in metres, or None
    if there is not yet enough information to act (the caller then leaves the
    QP range task disabled and the head falls back to pure posture-driven
    distance). All intermediate quantities are stored in ``self.metrics`` for
    the console debug window / telemetry.
"""

import numpy as np

import triago_control.head_control.config as cfg


class ActivePerceptionPlanner:
    def __init__(self):
        self.d_star = None                  # filtered desired standoff [m]
        self._d_star_raw = None             # pre-LPF setpoint [m]
        self._contain_ticks = 0             # consecutive "contained" ticks (hysteresis)
        self.last_action = "INIT"           # APPROACH | RETREAT | HOLD | INIT
        self.last_reason = "waiting for data"
        self.metrics = {}                   # populated every update() for debug

    # ------------------------------------------------------------------ #
    # Geometry helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_camera_frame(pts_base, R_cam_base, t_cam_base):
        """Map (N,3) base-frame points into the camera OPTICAL frame.

        R_cam_base, t_cam_base map camera->base (p_base = R p_cam + t), so the
        inverse is p_cam = R^T (p_base - t) = (p_base - t) @ R.
        """
        return (pts_base - t_cam_base) @ R_cam_base

    @staticmethod
    def _project(pts_cam, fx, fy, cx, cy):
        """Pinhole-project (N,3) optical-frame points to pixel (u, v).

        Returns (u, v) for points strictly in front of the camera (z > 0).
        """
        z = pts_cam[:, 2]
        front = z > 1e-6
        z = z[front]
        u = fx * pts_cam[front, 0] / z + cx
        v = fy * pts_cam[front, 1] / z + cy
        return u, v

    # ------------------------------------------------------------------ #
    # Main update                                                         #
    # ------------------------------------------------------------------ #
    def update(self, result, R_cam_base, t_cam_base, target_base, intr, wh):
        """Update and return the desired standoff distance d* [m] (or None).

        Parameters
        ----------
        result       : PerceptionResult (cropped_points, plane, objects)
        R_cam_base   : (3,3) camera-optical -> base rotation
        t_cam_base   : (3,)  camera-optical -> base translation
        target_base  : (3,)  current look-at point (table top centre) in base
        intr         : (fx, fy, cx, cy) scaled to the actual depth resolution
        wh           : (W, H) actual depth image resolution
        """
        m = {}
        self.metrics = m

        # --- Current camera-to-target range along the viewing ray -------
        p_cam_target = R_cam_base.T @ (np.asarray(target_base, float) - t_cam_base)
        cur_range = float(np.linalg.norm(p_cam_target))
        m["range"] = cur_range

        fx, fy, cx, cy = intr
        W, H = wh
        m["fx"] = fx
        m["img_wh"] = (W, H)

        # Lazy init: anchor d* at the current range so turning the loop on is
        # bump-free.
        if self.d_star is None:
            self.d_star = float(np.clip(cur_range, cfg.VIEW_D_STAR_MIN, cfg.VIEW_D_STAR_MAX))
            self._d_star_raw = self.d_star

        # --- (1) Framing / containment via reprojected border occupancy -
        edge_occ = {"L": 0.0, "R": 0.0, "T": 0.0, "B": 0.0}
        n_proj = 0
        fill = 0.0
        framing_ok = False   # enough points to trust the framing signal
        pts = result.cropped_points
        if pts is not None and len(pts) >= cfg.VIEW_MIN_PROJ_POINTS:
            pts_cam = self._to_camera_frame(pts, R_cam_base, t_cam_base)
            u, v = self._project(pts_cam, fx, fy, cx, cy)
            # Keep only projections that land on the sensor (they all should,
            # since they came from the image, but guard against numerical /
            # extrinsic drift).
            on = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v = u[on], v[on]
            n_proj = int(u.size)
            if n_proj >= cfg.VIEW_MIN_PROJ_POINTS:
                framing_ok = True
                margin = cfg.VIEW_BORDER_MARGIN_FRAC * min(W, H)
                inv = 1.0 / n_proj
                edge_occ["L"] = float(np.count_nonzero(u < margin) * inv)
                edge_occ["R"] = float(np.count_nonzero(u > W - margin) * inv)
                edge_occ["T"] = float(np.count_nonzero(v < margin) * inv)
                edge_occ["B"] = float(np.count_nonzero(v > H - margin) * inv)
                # Framing "fill": bounding-box coverage of the ROI in the image
                # (informational — how much of the frame the table occupies).
                bbox_w = float(u.max() - u.min())
                bbox_h = float(v.max() - v.min())
                fill = (bbox_w * bbox_h) / float(W * H)
        m["edge_occ"] = edge_occ
        m["n_proj"] = n_proj
        m["fill"] = fill
        m["framing_ok"] = framing_ok

        max_edge = max(edge_occ.values()) if framing_ok else 0.0
        clipping = framing_ok and (max_edge > cfg.VIEW_BORDER_HIGH)
        contained = framing_ok and (max_edge < cfg.VIEW_BORDER_LOW)
        m["clipping"] = clipping
        m["contained"] = contained
        m["max_edge"] = max_edge

        # --- (2) Resolution sufficiency from tracked objects ------------
        obj_res = []
        for o in result.objects:
            rng = float(np.linalg.norm(R_cam_base.T @ (o.center - t_cam_base)))
            r_px = fx * o.radius / rng if rng > 1e-6 else 0.0
            obj_res.append({
                "label": o.label,
                "r_px": float(r_px),
                "range": rng,
                "fit_rms": float(getattr(o, "fit_rms", 0.0)),
                "n_points": int(getattr(o, "n_points", 0)),
                "coverage": float(getattr(o, "arc_coverage", 0.0)),
                "confidence": float(getattr(o, "confidence", 0.0)),
            })
        m["objects"] = obj_res

        have_objs = len(obj_res) > 0
        mean_r_px = float(np.mean([o["r_px"] for o in obj_res])) if have_objs else 0.0
        worst_rms = float(np.max([o["fit_rms"] for o in obj_res])) if have_objs else 0.0
        m["mean_r_px"] = mean_r_px
        m["worst_rms"] = worst_rms
        # Under-resolved if the objects are too small in the image OR the rim
        # fit is still loose (and we actually have objects to judge).
        under_resolved = have_objs and (
            mean_r_px < cfg.VIEW_RES_RADIUS_PX_TARGET
            or worst_rms > cfg.VIEW_RES_FIT_RMS_OK
        )
        m["under_resolved"] = under_resolved

        # --- No table at all: widen the view to reacquire ---------------
        no_table = result.plane is None

        # --- Decision (priority: framing > resolution) ------------------
        if no_table:
            self._d_star_raw = min(self._d_star_raw + cfg.VIEW_STEP_OUT, cfg.VIEW_D_STAR_MAX)
            self._contain_ticks = 0
            action, reason = "RETREAT", "no table detected -> widen FOV to reacquire"
        elif not framing_ok:
            action, reason = "HOLD", "too few reprojected points to judge framing"
        elif clipping:
            edges = [k for k, val in edge_occ.items() if val > cfg.VIEW_BORDER_HIGH]
            self._d_star_raw = min(self._d_star_raw + cfg.VIEW_STEP_OUT, cfg.VIEW_D_STAR_MAX)
            self._contain_ticks = 0
            action, reason = "RETREAT", f"ROI clipping frame edge(s) {'+'.join(edges)}"
        elif contained and under_resolved:
            # Hysteresis: only approach after N consecutive "contained" ticks to
            # avoid oscillation when the scan shifts between clip/contain.
            self._contain_ticks += 1
            if self._contain_ticks >= cfg.VIEW_CONTAIN_HYSTERESIS:
                self._d_star_raw = max(self._d_star_raw - cfg.VIEW_STEP_IN, cfg.VIEW_D_STAR_MIN)
                if mean_r_px < cfg.VIEW_RES_RADIUS_PX_TARGET:
                    reason = (f"ROI contained, objects under-resolved "
                              f"(r={mean_r_px:.0f}<{cfg.VIEW_RES_RADIUS_PX_TARGET:.0f}px) -> approach")
                else:
                    reason = (f"ROI contained, rim fit loose "
                              f"(rms={worst_rms*1e3:.1f}mm) -> approach")
                action = "APPROACH"
            else:
                action = "HOLD"
                reason = (f"contained+under-resolved but hysteresis "
                          f"({self._contain_ticks}/{cfg.VIEW_CONTAIN_HYSTERESIS} ticks)")
        elif contained and not have_objs:
            self._contain_ticks += 1
            action, reason = "HOLD", "ROI contained, no objects yet (holding for detection)"
        else:
            # "partial" or contained+resolved — don't approach, don't retreat
            if contained:
                self._contain_ticks += 1
            else:
                self._contain_ticks = 0
            action, reason = "HOLD", "framing + resolution within targets"

        self.last_action = action
        self.last_reason = reason
        m["action"] = action
        m["reason"] = reason
        m["d_star_raw"] = self._d_star_raw

        # --- Low-pass the setpoint for a smooth QP target ---------------
        a = cfg.VIEW_STANDOFF_LPF
        self.d_star = a * self._d_star_raw + (1.0 - a) * self.d_star
        m["d_star"] = self.d_star

        return self.d_star
