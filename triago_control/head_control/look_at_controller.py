"""
Look-at controller: drive the 7-DOF head so the camera optical +Z axis points
at a target (the table), with an optional gentle scan to improve coverage.

CONTROL LAW (Position-Based Visual Servoing, rotational only)
------------------------------------------------------------
We do NOT need the full 2.5D IBVS of qp_head_visual_servo.py here — there is no
pixel target, only a 3D point to fixate. So we use a clean rotational look-at:

    1. Express the target point in the camera optical frame: p = T_cam_base^-1 * P.
    2. The current optical axis is z = [0,0,1]. The desired axis is d = p/|p|.
    3. The angular velocity that rotates z onto d is  omega_des = lambda * (z x d),
       expressed in the camera frame. (Its magnitude ~ sin(angle), a natural
       proportional law that vanishes at alignment.)
    4. Degenerate case (target directly behind, d.z < -0.95): push a fixed pitch
       to escape the singularity.

QP (same structure & solver as the arm controllers, for consistency)
--------------------------------------------------------------------
    decision x = [dq_head (7), slack (3)]
    minimise   1/2 xᵀ H x  -  gᵀ x
       H: per-joint velocity weights (proximal joints heavier) + slack penalty
       g: null-space posture spring toward joint mid-range
    subject to (Cᵀ x >= b, first meq are equalities):
       equality : J_rot · dq - slack = omega_des     (the look-at task)
       inequality: velocity-aware joint-limit CBF + hard velocity caps
    solved with quadprog.solve_qp (active-set), exactly like the arm QP.
"""

import numpy as np
import quadprog

import triago_control.head_control.config as cfg


class LookAtController:
    def __init__(self, kin):
        self.kin = kin              # HeadKinematics
        self.n = len(cfg.HEAD_JOINTS)
        self.last_angle_deg = 180.0  # most recent look-at error (for telemetry)
        self.last_slack_norm = 0.0   # QP slack magnitude (>0 means joint limits bite)

        # --- Active-vision state (perception-driven head motion) ----------
        self.phase = 0               # 0 SWEEP, 1 REFINE, 2 HOLD (telemetry)
        self.weakest_coverage = 0.0  # arc coverage of the least-observed object (telemetry)
        self._wp_idx = 0             # current SCAN_WAYPOINTS index in Phase 0
        self._noprog = 0             # consecutive aligned+fresh frames with no coverage gain
        self._prev_total_cov = None  # total arc coverage at the current dwell (reset on retarget)
        self._refine_done = set()    # colours whose refinement has plateaued
        self._last_result_id = None  # detect a fresh perception frame (vs. control-rate re-calls)
        self._last_target = cfg.TABLE_TOP_CENTER_BASE.copy()

    # ------------------------------------------------------------------ #
    # Scan target generation                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def scan_target(t: float):
        """FIXED time-based scan (fallback when ENABLE_ACTIVE_VISION is False).

        STEPPED scan: hold each waypoint for SCAN_DWELL_S so the head settles
        and the velocity-gated accumulation can fuse clean, well-registered
        frames before moving on. Cycles through the waypoint list.
        """
        base = cfg.TABLE_TOP_CENTER_BASE.copy()
        if not cfg.ENABLE_SCAN:
            return base
        wps = cfg.SCAN_WAYPOINTS
        idx = int(t / cfg.SCAN_DWELL_S) % len(wps)
        ox, oy = wps[idx]
        return base + np.array([ox, oy, 0.0])

    # ------------------------------------------------------------------ #
    # Active vision: perception-driven look-at target                     #
    # ------------------------------------------------------------------ #
    def active_vision_target(self, t: float, latest_result):
        """Return the look-at point in base_footprint, LED BY PERCEPTION.

        Two-phase policy (see config §5c):
          Phase 0 SWEEP : cycle SCAN_WAYPOINTS, advancing off a waypoint once its
                          incremental arc-coverage gain plateaus (not a fixed dwell).
          Phase 1 REFINE: aim at the weakest (least-covered) object, nudged toward
                          its unseen-sector azimuth, until it plateaus / clears the
                          floor; then re-pick the next weakest.
          Phase 2 HOLD  : freeze the aim so the rate-based convergence check can run.

        Progress is only accounted on a FRESH perception frame while the head is
        ALIGNED (arrived + holding) — so slewing between targets is never mistaken
        for a coverage plateau, and the move->settle->fuse discipline is preserved.
        """
        base = cfg.TABLE_TOP_CENTER_BASE.copy()
        if not cfg.ENABLE_ACTIVE_VISION:
            return self.scan_target(t)

        objs = list(latest_result.objects) if latest_result is not None else []
        cov_by_color, obj_by_color = {}, {}
        for o in objs:
            cname = getattr(o, "color_name", "unknown")
            if cname in ("red", "blue"):
                cov_by_color[cname] = float(getattr(o, "arc_coverage", 0.0))
                obj_by_color[cname] = o
        if cov_by_color:
            self.weakest_coverage = min(cov_by_color.values())

        # Budget cap: stop searching and hold wherever we are (real hardware can't
        # scan forever). A never-covered object is left to the convergence gate,
        # which will simply not converge until re-armed / re-observed.
        if t > cfg.ACTIVE_VISION_TIME_BUDGET_S:
            self.phase = 2
            return self._last_target

        # Only account progress on a genuinely NEW perception frame while aligned.
        fresh = latest_result is not None and id(latest_result) != self._last_result_id
        aligned = self.is_aligned()
        self._last_result_id = id(latest_result) if latest_result is not None else self._last_result_id
        total_cov = sum(cov_by_color.values())
        plateaued = self._account_progress(fresh and aligned, total_cov)

        # -------- Phase 0: SWEEP to discover plane + all objects ----------
        if self.phase == 0:
            if plateaued:
                self._retarget()
                self._wp_idx += 1
                if self._wp_idx >= len(cfg.SCAN_WAYPOINTS):
                    self._wp_idx = 0
                    if len(cov_by_color) >= cfg.WORLD_EXPECTED_CYLINDERS:
                        self.phase = 1          # everything seen -> refine the weakest
            ox, oy = cfg.SCAN_WAYPOINTS[self._wp_idx]
            self._last_target = base + np.array([ox, oy, 0.0])
            return self._last_target

        # -------- Phase 1: REFINE the weakest object ----------------------
        if self.phase == 1:
            # Which colours still need work (below the coverage floor + not done)?
            pending = [c for c, cov in cov_by_color.items()
                       if cov < cfg.WORLD_MIN_ARC_COVERAGE and c not in self._refine_done]
            if not pending:
                self.phase = 2                  # nothing left to improve -> hold
                return self._last_target
            weak = min(pending, key=lambda c: cov_by_color[c])
            if plateaued:
                self._refine_done.add(weak)     # this one stopped improving; move on
                self._retarget()
                return self._last_target
            o = obj_by_color[weak]
            center = np.asarray(o.center, dtype=float)
            off = cfg.ACTIVE_VISION_REFINE_OFFSET * self._unseen_azimuth_dir(
                getattr(o, "arc_bins", None))
            self._last_target = np.array([center[0] + off[0], center[1] + off[1], base[2]])
            return self._last_target

        # -------- Phase 2: HOLD (let convergence run) ---------------------
        return self._last_target

    def _account_progress(self, evaluate: bool, total_cov: float) -> bool:
        """Track coverage-gain plateau at the current dwell. Returns True once
        PLATEAU_PATIENCE consecutive evaluated frames show < EPS gain."""
        if not evaluate:
            return False
        if self._prev_total_cov is None:
            self._prev_total_cov = total_cov
            return False
        gain = total_cov - self._prev_total_cov
        self._prev_total_cov = total_cov
        if gain < cfg.ACTIVE_VISION_PLATEAU_EPS:
            self._noprog += 1
        else:
            self._noprog = 0
        return self._noprog >= cfg.ACTIVE_VISION_PLATEAU_PATIENCE

    def _retarget(self):
        """Reset the plateau accounting when the aim is about to change (the head
        will slew, so within-dwell coverage progress starts fresh)."""
        self._noprog = 0
        self._prev_total_cov = None

    @staticmethod
    def _unseen_azimuth_dir(arc_bins):
        """Mean base-frame XY unit direction of the UNSEEN arc sectors, i.e. the
        side of the object we have NOT yet observed. Returns [0,0] if fully seen or
        unknown. arc_bins[i] marks azimuth (-pi + (i+0.5)*2pi/N) about the object
        centre where points WERE seen (object_detector.py)."""
        if arc_bins is None:
            return np.zeros(2)
        arc_bins = np.asarray(arc_bins, dtype=bool)
        unseen = ~arc_bins
        if not unseen.any() or unseen.all():
            return np.zeros(2)
        n = len(arc_bins)
        idx = np.nonzero(unseen)[0]
        ang = -np.pi + (idx + 0.5) * (2.0 * np.pi / n)
        v = np.array([np.cos(ang).sum(), np.sin(ang).sum()])
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else np.zeros(2)

    # ------------------------------------------------------------------ #
    # Main solve                                                          #
    # ------------------------------------------------------------------ #
    def compute(self, T_cam_base, J_cam, target_base):
        """Compute head joint velocities (7,) to look at target_base.

        Parameters
        ----------
        T_cam_base : pin.SE3   camera pose in base_footprint (from kin.forward)
        J_cam      : (6,7)     LOCAL camera Jacobian w.r.t. head joints
        target_base: (3,)      fixation point in base_footprint

        Returns
        -------
        dq : (7,) joint velocity command  (rad/s)
        """
        # --- 1. Target into camera optical frame -----------------------
        p_cam = T_cam_base.inverse().act(np.asarray(target_base, dtype=float))
        norm = np.linalg.norm(p_cam)
        if norm < 1e-6:
            return np.zeros(self.n)
        d = p_cam / norm

        # Angular error (for telemetry / "aligned?" check).
        cos_a = np.clip(d[2], -1.0, 1.0)        # angle between current z and d
        self.last_angle_deg = float(np.degrees(np.arccos(cos_a)))

        # --- 2. Desired angular velocity (camera frame) ----------------
        z_axis = np.array([0.0, 0.0, 1.0])
        omega_des = cfg.LOOKAT_LAMBDA * np.cross(z_axis, d)
        if d[2] < -0.95:                         # target behind -> escape
            omega_des[1] = cfg.LOOKAT_LAMBDA

        # --- 3. Build the QP -------------------------------------------
        n_vars = self.n + 3
        H = np.zeros((n_vars, n_vars))
        H[:self.n, :self.n] = np.diag(cfg.HEAD_JOINT_WEIGHTS)
        H[self.n:, self.n:] = np.eye(3) * cfg.LOOKAT_SLACK_WEIGHT
        # Tiny regularisation so H stays strictly positive-definite for quadprog.
        H += np.eye(n_vars) * 1e-6

        # Null-space posture spring toward a known-good observation posture
        # (keeps the redundant DOF at a forward, top-down viewpoint instead of
        # drifting to the grazing mid-range config).
        g = np.zeros(n_vars)
        q = self.kin.get_head_joint_positions()
        q_min, q_max = self.kin.get_head_joint_limits()
        dq_posture = -cfg.POSTURE_GAIN * (q - cfg.HEAD_POSTURE_TARGET)
        g[:self.n] = H[:self.n, :self.n] @ dq_posture   # scale by H so it isn't drowned

        # Soft roll-alignment ("up-righting", see config.py §4): keep image-right
        # (camera +X) aligned with world-right so the horizon stays level, added
        # AFTER the posture g above so it only shapes the LEFTOVER redundancy.
        # Low-weight least-squares COST (subordinate to the pointing equality
        # below, never competes with it) + DEADBAND so it is completely silent
        # near level -- guaranteeing a settled dwell drops below
        # INTEGRATE_VEL_THRESH and fusion proceeds. `wz` targets the camera roll
        # rate (J_cam angular row about the optical +Z axis).
        R_cam = np.asarray(T_cam_base.rotation)
        d_roll = R_cam.T @ cfg.WORLD_RIGHT
        in_plane = float(np.hypot(d_roll[0], d_roll[1]))
        if in_plane > cfg.ROLL_ALIGN_EPS:
            theta_roll = float(np.arctan2(d_roll[1], d_roll[0]))
            gate = float(np.clip(1.0 - self.last_angle_deg / cfg.ROLL_GATE_ANGLE_DEG,
                                 cfg.ROLL_GATE_FLOOR, 1.0))
            # Fully silent (no H/g contribution at all) both within the deadband
            # AND while slewing far off-target (gate==0) -- only shapes roll once
            # aligned and meaningfully tilted, so it never stiffens a slew nor
            # holds a dwell above the fusion velocity gate.
            if abs(np.degrees(theta_roll)) > cfg.ROLL_DEADBAND_DEG and gate > 1e-3:
                wz = float(np.clip(cfg.K_ROLL_ALIGN * theta_roll,
                                   -cfg.ROLL_WZ_MAX, cfg.ROLL_WZ_MAX)) * gate
                J_roll = J_cam[5, :]                  # roll rate about the optical (+Z) axis
                H[:self.n, :self.n] += cfg.W_ROLL_ALIGN * np.outer(J_roll, J_roll)
                g[:self.n] += cfg.W_ROLL_ALIGN * wz * J_roll

        # Equality: look-at task  (J_rot dq - slack = omega_des)
        J_rot = J_cam[3:, :]                     # angular rows
        A_eq = np.zeros((3, n_vars))
        A_eq[:, :self.n] = J_rot
        A_eq[:, self.n:] = -np.eye(3)
        b_eq = omega_des

        # Inequalities: velocity-aware joint-limit CBF + velocity caps.
        C_rows, b_vals = [], []
        for i in range(self.n):
            if (q_max[i] - q_min[i]) < 0.01:
                up, lo = cfg.MAX_HEAD_VELOCITY, -cfg.MAX_HEAD_VELOCITY
            else:
                buf = min(cfg.JOINT_LIMIT_BUFFER, (q_max[i] - q_min[i]) * 0.1)
                v_up = cfg.JOINT_LIMIT_GAMMA * (q_max[i] - q[i] - buf)
                v_lo = -cfg.JOINT_LIMIT_GAMMA * (q[i] - q_min[i] - buf)
                up = min(cfg.MAX_HEAD_VELOCITY, v_up)
                lo = max(-cfg.MAX_HEAD_VELOCITY, v_lo)
                if lo >= up:                     # over the buffer: freeze joint
                    mid = 0.5 * (up + lo)
                    up, lo = mid + 0.01, mid - 0.01
            # dq_i <= up   ->  -dq_i >= -up
            row = np.zeros(n_vars); row[i] = -1.0
            C_rows.append(row); b_vals.append(-up)
            # dq_i >= lo
            row = np.zeros(n_vars); row[i] = 1.0
            C_rows.append(row); b_vals.append(lo)

        C_ineq = np.array(C_rows).T if C_rows else np.zeros((n_vars, 0))
        b_ineq = np.array(b_vals) if b_vals else np.zeros(0)

        C = np.hstack((A_eq.T, C_ineq))
        b = np.hstack((b_eq, b_ineq))

        # --- 4. Solve --------------------------------------------------
        try:
            sol = quadprog.solve_qp(H, g, C, b, meq=3)[0]
            dq = sol[:self.n]
            slack = sol[self.n:]
            self.last_slack_norm = float(np.linalg.norm(slack))
        except ValueError:
            # Infeasible (rare, e.g. at hard limits) -> command zero, stay safe.
            dq = np.zeros(self.n)
            self.last_slack_norm = -1.0
        return dq

    def is_aligned(self) -> bool:
        return self.last_angle_deg < cfg.LOOKAT_ALIGNED_DEG
