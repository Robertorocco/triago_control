# reference_governor.py
"""
Reference Governor — the CLF-safety intermediate layer.

Sits between the raw cartesian reference (from teleop / trajectory_generator /
planner) and the CLF's actual perceived reference inside the QP-CLF-CBF solver.
Its job is to guarantee that the reference the CLF sees ALWAYS lives inside a
*constraint-admissible set* — bounded position error, bounded orientation error,
bounded velocity, bounded acceleration — so the QP's feasibility is never
threatened by aggressive, discontinuous, or far-away commands.

Design principles:
    1. TRANSPARENT when the raw reference is already well-behaved: if it's close,
       slow, and continuous, the output ≈ input (no perceptible lag or damping).
    2. ACTIVE only when needed: clamps/shapes kick in only when a bound is about
       to be violated — the operator never feels restricted in normal operation.
    3. STATEFUL per arm: each arm has its own governor instance (velocity memory
       for acceleration limiting). No cross-arm coupling.
    4. PRESERVES the QP's math: the governor does NOT modify the QP formulation
       itself (no extra rows/columns); it only pre-conditions the CLF's input.
    5. FUTURE: exposes a `set_waypoint` interface for high-level planning injection
       (steer the reference out of local minima without violating the bounds).

The four active features (each independently configurable via config.py):

    A. VELOCITY SHAPING: clamp the reference velocity magnitude (direction
       preserved) to GOV_V_MAX_LIN / GOV_V_MAX_ANG.

    B. POSITION ERROR BOUNDING: if ||x_ref - x_real|| > GOV_E_MAX_POS, project
       x_ref onto the ball of radius GOV_E_MAX_POS centered at x_real. The CLF
       never sees a position error larger than GOV_E_MAX_POS.

    C. ACCELERATION LIMITING: rate-limit the velocity change between ticks
       (||Δv|| ≤ GOV_A_MAX * dt). Prevents discontinuous jumps from propagating
       as infinite jerk demands.

    D. ORIENTATION GEODESIC CLAMPING: if the rotation error
       ||log3(R_des · R_real^T)|| > GOV_E_MAX_ORI, interpolate R_des toward
       R_real via the exponential map so the angular error stays bounded.
       Prevents the 180° ambiguity/singularity issue and keeps the CLF's
       orientation row demand bounded.

All four operate on the SE(3) reference BEFORE it reaches extract_task_errors.
"""

import numpy as np
import pinocchio as pin
from collections import deque
import time
import triago_control.qp_controller.config as cfg


class ReferenceGovernor:
    """Per-arm reference governor instance. Stateful (velocity memory for accel limiting)."""

    def __init__(self, arm_side: str, model=None, ee_frame_name=None):
        """
        Args:
            arm_side: 'right' or 'left' (used only for logging/debug identity).
            model: Pinocchio model. Kept as a constructor parameter for
                   interface stability; not currently used internally (the
                   RRT-Connect planner that once consumed it was removed --
                   see the local-minima-escape docstring below).
            ee_frame_name: e.g. 'gripper_right_grasping_link'. Same note as above.
        """
        self.arm_side = arm_side

        # --- Internal state for acceleration limiting ---
        # The LAST governed velocity (linear + angular) output by this instance,
        # used to compute the velocity *change* for acceleration clamping.
        self._v_lin_prev = np.zeros(3)
        self._v_ang_prev = np.zeros(3)
        self._initialized = False  # First tick: skip accel limiting (no history yet)

        # --- Waypoint injection (STUB, 2026-07-01) ---
        # When active, the governor blends the raw reference toward this target
        # SE(3) pose, respecting all bounds. None = inactive (pure passthrough
        # from the raw reference after clamping).
        self._waypoint_pos = None       # np.ndarray(3) or None
        self._waypoint_rpy = None       # np.ndarray(3) or None
        self._waypoint_priority = 0.0   # [0, 1]: 0 = follow user, 1 = follow waypoint

        # --- Local minima escape (2026-07-01) ---
        # See update_local_minima_escape for the full state machine. Uses an
        # internally-accumulated virtual clock (sum of dt) rather than wall
        # time, so it stays consistent regardless of sim/real-time settings.
        self._lme_elapsed_time = 0.0
        self._lme_error_history = deque()      # [(t, error_norm), ...] pruned to the trigger window
        self._lme_state = 'normal'             # 'normal' | 'escaping'
        self._lme_category = None              # None | 'obstacle' | 'joint' | 'unknown'
        self._lme_escape_elapsed = 0.0
        self._lme_posture_scale = 1.0          # ramped actual multiplier (returned to the caller)
        self._lme_last_console_time = -1e9     # forces an immediate print on the first report

        # NOTE (2026-07-03): an RRT-Connect joint-space planner was attempted
        # here as a local-minima escape fallback (background planning thread,
        # Cartesian waypoint queue consumed by govern()). The attempt was
        # unsuccessful and has been fully removed -- ENABLE_LOCAL_MINIMA_ESCAPE
        # now offers ONLY the posture-weight + task_dim correction below.

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def govern(self, x_ref, rpy_ref, v_ref, w_ref, x_real, R_real, dt):
        """Apply all governor features and return the governed reference.

        Args:
            x_ref:  (3,) raw reference position [m] (may be None → passthrough)
            rpy_ref: (3,) raw reference orientation as RPY [rad] (may be None)
            v_ref:  (3,) raw reference linear velocity [m/s]
            w_ref:  (3,) raw reference angular velocity [rad/s]
            x_real: (3,) current real EE position from FK [m]
            R_real: (3,3) current real EE rotation matrix from FK
            dt:     scalar control timestep [s]

        Returns:
            (x_gov, rpy_gov, v_gov, w_gov) — the governed reference, same shapes.
            Any input that was None is returned as None (governor skips that DOF).
        """
        if x_ref is None:
            return None, None, None, None

        # Work on copies so we never mutate the caller's arrays
        x_gov = np.array(x_ref, dtype=float)
        rpy_gov = np.array(rpy_ref, dtype=float) if rpy_ref is not None else None
        v_gov = np.array(v_ref, dtype=float) if v_ref is not None else np.zeros(3)
        w_gov = np.array(w_ref, dtype=float) if w_ref is not None else np.zeros(3)

        # --- A. VELOCITY SHAPING (direction-preserving magnitude clamp) ---
        v_gov = self._clamp_velocity(v_gov, cfg.GOV_V_MAX_LIN)
        w_gov = self._clamp_velocity(w_gov, cfg.GOV_V_MAX_ANG)

        # --- B. POSITION ERROR BOUNDING ---
        x_gov = self._bound_position_error(x_gov, x_real, cfg.GOV_E_MAX_POS)

        # --- C. ACCELERATION LIMITING ---
        v_gov, w_gov = self._limit_acceleration(v_gov, w_gov, dt)

        # --- D. ORIENTATION GEODESIC CLAMPING ---
        if rpy_gov is not None and R_real is not None:
            rpy_gov = self._clamp_orientation_error(rpy_gov, R_real, cfg.GOV_E_MAX_ORI)

        return x_gov, rpy_gov, v_gov, w_gov

    def reset(self):
        """Reset internal state (call on arm switch / re-anchor / watchdog freeze)."""
        self._v_lin_prev = np.zeros(3)
        self._v_ang_prev = np.zeros(3)
        self._initialized = False
        self._waypoint_pos = None
        self._waypoint_rpy = None
        self._waypoint_priority = 0.0
        # Local minima escape: a freshly-frozen/re-anchored arm has no meaningful
        # tracking error to speak of, so drop any in-progress escape and clear
        # the error history (stale samples from before the freeze would otherwise
        # bias the next "stuck" detection).
        self._lme_error_history.clear()
        self._lme_state = 'normal'
        self._lme_category = None
        self._lme_escape_elapsed = 0.0
        self._lme_posture_scale = 1.0

    # =====================================================================
    # WAYPOINT INJECTION INTERFACE (STUB — future planning hook)
    # =====================================================================

    def set_waypoint(self, pos, rpy, priority=1.0):
        """Set a planning waypoint that the governor should steer toward.

        When active (priority > 0), the governor BLENDS the raw user/teleop
        reference with a smooth trajectory toward this waypoint, respecting all
        velocity/acceleration/error bounds. This is the mechanism by which a
        high-level planner (PRM, potential-field escape, etc.) can drive
        the robot OUT of QP local minima without ever presenting the CLF with
        an infeasible/discontinuous command. (An RRT-Connect planner was
        attempted for this role and removed -- 2026-07-03, see the class-level
        note near __init__.)

        Args:
            pos: (3,) target position in base_footprint [m], or None to clear.
            rpy: (3,) target orientation as RPY [rad], or None to clear.
            priority: float in [0, 1]. 0 = pure user following (waypoint ignored),
                      1 = pure waypoint following (user reference ignored).
                      Intermediate values = convex blend.

        Clearing: call set_waypoint(None, None, 0.0) to deactivate.

        NOTE (2026-07-01): this interface is DEFINED but NOT YET WIRED into the
        govern() output — the blending logic will be implemented when the
        high-level planning module is built. Currently govern() ignores the
        waypoint entirely and operates purely on the raw reference.
        """
        self._waypoint_pos = np.array(pos, dtype=float) if pos is not None else None
        self._waypoint_rpy = np.array(rpy, dtype=float) if rpy is not None else None
        self._waypoint_priority = float(np.clip(priority, 0.0, 1.0))

    def clear_waypoint(self):
        """Convenience: deactivate the planning waypoint."""
        self.set_waypoint(None, None, 0.0)

    @property
    def waypoint_active(self):
        """True if a planning waypoint is currently set and has nonzero priority."""
        return self._waypoint_pos is not None and self._waypoint_priority > 0.0

    # =====================================================================
    # LOCAL MINIMA ESCAPE (2026-07-01)
    # =====================================================================
    #
    # PROBLEM: the CLF-CBF-QP can reach a local minimum of the CLF's landscape
    # restricted to the CBF-safe set: the CLF's gradient direction (toward the
    # reference) is blocked by a hard constraint, and no feasible joint
    # velocity can simultaneously decrease the tracking error AND satisfy every
    # barrier -- the QP settles on q_dot=0 (or a small residual) with a large,
    # non-decreasing 3D position error. Two known causes (per design review):
    #   1. A CBF obstacle blocks the direct path (lambda_cbf shadow price high).
    #   2. A joint-limit barrier blocks the required joint rotation
    #      (lambda_joints shadow price high).
    #   3. Both simultaneously (obstacle takes priority per instruction).
    #
    # DETECTION: the 3D position error norm is tracked over a rolling window
    # (LME_ERROR_STUCK_WINDOW). "Stuck" = error > LME_ERROR_TRIGGER AND has
    # varied by less than LME_ERROR_STUCK_TOLERANCE across the whole window
    # (i.e. the QP is not making progress, not just moving slowly).
    #
    # CATEGORIZATION: read from the shadow prices produced by the QP's
    # PREVIOUS solve (fed in by the caller each tick -- see
    # main_qp_controller.py). Thresholds are OPERATOR-TUNED for the current
    # parameter set (see config.py's LME_* comment).
    #
    # ESCAPE ACTION (posture task ONLY -- verified no other module writes to
    # qp.posture_scale_right/left, so no conflict):
    #   - Obstacle:    posture weight -> x0.2 (more redundancy to slip past),
    #                  task_dim forced to 3.0 (position-only CLF, orientation
    #                  fully relaxed) for the duration of the escape.
    #   - Joint limit: posture weight -> x5.0 (push harder away from the
    #                  limit), task_dim UNCHANGED.
    #   The posture-scale multiplier is RAMPED (first-order low-pass, same
    #   technique as the existing grasp-phase POSTURE_SCALE_TAU ramp in
    #   main_qp_controller.solve_and_publish) rather than stepped, so the QP
    #   never sees a cost-function discontinuity.
    #
    # EXIT: escape ends (returns to 'normal') when the error drops below
    # LME_ERROR_RECOVERED (success) OR LME_MAX_ESCAPE_DURATION elapses (give
    # up, avoid holding a distorted posture weight forever if the correction
    # didn't work) -- whichever comes first. On exit the governor immediately
    # resumes checking for a NEW local minimum (no cooldown).

    def update_local_minima_escape(self, error_norm, lambda_cbf, lambda_joints, dt, logger=None):
        """Advance the local-minima detection/escape state machine by one tick.

        Args:
            error_norm: scalar, ||x_ref - x_real|| (3D position error norm,
                        computed by the caller from the SAME x_ref/x_real used
                        for the CLF -- per instruction, 3D-only for now).
            lambda_cbf: this arm's CBF shadow price from the QP's PREVIOUS
                        solve (qp.last_lambda_cbf_right / _left).
            lambda_joints: this arm's joint-limit shadow price from the QP's
                        PREVIOUS solve (qp.last_lambda_joints_right / _left).
            dt: control timestep [s] (used both for the ramp and the internal
                        virtual clock -- NOT wall time, so it stays correct
                        under SIMULATE_IDEAL_KINEMATICS or non-realtime sim).
            logger: optional callable(str) for the throttled console report
                        (e.g. a rclpy logger's .info, or plain print). If None,
                        no console output is produced.

        Returns:
            (posture_scale_multiplier, task_dim_override)
              posture_scale_multiplier: float, RAMPED multiplier to apply on
                  top of the existing POSTURE_GRASP_SCALE ramp (1.0 = no
                  correction active). Multiply, don't replace, so this
                  composes cleanly with the existing grasp-phase scale.
              task_dim_override: None (no override) or 3.0 (force
                  position-only CLF) -- the caller should use this INSTEAD OF
                  the raw task_dim_right/left while non-None.
        """
        if not cfg.ENABLE_LOCAL_MINIMA_ESCAPE:
            return 1.0, None

        self._lme_elapsed_time += dt

        # --- 1. Maintain the rolling error-history window ---
        t_now = self._lme_elapsed_time
        self._lme_error_history.append((t_now, error_norm))
        cutoff = t_now - cfg.LME_ERROR_STUCK_WINDOW
        while self._lme_error_history and self._lme_error_history[0][0] < cutoff:
            self._lme_error_history.popleft()

        if self._lme_state == 'normal':
            # --- 2. STUCK DETECTION: only evaluate once the window is full ---
            window_full = (len(self._lme_error_history) >= 2 and
                          (self._lme_error_history[-1][0] - self._lme_error_history[0][0])
                          >= cfg.LME_ERROR_STUCK_WINDOW * 0.9)
            if window_full and error_norm > cfg.LME_ERROR_TRIGGER:
                errs = [e for _, e in self._lme_error_history]
                variation = max(errs) - min(errs)
                if variation <= cfg.LME_ERROR_STUCK_TOLERANCE:
                    # Possible local minimum -- categorize via shadow prices.
                    # Obstacle takes PRIORITY if both conditions are met.
                    if lambda_cbf > cfg.LME_LAMBDA_CBF_THRESHOLD:
                        category = 'obstacle'
                    elif lambda_joints > cfg.LME_LAMBDA_JOINT_THRESHOLD:
                        category = 'joint'
                    else:
                        category = 'unknown'
                    self._lme_state = 'escaping'
                    self._lme_category = category
                    self._lme_escape_elapsed = 0.0
                    self._lme_last_console_time = -1e9  # force an immediate report
                    if logger is not None:
                        if category == 'obstacle':
                            logger(f"\033[93m[LOCAL MINIMA][{self.arm_side.upper()}] Possible local minimum "
                                  f"DETECTED (|e|={error_norm:.3f}m stuck, lambda_cbf={lambda_cbf:.2f}). "
                                  f"Category: OBSTACLE. Applying escape: posture x{cfg.LME_POSTURE_SCALE_OBSTACLE}, "
                                  f"task_dim -> {cfg.LME_TASK_DIM_OBSTACLE} (orientation relaxed).\033[0m")
                        elif category == 'joint':
                            logger(f"\033[93m[LOCAL MINIMA][{self.arm_side.upper()}] Possible local minimum "
                                  f"DETECTED (|e|={error_norm:.3f}m stuck, lambda_joints={lambda_joints:.2f}). "
                                  f"Category: JOINT LIMIT. Applying escape: posture x{cfg.LME_POSTURE_SCALE_JOINT}.\033[0m")
                        else:
                            logger(f"\033[93m[LOCAL MINIMA][{self.arm_side.upper()}] Possible local minimum "
                                  f"DETECTED (|e|={error_norm:.3f}m stuck) but neither lambda_cbf "
                                  f"({lambda_cbf:.2f}) nor lambda_joints ({lambda_joints:.2f}) exceeds its "
                                  f"threshold -- category UNKNOWN, no escape action applied.\033[0m")

        if self._lme_state == 'escaping':
            self._lme_escape_elapsed += dt

            # --- 3. EXIT CONDITIONS ---
            recovered = error_norm < cfg.LME_ERROR_RECOVERED
            timed_out = self._lme_escape_elapsed >= cfg.LME_MAX_ESCAPE_DURATION
            if recovered or timed_out:
                if logger is not None:
                    reason = "RECOVERED" if recovered else "TIMED OUT"
                    logger(f"\033[92m[LOCAL MINIMA][{self.arm_side.upper()}] Escape ended ({reason}, "
                          f"|e|={error_norm:.3f}m after {self._lme_escape_elapsed:.1f}s). "
                          f"Resuming normal posture weighting.\033[0m")
                self._lme_state = 'normal'
                self._lme_category = None
                self._lme_error_history.clear()  # don't immediately re-trigger on stale samples

        # --- 4. COMPUTE the target multiplier + ramp it smoothly ---
        # The ONLY corrective action available is the posture-weight nudge
        # (+ task_dim=3 for an obstacle-induced minimum). An RRT-Connect
        # planner fallback was attempted and removed (2026-07-03) -- see the
        # class-level note near __init__.
        if self._lme_state == 'escaping' and self._lme_category == 'obstacle':
            target_scale = cfg.LME_POSTURE_SCALE_OBSTACLE
            task_dim_override = cfg.LME_TASK_DIM_OBSTACLE
        elif self._lme_state == 'escaping' and self._lme_category == 'joint':
            target_scale = cfg.LME_POSTURE_SCALE_JOINT
            task_dim_override = None
        else:
            # 'normal' state, or 'escaping' with category 'unknown' (detected
            # but no actionable correction -- ramp back to nominal so an
            # unknown-cause stall doesn't leave a stale distorted weight).
            target_scale = 1.0
            task_dim_override = None

        # Smooth first-order ramp (same technique as POSTURE_SCALE_TAU elsewhere)
        a_ramp = dt / (cfg.LME_RAMP_TAU + dt)
        self._lme_posture_scale += a_ramp * (target_scale - self._lme_posture_scale)

        # --- 5. Non-spam console reporting while an escape is in progress ---
        if logger is not None and self._lme_state == 'escaping':
            if (self._lme_elapsed_time - self._lme_last_console_time) >= cfg.LME_CONSOLE_PERIOD:
                self._lme_last_console_time = self._lme_elapsed_time
                if self._lme_category == 'unknown':
                    logger(f"[LOCAL MINIMA][{self.arm_side.upper()}] Still stuck (|e|={error_norm:.3f}m), "
                          f"cause UNKNOWN (no shadow-price threshold exceeded) -- no correction applied. "
                          f"elapsed={self._lme_escape_elapsed:.1f}/{cfg.LME_MAX_ESCAPE_DURATION:.0f}s")
                else:
                    logger(f"[LOCAL MINIMA][{self.arm_side.upper()}] Escaping ({self._lme_category}): "
                          f"posture_scale={self._lme_posture_scale:.3f}, |e|={error_norm:.3f}m, "
                          f"elapsed={self._lme_escape_elapsed:.1f}/{cfg.LME_MAX_ESCAPE_DURATION:.0f}s")

        return self._lme_posture_scale, task_dim_override

    @property
    def local_minima_state(self):
        """('normal'|'escaping', category or None) -- for external telemetry/logging."""
        return self._lme_state, self._lme_category

    # =====================================================================
    # INTERNAL FEATURE IMPLEMENTATIONS
    # =====================================================================

    @staticmethod
    def _clamp_velocity(v, v_max):
        """Clamp a 3-vector's magnitude to v_max, preserving direction.

        If ||v|| <= v_max, returns v unchanged. Otherwise scales v to exactly
        v_max along its original direction.
        """
        norm = np.linalg.norm(v)
        if norm > v_max and norm > 1e-9:
            return v * (v_max / norm)
        return v

    @staticmethod
    def _bound_position_error(x_ref, x_real, e_max):
        """Project x_ref onto the ball of radius e_max centered at x_real.

        If ||x_ref - x_real|| <= e_max, returns x_ref unchanged. Otherwise
        returns the point on the ball's surface closest to x_ref (same direction
        from x_real, clamped distance). The CLF never sees a position error
        larger than e_max.
        """
        error = x_ref - x_real
        norm = np.linalg.norm(error)
        if norm > e_max and norm > 1e-9:
            return x_real + error * (e_max / norm)
        return x_ref

    def _limit_acceleration(self, v_lin, v_ang, dt):
        """Rate-limit the velocity change between ticks.

        ||v_new - v_prev|| is clamped to A_MAX * dt (separately for linear and
        angular). On the first tick after a reset, the previous velocity is
        assumed zero (the arm was stationary) and acceleration limiting applies
        normally — this handles the re-anchor-after-clutch case gracefully
        (the first governed velocity ramps from zero instead of jumping).
        """
        if not self._initialized:
            # First tick: apply limiting from zero (arm was stationary)
            self._initialized = True

        # Linear acceleration clamping
        dv_lin = v_lin - self._v_lin_prev
        dv_lin_norm = np.linalg.norm(dv_lin)
        max_dv_lin = cfg.GOV_A_MAX_LIN * dt
        if dv_lin_norm > max_dv_lin and dv_lin_norm > 1e-9:
            dv_lin = dv_lin * (max_dv_lin / dv_lin_norm)
        v_lin_out = self._v_lin_prev + dv_lin

        # Angular acceleration clamping
        dv_ang = v_ang - self._v_ang_prev
        dv_ang_norm = np.linalg.norm(dv_ang)
        max_dv_ang = cfg.GOV_A_MAX_ANG * dt
        if dv_ang_norm > max_dv_ang and dv_ang_norm > 1e-9:
            dv_ang = dv_ang * (max_dv_ang / dv_ang_norm)
        v_ang_out = self._v_ang_prev + dv_ang

        # Re-apply velocity magnitude clamp AFTER acceleration shaping
        # (acceleration limiting alone could accumulate beyond V_MAX over many ticks
        # if the raw reference sustains a high velocity for a long time)
        v_lin_out = self._clamp_velocity(v_lin_out, cfg.GOV_V_MAX_LIN)
        v_ang_out = self._clamp_velocity(v_ang_out, cfg.GOV_V_MAX_ANG)

        # Update memory for next tick
        self._v_lin_prev = v_lin_out.copy()
        self._v_ang_prev = v_ang_out.copy()

        return v_lin_out, v_ang_out

    @staticmethod
    def _clamp_orientation_error(rpy_ref, R_real, theta_max):
        """Clamp the orientation error on the SO(3) geodesic.

        If ||log3(R_des · R_real^T)|| > theta_max, interpolate R_des toward
        R_real via the exponential map so the angular error is exactly theta_max
        (same rotation axis, shortened angle). Returns the clamped orientation
        as RPY.

        This prevents two issues:
        1. The 180° singularity of log3 (axis undefined at exactly π).
        2. Unbounded CLF orientation-row demand (which can cause infeasibility
           or wild joint motion when the target is far in SO(3)).
        """
        R_des = pin.rpy.rpyToMatrix(rpy_ref[0], rpy_ref[1], rpy_ref[2])
        R_error = R_des @ R_real.T
        log_error = pin.log3(R_error)
        angle = np.linalg.norm(log_error)

        if angle > theta_max and angle > 1e-9:
            # Scale the rotation vector to exactly theta_max (same axis, shorter angle)
            log_clamped = log_error * (theta_max / angle)
            R_clamped = pin.exp3(log_clamped) @ R_real
            return pin.rpy.matrixToRpy(R_clamped)
        return rpy_ref
