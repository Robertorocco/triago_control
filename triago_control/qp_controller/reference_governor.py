# reference_governor.py
"""Pre-CLF reference shaping: bounds velocity, position error, acceleration and orientation error
so the CLF always sees a constraint-admissible target; transparent when the reference is well-behaved.
Per-arm and stateful (velocity memory); never touches the QP formulation itself."""

import numpy as np
import pinocchio as pin
import triago_control.qp_controller.config as cfg


class ReferenceGovernor:
    """Per-arm reference governor instance. Stateful (velocity memory for accel limiting)."""

    def __init__(self, arm_side: str):
        """Creates a governor for one arm; arm_side is identity only."""
        self.arm_side = arm_side

        # Last governed velocity, the memory for acceleration clamping.
        self._v_lin_prev = np.zeros(3)
        self._v_ang_prev = np.zeros(3)
        self._initialized = False

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def govern(self, x_ref, rpy_ref, v_ref, w_ref, x_real, R_real, dt):
        """Applies all four bounds and returns the governed (x, rpy, v, w); None inputs pass through as None."""
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

    # =====================================================================
    # INTERNAL FEATURE IMPLEMENTATIONS
    # =====================================================================

    @staticmethod
    def _clamp_velocity(v, v_max):
        """Clamps a 3-vector's magnitude to v_max, preserving direction."""
        norm = np.linalg.norm(v)
        if norm > v_max and norm > 1e-9:
            return v * (v_max / norm)
        return v

    @staticmethod
    def _bound_position_error(x_ref, x_real, e_max):
        """Projects x_ref onto the ball of radius e_max centered at x_real (bounded CLF position error)."""
        error = x_ref - x_real
        norm = np.linalg.norm(error)
        if norm > e_max and norm > 1e-9:
            return x_real + error * (e_max / norm)
        return x_ref

    def _limit_acceleration(self, v_lin, v_ang, dt):
        """Rate-limits ||dv|| to A_MAX*dt per tick; after a reset the first tick ramps from zero."""
        if not self._initialized:
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

        # Re-clamp magnitude after acceleration shaping (ramping alone could exceed V_MAX over time).
        v_lin_out = self._clamp_velocity(v_lin_out, cfg.GOV_V_MAX_LIN)
        v_ang_out = self._clamp_velocity(v_ang_out, cfg.GOV_V_MAX_ANG)

        # Update memory for next tick
        self._v_lin_prev = v_lin_out.copy()
        self._v_ang_prev = v_ang_out.copy()

        return v_lin_out, v_ang_out

    @staticmethod
    def _clamp_orientation_error(rpy_ref, R_real, theta_max):
        """Clamps the SO(3) geodesic error to theta_max (same axis, shorter angle); avoids the pi singularity."""
        R_des = pin.rpy.rpyToMatrix(rpy_ref[0], rpy_ref[1], rpy_ref[2])
        R_error = R_des @ R_real.T
        log_error = pin.log3(R_error)
        angle = np.linalg.norm(log_error)

        if angle > theta_max and angle > 1e-9:
            log_clamped = log_error * (theta_max / angle)
            R_clamped = pin.exp3(log_clamped) @ R_real
            return pin.rpy.matrixToRpy(R_clamped)
        return rpy_ref
