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

    # ------------------------------------------------------------------ #
    # Scan target generation                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def scan_target(t: float):
        """Return the look-at point in base_footprint at time t [s].

        A slow Lissajous sweep across the table top. Periods are deliberately
        non-commensurate so the camera covers the whole surface over time
        rather than retracing one line.
        """
        base = cfg.TABLE_TOP_CENTER_BASE.copy()
        if not cfg.ENABLE_SCAN:
            return base
        ox = cfg.SCAN_AMPLITUDE_X * np.sin(2.0 * np.pi * t / cfg.SCAN_PERIOD_X)
        oy = cfg.SCAN_AMPLITUDE_Y * np.sin(2.0 * np.pi * t / cfg.SCAN_PERIOD_Y)
        return base + np.array([ox, oy, 0.0])

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

        # Null-space posture spring toward joint mid-range (keeps the head tidy).
        g = np.zeros(n_vars)
        q = self.kin.get_head_joint_positions()
        q_min, q_max = self.kin.get_head_joint_limits()
        dq_posture = np.zeros(self.n)
        for i in range(self.n):
            if (q_max[i] - q_min[i]) > 0.01:
                q_center = 0.5 * (q_max[i] + q_min[i])
                dq_posture[i] = -cfg.POSTURE_GAIN * (q[i] - q_center)
        g[:self.n] = H[:self.n, :self.n] @ dq_posture   # scale by H so it isn't drowned

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
