# qp_formulator.py
"""
The Mathematical Core.

Builds the strictly-convex Quadratic Program solved every control tick and
hands it to `quadprog`. The decision vector is:

    x = [ q_dot (nv) , delta_right , delta_left ]

i.e. the full joint velocity plus one scalar CLF slack per arm. The QP is:

    min  1/2 x' H x + g' x
    s.t. C' x >= b

Cost  H/g : joint regularization (damping) + posture-hold spring + slack penalty.
Constraints C/b :
    * Task   : Perfect Scalar Inequality CLF  (per arm, normalized or raw)
    * Safety : SoftMin CBF collision avoidance (padded with zero slack)
    * Limits : velocity-aware joint position limits (upper + lower)

ADAPTIVE SCHEDULING (PRESERVED EXACTLY):
    * Decoupled dynamic slack weighting:
          alpha = exp(-beta * lambda^2)
          w_slack = base_weight_slack + alpha * (max_weight_slack - base_weight_slack)
      computed independently per arm from that arm's worst shadow price lambda.
    * Dynamic gamma (CLF) scheduling, low-pass filtered:
          target_gamma = gamma_min + exp(-beta_gamma * lambda_col) * (gamma_max - gamma_min)
      so CLF tracking authority decays exponentially near obstacles, using the
      Lagrange multipliers (shadow prices) carried over from the previous loop.
"""

import numpy as np
import quadprog
import pinocchio as pin
import triago_control.qp_controller.config as cfg


class QPFormulator:
    """Assembles H, g, C, b, solves the CLF-CBF-QP and tracks shadow prices."""

    def __init__(self, model):
        self.model = model
        self.n_joints = model.nv       # Full robot velocity dimension
        self.n_slacks = 2              # One scalar CLF slack per arm (right, left)
        self.n_total = self.n_joints + self.n_slacks

        # Pre-allocated QP matrices (filled in-place each tick)
        self.H = np.zeros((self.n_total, self.n_total))
        self.g = np.zeros(self.n_total)

        # Lagrange multiplier memory (shadow prices fed back into scheduling)
        self.last_lambda_col = 0.0
        self.last_lambda_joints_right = 0.0
        self.last_lambda_joints_left = 0.0

        # Low-pass-filtered shadow prices used by the slack scheduler (the raw
        # multipliers are noisy tick-to-tick and made the slack weight jump).
        self._lam_col_f = 0.0
        self._lam_jr_f = 0.0
        self._lam_jl_f = 0.0

        # CLF convergence rate (updated by the dynamic scheduler)
        self.gamma_clf = cfg.GAMMA_CLF_DEFAULT

        # Telemetry: the representative slack weight published to the dashboard
        self.weight_slack = cfg.BASE_WEIGHT_SLACK

        # Joint-limit helpers + constant constraint blocks for the limit rows
        self.dq_max_safe = np.zeros(self.n_joints)
        self.dq_min_safe = np.zeros(self.n_joints)
        I_joints = np.eye(self.n_joints)
        Zero_block = np.zeros((self.n_joints, self.n_slacks))
        self.C_max = np.hstack([-I_joints, Zero_block])  # -dq <= dq_max  ->  -dq >= -dq_max ... (quadprog form)
        self.C_min = np.hstack([I_joints, Zero_block])
        print("[Controller] QP Memory Pre-Allocated Successfully.")

    @staticmethod
    def _solve_qp(H, g, C, b):
        # Wrapper around quadprog: min 1/2 x'Hx + g'x s.t. C'x >= b. Returns (x, lagrangians).
        try:
            g_flat = -g.flatten().astype(np.float64)  # quadprog maximizes a'x -> a = -g
            b_flat = b.flatten().astype(np.float64)
            xf, f, vu, imeq, lagrangians, iact = quadprog.solve_qp(
                H.astype(np.float64), g_flat, C.astype(np.float64), b_flat, meq=0)
            return np.array(xf), lagrangians
        except Exception as e:
            print(f"\033[91m[QP Error] No solution: {e}\033[0m")
            return None, None

    def _schedule_weights(self, dt):
        # Update per-arm slack weights and the shared CLF gamma from last loop's shadow prices.
        # Smooth the (noisy) shadow prices with a first-order LPF before they drive
        # the slack weights — this is what removes the abrupt weight jumps.
        filter_alpha = np.exp(-dt / cfg.SLACK_FILTER_TAU)
        self._lam_col_f = filter_alpha * self._lam_col_f + (1.0 - filter_alpha) * self.last_lambda_col
        self._lam_jr_f = filter_alpha * self._lam_jr_f + (1.0 - filter_alpha) * self.last_lambda_joints_right
        self._lam_jl_f = filter_alpha * self._lam_jl_f + (1.0 - filter_alpha) * self.last_lambda_joints_left

        # Decoupled dynamic slack weighting (per arm), driven by the SMOOTHED prices
        if cfg.DYNAMIC_SLACK_WEIGHT:
            max_shadow_r = max(self._lam_col_f, self._lam_jr_f)
            alpha_r = np.exp(-cfg.BETA * (max_shadow_r ** 2))
            weight_slack_r = cfg.BASE_WEIGHT_SLACK + alpha_r * (cfg.MAX_WEIGHT_SLACK - cfg.BASE_WEIGHT_SLACK)

            max_shadow_l = max(self._lam_col_f, self._lam_jl_f)
            alpha_l = np.exp(-cfg.BETA * (max_shadow_l ** 2))
            weight_slack_l = cfg.BASE_WEIGHT_SLACK + alpha_l * (cfg.MAX_WEIGHT_SLACK - cfg.BASE_WEIGHT_SLACK)
        else:
            weight_slack_r = cfg.BASE_WEIGHT_SLACK
            weight_slack_l = cfg.BASE_WEIGHT_SLACK

        # Dynamic gamma (CLF) scheduling with a time-explicit low-pass filter
        if cfg.DYNAMIC_GAMMA_CLF:
            alpha_gamma = np.exp(-cfg.BETA_GAMMA * self._lam_col_f)
            target_gamma = cfg.GAMMA_MIN + alpha_gamma * (cfg.GAMMA_MAX - cfg.GAMMA_MIN)
            filter_alpha_g = np.exp(-dt / cfg.GAMMA_FILTER_TAU)
            self.gamma_clf = (filter_alpha_g * self.gamma_clf) + ((1.0 - filter_alpha_g) * target_gamma)
        else:
            self.gamma_clf = cfg.GAMMA_CLF_DEFAULT

        return weight_slack_r, weight_slack_l

    def build_and_solve(self, kin, J_soft, h_soft, d_safe_dynamic,
                        right_motion, left_motion, xdot_r, xdot_l,
                        e_r, v_r, e_l, v_l, dt):
        """
        Build and solve the CLF-CBF-QP for this tick.

        Returns: (q_dot_safe, slack_r, slack_l, b_col, lambda_joints_total)
        """
        self.H.fill(0.0)
        self.g.fill(0.0)

        # --- Adaptive scheduling from previous loop's shadow prices ---
        weight_slack_r, weight_slack_l = self._schedule_weights(dt)
        # Representative slack weight for telemetry (average of both arms)
        self.weight_slack = (weight_slack_r + weight_slack_l) / 2.0

        # =========================================================
        # A. COST FUNCTION (damping + posture spring + slack penalty)
        # =========================================================
        H_brake = np.eye(self.n_joints) * cfg.DAMP

        # Posture centering term: a soft virtual spring toward q_neutral on the arm joints
        mask_center = np.zeros(self.n_joints)
        if kin.idx_right:
            mask_center[kin.idx_right] = 1.0
        if kin.idx_left:
            mask_center[kin.idx_left] = 1.0
        error_center = pin.difference(self.model, kin.q_neutral, kin.current_q)
        v_ref_center = -cfg.KP_POSTURE * error_center

        # --- Joint-limit avoidance secondary task (augmented posture) ---
        # Add a velocity that pushes each ACTIVE joint away from its nearer limit,
        # growing quadratically once past LIMIT_AVOID_THRESH of its half-range and
        # zero in the comfortable mid-range (Chan & Dubey 1995 spirit). This uses
        # the 7-DOF redundancy to reconfigure away from limits so the EE can reach
        # poses the hard limit-CBF would otherwise block. COST-only: the hard
        # constraints and the CLF/CBF math are untouched.
        if cfg.JOINT_LIMIT_AVOID:
            active_v = set(kin.idx_right + kin.idx_left)
            for joint in self.model.joints:
                if joint.id == 0 or joint.nq != 1:
                    continue
                idx_v = joint.idx_v
                if idx_v not in active_v:
                    continue
                idx_q = joint.idx_q
                q_u = self.model.upperPositionLimit[idx_q]
                q_l = self.model.lowerPositionLimit[idx_q]
                if not (q_u < 1e10 and q_l > -1e10):
                    continue
                rng = q_u - q_l
                if rng < 1e-6:
                    continue
                mid = 0.5 * (q_u + q_l)
                p = 2.0 * (kin.current_q[idx_q] - mid) / rng   # normalized pos in [-1, 1]
                ap = abs(p)
                if ap > cfg.LIMIT_AVOID_THRESH:
                    s = (ap - cfg.LIMIT_AVOID_THRESH) / (1.0 - cfg.LIMIT_AVOID_THRESH)
                    v_ref_center[idx_v] += -cfg.KP_LIMIT_AVOID * np.sign(p) * (s * s)

        H_center = np.diag(mask_center * cfg.W_CENTER)
        g_center = -(mask_center * cfg.W_CENTER) * v_ref_center

        # Top-left (joint) block
        self.H[:self.n_joints, :self.n_joints] = H_brake + H_center
        self.g[:self.n_joints] = g_center

        # Bottom-right (slack) block: first half -> right arm, second half -> left arm
        half_slacks = self.n_slacks // 2
        for i in range(half_slacks):
            self.H[self.n_joints + i, self.n_joints + i] = weight_slack_r
            self.H[self.n_joints + half_slacks + i, self.n_joints + half_slacks + i] = weight_slack_l

        C_stack, b_stack = [], []

        # =========================================================
        # B. TASK CONSTRAINTS (Perfect Scalar Inequality CLF)
        #    e^T (J dq) + delta >= e^T xdot_ref + gamma * V(e)
        # =========================================================
        def add_perfect_scalar_clf(ee_id, e_vec, xdot_ref_vec, slack_idx):
            if ee_id is None:
                return
            J_6D = pin.getFrameJacobian(self.model, kin.data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dim = len(e_vec)
            J_task = J_6D[:dim, :]

            # Diagonal task weights (heavily penalize position, barely orientation)
            W = cfg.TASK_WEIGHTS_6D[:dim]
            e_w = e_vec * W  # element-wise == W @ e for a diagonal W

            if cfg.COMPARISON_CLF:
                # Normalized (unit-error) formulation
                e_norm = np.linalg.norm(e_w)
                if e_norm > 1e-5:
                    e_unit = e_w / e_norm
                    row_q = np.dot(e_unit.T, J_task)
                    row_slack = np.zeros(self.n_slacks)
                    row_slack[slack_idx] = 1.0
                    C_stack.append(np.concatenate([row_q, row_slack]))
                    # b = e_unit^T xdot_ref + (gamma/2) ||e_w||
                    b_stack.append(np.dot(e_unit, xdot_ref_vec) + (0.5 * self.gamma_clf * e_norm))
            else:
                # Raw (un-normalized) formulation
                row_q = np.dot(e_w.T, J_task)
                row_slack = np.zeros(self.n_slacks)
                row_slack[slack_idx] = 1.0
                C_stack.append(np.concatenate([row_q, row_slack]))
                # b = (W e)^T xdot_ref + gamma * V(e),  V(e) = 0.5 e^T W e
                b_stack.append(np.dot(e_w, xdot_ref_vec) + 0.5 * self.gamma_clf * np.dot(e_vec, e_w))

        # Inject per-arm CLF rows only when that arm is tracking a reference
        if right_motion or xdot_r is not None:
            add_perfect_scalar_clf(kin.ee_id_right, e_r, v_r, 0)
        if left_motion or xdot_l is not None:
            add_perfect_scalar_clf(kin.ee_id_left, e_l, v_l, 1)

        # =========================================================
        # C. SAFETY CONSTRAINT (SoftMin CBF collision avoidance)
        #    J_soft dq >= -gamma_cbf * (h_soft - d_safe_dynamic)
        # =========================================================
        C_col_padded = np.concatenate([J_soft, np.zeros(self.n_slacks)])
        if cfg.DISABLE_CBF:
            b_col = -10000.0  # Practically infinite slack -> barrier never activates
        else:
            b_col = -cfg.GAMMA_CBF * (h_soft - d_safe_dynamic)

        # =========================================================
        # D. JOINT LIMITS (velocity-aware CBF buffer, upper + lower)
        # =========================================================
        self.dq_max_safe.fill(0.0)
        self.dq_min_safe.fill(0.0)
        active_indices = kin.idx_right + kin.idx_left
        for joint in self.model.joints:
            if joint.id == 0 or joint.nq != 1:
                continue
            idx_v = joint.idx_v
            if idx_v not in active_indices:  # Locked joints stay at zero (masked out)
                continue
            v_limit = self.model.velocityLimit[idx_v]
            self.dq_max_safe[idx_v] = v_limit
            self.dq_min_safe[idx_v] = -v_limit

            idx_q = joint.idx_q
            q_u = self.model.upperPositionLimit[idx_q]
            q_l = self.model.lowerPositionLimit[idx_q]
            q_now = kin.current_q[idx_q]
            v_now = kin.current_v[idx_v]

            # Velocity-aware buffer: dynamic_buffer = base + K_v * |v|
            dynamic_buffer = cfg.JOINT_LIMIT_BUFFER_BASE + (cfg.JOINT_LIMIT_K_V * abs(v_now))
            if q_u < 1e10:
                self.dq_max_safe[idx_v] = min(self.dq_max_safe[idx_v],
                                              cfg.P_GAIN_LIMITS * (q_u - q_now - dynamic_buffer))
            if q_l > -1e10:
                self.dq_min_safe[idx_v] = max(self.dq_min_safe[idx_v],
                                              -cfg.P_GAIN_LIMITS * (q_now - q_l - dynamic_buffer))

        # =========================================================
        # E. ASSEMBLE ALL CONSTRAINTS (collision, limits, task)
        # =========================================================
        C_all = [C_col_padded.reshape(1, -1)]
        b_all = [np.array([b_col])]
        C_all.append(self.C_max)
        b_all.append(-self.dq_max_safe)
        C_all.append(self.C_min)
        b_all.append(self.dq_min_safe)
        if C_stack:  # Task CLF rows (may be empty if no reference yet)
            C_all.append(np.vstack(C_stack))
            b_all.append(np.array(b_stack))

        # quadprog convention: C.T x >= b
        C_final = np.vstack(C_all).T
        b_final = np.concatenate(b_all)

        # =========================================================
        # F. SOLVE + SHADOW-PRICE EXTRACTION
        # =========================================================
        sol, lagrangians = self._solve_qp(self.H, self.g, C_final, b_final)
        if sol is None:
            # Infeasible: halt motion and reset the collision shadow-price memory
            self.last_lambda_col = 0.0
            return np.zeros(self.n_joints), 0.0, 0.0, b_col, np.zeros(self.n_joints)

        q_dot_safe = sol[:self.n_joints]
        slack_r = sol[-2]
        slack_l = sol[-1]

        # 1. Collision shadow price (row 0)
        self.last_lambda_col = float(lagrangians[0])
        # 2. Joint-limit shadow prices: upper rows then lower rows
        lambda_upper = np.array(lagrangians[1:1 + self.n_joints])
        lambda_lower = np.array(lagrangians[1 + self.n_joints:1 + 2 * self.n_joints])
        lambda_joints_total = lambda_upper + lambda_lower

        # Extract per-arm worst shadow price only when that arm is actively tracking
        if kin.idx_right and right_motion:
            self.last_lambda_joints_right = float(np.max(lambda_joints_total[kin.idx_right]))
        else:
            self.last_lambda_joints_right = 0.0
        if kin.idx_left and left_motion:
            self.last_lambda_joints_left = float(np.max(lambda_joints_total[kin.idx_left]))
        else:
            self.last_lambda_joints_left = 0.0

        return q_dot_safe, slack_r, slack_l, b_col, lambda_joints_total
