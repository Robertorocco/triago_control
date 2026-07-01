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

        # Lagrange multiplier memory (shadow prices fed back into scheduling).
        # last_lambda_col is now the MAX of the two per-arm CBF shadow prices
        # (kept for backward-compat with the slack scheduler, which wants "how
        # hard is EITHER barrier pushing"); the two are also tracked separately
        # for telemetry/plotting.
        self.last_lambda_col = 0.0
        self.last_lambda_cbf_right = 0.0
        self.last_lambda_cbf_left = 0.0
        self.last_lambda_joints_right = 0.0
        self.last_lambda_joints_left = 0.0

        # Lazy cache for the posture-field joint indexing (built on first solve).
        self._posture_cache = None

        # Live scale on the posture-task weight (1.0 = nominal). Dropped toward
        # POSTURE_GRASP_SCALE during autonomous precision phases (grasp/lift) so
        # the QP devotes the redundancy to precise tracking instead of posture.
        self.posture_scale = 1.0

        # Soft-task cost decomposition at the last solution (telemetry):
        # [E_damp, E_posture, E_slack] weighted squared energies. See build_and_solve.
        self.task_energies = np.zeros(3)

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

    def _posture_indices(self, kin):
        """Lazily build & cache the per-joint arrays for the posture field.

        Returns (v_idx, q_idx, mids, half_ranges) over the ACTIVE arm joints that
        are single-DOF and have finite limits. Cached after the first call since
        the model topology and active arm-joint set are static.
        """
        if self._posture_cache is not None:
            return self._posture_cache
        v_idx, q_idx, mids, half_ranges = [], [], [], []
        active_v = set(kin.idx_right + kin.idx_left)
        for joint in self.model.joints:
            if joint.id == 0 or joint.nq != 1:
                continue
            if joint.idx_v not in active_v:
                continue
            q_u = self.model.upperPositionLimit[joint.idx_q]
            q_l = self.model.lowerPositionLimit[joint.idx_q]
            if not (q_u < 1e10 and q_l > -1e10):
                continue
            rng = q_u - q_l
            if rng < 1e-6:
                continue
            v_idx.append(joint.idx_v)
            q_idx.append(joint.idx_q)
            mids.append(0.5 * (q_u + q_l))
            half_ranges.append(0.5 * rng)
        self._posture_cache = (
            np.array(v_idx, dtype=int), np.array(q_idx, dtype=int),
            np.array(mids, dtype=float), np.array(half_ranges, dtype=float))
        return self._posture_cache

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

    def build_and_solve(self, kin, J_soft_r, h_soft_r, J_soft_l, h_soft_l, d_safe_dynamic,
                        right_motion, left_motion, xdot_r, xdot_l,
                        e_r, v_r, e_l, v_l, dt, right_frozen=False, left_frozen=False,
                        tracking_boost_arm=None):
        """
        Build and solve the CLF-CBF-QP for this tick.

        Returns: (q_dot_safe, slack_r, slack_l, (b_col_r, b_col_l), lambda_joints_total)
        """
        self.H.fill(0.0)
        self.g.fill(0.0)

        # --- Adaptive scheduling from previous loop's shadow prices ---
        weight_slack_r, weight_slack_l = self._schedule_weights(dt)
        # COST DECOUPLING (single-arm teleop): an INACTIVE (frozen) arm gets a
        # FIXED maximal slack weight (no dynamic update), a fixed GAMMA_MAX CLF
        # convergence rate, and doubled joint damping — so its hold is rigid and
        # its solution is independent of whatever the active arm is doing. When
        # BOTH arms are active (neither frozen) nothing changes (kept dynamic).
        if right_frozen:
            weight_slack_r = cfg.MAX_WEIGHT_SLACK
        if left_frozen:
            weight_slack_l = cfg.MAX_WEIGHT_SLACK
        # Per-arm CLF convergence rate: frozen arm holds tight at GAMMA_MAX.
        gamma_r = cfg.GAMMA_MAX if right_frozen else self.gamma_clf
        gamma_l = cfg.GAMMA_MAX if left_frozen else self.gamma_clf
        # GRASP TRACKING BOOST: during autonomous grasp execution the grasping
        # (active) arm must converge tightly to the standoff/insertion reference,
        # so it is pinned to the MAX dynamic values — highest slack weight (track
        # hard, barely yield) and highest CLF gamma (fast error decay). This is
        # what lets GRASP_ALIGN converge inside tolerance instead of timing out.
        if tracking_boost_arm == 'right':
            weight_slack_r = cfg.MAX_WEIGHT_SLACK
            gamma_r = cfg.GAMMA_MAX
        elif tracking_boost_arm == 'left':
            weight_slack_l = cfg.MAX_WEIGHT_SLACK
            gamma_l = cfg.GAMMA_MAX
        # Representative slack weight for telemetry (average of both arms)
        self.weight_slack = (weight_slack_r + weight_slack_l) / 2.0

        # =========================================================
        # A. COST FUNCTION (damping + posture spring + slack penalty)
        # =========================================================
        # Joint velocity regularization (damping). Per-arm: an INACTIVE (frozen)
        # arm gets DOUBLE damping so its motion is heavily penalized and its QP
        # solution is decoupled from the active arm's demands.
        damp_vec = np.full(self.n_joints, cfg.DAMP)
        if right_frozen and kin.idx_right:
            damp_vec[kin.idx_right] = 2.0 * cfg.DAMP
        if left_frozen and kin.idx_left:
            damp_vec[kin.idx_left] = 2.0 * cfg.DAMP
        H_brake = np.diag(damp_vec)

        # Posture / joint-limit avoidance: repulsive POTENTIAL-FIELD reference.
        # ------------------------------------------------------------------
        # Replaces the old q_neutral spring + Chan&Dubey piecewise ramp. The
        # reference velocity is the negative gradient of a barrier potential that
        # diverges at each joint limit, evaluated on the NORMALIZED position
        # p = (q - mid)/half_range in [-1, 1] (range-independent, so every joint
        # is defended equally at the same fraction of its travel):
        #     H(p)       = 1/(1-p)^2 + 1/(1+p)^2
        #     dH/dp      = 2/(1-p)^3 - 2/(1+p)^3
        #     v_ref      = -K_GRADIENT * dH/dp        (clamped to +/- V_MAX_POSTURE)
        # Near-zero in the comfortable mid-range (CLF keeps tracking priority) and
        # explodes (clamped) only near a limit, reconfiguring the redundant DOF.
        # COST-only: the hard CLF/CBF/limit constraints are untouched.
        mask_center = np.zeros(self.n_joints)
        v_ref_center = np.zeros(self.n_joints)
        v_idx, q_idx, mids, half_ranges = self._posture_indices(kin)
        if v_idx.size > 0:
            # Normalized position, GUARDED strictly inside (-1, 1). This guard is
            # essential: at/over a limit the raw cube would flip sign and PUSH the
            # joint further out (runaway). Clamping p keeps the gradient finite and
            # correctly-signed; the output clamp below bounds the magnitude.
            EPS = 1e-3
            p = (kin.current_q[q_idx] - mids) / half_ranges
            p = np.clip(p, -1.0 + EPS, 1.0 - EPS)
            gap_hi = 1.0 - p     # > 0 by the clamp
            gap_lo = 1.0 + p     # > 0 by the clamp
            grad = 2.0 / gap_hi**3 - 2.0 / gap_lo**3      # dH/dp
            v = np.clip(-cfg.K_GRADIENT * grad, -cfg.V_MAX_POSTURE, cfg.V_MAX_POSTURE)
            mask_center[v_idx] = 1.0
            v_ref_center[v_idx] = v

        # Effective posture weight (scaled down during autonomous precision phases
        # via self.posture_scale, set by the controller from the grasp state).
        w_center = cfg.W_CENTER * self.posture_scale
        H_center = np.diag(mask_center * w_center)
        g_center = -(mask_center * w_center) * v_ref_center

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
        def add_perfect_scalar_clf(ee_id, e_vec, xdot_ref_vec, slack_idx, gamma):
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
                    b_stack.append(np.dot(e_unit, xdot_ref_vec) + (0.5 * gamma * e_norm))
            else:
                # Raw (un-normalized) formulation
                row_q = np.dot(e_w.T, J_task)
                row_slack = np.zeros(self.n_slacks)
                row_slack[slack_idx] = 1.0
                C_stack.append(np.concatenate([row_q, row_slack]))
                # b = (W e)^T xdot_ref + gamma * V(e),  V(e) = 0.5 e^T W e
                b_stack.append(np.dot(e_w, xdot_ref_vec) + 0.5 * gamma * np.dot(e_vec, e_w))

        # Inject per-arm CLF rows only when that arm is tracking a reference.
        # The frozen arm uses GAMMA_MAX (gamma_r / gamma_l set above) so it holds
        # its pose tightly and independently of the active arm.
        if right_motion or xdot_r is not None:
            add_perfect_scalar_clf(kin.ee_id_right, e_r, v_r, 0, gamma_r)
        if left_motion or xdot_l is not None:
            add_perfect_scalar_clf(kin.ee_id_left, e_l, v_l, 1, gamma_l)

        # =========================================================
        # C. SAFETY CONSTRAINTS (TWO INDEPENDENT per-arm SoftMin CBFs)
        #    J_soft_X dq >= -gamma_cbf * (h_soft_X - d_safe_dynamic)   for X in {R, L}
        #
        # Replaces the single combined SoftMin row. Each row's gradient only has
        # nonzero columns in the joints of the pairs that actually contributed to
        # IT (see CollisionManager.compute_softmin_jacobian): an inter-arm pair
        # (or two held cylinders) appears in BOTH rows (preserves "arm A may
        # yield to help arm B"), but a pair touching only arm A's geometry vs. a
        # static obstacle NEVER appears in arm B's row (eliminates the spurious
        # coupling/oscillation where an idle arm twitched because the OTHER arm
        # neared an unrelated obstacle).
        # =========================================================
        C_col_r_padded = np.concatenate([J_soft_r, np.zeros(self.n_slacks)])
        C_col_l_padded = np.concatenate([J_soft_l, np.zeros(self.n_slacks)])
        if cfg.DISABLE_CBF:
            b_col_r = -10000.0  # Practically infinite slack -> barrier never activates
            b_col_l = -10000.0
        else:
            b_col_r = -cfg.GAMMA_CBF * (h_soft_r - d_safe_dynamic)
            b_col_l = -cfg.GAMMA_CBF * (h_soft_l - d_safe_dynamic)

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
        # E. ASSEMBLE ALL CONSTRAINTS (collision x2, limits, task)
        # =========================================================
        C_all = [C_col_r_padded.reshape(1, -1), C_col_l_padded.reshape(1, -1)]
        b_all = [np.array([b_col_r]), np.array([b_col_l])]
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
            self.last_lambda_cbf_right = 0.0
            self.last_lambda_cbf_left = 0.0
            self.task_energies = np.zeros(3)
            return np.zeros(self.n_joints), 0.0, 0.0, (b_col_r, b_col_l), np.zeros(self.n_joints)

        q_dot_safe = sol[:self.n_joints]
        slack_r = sol[-2]
        slack_l = sol[-1]

        # 1. Collision shadow prices (rows 0 = right CBF, 1 = left CBF)
        self.last_lambda_cbf_right = float(lagrangians[0])
        self.last_lambda_cbf_left = float(lagrangians[1])
        # Backward-compat scalar (used by the slack scheduler's "how hard is
        # EITHER barrier pushing" logic): the max of the two per-arm prices.
        self.last_lambda_col = max(self.last_lambda_cbf_right, self.last_lambda_cbf_left)
        # 2. Joint-limit shadow prices: upper rows then lower rows (offset by +2
        #    now that there are TWO collision rows instead of one).
        lambda_upper = np.array(lagrangians[2:2 + self.n_joints])
        lambda_lower = np.array(lagrangians[2 + self.n_joints:2 + 2 * self.n_joints])
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

        # Soft-task cost decomposition (telemetry / authority plot). Weighted
        # squared energies actually realised at the QP solution:
        #   E_damp    = DAMP * ||q_dot||^2                  (regularisation effort)
        #   E_posture = W_CENTER * ||q_dot_arm - v_ref||^2  (posture/limit task)
        #   E_slack   = w_r*delta_r^2 + w_l*delta_l^2        (CLF task relaxation)
        # Their normalised shares (computed in the plotter) show where the QP's
        # objective effort/conflict concentrates each tick. The HARD-constraint
        # authority is the KKT dual (shadow prices) already published separately.
        dq_post = (q_dot_safe - v_ref_center) * mask_center
        e_damp = cfg.DAMP * float(q_dot_safe @ q_dot_safe)
        e_posture = w_center * float(dq_post @ dq_post)
        e_slack = float(weight_slack_r * slack_r ** 2 + weight_slack_l * slack_l ** 2)
        self.task_energies = np.array([e_damp, e_posture, e_slack])

        return q_dot_safe, slack_r, slack_l, (b_col_r, b_col_l), lambda_joints_total
