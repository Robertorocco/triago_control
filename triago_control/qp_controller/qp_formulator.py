# qp_formulator.py
"""CLF-CBF-QP core: builds min 1/2 x'Hx + g'x s.t. C'x >= b over x = [q_dot(nv), delta_R, delta_L] and solves with quadprog."""

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

        # Shadow-price memory fed back into scheduling; last_lambda_col = max of the two per-arm CBF prices.
        self.last_lambda_col = 0.0
        self.last_lambda_cbf_right = 0.0
        self.last_lambda_cbf_left = 0.0
        self.last_lambda_joints_right = 0.0
        self.last_lambda_joints_left = 0.0

        # Lazy cache for the posture-field joint indexing (built on first solve).
        self._posture_cache = None

        # Lazy active-arm mask for the rate-damping term (must stay masked to the arm joints).
        self._rate_mask = None

        # Global posture-weight scale, dropped toward POSTURE_GRASP_SCALE during precision phases.
        self.posture_scale = 1.0

        # Previous tick's solved joint velocity (rate-damping anchor fallback), zero at startup.
        self.last_dq_safe = np.zeros(self.n_joints)

        # Telemetry: [0:4] combined [E_damp, E_posture, E_slack, E_rate], [4:8] right, [8:12] left.
        self.task_energies = np.zeros(12)

        # Per-arm LPF'd shadow prices so one arm's CBF pressure never leaks into the other's schedule.
        self._lam_col_r_f = 0.0
        self._lam_col_l_f = 0.0
        self._lam_jr_f = 0.0
        self._lam_jl_f = 0.0

        # CLF convergence rate, updated per arm by _schedule_weights.
        self.gamma_clf_r = cfg.GAMMA_CLF_DEFAULT
        self.gamma_clf_l = cfg.GAMMA_CLF_DEFAULT

        # weight_slack_r/l are what the QP cost uses; weight_slack is their average (telemetry).
        self.weight_slack = cfg.BASE_WEIGHT_SLACK
        self.weight_slack_r = cfg.BASE_WEIGHT_SLACK
        self.weight_slack_l = cfg.BASE_WEIGHT_SLACK

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
        """Updates per-arm slack weights and CLF gamma from the LPF'd shadow prices of the last solve."""
        filter_alpha = np.exp(-dt / cfg.SLACK_FILTER_TAU)
        self._lam_col_r_f = filter_alpha * self._lam_col_r_f + (1.0 - filter_alpha) * self.last_lambda_cbf_right
        self._lam_col_l_f = filter_alpha * self._lam_col_l_f + (1.0 - filter_alpha) * self.last_lambda_cbf_left
        self._lam_jr_f = filter_alpha * self._lam_jr_f + (1.0 - filter_alpha) * self.last_lambda_joints_right
        self._lam_jl_f = filter_alpha * self._lam_jl_f + (1.0 - filter_alpha) * self.last_lambda_joints_left

        # Per-arm worst shadow price: that arm's own collision OR joint-limit price, never the other's.
        max_shadow_r = max(self._lam_col_r_f, self._lam_jr_f)
        max_shadow_l = max(self._lam_col_l_f, self._lam_jl_f)

        # Decoupled per-arm slack weighting; a second LPF on the weight itself smooths
        # BETA's steep squared-exponential response to fast-rising lambdas.
        if cfg.DYNAMIC_SLACK_WEIGHT:
            alpha_r = np.exp(-cfg.BETA * (max_shadow_r ** 2))
            target_slack_r = cfg.BASE_WEIGHT_SLACK + alpha_r * (cfg.MAX_WEIGHT_SLACK - cfg.BASE_WEIGHT_SLACK)

            alpha_l = np.exp(-cfg.BETA * (max_shadow_l ** 2))
            target_slack_l = cfg.BASE_WEIGHT_SLACK + alpha_l * (cfg.MAX_WEIGHT_SLACK - cfg.BASE_WEIGHT_SLACK)

            filter_alpha_w = np.exp(-dt / cfg.WEIGHT_SLACK_FILTER_TAU)
            self.weight_slack_r = filter_alpha_w * self.weight_slack_r + (1.0 - filter_alpha_w) * target_slack_r
            self.weight_slack_l = filter_alpha_w * self.weight_slack_l + (1.0 - filter_alpha_w) * target_slack_l
        else:
            self.weight_slack_r = cfg.BASE_WEIGHT_SLACK
            self.weight_slack_l = cfg.BASE_WEIGHT_SLACK
        weight_slack_r, weight_slack_l = self.weight_slack_r, self.weight_slack_l

        # Same quadratic-tolerance law as the slack weight, per arm, over [GAMMA_MIN, GAMMA_MAX].
        if cfg.DYNAMIC_GAMMA_CLF:
            alpha_gamma_r = np.exp(-cfg.BETA_GAMMA * (max_shadow_r ** 2))
            target_gamma_r = cfg.GAMMA_MIN + alpha_gamma_r * (cfg.GAMMA_MAX - cfg.GAMMA_MIN)
            alpha_gamma_l = np.exp(-cfg.BETA_GAMMA * (max_shadow_l ** 2))
            target_gamma_l = cfg.GAMMA_MIN + alpha_gamma_l * (cfg.GAMMA_MAX - cfg.GAMMA_MIN)

            filter_alpha_g = np.exp(-dt / cfg.GAMMA_FILTER_TAU)
            self.gamma_clf_r = filter_alpha_g * self.gamma_clf_r + (1.0 - filter_alpha_g) * target_gamma_r
            self.gamma_clf_l = filter_alpha_g * self.gamma_clf_l + (1.0 - filter_alpha_g) * target_gamma_l
        else:
            self.gamma_clf_r = cfg.GAMMA_CLF_DEFAULT
            self.gamma_clf_l = cfg.GAMMA_CLF_DEFAULT

        return weight_slack_r, weight_slack_l

    def _joint_name_by_idx_v(self, idx_v):
        """Best-effort joint name for a velocity index (cached); resolves name->id, never fatal."""
        if not hasattr(self, '_idx_v_name_map'):
            self._idx_v_name_map = {}
            for name in self.model.names:
                if name == 'universe':
                    continue
                try:
                    jid = self.model.getJointId(name)
                    joint = self.model.joints[jid]
                    if joint.nv != 1:
                        continue
                    self._idx_v_name_map[joint.idx_v] = name
                except Exception as e:  # noqa: BLE001 -- best-effort label, never fatal
                    print(f"  (skipping joint '{name}' in idx_v name cache: {e})")
        return self._idx_v_name_map.get(idx_v, f"idx_v={idx_v}")

    def _diagnose_infeasibility(self, kin, C_final, b_final,
                                J_soft_r, b_col_r, h_soft_r, d_safe_r,
                                J_soft_l, b_col_l, h_soft_l, d_safe_l):
        """Throttled console post-mortem for an infeasible solve (CLF rows have slack, so only
        non-finite inputs, inverted velocity boxes, or unreachable/conflicting CBF rows can fail)."""
        import time as _time
        now = _time.monotonic()
        if now - getattr(self, '_last_infeas_diag', 0.0) < 1.0:
            return
        self._last_infeas_diag = now

        print("\033[93m[QP Infeasibility Diagnosis]\033[0m")
        print(f"  live flags: ENABLE_RATE_DAMPING={cfg.ENABLE_RATE_DAMPING}  "
              f"RATE_WEIGHT={cfg.RATE_WEIGHT}  DAMP={cfg.DAMP}  "
              f"(if these do not match config.py on disk, the running node was "
              f"never restarted after the edit)")

        # 1. Non-finite values (a single NaN in C or b breaks quadprog outright)
        clean = True
        for name, arr in (("C", C_final), ("b", b_final), ("H", self.H), ("g", self.g)):
            n_bad = int((~np.isfinite(np.asarray(arr))).sum())
            if n_bad:
                clean = False
                print(f"  -> {n_bad} NON-FINITE entries in {name} -- upstream "
                      f"numerics (h_soft_r={h_soft_r}, h_soft_l={h_soft_l})")

        # 2. Inverted per-joint velocity boxes
        inverted = np.where(self.dq_min_safe > self.dq_max_safe + 1e-12)[0]
        for idx_v in inverted:
            clean = False
            v_now = float(kin.current_v[idx_v]) if kin.current_v is not None else float('nan')
            print(f"  -> INVERTED velocity box on {self._joint_name_by_idx_v(idx_v)}: "
                  f"dq_min={self.dq_min_safe[idx_v]:+.4f} > dq_max={self.dq_max_safe[idx_v]:+.4f}  "
                  f"(measured v={v_now:+.4f} rad/s -- a large |v| inflates the "
                  f"joint-limit buffer on BOTH sides; a velocity SPIKE from a "
                  f"q jump, e.g. a sim reset, can invert the box transiently)")

        # 3. Per-CBF-row achievability over the velocity box
        for side, J, b_c, h, dsafe in (("RIGHT", J_soft_r, b_col_r, h_soft_r, d_safe_r),
                                       ("LEFT", J_soft_l, b_col_l, h_soft_l, d_safe_l)):
            J = np.asarray(J, dtype=float)
            if not np.isfinite(J).all() or not np.isfinite(b_c):
                continue  # already reported in check 1
            best = float(np.sum(np.where(J > 0, J * self.dq_max_safe, J * self.dq_min_safe)))
            if best + 1e-12 < b_c:
                clean = False
                print(f"  -> {side} CBF row UNSATISFIABLE within the velocity box: "
                      f"needs J.dq >= {b_c:+.4f}, best achievable {best:+.4f}  "
                      f"(h_soft={h:.4f}, d_safe={dsafe:.4f}, barrier deficit "
                      f"{dsafe - h:+.4f} m -- the box cannot brake/retreat fast enough)")

        if clean:
            print("  -> no single check fired: the conflict is JOINT -- most "
                  "likely the two CBF rows demand incompatible directions inside "
                  "the shared velocity box (inter-arm geometry), or a numerical "
                  "degeneracy in quadprog's active-set path. Log the current "
                  "q/h values and inspect fig2/fig4 for this instant.")

    def _safe_zero_solution(self, b_col_r=0.0, b_col_l=0.0):
        """Fallback for any failed solve: halt motion, reset shadow prices, sanitize the CBF telemetry."""
        # last_dq_safe is deliberately NOT zeroed: anchoring on a zeroed value makes one bad tick self-sustain.
        b_col_r = float(b_col_r) if np.isfinite(b_col_r) else 0.0
        b_col_l = float(b_col_l) if np.isfinite(b_col_l) else 0.0
        self.last_lambda_col = 0.0
        self.last_lambda_cbf_right = 0.0
        self.last_lambda_cbf_left = 0.0
        self.task_energies = np.zeros(4)
        return (np.zeros(self.n_joints), 0.0, 0.0,
               (b_col_r, b_col_l), np.zeros(self.n_joints))

    def build_and_solve(self, kin, J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                        d_safe_dynamic_r, d_safe_dynamic_l,
                        right_motion, left_motion, xdot_r, xdot_l,
                        e_r, v_r, e_l, v_l, dt, right_frozen=False, left_frozen=False,
                        tracking_boost_arm=None, orient_boost_arms=()):
        """Builds and solves this tick's QP; returns (q_dot_safe, slack_r, slack_l, (b_col_r, b_col_l), lambda_joints)."""
        self.H.fill(0.0)
        self.g.fill(0.0)

        weight_slack_r, weight_slack_l = self._schedule_weights(dt)
        # A frozen (inactive) arm is pinned to max slack weight, GAMMA_MAX and doubled damping,
        # so its hold is rigid and decoupled from whatever the active arm is doing.
        if right_frozen:
            weight_slack_r = cfg.MAX_WEIGHT_SLACK
        if left_frozen:
            weight_slack_l = cfg.MAX_WEIGHT_SLACK
        gamma_r = cfg.GAMMA_MAX if right_frozen else self.gamma_clf_r
        gamma_l = cfg.GAMMA_MAX if left_frozen else self.gamma_clf_l
        # Grasp tracking boost: the autonomously-grasping arm is pinned to the max dynamic values
        # so alignment converges inside tolerance instead of timing out.
        if tracking_boost_arm == 'right':
            weight_slack_r = cfg.MAX_WEIGHT_SLACK
            gamma_r = cfg.GAMMA_MAX
        elif tracking_boost_arm == 'left':
            weight_slack_l = cfg.MAX_WEIGHT_SLACK
            gamma_l = cfg.GAMMA_MAX
        # Telemetry: per-arm slack weights are exactly what the Hessian's slack block uses.
        self.weight_slack = (weight_slack_r + weight_slack_l) / 2.0
        self.weight_slack_r = weight_slack_r
        self.weight_slack_l = weight_slack_l

        # =========================================================
        # A. COST FUNCTION (damping + posture spring + slack penalty)
        # =========================================================
        # Joint-velocity damping; a frozen arm gets double damping to stay decoupled.
        damp_vec = np.full(self.n_joints, cfg.DAMP)
        if right_frozen and kin.idx_right:
            damp_vec[kin.idx_right] = 2.0 * cfg.DAMP
        if left_frozen and kin.idx_left:
            damp_vec[kin.idx_left] = 2.0 * cfg.DAMP
        H_brake = np.diag(damp_vec)

        # Posture field: v_ref = -K_GRADIENT * dH/dp with H(p) = 1/(1-p)^2 + 1/(1+p)^2 on the
        # normalized position p = (q-mid)/half_range -- near-zero mid-range, diverges at limits.
        # Cost-only: the hard CLF/CBF/limit constraints are untouched.
        mask_center = np.zeros(self.n_joints)
        v_ref_center = np.zeros(self.n_joints)
        v_idx, q_idx, mids, half_ranges = self._posture_indices(kin)
        if v_idx.size > 0:
            # Clamp p strictly inside (-1,1): at/over a limit the raw cube flips sign and pushes OUT.
            EPS = 1e-3
            p = (kin.current_q[q_idx] - mids) / half_ranges
            p = np.clip(p, -1.0 + EPS, 1.0 - EPS)
            gap_hi = 1.0 - p     # > 0 by the clamp
            gap_lo = 1.0 + p     # > 0 by the clamp
            grad = 2.0 / gap_hi**3 - 2.0 / gap_lo**3      # dH/dp
            v = np.clip(-cfg.K_GRADIENT * grad, -cfg.V_MAX_POSTURE, cfg.V_MAX_POSTURE)
            mask_center[v_idx] = 1.0
            v_ref_center[v_idx] = v

        # Effective posture weight, scaled down during autonomous precision phases.
        w_center = cfg.W_CENTER * self.posture_scale
        H_center = np.diag(mask_center * w_center)
        g_center = -(mask_center * w_center) * v_ref_center

        # Rate damping ||dq - dq_measured||^2, arm joints ONLY: locked joints are pinned to dq=0 by
        # two opposing box rows, and an unmasked rate gradient contests that pin -- two linearly
        # dependent rows enter quadprog's active set and the dual method fails outright.
        if self._rate_mask is None:
            self._rate_mask = np.zeros(self.n_joints)
            if kin.idx_right:
                self._rate_mask[kin.idx_right] = 1.0
            if kin.idx_left:
                self._rate_mask[kin.idx_left] = 1.0
        if cfg.ENABLE_RATE_DAMPING:
            if cfg.RATE_DAMPING_VS_MEASURED and kin.current_v is not None:
                dq_prev_rate = kin.current_v
            else:
                dq_prev_rate = self.last_dq_safe
            # Relaxed RATE_WEIGHT_GRASP on the boosted arm and whenever tracking error is large:
            # full RATE_WEIGHT anchored to a near-zero measured velocity self-reinforces a freeze
            # (every command pulled back to ~zero); full anti-oscillation weight only matters near convergence.
            err_r = float(np.linalg.norm(e_r[:3]))
            err_l = float(np.linalg.norm(e_l[:3]))
            rate_weight_r = (cfg.RATE_WEIGHT_GRASP
                            if (tracking_boost_arm == 'right' or err_r > cfg.STALL_ERR_POS_THRESH)
                            else cfg.RATE_WEIGHT)
            rate_weight_l = (cfg.RATE_WEIGHT_GRASP
                            if (tracking_boost_arm == 'left' or err_l > cfg.STALL_ERR_POS_THRESH)
                            else cfg.RATE_WEIGHT)
            rate_weight_vec = np.zeros(self.n_joints)
            if kin.idx_right:
                rate_weight_vec[kin.idx_right] = rate_weight_r
            if kin.idx_left:
                rate_weight_vec[kin.idx_left] = rate_weight_l
            H_rate = np.diag(rate_weight_vec)
            # Built by SELECTING at arm indices, never mask-multiplying: 0*nan == nan, so a non-finite
            # entry elsewhere in dq_prev_rate (e.g. an unsensed wheel joint) would poison the whole solve.
            g_rate = np.zeros(self.n_joints)
            if kin.idx_right:
                g_rate[kin.idx_right] = -rate_weight_r * dq_prev_rate[kin.idx_right]
            if kin.idx_left:
                g_rate[kin.idx_left] = -rate_weight_l * dq_prev_rate[kin.idx_left]
        else:
            dq_prev_rate = self.last_dq_safe
            H_rate = 0.0
            g_rate = 0.0

        # Top-left (joint) block
        self.H[:self.n_joints, :self.n_joints] = H_brake + H_center + H_rate
        self.g[:self.n_joints] = g_center + g_rate

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
        def add_perfect_scalar_clf(ee_id, e_vec, xdot_ref_vec, slack_idx, gamma, task_weights):
            if ee_id is None:
                return
            J_6D = pin.getFrameJacobian(self.model, kin.data, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dim = len(e_vec)
            J_task = J_6D[:dim, :]

            # Diagonal task weights: position-dominant nominally, orientation-boosted during grasp work.
            W = task_weights[:dim]
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

        # Static per-phase weight swap: arms in orient_boost_arms (grasp work or carrying an
        # object) use the orientation-boosted grasp weights; decoupled from tracking_boost_arm.
        W_task_r = cfg.TASK_WEIGHTS_6D_GRASP if 'right' in orient_boost_arms else cfg.TASK_WEIGHTS_6D
        W_task_l = cfg.TASK_WEIGHTS_6D_GRASP if 'left' in orient_boost_arms else cfg.TASK_WEIGHTS_6D

        # Per-arm CLF rows are injected only when that arm tracks a reference.
        if right_motion or xdot_r is not None:
            add_perfect_scalar_clf(kin.ee_id_right, e_r, v_r, 0, gamma_r, W_task_r)
        if left_motion or xdot_l is not None:
            add_perfect_scalar_clf(kin.ee_id_left, e_l, v_l, 1, gamma_l, W_task_l)

        # =========================================================
        # C. SAFETY CONSTRAINTS: two independent per-arm SoftMin CBF rows,
        #    J_soft_X dq >= -gamma_cbf * (h_soft_X - d_safe_dynamic_X).
        # An inter-arm pair contributes to BOTH rows (either arm may yield for the other);
        # a pair touching only one arm never recruits the other arm's joints.
        # =========================================================
        C_col_r_padded = np.concatenate([J_soft_r, np.zeros(self.n_slacks)])
        C_col_l_padded = np.concatenate([J_soft_l, np.zeros(self.n_slacks)])
        if cfg.DISABLE_CBF:
            b_col_r = -10000.0  # practically infinite slack: the barrier never activates
            b_col_l = -10000.0
        else:
            b_col_r = -cfg.GAMMA_CBF * (h_soft_r - d_safe_dynamic_r)
            b_col_l = -cfg.GAMMA_CBF * (h_soft_l - d_safe_dynamic_l)

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
            if idx_v not in active_indices:  # locked joints stay pinned at zero
                continue
            v_limit = self.model.velocityLimit[idx_v]
            self.dq_max_safe[idx_v] = v_limit
            self.dq_min_safe[idx_v] = -v_limit

            idx_q = joint.idx_q
            q_u = self.model.upperPositionLimit[idx_q]
            q_l = self.model.lowerPositionLimit[idx_q]
            q_now = kin.current_q[idx_q]
            v_now = kin.current_v[idx_v]

            # Velocity-aware buffer: a fast joint starts braking earlier (base + K_v * |v|).
            dynamic_buffer = cfg.JOINT_LIMIT_BUFFER_BASE + (cfg.JOINT_LIMIT_K_V * abs(v_now))
            if q_u < 1e10:
                self.dq_max_safe[idx_v] = min(self.dq_max_safe[idx_v],
                                              cfg.P_GAIN_LIMITS * (q_u - q_now - dynamic_buffer))
            if q_l > -1e10:
                self.dq_min_safe[idx_v] = max(self.dq_min_safe[idx_v],
                                              -cfg.P_GAIN_LIMITS * (q_now - q_l - dynamic_buffer))
            # No hard slew-rate box here: it cannot track a fast-turning CBF row and goes infeasible.

        # =========================================================
        # E. ASSEMBLE ALL CONSTRAINTS (collision x2, limits, task)
        # =========================================================
        C_all = [C_col_r_padded.reshape(1, -1), C_col_l_padded.reshape(1, -1)]
        b_all = [np.array([b_col_r]), np.array([b_col_l])]
        C_all.append(self.C_max)
        b_all.append(-self.dq_max_safe)
        C_all.append(self.C_min)
        b_all.append(self.dq_min_safe)
        if C_stack:  # task CLF rows (empty until a reference arrives)
            C_all.append(np.vstack(C_stack))
            b_all.append(np.array(b_stack))

        # quadprog convention: C.T x >= b
        C_final = np.vstack(C_all).T
        b_final = np.concatenate(b_all)

        # =========================================================
        # F. SOLVE + SHADOW-PRICE EXTRACTION
        # =========================================================
        # NaN/Inf guard BEFORE the solve: quadprog does not reliably raise on non-finite input,
        # it can silently return a NaN "solution" that would sail straight to the hardware command.
        if not (np.isfinite(self.H).all() and np.isfinite(self.g).all()
                and np.isfinite(C_final).all() and np.isfinite(b_final).all()):
            print("\033[91m[QP Error] NON-FINITE H/g/C/b -- refusing to call quadprog "
                  "(would silently return garbage, not raise).\033[0m")
            self._diagnose_infeasibility(kin, C_final, b_final,
                                         J_soft_r, b_col_r, h_soft_r, d_safe_dynamic_r,
                                         J_soft_l, b_col_l, h_soft_l, d_safe_dynamic_l)
            return self._safe_zero_solution(b_col_r, b_col_l)

        sol, lagrangians = self._solve_qp(self.H, self.g, C_final, b_final)
        if sol is None:
            # Infeasible tick: safe halt (last_dq_safe intentionally kept, see _safe_zero_solution).
            self._diagnose_infeasibility(kin, C_final, b_final,
                                         J_soft_r, b_col_r, h_soft_r, d_safe_dynamic_r,
                                         J_soft_l, b_col_l, h_soft_l, d_safe_dynamic_l)
            return self._safe_zero_solution(b_col_r, b_col_l)

        if not np.isfinite(sol).all():
            # Finite inputs can still yield a non-finite solution (ill-conditioned active set).
            print("\033[91m[QP Error] quadprog returned a NON-FINITE solution from "
                  "finite inputs (numerical ill-conditioning) -- discarding it.\033[0m")
            return self._safe_zero_solution(b_col_r, b_col_l)

        q_dot_safe = sol[:self.n_joints]
        slack_r = sol[-2]
        slack_l = sol[-1]

        # Collision shadow prices (row 0 = right CBF, row 1 = left CBF).
        self.last_lambda_cbf_right = float(lagrangians[0])
        self.last_lambda_cbf_left = float(lagrangians[1])
        # Backward-compat scalar for the slack scheduler: the worse of the two barriers.
        self.last_lambda_col = max(self.last_lambda_cbf_right, self.last_lambda_cbf_left)
        # Joint-limit shadow prices: upper rows then lower rows, offset by the 2 collision rows.
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

        # Telemetry cost decomposition per arm: E_damp, E_posture, E_slack, E_rate.
        def _sum_sq(vec, idx):
            return float(np.sum(vec[idx] ** 2)) if idx else 0.0

        idx_r, idx_l = kin.idx_right, kin.idx_left
        dq_post = (q_dot_safe - v_ref_center) * mask_center
        e_damp_r = cfg.DAMP * _sum_sq(q_dot_safe, idx_r)
        e_damp_l = cfg.DAMP * _sum_sq(q_dot_safe, idx_l)
        e_posture_r = w_center * _sum_sq(dq_post, idx_r)
        e_posture_l = w_center * _sum_sq(dq_post, idx_l)
        e_slack_r = float(weight_slack_r * slack_r ** 2)
        e_slack_l = float(weight_slack_l * slack_l ** 2)
        # Selected at arm indices for the same NaN-poisoning reason as g_rate above.
        if cfg.ENABLE_RATE_DAMPING:
            def _rate_energy(idx):
                if not idx:
                    return 0.0
                resid = q_dot_safe[idx] - dq_prev_rate[idx]
                return float(cfg.RATE_WEIGHT * np.sum(resid ** 2))
            e_rate_r = _rate_energy(idx_r)
            e_rate_l = _rate_energy(idx_l)
        else:
            e_rate_r = 0.0
            e_rate_l = 0.0
        # [0:4] combined totals (for plotter.py); [4:8] right only; [8:12] left only.
        self.task_energies = np.array([
            e_damp_r + e_damp_l, e_posture_r + e_posture_l,
            e_slack_r + e_slack_l, e_rate_r + e_rate_l,
            e_damp_r, e_posture_r, e_slack_r, e_rate_r,
            e_damp_l, e_posture_l, e_slack_l, e_rate_l,
        ])
        self.last_dq_safe = q_dot_safe.copy()

        return q_dot_safe, slack_r, slack_l, (b_col_r, b_col_l), lambda_joints_total
