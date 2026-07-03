# rrt_planner.py
"""
Joint-Space RRT-Connect Planner for Local Minima Escape.

Bidirectional RRT-Connect in the 7D joint-space of one arm, using the SAME
hppfcl collision model the CBF uses (so the planner and the real-time
controller agree on what's safe). Runs in a background thread, produces a
sequence of Cartesian waypoints (derived from collision-free joint configs
via FK) that the Reference Governor feeds to the CLF.

Key design choices:
    - Plans in JOINT SPACE (7D for one arm), not task space — this correctly
      handles chain-level collisions (elbow/forearm near obstacles) that a
      Cartesian planner would miss.
    - Uses the EXISTING Pinocchio model + hppfcl collision model — no new
      dependencies, no model mismatch with the CBF.
    - Output is Cartesian waypoints (not joint commands) — the QP-CLF-CBF
      still drives execution, preserving ALL safety guarantees.
    - Abortable: an external flag can cancel the planning thread at any time
      (e.g., user moves the reference → abort).
    - Thread-safe: communicates results via a simple list + lock pattern.
"""

import numpy as np
import pinocchio as pin
import time
import threading
try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl
import triago_control.qp_controller.config as cfg


class RRTNode:
    """A single node in the RRT tree."""
    __slots__ = ('q', 'parent_idx')

    def __init__(self, q, parent_idx=-1):
        self.q = np.array(q, dtype=float)
        self.parent_idx = parent_idx


class RRTPlannerResult:
    """Container for a completed planning result."""
    def __init__(self):
        self.success = False
        self.cartesian_waypoints = []   # list of (pos_3d,) tuples
        self.joint_waypoints = []       # list of q_7d arrays
        self.planning_time_s = 0.0
        self.samples_used = 0
        self.raw_path_length = 0
        self.smoothed_path_length = 0
        self.total_cartesian_length = 0.0  # [m] sum of consecutive EE distances


class RRTPlanner:
    """Joint-space RRT-Connect planner for one arm.

    Instantiated ONCE per arm by the Reference Governor. The plan() method is
    meant to be called from a background thread (see plan_async / abort).
    """

    def __init__(self, arm_side, model, ee_frame_name):
        """
        Args:
            arm_side: 'right' or 'left'
            model: the Pinocchio model (shared, read-only during planning)
            ee_frame_name: e.g. 'gripper_right_grasping_link'
        """
        self.arm_side = arm_side
        self.model = model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = model.getFrameId(ee_frame_name) if model.existFrame(ee_frame_name) else None

        # Joint indices for THIS arm (velocity-vector indices, same as kin.idx_right/left)
        joint_names = cfg.RIGHT_JOINTS if arm_side == 'right' else cfg.LEFT_JOINTS
        self.arm_joint_ids = []  # Pinocchio joint IDs (for idx_q / idx_v lookup)
        self.arm_idx_q = []      # position indices in q
        self.arm_idx_v = []      # velocity indices in v (same length as arm_joint_ids)
        for name in joint_names:
            if model.existJointName(name):
                jid = model.getJointId(name)
                self.arm_joint_ids.append(jid)
                self.arm_idx_q.append(model.joints[jid].idx_q)
                self.arm_idx_v.append(model.joints[jid].idx_v)
        self.n_dof = len(self.arm_idx_q)

        # Joint limits for this arm (from the model)
        self.q_lower = np.array([model.lowerPositionLimit[i] for i in self.arm_idx_q])
        self.q_upper = np.array([model.upperPositionLimit[i] for i in self.arm_idx_q])
        # Shrink by the joint-limit buffer so planned configs stay away from limits
        buf = cfg.JOINT_LIMIT_BUFFER_BASE
        self.q_lower_plan = self.q_lower + buf
        self.q_upper_plan = self.q_upper - buf

        # Threading / abort control
        self._abort_flag = threading.Event()
        self._thread = None
        self._result_lock = threading.Lock()
        self._last_result = None  # RRTPlannerResult or None
        # Monotonic planning epoch: every plan_async / abort bumps it. A running
        # background thread captures its epoch and fences itself out (stops, and
        # its result is discarded) the moment the epoch no longer matches. This
        # lets abort() be fully NON-BLOCKING -- we never join the thread from the
        # 300Hz control loop, which previously could stall the real-time loop.
        self._plan_epoch = 0
        self._last_result_epoch = -1

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def plan_async(self, q_start_full, x_goal, cmodel, cdata, logger=None):
        """Launch the planning in a background thread. Non-blocking.

        Args:
            q_start_full: full-model configuration (pin.neutral size) at the
                          moment planning starts (other-arm joints frozen here).
            x_goal: (3,) target EE position in base_footprint.
            cmodel: the collision GeometryModel (shared, read-only).
            cdata: a FRESH GeometryData created for this planning call (NOT the
                   live one used by the control loop — avoids data races).
            logger: optional callable(str) for verbose progress diagnostics
                    (goal-IK per-restart trace, RRT-Connect periodic progress).
                    None = silent (still returns a full RRTPlannerResult).
        """
        # Bump the epoch (fences any previous run) and start fresh. We do NOT
        # join the old thread -- it is a daemon that will notice the epoch change
        # and exit on its own; joining here would block the control loop.
        self._plan_epoch += 1
        epoch = self._plan_epoch
        self._abort_flag.clear()
        with self._result_lock:
            self._last_result = None
        self._thread = threading.Thread(
            target=self._plan_thread,
            args=(q_start_full.copy(), x_goal.copy(), cmodel, cdata, epoch, logger),
            daemon=True)
        self._thread.start()

    def abort(self):
        """Signal the planning thread to stop. NON-BLOCKING (never joins).

        Sets the abort flag (fast early-out for the running thread) and bumps the
        epoch so any in-flight thread fences itself out and its result is
        discarded by _finish. Safe to call from the 300Hz control loop.
        """
        self._abort_flag.set()
        self._plan_epoch += 1

    def _should_abort(self, epoch):
        """True if this thread (identified by `epoch`) must stop: either an
        explicit abort was requested, or a newer plan/abort superseded it."""
        return self._abort_flag.is_set() or epoch != self._plan_epoch

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def result(self):
        """Retrieve the latest planning result (None if not yet available)."""
        with self._result_lock:
            return self._last_result

    # =====================================================================
    # PLANNING THREAD
    # =====================================================================

    def _plan_thread(self, q_start_full, x_goal, cmodel, cdata, epoch, logger=None):
        """The actual RRT-Connect algorithm (runs in background thread).

        Wrapped in a blanket try/except so that ANY failure — an unreachable
        target, a numerical issue, an aborted run — resolves to a well-defined
        `success=False` result and can NEVER propagate out of the thread. The
        control loop only ever READS the result, so the QP-CLF-CBF is fully
        insulated from whatever happens here.

        `logger`, if given, receives verbose progress lines (goal-IK search,
        RRT-Connect growth) so the operator can see WHY a run failed/succeeded.
        There is NO time requirement on this thread other than the configured
        budgets below — running for several seconds is fine; the QP keeps
        driving the arm with the (already-applied) posture correction the
        whole time.
        """
        t0 = time.perf_counter()
        result = RRTPlannerResult()
        tag = f"[RRT][{self.arm_side.upper()}]"

        def log(msg):
            if logger is not None:
                logger(msg)

        try:
            log(f"{tag} --- Planning thread started. Target x_goal={np.round(x_goal, 4).tolist()} "
                f"(base_footprint). IK budget={cfg.RRT_IK_TIME_BUDGET_S:.1f}s "
                f"(max {cfg.RRT_IK_MAX_RESTARTS} restarts x {cfg.RRT_IK_ITERS_PER_RESTART} iters), "
                f"RRT-Connect budget={cfg.RRT_PLANNING_BUDGET_S:.1f}s. ---")
            # Extract this arm's joint positions from the full config
            q_arm_start = np.array([q_start_full[i] for i in self.arm_idx_q])

            # Local Pinocchio data for FK (thread-safe, not shared with control loop)
            data = self.model.createData()

            # Helper: set arm joints in a full q vector
            def set_arm_q(q_full, q_arm):
                for i, idx in enumerate(self.arm_idx_q):
                    q_full[idx] = q_arm[i]

            # Helper: FK for this arm's EE
            def ee_pos(q_arm):
                q_full = q_start_full.copy()
                set_arm_q(q_full, q_arm)
                pin.forwardKinematics(self.model, data, q_full)
                pin.updateFramePlacements(self.model, data)
                return np.array(data.oMf[self.ee_frame_id].translation)

            # Helper: collision check (True = collision-free)
            def is_collision_free(q_arm):
                q_full = q_start_full.copy()
                set_arm_q(q_full, q_arm)
                pin.updateGeometryPlacements(self.model, data, cmodel, cdata, q_full)
                pin.computeDistances(cmodel, cdata)
                min_dist = min(r.min_distance for r in cdata.distanceResults)
                return min_dist > (cfg.D_SAFE_BASE + cfg.RRT_COLLISION_MARGIN)

            # Helper: random sample in joint limits
            def random_sample():
                return np.random.uniform(self.q_lower_plan, self.q_upper_plan)

            # Helper: steer from q_near toward q_target by at most step_size
            def steer(q_near, q_target):
                diff = q_target - q_near
                dist = np.linalg.norm(diff)
                if dist <= cfg.RRT_STEP_SIZE:
                    return q_target.copy()
                return q_near + diff * (cfg.RRT_STEP_SIZE / dist)

            # Helper: nearest node in a tree
            def nearest(tree, q):
                dists = [np.linalg.norm(node.q - q) for node in tree]
                return int(np.argmin(dists))

            # Helper: extend tree toward q_target, returns (new_node_idx, reached_exactly)
            def extend(tree, q_target):
                idx_near = nearest(tree, q_target)
                q_new = steer(tree[idx_near].q, q_target)
                if not is_collision_free(q_new):
                    return -1, False
                new_idx = len(tree)
                tree.append(RRTNode(q_new, idx_near))
                reached = np.linalg.norm(q_new - q_target) < 1e-3
                return new_idx, reached

            # Helper: connect tree to q_target (repeated extend until stuck or reached)
            def connect(tree, q_target):
                while True:
                    if self._should_abort(epoch):
                        return -1, False
                    idx, reached = extend(tree, q_target)
                    if idx < 0:
                        return -1, False
                    if reached:
                        return idx, True

            # --- Find a goal configuration via damped least-squares IK ---
            # (position-only, per design). Uniform random rejection sampling in
            # 7D essentially NEVER lands the EE inside a small Cartesian ball,
            # which is exactly why the previous finder returned samples=0 and
            # ALWAYS failed. DLS IK converges in a handful of iterations whenever
            # the target is actually reachable; if it can't (target inside an
            # obstacle / out of the collision-free workspace) the planner fails
            # cleanly and the caller keeps the posture-only correction.
            t_ik0 = time.perf_counter()
            q_goal_arm = self._find_goal_config_ik(
                q_start_full, x_goal, data, is_collision_free, set_arm_q, epoch, log)
            ik_time = time.perf_counter() - t_ik0
            if q_goal_arm is None:
                log(f"{tag} Goal-IK FAILED after {ik_time*1000:.0f}ms — no collision-free "
                    f"configuration found that places the EE within "
                    f"{cfg.RRT_GOAL_POS_TOLERANCE*100:.1f}cm of the target. RRT-Connect will "
                    f"NOT run (no goal to grow toward). Target is likely unreachable / inside "
                    f"an obstacle / outside the joint-limited workspace.")
                result.success = False
                self._finish(result, t0, epoch)
                return
            log(f"{tag} Goal-IK SUCCESS after {ik_time*1000:.0f}ms — collision-free goal config "
                f"found (EE within {cfg.RRT_GOAL_POS_TOLERANCE*100:.1f}cm of target). "
                f"Starting bidirectional RRT-Connect...")

            # --- Bidirectional RRT-Connect ---
            tree_start = [RRTNode(q_arm_start)]
            tree_goal = [RRTNode(q_goal_arm)]
            samples = 0
            path_found = False
            connect_idx_start = -1
            connect_idx_goal = -1

            deadline = t0 + cfg.RRT_PLANNING_BUDGET_S
            last_progress_log = t0

            while samples < cfg.RRT_MAX_SAMPLES and time.perf_counter() < deadline:
                if self._should_abort(epoch):
                    log(f"{tag} RRT-Connect ABORTED at sample {samples} "
                        f"(reference resumed motion / superseded).")
                    self._finish(result, t0, epoch)
                    return
                samples += 1

                # Sample (with goal bias toward the other tree's root)
                if np.random.rand() < cfg.RRT_GOAL_BIAS:
                    q_rand = tree_goal[0].q.copy()
                else:
                    q_rand = random_sample()

                # Extend tree_start toward q_rand
                idx_new, _ = extend(tree_start, q_rand)
                if idx_new < 0:
                    # Swap trees and retry
                    tree_start, tree_goal = tree_goal, tree_start
                else:
                    # Try to connect tree_goal to the new node in tree_start
                    q_new = tree_start[idx_new].q
                    idx_connect, connected = connect(tree_goal, q_new)
                    if connected:
                        connect_idx_start = idx_new
                        connect_idx_goal = idx_connect
                        path_found = True
                        break
                    # Swap trees for balanced growth
                    tree_start, tree_goal = tree_goal, tree_start

                # Periodic progress report (non-spam) so a multi-second run is
                # visible in the console instead of looking hung.
                now = time.perf_counter()
                if now - last_progress_log >= cfg.RRT_PROGRESS_LOG_PERIOD_S:
                    last_progress_log = now
                    log(f"{tag} RRT-Connect growing... samples={samples}, "
                        f"tree sizes=({len(tree_start)},{len(tree_goal)}), "
                        f"elapsed={now - t0:.1f}s / budget={cfg.RRT_PLANNING_BUDGET_S:.1f}s")

            if not path_found:
                log(f"{tag} RRT-Connect EXHAUSTED budget/samples without connecting the two "
                    f"trees (samples={samples}, tree sizes=({len(tree_start)},{len(tree_goal)}), "
                    f"elapsed={time.perf_counter() - t0:.1f}s). The obstacle likely splits the "
                    f"reachable joint-space into disconnected regions the step size couldn't "
                    f"bridge in time.")
                result.success = False
                result.samples_used = samples
                self._finish(result, t0, epoch)
                return
            log(f"{tag} RRT-Connect CONNECTED after samples={samples} "
                f"(elapsed={time.perf_counter() - t0:.1f}s). Extracting + smoothing path...")

            # --- Extract path (backtrack from both trees to their roots) ---
            start_is_original = np.allclose(tree_start[0].q, q_arm_start, atol=1e-6)

            def backtrack(tree, idx):
                path = []
                while idx >= 0:
                    path.append(tree[idx].q)
                    idx = tree[idx].parent_idx
                path.reverse()
                return path

            if start_is_original:
                path_from_start = backtrack(tree_start, connect_idx_start)
                path_from_goal = backtrack(tree_goal, connect_idx_goal)
                path_from_goal.reverse()
                raw_path = path_from_start + path_from_goal
            else:
                path_from_goal_tree = backtrack(tree_start, connect_idx_start)
                path_from_start_tree = backtrack(tree_goal, connect_idx_goal)
                path_from_start_tree.reverse()
                raw_path = path_from_start_tree + path_from_goal_tree

            result.raw_path_length = len(raw_path)

            # --- Shortcut smoothing ---
            smoothed = self._shortcut_smooth(raw_path, is_collision_free)
            result.smoothed_path_length = len(smoothed)

            # --- Convert to Cartesian waypoints ---
            cartesian_wps = [ee_pos(q_arm) for q_arm in smoothed]

            result.success = True
            result.cartesian_waypoints = cartesian_wps
            result.joint_waypoints = smoothed
            result.samples_used = samples
            # Compute total Cartesian path length
            total_len = 0.0
            for i in range(1, len(cartesian_wps)):
                total_len += np.linalg.norm(cartesian_wps[i] - cartesian_wps[i - 1])
            result.total_cartesian_length = total_len

            self._finish(result, t0, epoch)
        except Exception as e:
            # Blanket guard: a background-thread failure must NEVER reach the QP.
            result.success = False
            self._finish(result, t0, epoch)
            print(f"\033[91m[RRT][{self.arm_side.upper()}] Planner thread exception "
                  f"(handled, planning marked failed): {e}\033[0m", flush=True)

    def _find_goal_config_ik(self, q_start_full, x_goal, data, is_collision_free,
                             set_arm_q, epoch, log=lambda msg: None):
        """Position-only damped least-squares IK for a collision-free goal config.

        Iteratively drives THIS arm's joints so FK(q).translation → x_goal, then
        verifies the converged config is collision-free. On non-convergence or a
        collision, restarts from a fresh random seed (the local minimum caused by
        the obstacle usually still leaves a valid goal config, just not reachable
        by the greedy CLF gradient). Returns the arm's 7-vector or None.

        Fixes the previous uniform-random-rejection finder, which could not hit a
        3cm Cartesian ball in 7D and therefore always failed with samples=0.

        Budgeted by WALL-CLOCK time (cfg.RRT_IK_TIME_BUDGET_S), not just a fixed
        restart count — there is no hard millisecond requirement on this search;
        the QP keeps driving the arm with the posture correction while this runs,
        for up to several seconds if needed.
        """
        tag = f"[RRT][{self.arm_side.upper()}]"
        damp2 = cfg.RRT_IK_DAMPING ** 2
        eye3 = np.eye(3)
        q_full = q_start_full.copy()
        t_budget_end = time.perf_counter() + cfg.RRT_IK_TIME_BUDGET_S
        restart = 0
        best_err_norm = None
        while restart < cfg.RRT_IK_MAX_RESTARTS and time.perf_counter() < t_budget_end:
            if self._should_abort(epoch):
                return None
            if restart == 0:
                # First attempt seeds from the current (stuck) configuration.
                q_arm = np.array([q_start_full[i] for i in self.arm_idx_q])
            else:
                q_arm = np.random.uniform(self.q_lower_plan, self.q_upper_plan)
            err_norm = None
            for it in range(cfg.RRT_IK_ITERS_PER_RESTART):
                if self._should_abort(epoch):
                    return None
                set_arm_q(q_full, q_arm)
                pin.forwardKinematics(self.model, data, q_full)
                pin.updateFramePlacements(self.model, data)
                x_cur = np.array(data.oMf[self.ee_frame_id].translation)
                err = x_goal - x_cur
                err_norm = float(np.linalg.norm(err))
                if err_norm < cfg.RRT_GOAL_POS_TOLERANCE:
                    # Converged in position -- accept only if collision-free.
                    if is_collision_free(q_arm):
                        log(f"{tag} Goal-IK restart {restart+1}/{cfg.RRT_IK_MAX_RESTARTS}: "
                            f"CONVERGED at iter {it} (|err|={err_norm*1000:.1f}mm), "
                            f"collision-free -> ACCEPTED.")
                        return q_arm.copy()
                    log(f"{tag} Goal-IK restart {restart+1}/{cfg.RRT_IK_MAX_RESTARTS}: "
                        f"reached target position at iter {it} but the config COLLIDES "
                        f"-> reseeding.")
                    break  # good position but colliding -> reseed
                # Position-only frame Jacobian, restricted to this arm's columns.
                J6 = pin.computeFrameJacobian(
                    self.model, data, q_full, self.ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J = J6[:3, :][:, self.arm_idx_v]
                # Damped least-squares step: dq = J^T (J J^T + damp^2 I)^-1 err
                dq = J.T @ np.linalg.solve(J @ J.T + damp2 * eye3, err)
                # Cap the step for numerical stability near singularities.
                step = np.linalg.norm(dq)
                if step > cfg.RRT_IK_MAX_STEP:
                    dq *= cfg.RRT_IK_MAX_STEP / step
                q_arm = np.clip(q_arm + dq, self.q_lower_plan, self.q_upper_plan)
            else:
                # Loop exhausted without breaking (i.e. never converged in position)
                log(f"{tag} Goal-IK restart {restart+1}/{cfg.RRT_IK_MAX_RESTARTS}: did NOT "
                    f"converge in {cfg.RRT_IK_ITERS_PER_RESTART} iters (final |err|="
                    f"{err_norm*1000:.1f}mm) -- likely stuck at a singularity/joint-limit "
                    f"wall -> reseeding.")
            if best_err_norm is None or (err_norm is not None and err_norm < best_err_norm):
                best_err_norm = err_norm
            restart += 1
        log(f"{tag} Goal-IK exhausted ({restart} restarts, "
            f"best |err| reached={('%.1fmm' % (best_err_norm*1000)) if best_err_norm is not None else 'n/a'}) "
            f"without a collision-free convergence.")
        return None

    def _shortcut_smooth(self, path, is_collision_free):
        """Shortcut smoothing: try to directly connect non-adjacent waypoints."""
        if len(path) <= 2:
            return list(path)
        smoothed = list(path)
        for _ in range(cfg.RRT_SHORTCUT_ITERS):
            if len(smoothed) <= 2:
                break
            i = np.random.randint(0, len(smoothed) - 2)
            j = np.random.randint(i + 2, len(smoothed))
            # Check if the straight segment i→j is collision-free
            n_checks = max(2, int(np.linalg.norm(smoothed[j] - smoothed[i]) / cfg.RRT_STEP_SIZE) + 1)
            segment_ok = True
            for k in range(1, n_checks):
                alpha = k / n_checks
                q_interp = smoothed[i] * (1.0 - alpha) + smoothed[j] * alpha
                if not is_collision_free(q_interp):
                    segment_ok = False
                    break
            if segment_ok:
                smoothed = smoothed[:i + 1] + smoothed[j:]
        return smoothed

    def _finish(self, result, t0, epoch):
        result.planning_time_s = time.perf_counter() - t0
        with self._result_lock:
            # Only publish the result if this thread is still the current plan.
            # A superseded/aborted thread (older epoch) silently drops its result.
            if epoch == self._plan_epoch:
                self._last_result = result
                self._last_result_epoch = epoch
