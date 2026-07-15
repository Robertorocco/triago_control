#!/usr/bin/env python3
# main_qp_controller_real.py
"""
Real-hardware QP-CLF-CBF controller — SAME safety math as the base, but the two
heaviest per-tick jobs run on WORKER THREADS so the control loop can hit a higher
rate on the robot.

This is a THIN SUBCLASS of SafetyQPController (main_qp_controller.py). On the real
TIAGo Pro the single dominant per-tick cost is the SoftMin CBF aggregation
(CollisionManager.compute_softmin_jacobian) — measured larger than the entire
300 Hz budget on its own. This subclass moves it OFF the control tick:

    * CBF worker thread  — runs the full FK + geometry + SoftMin CBF on its OWN
      private pin.Data / GeometryData (so it never races the main thread's
      kinematics refresh). The main tick reads the most-recent result and feeds it
      to the QP. A STALENESS WATCHDOG freezes BOTH arms (zero velocity command,
      auto-resume) if no fresh CBF has arrived for cfg.CBF_STALENESS_MAX_TICKS
      consecutive ticks — so the QP can never act on an unboundedly-old barrier.
    * RViz-overlay worker thread — publishes the collision-witness line + teleop
      tethers (which need only a small witness snapshot; QPVisualizer reads
      everything else from its own ROS subscriptions).

EVERYTHING ELSE is inherited verbatim from the base, so this file is just the four
overridden seams (_compute_cbf / _gate_command / _publish_visual_overlays /
_process_deferred_topology), the two worker loops, and a main().

Isolation invariant (why this is race-free):
    * The rclpy executor is single-threaded, so current_q/current_v and the HRI
      grasp dicts have exactly ONE writer (the executor). The CBF worker only ever
      READS a snapshot the main tick hands it.
    * The worker computes on private pin.Data / GeometryData; the shared self.data
      / self.cdata stay owned by the main tick (telemetry/markers keep working).
    * The ONLY shared mutable structure is the collision MODEL (cmodel), which is
      structurally mutated on grasp attach/detach. That is serialized against the
      worker via CollisionManager.geom_lock (held by the worker around its compute
      and by _process_deferred_topology around the mutation).
    * The CBF the QP consumes is delivered via an immutable CbfResult under a lock
      with a monotonic sequence number; the witness/top-pairs telemetry is carried
      inside that same record, so self.col.witness_* is never read cross-thread.

Config (all in config.py, section 4 — the user tunes these, not this file):
    REAL_ASYNC_CBF, REAL_ASYNC_VIZ, CBF_STALENESS_MAX_TICKS. Setting either flag
    False, OR running where REAL_HARDWARE is not detected (e.g. Gazebo), makes
    this subclass fall back to the base's synchronous behavior — so it is safe to
    launch anywhere and is a drop-in A/B against main_qp_controller.py.

Run (on the robot, in place of main_qp_controller.py):
    ros2 run triago_control main_qp_controller_real.py
"""

import os
import sys
import threading

import numpy as np
import pinocchio as pin
import rclpy
from std_msgs.msg import Float64, Float64MultiArray

# The QP entrypoints install flat into lib/triago_control/ alongside each other;
# add this file's dir so the sibling `main_qp_controller` module imports cleanly
# whether run from the install tree or source (mirrors main_qp_controller_perceived.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_qp_controller import SafetyQPController, CbfResult          # noqa: E402
import triago_control.qp_controller.config as cfg                     # noqa: E402


class RealQPController(SafetyQPController):
    """SafetyQPController with the SoftMin CBF + RViz overlays on worker threads."""

    def __init__(self):
        super().__init__()

        # Go async only on REAL hardware AND when the config flags allow it.
        # Otherwise every override below falls through to super() -> byte-identical
        # to the base synchronous controller (safe to run in sim / for A/B).
        self._async_cbf = bool(self.REAL_HARDWARE and cfg.REAL_ASYNC_CBF)
        self._async_viz = bool(self.REAL_HARDWARE and cfg.REAL_ASYNC_VIZ)

        # Watchdog state (referenced by _gate_command even when async is off).
        self._cbf_fresh = False          # False until the first worker result lands
        self._stale_ticks = 0
        self._cbf_last_seen_seq = 0
        self._cbf_thread = None
        self._viz_thread = None

        if not self._async_cbf and not self._async_viz:
            self.get_logger().info(
                "\033[93m[REAL] Async disabled (not real hardware or flags off) -- "
                "running synchronously, identical to the base controller.\033[0m")
            return

        if self._async_cbf:
            # Private kinematics + geometry data the worker owns exclusively.
            self._data_cbf = self.kin.model.createData()
            self._cdata_cbf = self.col.cmodel.createData()
            for req in self._cdata_cbf.distanceRequests:
                req.enable_nearest_points = True
            self._cbf_pair_count = len(self.col.cmodel.collisionPairs)

            self._cbf_cond = threading.Condition()
            self._cbf_input = None       # latest input snapshot (dict) posted by the tick
            self._cbf_in_seq = 0
            self._cbf_result = None      # latest CbfResult produced by the worker
            self._cbf_out_seq = 0
            self._cbf_stop = False
            self._cbf_thread = threading.Thread(
                target=self._cbf_worker, name="cbf_worker", daemon=True)
            self._cbf_thread.start()
            self.get_logger().info(
                f"\033[96m[REAL] Async CBF worker started (freeze both arms after "
                f"{cfg.CBF_STALENESS_MAX_TICKS} stale ticks, auto-resume on refresh).\033[0m")

        if self._async_viz:
            self._viz_cond = threading.Condition()
            self._viz_input = None
            self._viz_in_seq = 0
            self._viz_stop = False
            self._viz_thread = threading.Thread(
                target=self._viz_worker, name="viz_worker", daemon=True)
            self._viz_thread.start()
            self.get_logger().info(
                "\033[96m[REAL] Async RViz-overlay worker started.\033[0m")

    # =====================================================================
    # SEAM OVERRIDES (see SafetyQPController's "OVERRIDABLE SEAMS" section)
    # =====================================================================
    def _process_deferred_topology(self):
        # Grasp attach/detach STRUCTURALLY mutates cmodel; hold geom_lock so the
        # CBF worker (which reads cmodel / runs computeDistances on it) is never
        # doing so mid-mutation. Rare (grasp events only) -> ~uncontended.
        if not self._async_cbf:
            return super()._process_deferred_topology()
        with self.col.geom_lock:
            super()._process_deferred_topology()

    def _compute_cbf(self):
        # Post THIS tick's snapshot for the worker to chew on, then return the
        # most-recent result it has produced (from a prior tick), driving the
        # staleness watchdog. Synchronous fallback when async is off.
        if not self._async_cbf:
            return super()._compute_cbf()

        snap = {
            'q': self.kin.current_q.copy(),
            'v': (self.kin.current_v.copy() if self.kin.current_v is not None else None),
            'idx_r': self.kin.idx_right,
            'idx_l': self.kin.idx_left,
            # Shallow copies: the single-threaded executor is the sole mutator of
            # these HRI containers, and grasp events REPLACE them (never in-place
            # mutate), so a top-level copy is a consistent snapshot.
            'margin_targets': dict(self.hri.grasp_margin_targets),
            'attached_objs': set(self.hri.attached_objects),
            'attached_adjacency': dict(self.hri.attached_adjacency),
            'ignored_targets': set(self.hri.ignored_targets),
            'ramp_shifts': dict(self.hri.get_attach_ramp_shifts() or {}),
            'attached_object_arm': dict(self.hri.attached_object_arm),
            'publish_counter': self.publish_counter,
        }
        with self._cbf_cond:
            self._cbf_input = snap
            self._cbf_in_seq += 1
            self._cbf_cond.notify()
            result = self._cbf_result
            out_seq = self._cbf_out_seq

        if result is None:
            # No CBF computed yet (startup): well-formed empty barrier, NOT fresh
            # -> _gate_command freezes the arms until the first result arrives.
            self._set_fresh(False)
            return self._safe_empty_cbf()

        if out_seq != self._cbf_last_seen_seq:
            self._cbf_last_seen_seq = out_seq
            self._stale_ticks = 0
        else:
            self._stale_ticks += 1
        self._set_fresh(self._stale_ticks < cfg.CBF_STALENESS_MAX_TICKS)
        return result._replace(fresh=self._cbf_fresh)

    def _gate_command(self, q_dot_safe):
        # Freeze BOTH arms (zero velocity) whenever the async CBF result is stale;
        # identity otherwise (and always, when async is off).
        if not self._async_cbf or self._cbf_fresh:
            return q_dot_safe
        return np.zeros_like(q_dot_safe)

    def _publish_visual_overlays(self, cbf, q_dot_safe):
        # Hand the RViz overlays to the viz worker. It needs only the witness +
        # safety-margin snapshot; publish_debug/publish_teleop_tether otherwise
        # read their live state from QPVisualizer's own ROS subscriptions.
        if not self._async_viz:
            return super()._publish_visual_overlays(cbf, q_dot_safe)
        margin = None
        if not cfg.DISABLE_CBF:
            # Per-arm margin; NaN when that arm has no active pair (not the
            # h_soft=1.0 QP sentinel, which isn't a real margin).
            margin_r = (cbf.h_soft_r - cbf.d_safe_r) if cbf.active_r else float('nan')
            margin_l = (cbf.h_soft_l - cbf.d_safe_l) if cbf.active_l else float('nan')
            margin = (float(margin_r), float(margin_l))
        snap = {'witness_distance': cbf.witness_distance,
                'witness_points': cbf.witness_points,
                'margin': margin}
        with self._viz_cond:
            self._viz_input = snap
            self._viz_in_seq += 1
            self._viz_cond.notify()

    # =====================================================================
    # WORKER THREADS
    # =====================================================================
    def _safe_empty_cbf(self):
        # A well-formed "no interaction" CbfResult (exactly what compute_softmin_
        # jacobian returns in open space), tagged NOT fresh. Used before the first
        # worker result exists; the QP accepts it but _gate_command zeroes the cmd.
        nv = self.kin.model.nv
        return CbfResult(
            np.zeros(nv), 1.0, np.zeros(nv), 1.0,
            cfg.D_SAFE_BASE, cfg.D_SAFE_BASE, float('nan'), False, False,
            float('nan'), None, [], False)

    def _set_fresh(self, fresh):
        # Edge-triggered logging on the freeze/resume transition (no per-tick spam).
        if fresh and not self._cbf_fresh:
            self.get_logger().info(
                "\033[92m[SAFETY] CBF result live -- arms under normal control.\033[0m")
        elif not fresh and self._cbf_fresh:
            self.get_logger().warn(
                f"\033[91m[SAFETY] CBF stale >= {cfg.CBF_STALENESS_MAX_TICKS} ticks "
                f"-- FREEZING both arms (zero velocity) until it refreshes.\033[0m")
        self._cbf_fresh = fresh

    def _cbf_worker(self):
        # Sole caller of compute_softmin_jacobian in async mode. Waits for a new
        # input snapshot, computes the CBF on its OWN pin.Data/GeometryData under
        # geom_lock (serialized vs. grasp cmodel mutation), publishes a CbfResult.
        model = self.kin.model
        cmodel = self.col.cmodel
        last_seq = 0
        while not self._cbf_stop:
            with self._cbf_cond:
                while self._cbf_in_seq == last_seq and not self._cbf_stop:
                    self._cbf_cond.wait(timeout=1.0)
                if self._cbf_stop:
                    return
                snap = self._cbf_input
                last_seq = self._cbf_in_seq
            if snap is None or snap['q'] is None or snap['v'] is None:
                continue
            try:
                q, v = snap['q'], snap['v']
                with self.col.geom_lock:
                    # Grasp attach/detach changed the pair set -> rebuild our cdata.
                    if self._cbf_pair_count != len(cmodel.collisionPairs):
                        self._cdata_cbf = cmodel.createData()
                        for req in self._cdata_cbf.distanceRequests:
                            req.enable_nearest_points = True
                        self._cbf_pair_count = len(cmodel.collisionPairs)
                    # Mirror RobotKinematics.update_kinematics on our OWN data.
                    pin.forwardKinematics(model, self._data_cbf, q, v)
                    pin.updateFramePlacements(model, self._data_cbf)
                    pin.computeJointJacobians(model, self._data_cbf, q)
                    self.col.update_geometry(q, data=self._data_cbf, cdata=self._cdata_cbf)
                    (J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                     d_safe_r, d_safe_l, abs_min,
                     active_r, active_l) = self.col.compute_softmin_jacobian(
                        v, snap['idx_r'], snap['idx_l'],
                        snap['margin_targets'], snap['attached_objs'],
                        snap['attached_adjacency'], snap['ignored_targets'],
                        snap['publish_counter'],
                        attach_ramp_shifts=snap['ramp_shifts'],
                        attached_object_arm=snap['attached_object_arm'],
                        data=self._data_cbf, cdata=self._cdata_cbf)
                    # Read the witness/top written by the call above SAME-THREAD,
                    # still under the lock, and carry them in the result so the
                    # main tick never reads self.col.* cross-thread.
                    witness_d = self.col.witness_min_distance
                    witness_p = self.col.witness_min_points
                    top = list(self.col.top_active_pairs)
                result = CbfResult(J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                                   d_safe_r, d_safe_l, abs_min, active_r, active_l,
                                   witness_d, witness_p, top, True)
                with self._cbf_cond:
                    self._cbf_result = result
                    self._cbf_out_seq += 1
            except Exception as e:  # noqa: BLE001 -- keep the loop alive; watchdog freezes
                self.get_logger().error(f"[REAL] CBF worker error (will freeze if it persists): {e}")

    def _viz_worker(self):
        # Publishes the RViz collision-witness line + teleop tethers off the tick.
        last_seq = 0
        while not self._viz_stop:
            with self._viz_cond:
                while self._viz_in_seq == last_seq and not self._viz_stop:
                    self._viz_cond.wait(timeout=1.0)
                if self._viz_stop:
                    return
                snap = self._viz_input
                last_seq = self._viz_in_seq
            if snap is None:
                continue
            try:
                if snap['margin'] is not None:
                    self.pub_debug_h.publish(Float64MultiArray(data=list(snap['margin'])))
                # publish_debug ignores model/data/q/q_dot/ids/limit today (it uses
                # only witness_info + its own ROS-subscription state), so we pass
                # None for the mutable ones -> zero cross-thread pinocchio reads.
                self.viz.publish_debug(
                    None, None,
                    (snap['witness_distance'], snap['witness_points']),
                    None, None, None, None,
                    self.kin.ee_id_right, self.kin.ee_id_left,
                    cfg.JOINT_LIMIT_BUFFER_BASE)
                self.viz.publish_teleop_tether()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"[REAL] Viz worker error: {e}")

    def shutdown_workers(self):
        # Signal + join both worker threads (idempotent; safe when async is off).
        if self._cbf_thread is not None:
            with self._cbf_cond:
                self._cbf_stop = True
                self._cbf_cond.notify_all()
            self._cbf_thread.join(timeout=2.0)
        if self._viz_thread is not None:
            with self._viz_cond:
                self._viz_stop = True
                self._viz_cond.notify_all()
            self._viz_thread.join(timeout=2.0)


def main():
    rclpy.init()
    node = RealQPController()

    # Wall-clock control loop by design -- do not set use_sim_time here.

    # --- PHASE 1: wait for TF, then verify controller state ---
    node.get_logger().info("[Main] Waiting for TF...")
    node.wait_for_tf()

    node.get_logger().info("[Main] Verifying Controller State...")
    if node.check_and_switch_controllers():
        print("------------------------------------------------")
        print("REAL-HARDWARE SAFETY CONTROLLER RUNNING (Async CBF + Velocity Mode)")
        print("------------------------------------------------")
    else:
        print("[Error] Could not switch controllers. Exiting.")
        node.shutdown_workers()
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- PHASE 2: visualization + diagnostics (Meshcat suspended on real HW) ---
    if node.REAL_HARDWARE:
        node.get_logger().info(
            "\033[93m[Viz] REAL HARDWARE: Meshcat suspended (RViz/plotters unaffected).\033[0m")
    else:
        node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    node.start_control_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_workers()
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
