#!/usr/bin/env python3
# main_qp_controller.py
"""Safety-controller orchestrator: wires RobotKinematics + CollisionManager +
SharedAutonomyHandler + QPFormulator + VisualizationEngine and drives the wall-clock
control loop (references in, safe joint velocities + telemetry out)."""

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, String, Bool
from controller_manager_msgs.srv import SwitchController, ListControllers
from rcl_interfaces.srv import GetParameters
from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import pinocchio as pin
import numpy as np
import time
import tempfile
import os
import sys
import traceback
import faulthandler
from collections import namedtuple

# Catch fatal native-level crashes (segfault/abort/bus error -- e.g. a
# malformed problem reaching quadprog's/pinocchio's C/Fortran internals) that
# a plain Python try/except CANNOT catch: dumps the Python-level traceback of
# whichever thread was executing to stderr right as the signal is caught,
# BEFORE the process actually dies. Module-level so it's active regardless of
# which entrypoint imports this file (main_qp_controller[_real|_perceived].py).
faulthandler.enable(file=sys.stderr, all_threads=True)

import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.qp_controller.collision_manager import CollisionManager
from triago_control.qp_controller.shared_autonomy_handler import SharedAutonomyHandler
from triago_control.qp_controller.qp_formulator import QPFormulator
from triago_control.qp_controller.visualization_engine import VisualizationEngine
from triago_control.qp_controller.reference_governor import ReferenceGovernor
from triago_control.qp_controller.world_loader import load_world


# One control tick's collision-barrier result: J_soft/h_soft/d_safe feed the
# QP; active_r/l marks a real pair (vs. h_soft's safe 1.0 sentinel) for
# telemetry only; fresh=False triggers the watchdog freeze.
CbfResult = namedtuple('CbfResult', [
    'J_soft_r', 'h_soft_r', 'J_soft_l', 'h_soft_l',
    'd_safe_r', 'd_safe_l', 'abs_min_distance', 'active_r', 'active_l',
    'witness_distance', 'witness_points', 'top_active_pairs', 'fresh',
    'n_eff_r', 'n_eff_l'])


class SafetyQPController(Node):
    """ROS 2 node orchestrating the bimanual QP-CLF-CBF safety controller."""

    def __init__(self):
        super().__init__('safety_qp_controller')

        self._control_freq = cfg.CONTROL_FREQ_DEFAULT
        self.loop_timer = None

        # Obstacle layout for the CBF collision model, independent of the
        # launched Gazebo world (see world_loader.py). Override with
        # --ros-args -p world_name:=no_obstacle.
        self.declare_parameter('world_name', 'movement_tutorial')
        self.world_scene = self._build_world_scene()
        self.get_logger().info(
            f"\033[96m[World] Loaded scene '{self.world_scene.world_name}' "
            f"({len(self.world_scene.static_obstacles)} static obstacles; "
            f"matching Gazebo world file: {self.world_scene.gazebo_world_file}).\033[0m")

        # --- TF (kept for the start-up transform wait) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- BUILD MODEL + SUB-MODULES ---
        urdf_str = self.get_urdf()
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
            f.write(urdf_str)
            self.urdf_path = f.name

        # =====================================================================
        # REAL_HARDWARE DETECTION
        # =====================================================================
        # The Gazebo URDF contains gripper_*_grasping_link frames natively.
        # The real TIAGo Pro URDF does NOT — this is the discriminator.
        # Detection happens BEFORE building the Pinocchio model so we can
        # inject the missing frames and adapt the velocity pipeline.
        self.REAL_HARDWARE = ('gripper_right_grasping_link' not in urdf_str or
                             'gripper_left_grasping_link' not in urdf_str)
        if self.REAL_HARDWARE:
            self.get_logger().info(
                "\033[96m[ENV] REAL HARDWARE detected (URDF lacks grasping frames). "
                "Using direct joint velocities + injecting TCP frames.\033[0m")
        else:
            self.get_logger().info(
                "\033[92m[ENV] SIMULATION detected (URDF contains grasping frames). "
                "Using EMA-filtered velocity from position differentiation.\033[0m")

        # --- STATIC TF: publish grasping frames ONLY on real hardware ---
        if self.REAL_HARDWARE:
            self._publish_grasping_link_tfs()

        self.kin = RobotKinematics(self.urdf_path, real_hardware=self.REAL_HARDWARE)
        self.col = CollisionManager(self.kin.model, self.kin.data)

        right_offsets = self.col.calculate_offsets(cfg.RIGHT_CHAIN, 'gripper_right_base_link')
        left_offsets = self.col.calculate_offsets(cfg.LEFT_CHAIN, 'gripper_left_base_link')
        # Head added as a quasi-static CBF obstacle only (cfg.HEAD_CHAIN);
        # never mapped into idx_right/idx_left, so it adds no QP DOF.
        head_offsets = None
        if self.kin.model.existFrame(cfg.HEAD_CHAIN[0]):
            head_offsets = self.col.calculate_offsets(cfg.HEAD_CHAIN, cfg.HEAD_TOOL_LINK)
        else:
            self.get_logger().warn("[Init] Head chain not found in URDF -- skipping head CBF obstacle.")
        self.col.build_collision_model(right_offsets, left_offsets, head_offsets,
                                       world_scene=self.world_scene)
        self.col.define_collision_pairs()

        self.viz = VisualizationEngine(self, self.kin.model, self.col.cmodel, self.urdf_path,
                                       world_scene=self.world_scene)
        self.viz.add_gripper_visual_boxes(self.col)
        self.hri = SharedAutonomyHandler(self, self.col, self.kin, self.viz)
        self.qp = QPFormulator(self.kin.model)

        # Bounds ref position/velocity/acceleration so the CLF demand stays
        # QP-feasible; one instance per arm (separate accel-limit memory).
        self.gov_right = ReferenceGovernor('right')
        self.gov_left = ReferenceGovernor('left')

        # --- CONTROL MODE / REFERENCES ---
        self.orientation_ctrl = cfg.ORIENTATION_CTRL
        self.x_ref_right = None; self.rpy_ref_right = None; self.xdot_ref_right = None; self.w_ref_right = None
        self.x_ref_left = None;  self.rpy_ref_left = None;  self.xdot_ref_left = None;  self.w_ref_left = None
        self.task_dim_right = 6.0
        self.task_dim_left = 6.0

        # --- WATCHDOGS ---
        self.right_imposed_motion = False
        self.left_imposed_motion = False
        self.last_right_msg_time = self._now()
        self.last_left_msg_time = self._now()

        # --- LOOP / SIM STATE ---
        self.active_controller_mode = False
        self.publish_counter = 0
        # On real hardware, telemetry/viz publishing is real CPU cost competing
        # with the control tick itself (see /qp_debug/loop_timing's telemetry_ms
        # phase) -- halve the rate. Sim has no such CPU-budget pressure, so the
        # default cadence is unchanged there.
        self.publish_every_n = cfg.PUBLISH_EVERY_N * (2 if self.REAL_HARDWARE else 1)
        self.last_freq_pub_time = time.perf_counter()
        self.last_sim_time = None
        self.last_qdot_cmd_14 = np.zeros(14)
        # Per-tick jitter/compute-time diagnostic (see /qp_debug/loop_timing below).
        self._last_tick_perf = None
        # Opt-in tick-cadence log (TRIAGO_DEBUG_TICKS=1): ticks/s, bunched
        # (<1ms gap) count, and max gap, once per wall second. Zero cost if unset.
        self._dbg_ticks = os.environ.get('TRIAGO_DEBUG_TICKS') == '1'
        self._dbg_win_start = None
        self._dbg_n = 0
        self._dbg_n_bunched = 0
        self._dbg_max_gap_ms = 0.0

        # --- COMMAND PUBLISHERS ---
        self.pub_right = self.create_publisher(Float64MultiArray, f'/{cfg.RIGHT_CONTROLLER}/joint_velocity_cmd', 1)
        self.pub_left = self.create_publisher(Float64MultiArray, f'/{cfg.LEFT_CONTROLLER}/joint_velocity_cmd', 1)

        # --- TELEMETRY PUBLISHERS ---
        self.pub_qdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_err', 10)
        self.pub_xdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/xdot_err', 10)
        self.pub_slacks = self.create_publisher(Float64MultiArray, '/qp_debug/slacks', 10)
        self.pub_ee_state = self.create_publisher(Float64MultiArray, '/qp_debug/ee_real', 10)
        # The ACTUAL per-arm reference the CLF tracks (truthful for BOTH arms):
        # for the active arm it is the commanded reference, for a frozen/inactive
        # arm it is the held pose (main_qp_controller._freeze_arm) -- unlike
        # /arm_*/cartesian_reference, which goes stale for the inactive arm since
        # teleop/shared_autonomy only publish the active one. Layout (12 floats):
        # [x_r(3), rpy_r(3), x_l(3), rpy_l(3)] (raw reference, pre-governor).
        self.pub_reference_effective = self.create_publisher(
            Float64MultiArray, '/qp_debug/reference_effective', 10)
        self.pub_debug_h = self.create_publisher(Float64MultiArray, '/qp_debug/safety_margin', 10)
        self.pub_loop_freq = self.create_publisher(Float64, '/qp_debug/loop_freq', 10)
        # Per-tick diagnostic (NOT downsampled -- every tick, unlike loop_freq's
        # running average): [tick_dt_ms, kin_ms, cbf_ms, gov_ms, solve_ms,
        # telemetry_ms, misc_ms]. tick_dt is wall-clock time since the previous
        # tick started (scheduling/jitter + everything in solve_and_publish); the
        # rest breaks that down by phase -- kinematics/FK refresh, SoftMin CBF
        # aggregation, reference governor + task error, QP build+solve, downstream
        # telemetry/command/viz publishing, and misc (unbracketed code + any
        # residual OS scheduling wait) -- so a real-hardware compute bottleneck can
        # be pinned to a specific phase instead of guessed at. See loop_timing_monitor.py.
        self.pub_loop_timing = self.create_publisher(Float64MultiArray, '/qp_debug/loop_timing', 10)
        self.pub_min_dist = self.create_publisher(Float64, '/qp_debug/min_distance', 10)
        self.pub_top_pairs = self.create_publisher(String, '/qp_debug/top_pairs', 10)
        self.pub_lambda_cbf = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_cbf', 10)
        self.pub_lambda_joints = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_joints', 10)
        self.pub_dynamic_weights = self.create_publisher(Float64MultiArray, '/qp_debug/dynamic_weights', 10)
        self.pub_d_safe_dynamic = self.create_publisher(Float64MultiArray, '/qp_debug/d_safe_dynamic', 10)
        self.pub_qdot_cmd = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_cmd', 10)
        # kin.current_v: always EMA-filtered diff from position, sim and real HW alike
        # (direct sensor velocity was noisier on real hardware -- see joint_callback).
        self.pub_qdot_measured = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_measured', 10)
        self.pub_task_authority = self.create_publisher(Float64MultiArray, '/qp_debug/task_authority', 10)
        self.pub_shared_col = self.create_publisher(Float64MultiArray, '/collision_constraints', 10)
        # [right_frozen, left_frozen] as 0.0/1.0, for the RViz gripper color.
        self.pub_arm_frozen = self.create_publisher(Float64MultiArray, '/qp_debug/arm_frozen', 10)
        # Live joint limits for the plotter's slider GUI (latched, published
        # once + on every late-subscriber via a slow timer): the REAL limits
        # from the Pinocchio model built from the live URDF -- the SAME
        # numbers the joint-limit CBF rows in qp_formulator enforce.
        self.pub_joint_limits = self.create_publisher(String, '/qp_debug/joint_limits', 10)
        self.timer_joint_limits = self.create_timer(
            4.0 if self.REAL_HARDWARE else 2.0, self._publish_joint_limits)

        # Raw-vs-governed reference diff, 24 floats: [pos/ori/vel/wvel]x[R,L], 3 each.
        self.pub_gov_telemetry = self.create_publisher(Float64MultiArray, '/qp_debug/governor', 10)

        # --- SUBSCRIBERS ---
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.ref_cb_right, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.ref_cb_left, 10)
        # Grasp-phase flag from shared autonomy: drops the posture-task weight
        # during autonomous precision phases (grasp/lift) for tighter tracking.
        self.grasp_active = False
        self._posture_scale = 1.0
        self.create_subscription(Bool, '/shared_autonomy/grasp_active', self.grasp_active_cb, 10)

        # Active-arm tracking (Option B bimanual): the INACTIVE arm is frozen at
        # its current EE pose (held by a zero-velocity CLF) with MAX_WEIGHT_SLACK,
        # GAMMA_MAX and doubled joint damping, but is NOT zeroed — its QP-computed
        # motion is ALWAYS sent to TSID so it can bend to help the active arm
        # avoid collisions.
        self.active_arm = 'right'
        self.right_frozen = False
        self.left_frozen = False
        self._refs_initialized = False
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # Local-minima stall escape (see _apply_stall_escape): per-arm timer +
        # latched escaping flag, reset only by that method itself.
        self._stall_state = {'right': {'timer': 0.0, 'escaping': False, 'v_filt': 0.0},
                              'left': {'timer': 0.0, 'escaping': False, 'v_filt': 0.0}}

        # Services for controller switching
        self.switch_srv = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_srv = self.create_client(ListControllers, '/controller_manager/list_controllers')
        # What check_and_switch_controllers() actually changed at startup, so
        # restore_controllers() can undo exactly that (and nothing else) on
        # shutdown -- see both methods below.
        self._ctrl_activated = []
        self._ctrl_deactivated = []

        # Low-rate RViz obstacle marker timer (matches original 0.5s cadence;
        # halved on real hardware -- see the publish_every_n comment above).
        self.timer_obs = self.create_timer(
            1.0 if self.REAL_HARDWARE else 0.5, lambda: self.viz.publish_obstacle_marker(self.hri))

    def _now(self):
        """Wall-clock time [s], used by the reference-staleness watchdog."""
        return self.get_clock().now().nanoseconds / 1e9

    def _dbg_tick_diag(self, tick_start, tick_dt_ms):
        """Log tick-cadence stats once per second (TRIAGO_DEBUG_TICKS=1)."""
        if self._dbg_win_start is None:
            self._dbg_win_start = tick_start
        self._dbg_n += 1
        if 0.0 < tick_dt_ms < 1.0:
            self._dbg_n_bunched += 1
        if tick_dt_ms > self._dbg_max_gap_ms:
            self._dbg_max_gap_ms = tick_dt_ms
        if tick_start - self._dbg_win_start >= 1.0:
            self.get_logger().warn(
                f"[TICK-DIAG] {self._dbg_n} ticks/s | bunched(<1ms)="
                f"{self._dbg_n_bunched} | max_gap={self._dbg_max_gap_ms:.1f}ms "
                "(bunched>0 or large max_gap => timer catch-up bursts)")
            self._dbg_win_start = tick_start
            self._dbg_n = 0
            self._dbg_n_bunched = 0
            self._dbg_max_gap_ms = 0.0

    # =====================================================================
    # WORLD-SCENE SOURCE (overridable seam)
    # =====================================================================
    def _build_world_scene(self):
        """Return the WorldScene the CBF collision model + viz are built from.

        This is the SINGLE place "where the obstacle world comes from" is decided;
        everything downstream (build_collision_model, VisualizationEngine, Meshcat,
        RViz markers) consumes self.world_scene generically. The base reads the
        YAML world named by the `world_name` ROS param (behavior unchanged). A
        subclass overrides ONLY this method to source the world elsewhere — e.g.
        main_qp_controller_perceived.py builds it from the head camera's latched
        perceived-world snapshot instead of a YAML file.
        """
        world_name = self.get_parameter('world_name').get_parameter_value().string_value
        return load_world(world_name)

    # =====================================================================
    # CONFIGURABLE FREQUENCY GOVERNOR
    # =====================================================================
    @property
    def control_frequency(self):
        # Current control loop frequency [Hz].
        return self._control_freq

    def set_control_frequency(self, freq_hz):
        # Dynamically govern the solve_and_publish loop, recreating the timer.
        if freq_hz <= 0:
            self.get_logger().warn(f"[FREQ] Ignoring non-positive frequency {freq_hz}.")
            return
        self._control_freq = float(freq_hz)
        if self.loop_timer is not None:
            self.destroy_timer(self.loop_timer)
        self.loop_timer = self.create_timer(1.0 / self._control_freq, self._control_tick_guarded)
        self.get_logger().info(f"[FREQ] Control loop set to {self._control_freq:.1f} Hz.")

    def _control_tick_guarded(self):
        # Timer entrypoint: NEVER let one bad tick silently take the whole loop
        # down. An unhandled exception in a timer callback propagates out of
        # rclpy.spin and looks like a frozen node with no error -- log the full
        # traceback (throttled) and keep the loop alive instead.
        try:
            self.solve_and_publish()
        except Exception:  # noqa: BLE001
            self.get_logger().error(
                "[LOOP] Unhandled exception in control tick (loop kept alive):\n"
                + traceback.format_exc(), throttle_duration_sec=1.0)

    def start_control_loop(self):
        # Engage the real-time loop at the configured frequency.
        self.set_control_frequency(self._control_freq)

    # =====================================================================
    # SETUP HELPERS
    # =====================================================================
    def _publish_grasping_link_tfs(self):
        """Broadcast static TFs for gripper grasping links if not already in the TF tree.

        On the real TIAGo Pro, the URDF may not include gripper_*_grasping_link.
        We publish the same transforms that would come from a manual
        static_transform_publisher:
            parent: gripper_{side}_base_link
            child:  gripper_{side}_grasping_link
            translation: [0, 0, 0.157]
            rotation:    RPY [0, -1.5708, 0]  (quaternion ~ [0, -0.7068, 0, 0.7074])
        """
        import math
        # Check if already available in TF (give 0.5s)
        need_right = not self.tf_buffer.can_transform(
            'gripper_right_base_link', 'gripper_right_grasping_link',
            rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
        need_left = not self.tf_buffer.can_transform(
            'gripper_left_base_link', 'gripper_left_grasping_link',
            rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))

        if not need_right and not need_left:
            self.get_logger().info("[TF] Grasping frames already in TF tree — no static publish needed.")
            return

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        transforms = []

        # RPY [0, -1.5708, 0] → quaternion
        # Ry(-pi/2): qx=0, qy=-sin(pi/4), qz=0, qw=cos(pi/4)
        pitch = -math.pi / 2.0
        qx, qy, qz, qw = 0.0, -math.sin(pitch / 2.0), 0.0, math.cos(pitch / 2.0)

        sides = []
        if need_right:
            sides.append(('gripper_right_base_link', 'gripper_right_grasping_link'))
        if need_left:
            sides.append(('gripper_left_base_link', 'gripper_left_grasping_link'))

        for parent, child in sides:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.157
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)

        self.static_tf_broadcaster.sendTransform(transforms)
        published_names = [s[1] for s in sides]
        self.get_logger().info(f"[TF] Published static grasping frames: {published_names}")

    def get_urdf(self):
        # Fetch the robot_description string from robot_state_publisher.
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value

    def wait_for_tf(self):
        # Block until the base->wrist transform is available (mirrors original startup).
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform(cfg.REF_FRAME, cfg.RIGHT_CHAIN[-1], rclpy.time.Time()):
                break

    def check_and_switch_controllers(self):
        # Smart switch: activate TSID JS velocity controllers, deactivate conflicts.
        if not self.list_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("List Controllers Service unavailable.")
            return False
        future = self.list_srv.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        if not future.result():
            return False

        current_state = {c.name: c.state for c in future.result().controller}
        to_activate = [t for t in [cfg.RIGHT_CONTROLLER, cfg.LEFT_CONTROLLER] if current_state.get(t) != "active"]
        to_deactivate = [c for c in cfg.CONFLICTING_CONTROLLERS if current_state.get(c) == "active"]

        if not to_activate and not to_deactivate:
            self.get_logger().info("Controllers already correct.")
            self.active_controller_mode = True
            return True
        if not self.switch_srv.wait_for_service(timeout_sec=1.0):
            return False

        req = SwitchController.Request()
        req.activate_controllers = to_activate
        req.deactivate_controllers = to_deactivate
        req.strictness = SwitchController.Request.STRICT
        self.get_logger().info(f"Switching: +{to_activate} | -{to_deactivate}")
        future = self.switch_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().ok:
            self.active_controller_mode = True
            # Remember exactly what we changed so restore_controllers() can
            # put the robot back the way it was found, on shutdown.
            self._ctrl_activated = to_activate
            self._ctrl_deactivated = to_deactivate
            return True
        self.get_logger().error("Switch Service Failed.")
        return False

    def restore_controllers(self):
        # Undo check_and_switch_controllers()'s switch: reactivate whatever we
        # deactivated at startup, deactivate whatever we newly activated -- so
        # a killed node leaves the robot in the SAME controller state it found
        # it in, not stuck on the QP's velocity controllers. Safe to call even
        # if the switch never happened (both lists empty -> no-op) or the
        # controller_manager is already gone (logged, not raised).
        if not self._ctrl_activated and not self._ctrl_deactivated:
            return
        # print(flush=True) alongside get_logger(): this fires during shutdown
        # (possibly mid-SIGINT), where a ROS logger can be suppressed/buffered
        # or the executor already tearing down -- print is the only channel
        # guaranteed to reach the console at this exact moment (same reason
        # trajectory_generator.py's banner/status lines use it too).
        print(f"[Shutdown] Restoring original controllers: "
              f"+{self._ctrl_deactivated} | -{self._ctrl_activated} ...", flush=True)
        # Belt-and-braces: this runs during shutdown (possibly right after a
        # SIGINT), a fragile moment for the rclpy context -- never let a
        # failure here block the rest of the cleanup in main()'s finally block.
        try:
            if not self.switch_srv.wait_for_service(timeout_sec=2.0):
                msg = ("[Shutdown] Switch Controller service unavailable -- cannot restore "
                      f"original controller state (was: +{self._ctrl_activated} "
                      f"-{self._ctrl_deactivated}).")
                print(msg, flush=True)
                self.get_logger().error(msg)
                return
            req = SwitchController.Request()
            req.activate_controllers = self._ctrl_deactivated
            req.deactivate_controllers = self._ctrl_activated
            req.strictness = SwitchController.Request.STRICT
            self.get_logger().info(
                f"[Shutdown] Restoring original controllers: "
                f"+{self._ctrl_deactivated} | -{self._ctrl_activated}")
            future = self.switch_srv.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None and future.result().ok:
                print("[Shutdown] Original controller state restored.", flush=True)
                self.get_logger().info("[Shutdown] Original controller state restored.")
            else:
                print("[Shutdown] Restore Switch Service Failed.", flush=True)
                self.get_logger().error("[Shutdown] Restore Switch Service Failed.")
        except Exception as e:  # noqa: BLE001 -- shutdown must proceed regardless
            print(f"[Shutdown] restore_controllers() raised: {e}", flush=True)
            self.get_logger().error(f"[Shutdown] restore_controllers() raised: {e}")

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def joint_callback(self, msg):
        # Parse physical joint positions and hand them to the kinematics filter.
        if self.kin.model is None:
            return
        q_physical = pin.neutral(self.kin.model)
        bad_fields = []  # [(joint_name, 'position'/'velocity', raw_value), ...]
        for i, name in enumerate(msg.name):
            if self.kin.model.existJointName(name):
                jid = self.kin.model.getJointId(name)
                idx_q = self.kin.model.joints[jid].idx_q
                idx_v = self.kin.model.joints[jid].idx_v
                if idx_q >= 0:
                    pos = msg.position[i]
                    if not np.isfinite(pos):
                        bad_fields.append((name, 'position', pos))
                    else:
                        q_physical[idx_q] = pos
                if self.REAL_HARDWARE and idx_v >= 0 and i < len(msg.velocity):
                    vel = msg.velocity[i]
                    if not np.isfinite(vel):
                        bad_fields.append((name, 'velocity', vel))
        if bad_fields:
            # Non-finite sensor reading on ONE joint (e.g. a mobile-base wheel
            # that structurally never reports a real velocity) -- sanitized to
            # the neutral/zero placeholder for JUST that joint, every other
            # joint in this same message still updates normally. Rejecting the
            # WHOLE message here would be worse: if the same joint is always
            # bad (a permanent characteristic, not a one-off fault), current_q/
            # current_v would freeze forever and the arms would never update again.
            # Downstream, the rate-damping term must still SELECT at arm indices
            # (0 * nan == nan: a mask-multiply would poison the whole solve).
            self.get_logger().error(
                f"[SENSOR] Non-finite /joint_states entr{'y' if len(bad_fields)==1 else 'ies'} "
                f"(that joint only; all others updated normally -- a bad position holds at "
                f"neutral, a bad velocity reading has no effect since velocity is derived "
                f"from position, not read directly): "
                + ", ".join(f"{n}.{f}={v}" for n, f, v in bad_fields),
                once=True)
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Real hardware's raw /joint_states velocity is unfiltered and was found
        # to inject sensor-noise ripple straight into the rate-damping term
        # (RATE_WEIGHT tracks it hard) and out into qdot_cmd. Always derive +
        # EMA-filter velocity from position (sim's path) instead of trusting v_direct.
        self.kin.update_from_joint_state(q_physical, time_stamp)

    def grasp_active_cb(self, msg):
        """Tracks whether shared autonomy is autonomously driving a grasp/lift."""
        self.grasp_active = bool(msg.data)

    def _publish_joint_limits(self):
        """Publish [name:lower:upper;...] for the slider GUI, the same limits
        the joint-limit CBF rows enforce. Repeats periodically (not one-shot)
        so late bag recorders still capture it."""
        if self.kin.model is None:
            return
        names = cfg.RIGHT_JOINTS + cfg.LEFT_JOINTS + cfg.HEAD_JOINTS + cfg.GRIPPER_FINGER_JOINTS
        lower, upper = self.kin.get_joint_limits(names)
        payload = ";".join(f"{n}:{lo:.4f}:{hi:.4f}" for n, lo, hi in zip(names, lower, upper))
        self.pub_joint_limits.publish(String(data=payload))

    def _freeze_arm(self, side):
        """Snapshot one arm's CURRENT EE pose as its held reference (zero velocity).

        Used when an arm becomes inactive (arm switch / stale teleop). The arm
        keeps imposed_motion=True so its CLF holds it at this pose; with its
        pinned MAX slack weight + doubled damping it stays put unless yielding
        helps the active arm. Requires up-to-date FK (call after kinematics update).
        """
        if self.kin.current_q is None:
            return
        ee_id = self.kin.ee_id_right if side == 'right' else self.kin.ee_id_left
        if ee_id is None:
            return
        pos = np.array(self.kin.data.oMf[ee_id].translation)
        rpy = pin.rpy.matrixToRpy(self.kin.data.oMf[ee_id].rotation)
        if side == 'right':
            self.x_ref_right = pos; self.rpy_ref_right = rpy
            self.xdot_ref_right = np.zeros(3); self.w_ref_right = np.zeros(3)
            self.right_imposed_motion = True
            self.last_right_msg_time = self._now()
            self.right_frozen = True
            self.gov_right.reset()  # Clear velocity memory (arm is now stationary)
        else:
            self.x_ref_left = pos; self.rpy_ref_left = rpy
            self.xdot_ref_left = np.zeros(3); self.w_ref_left = np.zeros(3)
            self.left_imposed_motion = True
            self.last_left_msg_time = self._now()
            self.left_frozen = True
            self.gov_left.reset()  # Clear velocity memory (arm is now stationary)

    def active_arm_cb(self, msg):
        """Arm switch: freeze the now-inactive arm at its current pose."""
        new_arm = msg.data
        if new_arm not in ('right', 'left') or new_arm == self.active_arm:
            return
        old_arm = self.active_arm
        self.active_arm = new_arm
        self._freeze_arm(old_arm)            # hold the arm we just left
        if new_arm == 'right':
            self.right_frozen = False        # the newly-active arm tracks teleop again
        else:
            self.left_frozen = False
        self.get_logger().info(f"[ARM] Active arm = {new_arm.upper()}; froze {old_arm.upper()} at current pose.")

    def ref_cb_right(self, msg):
        # Right-arm cartesian reference (12+ float protocol, 6-float fallback).
        # A fresh reference means this arm is being actively driven -> un-freeze it.
        if len(msg.data) >= 12:
            self.x_ref_right = np.array(msg.data[0:3])
            self.rpy_ref_right = np.array(msg.data[3:6])
            self.xdot_ref_right = np.array(msg.data[6:9])
            self.w_ref_right = np.array(msg.data[9:12])
            self.task_dim_right = msg.data[12] if len(msg.data) >= 13 else 6.0
            self.right_imposed_motion = True
            self.right_frozen = False
            self.last_right_msg_time = self._now()
        elif len(msg.data) >= 6:
            self.x_ref_right = np.array(msg.data[0:3])
            self.xdot_ref_right = np.array(msg.data[3:6])
            self.right_imposed_motion = True
            self.right_frozen = False
            self.last_right_msg_time = self._now()

    def ref_cb_left(self, msg):
        # Left-arm cartesian reference (12+ float protocol, 6-float fallback).
        # A fresh reference means this arm is being actively driven -> un-freeze it.
        if len(msg.data) >= 12:
            self.x_ref_left = np.array(msg.data[0:3])
            self.rpy_ref_left = np.array(msg.data[3:6])
            self.xdot_ref_left = np.array(msg.data[6:9])
            self.w_ref_left = np.array(msg.data[9:12])
            self.task_dim_left = msg.data[12] if len(msg.data) >= 13 else 6.0
            self.left_imposed_motion = True
            self.left_frozen = False
            self.last_left_msg_time = self._now()
        elif len(msg.data) >= 6:
            self.x_ref_left = np.array(msg.data[0:3])
            self.xdot_ref_left = np.array(msg.data[3:6])
            self.left_imposed_motion = True
            self.left_frozen = False
            self.last_left_msg_time = self._now()

    # =====================================================================
    # TASK ERROR EXTRACTION (5D / 6D / 3D)
    # =====================================================================
    def _arm_task_error(self, ee_id, x_ref, rpy_ref, xdot_ref, w_ref, task_dim):
        # Compute the (weighted-ready) task error + feedforward velocity for one arm.
        if x_ref is None or ee_id is None:
            return np.zeros(3), np.zeros(3)
        x_real = self.kin.data.oMf[ee_id].translation
        e_pos = x_ref - x_real
        # task_dim == 3.0: position-only CLF (orientation relaxed for a
        # local-minima escape); checked before orientation_ctrl so it applies
        # even when orientation_ctrl=True.
        if task_dim == 3.0:
            return e_pos, xdot_ref
        if self.orientation_ctrl and rpy_ref is not None:
            R_real = self.kin.data.oMf[ee_id].rotation
            R_des = pin.rpy.rpyToMatrix(rpy_ref[0], rpy_ref[1], rpy_ref[2])
            if task_dim == 5.0:
                e_R = np.cross(R_real[:, 0], R_des[:, 0])     # 5D: align X-axis only
            else:
                e_R = pin.log3(R_des @ R_real.T)              # 6D: full SO(3) error
            e_task = np.concatenate([e_pos, e_R])
            v_task = np.concatenate([xdot_ref, w_ref])
        else:
            e_task = e_pos                                    # 3D: translation only
            v_task = xdot_ref
        return e_task, v_task

    def extract_task_errors(self):
        # Build both arms' task errors for the CLF rows.
        e_r, v_r = self._arm_task_error(self.kin.ee_id_right, self.x_ref_right, self.rpy_ref_right,
                                        self.xdot_ref_right, self.w_ref_right, self.task_dim_right)
        e_l, v_l = self._arm_task_error(self.kin.ee_id_left, self.x_ref_left, self.rpy_ref_left,
                                        self.xdot_ref_left, self.w_ref_left, self.task_dim_left)
        return e_r, v_r, e_l, v_l

    def _ee_speed(self, ee_id):
        # Measured Cartesian EE speed (m/s) from kin.current_v -- same measured
        # joint velocities already used for rate-damping/tracking telemetry.
        if ee_id is None or self.kin.current_v is None:
            return 0.0
        J_pos = pin.getFrameJacobian(self.kin.model, self.kin.data, ee_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        return float(np.linalg.norm(J_pos @ self.kin.current_v))

    def _apply_stall_escape(self, side, ee_id, e_task, v_task, n_eff, active_interaction, dt):
        """Reactive CLF-CBF local-minima escape (real hardware: an arm reaching
        for a goal on the far side of a flat obstacle -- e.g. up and over a
        table from underneath -- can settle at a genuine saddle point where the
        CLF's goal-gradient and the CBF's repulsion gradient are anti-parallel).

        If this arm's measured EE speed stays near-zero for STALL_HOLD_S while
        an active barrier (n_eff not None) is engaged and the task error is
        still large, temporarily redirect the CLF's POSITION reference toward a
        nearby waypoint along the component of the goal direction that is
        TANGENTIAL to the blocking constraint's own softmax-aggregated contact
        normal (n_eff, same weights as the barrier itself -- see
        collision_manager.compute_softmin_jacobian). This reuses the existing,
        already-tuned CLF/QP machinery verbatim (it just tracks a different
        nearby point), so the escape motion stays subject to the same CBF and
        joint-limit constraints as any other reference. Orientation error/rate
        is left untouched.
        """
        st = self._stall_state[side]
        e_pos = e_task[:3]
        err_norm = float(np.linalg.norm(e_pos))
        ang_err = float(np.linalg.norm(e_task[3:])) if len(e_task) > 3 else 0.0
        # On real hardware kin.current_v is the RAW, unfiltered sensor velocity
        # (see robot_kinematics.py::update_from_joint_state) -- it never actually
        # settles near zero even while the arm is macroscopically still. Smooth
        # it LOCALLY (own EMA, doesn't touch the raw signal used elsewhere for
        # rate-damping/CBF margins) so the stall gate compares against genuine
        # motion, not sensor jitter.
        v_raw = self._ee_speed(ee_id)
        st['v_filt'] += (1.0 - cfg.STALL_SPEED_FILTER_ALPHA) * (v_raw - st['v_filt'])
        v_ee = st['v_filt']

        stalled_now = (v_ee < cfg.STALL_SPEED_THRESH and
                       (err_norm > cfg.STALL_ERR_POS_THRESH or ang_err > cfg.STALL_ERR_ANG_THRESH))
        st['timer'] = st['timer'] + dt if stalled_now else 0.0

        if not st['escaping']:
            if st['timer'] >= cfg.STALL_HOLD_S and active_interaction and n_eff is not None:
                st['escaping'] = True
                # Printed with flush=True (not get_logger, which can be
                # buffered/async) so this survives even a hard native crash
                # right after -- EDGE-TRIGGERED (once per escape episode, not
                # per-tick: a per-tick flushed print at control-loop rate is
                # itself a real perf hit, see .kiro/context.md).
                print(f"[ESCAPE-DEBUG] {side}: starting tangent computation -- "
                     f"e_pos={np.round(e_pos, 4).tolist()} err_norm={err_norm:.4f} "
                     f"n_eff={np.round(n_eff, 4).tolist()} v_ee={v_ee:.4f}", flush=True)
                self.get_logger().warn(
                    f"\033[93m[ESCAPE] {side} arm stalled >= {cfg.STALL_HOLD_S:.0f}s against an "
                    f"active barrier -- sliding tangentially to clear it.\033[0m")
        elif v_ee > cfg.STALL_RESUME_SPEED or not active_interaction:
            st['escaping'] = False

        if not st['escaping']:
            return e_task, v_task
        if n_eff is None:
            # Barrier still active but its aggregated normal happened to cancel
            # out THIS tick (opposing pairs) -- can't derive a tangent; hold the
            # escaping latch and just skip the bias for this one tick rather
            # than touch n_eff below.
            return e_task, v_task

        e_dir = e_pos / err_norm if err_norm > 1e-6 else np.zeros(3)
        tangent = e_dir - np.dot(e_dir, n_eff) * n_eff
        tnorm = float(np.linalg.norm(tangent))
        if tnorm < cfg.STALL_TANGENT_MIN:
            # Goal direction is (near-)parallel to the obstacle normal (the
            # straight-through-the-surface case) -- fall back to a horizontal
            # direction perpendicular to the normal, since the deadlock here is
            # almost always a flat/near-horizontal surface (table) blocking
            # straight-up motion.
            tangent = np.cross(n_eff, np.array([0.0, 0.0, 1.0]))
            tnorm = float(np.linalg.norm(tangent))
            if tnorm < 1e-6:
                return e_task, v_task
        tangent = tangent / tnorm

        e_pos_new = cfg.STALL_ESCAPE_STEP_M * tangent
        v_pos_new = cfg.STALL_ESCAPE_SPEED * tangent
        if len(e_task) > 3:
            return np.concatenate([e_pos_new, e_task[3:]]), np.concatenate([v_pos_new, v_task[3:]])
        return e_pos_new, v_pos_new

    # =====================================================================
    # OVERRIDABLE SEAMS (async-execution hooks)
    # =====================================================================
    # The only points where main_qp_controller_real.py diverges: it overrides
    # these to move CBF + overlays onto worker threads with a staleness watchdog.
    def _process_deferred_topology(self):
        # Apply any grasp attach/detach queued by shared autonomy (deferred to
        # here because it needs fresh oMi/oMg). This STRUCTURALLY mutates the
        # shared collision model, so the real subclass overrides it to hold
        # col.geom_lock around the mutation (serialized vs. the CBF worker).
        if self.hri.pending_attach is not None:
            arm_side, color = self.hri.pending_attach
            self.hri.pending_attach = None
            try:
                self.hri.attach_object_visually(arm_side, color)
            except Exception as e:
                self.get_logger().warn(f"[TOPOLOGY] Attach failed: {e}")
        if self.hri.pending_detach is not None:
            arm_side, color, world_pos = self.hri.pending_detach
            self.hri.pending_detach = None
            try:
                self.hri.detach_object_visually(arm_side, color, world_pos)
            except Exception as e:
                self.get_logger().warn(f"[TOPOLOGY] Detach failed: {e}")

    def _compute_cbf(self):
        # Synchronous SoftMin CBF for THIS tick, returned as a CbfResult (always
        # fresh here). The real subclass overrides this to return the latest
        # async worker result and drive the staleness watchdog.
        (J_soft_r, h_soft_r, J_soft_l, h_soft_l,
         d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
         active_r, active_l, n_eff_r, n_eff_l) = \
            self.col.compute_softmin_jacobian(
                self.kin.current_v, self.kin.idx_right, self.kin.idx_left,
                self.hri.grasp_margin_targets, self.hri.attached_objects,
                self.hri.attached_adjacency, self.hri.ignored_targets, self.publish_counter,
                attach_ramp_shifts=self.hri.get_attach_ramp_shifts(),
                attached_object_arm=self.hri.attached_object_arm)
        return CbfResult(
            J_soft_r, h_soft_r, J_soft_l, h_soft_l,
            d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance, active_r, active_l,
            self.col.witness_min_distance, self.col.witness_min_points,
            getattr(self.col, 'top_active_pairs', []), True, n_eff_r, n_eff_l)

    def _gate_command(self, q_dot_safe):
        # Final gate on the joint-velocity command actually sent to hardware.
        # Identity here; the real subclass zeroes it (freeze) while the async CBF
        # result is stale, so a stale barrier can never drive the arms.
        return q_dot_safe

    def _publish_visual_overlays(self, cbf, q_dot_safe):
        # RViz debug overlays: safety-margin scalar, collision-witness line and
        # teleop tethers. Runs inline here (already downsampled by the caller);
        # the real subclass dispatches it to a viz worker thread. Reads the
        # witness/margin from `cbf`, not self.col, so it is thread-snapshot-safe.
        if not cfg.DISABLE_CBF:
            # Per-arm margin; NaN when that arm has no active pair (not the
            # h_soft=1.0 QP sentinel, which isn't a real margin).
            margin_r = (cbf.h_soft_r - cbf.d_safe_r) if cbf.active_r else float('nan')
            margin_l = (cbf.h_soft_l - cbf.d_safe_l) if cbf.active_l else float('nan')
            self.pub_debug_h.publish(Float64MultiArray(data=[float(margin_r), float(margin_l)]))
        self.viz.publish_debug(
            self.kin.model, self.kin.data,
            (cbf.witness_distance, cbf.witness_points),
            self.kin.current_q,
            q_dot_safe, None, None, self.kin.ee_id_right, self.kin.ee_id_left,
            cfg.JOINT_LIMIT_BUFFER_BASE)
        self.viz.publish_teleop_tether()

    # =====================================================================
    # MAIN CONTROL LOOP
    # =====================================================================
    def solve_and_publish(self):
        # One control tick: kinematics -> SoftMin -> QP -> publish -> evolve twin.
        if self.kin.current_q is None:
            return

        # Tick-to-tick wall time (jitter diagnostic, see pub_loop_timing above).
        _tick_start = time.perf_counter()
        _tick_dt_ms = ((_tick_start - self._last_tick_perf) * 1000.0
                       if self._last_tick_perf is not None else 0.0)
        self._last_tick_perf = _tick_start
        if self._dbg_ticks:
            self._dbg_tick_diag(_tick_start, _tick_dt_ms)

        # --- Watchdog: a stale-reference arm is FROZEN at its current pose (held
        # by a zero-velocity CLF) rather than going limp. Option B keeps it under
        # QP control so it can still bend to help the active arm avoid collisions.
        if self.right_imposed_motion and not self.right_frozen \
                and (self._now() - self.last_right_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self._freeze_arm('right')
            print("[Safety] Watchdog: Right reference stale -> frozen at current pose.")
        if self.left_imposed_motion and not self.left_frozen \
                and (self._now() - self.last_left_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self._freeze_arm('left')
            print("[Safety] Watchdog: Left reference stale -> frozen at current pose.")

        # --- 0. Kinematics + geometry refresh ---
        _kin_start = time.perf_counter()
        self.kin.update_kinematics()
        self.kin.debug_interrogate()
        self.col.update_geometry(self.kin.current_q)
        _kin_ms = (time.perf_counter() - _kin_start) * 1000.0

        # One-time: freeze BOTH arms at their startup pose so every arm always has
        # a holding CLF (no limp/uncontrolled arm). Teleop overrides the active one.
        if not self._refs_initialized:
            self._freeze_arm('right')
            self._freeze_arm('left')
            self._refs_initialized = True

        # --- Deferred attach/detach (needs fresh oMi / oMg) ---
        self._process_deferred_topology()

        # --- Grasp contact distance telemetry ---
        self.hri.publish_contact_distances()

        # --- Low-level tracking error (commanded vs measured) ---
        qdot_err_14, xdot_err_6 = self.kin.compute_tracking_errors(self.last_qdot_cmd_14)

        # --- 1. SoftMin CBF aggregation (TWO independent per-arm barriers) ---
        # _compute_cbf() is the async seam: synchronous here, worker-fed on real
        # hardware. It returns a CbfResult carrying the barrier the QP consumes
        # plus the witness/top-pairs telemetry captured in that same pass. Names
        # are unpacked back into locals so the governor/QP block below is unchanged.
        _cbf_start = time.perf_counter()
        cbf = self._compute_cbf()
        J_soft_r, h_soft_r, J_soft_l, h_soft_l = cbf.J_soft_r, cbf.h_soft_r, cbf.J_soft_l, cbf.h_soft_l
        d_safe_dynamic_r, d_safe_dynamic_l = cbf.d_safe_r, cbf.d_safe_l
        abs_min_distance = cbf.abs_min_distance
        _cbf_ms = (time.perf_counter() - _cbf_start) * 1000.0

        # --- 2. Task errors ---
        # Governor applies velocity/error/acceleration bounds before the CLF
        # sees the reference; raw references are preserved for plotting.
        _gov_start = time.perf_counter()
        # Nominal period, not a measured delta -- the wall-clock timer already
        # fires at a steady real rate; measuring dt only adds executor jitter.
        dt = 1.0 / self._control_freq
        if cfg.ENABLE_REFERENCE_GOVERNOR:
            # Right arm
            x_real_r = np.array(self.kin.data.oMf[self.kin.ee_id_right].translation) if self.kin.ee_id_right else None
            R_real_r = self.kin.data.oMf[self.kin.ee_id_right].rotation if self.kin.ee_id_right else None
            x_gov_r, rpy_gov_r, v_gov_r, w_gov_r = self.gov_right.govern(
                self.x_ref_right, self.rpy_ref_right, self.xdot_ref_right, self.w_ref_right,
                x_real_r, R_real_r, dt)
            # Left arm
            x_real_l = np.array(self.kin.data.oMf[self.kin.ee_id_left].translation) if self.kin.ee_id_left else None
            R_real_l = self.kin.data.oMf[self.kin.ee_id_left].rotation if self.kin.ee_id_left else None
            x_gov_l, rpy_gov_l, v_gov_l, w_gov_l = self.gov_left.govern(
                self.x_ref_left, self.rpy_ref_left, self.xdot_ref_left, self.w_ref_left,
                x_real_l, R_real_l, dt)
        else:
            # Passthrough (no filtering): governed == raw
            x_gov_r, rpy_gov_r, v_gov_r, w_gov_r = (
                self.x_ref_right, self.rpy_ref_right, self.xdot_ref_right, self.w_ref_right)
            x_gov_l, rpy_gov_l, v_gov_l, w_gov_l = (
                self.x_ref_left, self.rpy_ref_left, self.xdot_ref_left, self.w_ref_left)

        # Compute CLF task errors from the GOVERNED references (not the raw ones).
        e_r, v_r = self._arm_task_error(self.kin.ee_id_right, x_gov_r, rpy_gov_r,
                                        v_gov_r, w_gov_r, self.task_dim_right)
        e_l, v_l = self._arm_task_error(self.kin.ee_id_left, x_gov_l, rpy_gov_l,
                                        v_gov_l, w_gov_l, self.task_dim_left)

        # Local-minima stall escape (see _apply_stall_escape) -- only while
        # actually tracking a real reference; a frozen/idle arm's error is
        # already ~0 so it never trips the stall condition regardless. This is
        # brand-new logic running inside the safety control loop: any bug in it
        # must degrade to "no bias" (never crash the whole node), so guard each
        # arm independently and self-disable on error instead of propagating.
        if cfg.ENABLE_STALL_ESCAPE:
            if x_gov_r is not None and not self.right_frozen:
                try:
                    e_r, v_r = self._apply_stall_escape('right', self.kin.ee_id_right, e_r, v_r,
                                                        cbf.n_eff_r, cbf.active_r, dt)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    print(f"[ESCAPE-DEBUG] right arm EXCEPTION -- disabling escape:\n{tb}",
                         flush=True)
                    self.get_logger().error(f"[ESCAPE] right arm error, disabling:\n{tb}")
                    cfg.ENABLE_STALL_ESCAPE = False
            if x_gov_l is not None and not self.left_frozen:
                try:
                    e_l, v_l = self._apply_stall_escape('left', self.kin.ee_id_left, e_l, v_l,
                                                        cbf.n_eff_l, cbf.active_l, dt)
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    print(f"[ESCAPE-DEBUG] left arm EXCEPTION -- disabling escape:\n{tb}",
                         flush=True)
                    self.get_logger().error(f"[ESCAPE] left arm error, disabling:\n{tb}")
                    cfg.ENABLE_STALL_ESCAPE = False
        _gov_ms = (time.perf_counter() - _gov_start) * 1000.0

        # --- 3. Build + solve the CLF-CBF-QP ---
        # Smoothly ramp the posture-task weight scale: drop toward POSTURE_GRASP_SCALE
        # during autonomous precision phases (grasp/lift), restore to 1.0 otherwise.
        target_scale = cfg.POSTURE_GRASP_SCALE if self.grasp_active else 1.0
        a_ps = dt / (cfg.POSTURE_SCALE_TAU + dt)
        self._posture_scale += a_ps * (target_scale - self._posture_scale)
        self.qp.posture_scale = self._posture_scale
        # Cost decoupling: a frozen (inactive) arm gets fixed MAX slack, GAMMA_MAX
        # CLF, and doubled damping inside the QP — but only when exactly that arm
        # is frozen while the other is active (both-active keeps the dynamic
        # coupling unchanged; both-frozen pins both, which is the idle hold).
        # During an autonomous grasp the ACTIVE arm is boosted to the max dynamic
        # values (slack + gamma) so it converges tightly to the grasp reference.
        boost_arm = self.active_arm if self.grasp_active else None
        # Orientation-weight boost applies to a SUPERSET of the slack/gamma boost:
        # the active arm during autonomous grasp/release (grasp_active) AND any arm
        # currently CARRYING an attached object (the HOLDING / placement-approach
        # phase). This keeps the gripper's approach-axis / placement orientation
        # tight whenever precision matters -- including steering the held object to
        # its release pose -- without touching the slack/gamma tracking boost.
        orient_boost_arms = set(self.hri.attached_object_arm.values())
        if self.grasp_active:
            orient_boost_arms.add(self.active_arm)
        _solve_start = time.perf_counter()
        q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total = self.qp.build_and_solve(
            self.kin, J_soft_r, h_soft_r, J_soft_l, h_soft_l,
            d_safe_dynamic_r, d_safe_dynamic_l,
            self.right_imposed_motion, self.left_imposed_motion,
            self.xdot_ref_right, self.xdot_ref_left, e_r, v_r, e_l, v_l, dt,
            right_frozen=self.right_frozen, left_frozen=self.left_frozen,
            tracking_boost_arm=boost_arm, orient_boost_arms=orient_boost_arms)
        _solve_ms = (time.perf_counter() - _solve_start) * 1000.0
        _telemetry_start = time.perf_counter()

        self.publish_counter += 1

        # Safety-critical for main_shared_autonomy.py's own CBF-gated policy solve
        # (not dashboard telemetry) -- must NOT be downsampled like the rest of
        # this tick's publishing, or its 3-tick staleness tolerance trips falsely.
        self._publish_shared_collision(b_col_pair, J_soft_r, J_soft_l)

        # --- 4. Downsampled telemetry publishing ---
        if self.publish_counter % self.publish_every_n == 0:
            self._publish_telemetry(q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total,
                                    J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                                    d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
                                    qdot_err_14, xdot_err_6, cbf.top_active_pairs)
            # Generic measured joint velocity (14 floats: R7+L7), same
            # environment-dependent signal the QP itself consumes -- see the
            # publisher's own comment above.
            if self.kin.idx_right and self.kin.idx_left and self.kin.current_v is not None:
                meas_v_14 = np.concatenate(
                    (self.kin.current_v[self.kin.idx_right], self.kin.current_v[self.kin.idx_left]))
                self.pub_qdot_measured.publish(Float64MultiArray(data=meas_v_14.tolist()))

        # --- 5. Command publishing ---
        # Always send the QP-computed velocity for both arms -- the inactive
        # arm is safely held by its frozen-pose CLF, not zeroed.
        # _gate_command: identity in sim; on real hardware zeroes both arms
        # while the async CBF result is stale (staleness watchdog).
        q_dot_cmd = self._gate_command(q_dot_safe)
        # LAST LINE OF DEFENSE: never let a non-finite value reach the actual
        # hardware velocity command, no matter which upstream stage produced it
        # (sensor fault, a NaN CBF pair, an ill-conditioned QP solve -- all are
        # guarded individually upstream, but this is the one place that can
        # never be bypassed, since every path to real hardware funnels through it).
        if not np.isfinite(q_dot_cmd).all():
            self.get_logger().error(
                "[SAFETY] Non-finite joint-velocity command BLOCKED before publish "
                "-- commanding zero instead. Check the [SENSOR]/[CBF Error]/[QP Error] "
                "logs above for the originating stage.", throttle_duration_sec=1.0)
            q_dot_cmd = np.zeros_like(q_dot_cmd)
        cmd_data_r = [0.0] * 7
        cmd_data_l = [0.0] * 7
        if self.active_controller_mode:
            if self.kin.idx_right:
                cmd_data_r = q_dot_cmd[self.kin.idx_right].tolist()
                self.pub_right.publish(Float64MultiArray(data=cmd_data_r))
            if self.kin.idx_left:
                cmd_data_l = q_dot_cmd[self.kin.idx_left].tolist()
                self.pub_left.publish(Float64MultiArray(data=cmd_data_l))

        # Save the exact command sent to hardware for next tick's tracking-error math
        self.last_qdot_cmd_14 = np.concatenate((cmd_data_r, cmd_data_l))
        if self.publish_counter % self.publish_every_n == 0:
            self.pub_qdot_cmd.publish(Float64MultiArray(data=self.last_qdot_cmd_14.tolist()))

            # Raw-vs-governed diff, 24 floats: [pos/ori/vel/wvel]x[R,L], 3 each.
            if cfg.ENABLE_REFERENCE_GOVERNOR:
                def _gov_diff(raw, gov):
                    if raw is None or gov is None:
                        return [0.0, 0.0, 0.0]
                    return (np.asarray(raw) - np.asarray(gov)).tolist()
                gov_data = (
                    _gov_diff(self.x_ref_right, x_gov_r) +
                    _gov_diff(self.rpy_ref_right, rpy_gov_r) +
                    _gov_diff(self.xdot_ref_right, v_gov_r) +
                    _gov_diff(self.w_ref_right, w_gov_r) +
                    _gov_diff(self.x_ref_left, x_gov_l) +
                    _gov_diff(self.rpy_ref_left, rpy_gov_l) +
                    _gov_diff(self.xdot_ref_left, v_gov_l) +
                    _gov_diff(self.w_ref_left, w_gov_l))
                self.pub_gov_telemetry.publish(Float64MultiArray(data=gov_data))
            else:
                self.pub_gov_telemetry.publish(Float64MultiArray(data=[0.0] * 24))

        # --- 6. Evolve the digital twin (ideal kinematics) ---
        if cfg.SIMULATE_IDEAL_KINEMATICS:
            current_time = self._now()
            if self.last_sim_time is None:
                dt_sim = 0.001
            else:
                dt_sim = current_time - self.last_sim_time
            if dt_sim > 0.1:
                dt_sim = 0.001
            # Use the GATED command so a staleness freeze halts the twin too (else
            # the internal state would drift while the hardware is held). Identity
            # to q_dot_safe in sim (the gate is a passthrough there).
            self.kin.integrate_simulated_state(q_dot_cmd, dt_sim)
            self.last_sim_time = current_time

        # --- 7. External debug visualizer (optional tethers / overlays) ---
        if self.publish_counter % self.publish_every_n == 0:
            self._publish_visual_overlays(cbf, q_dot_safe)

        # Per-tick timing breakdown (see pub_loop_timing above): localizes the
        # tick_dt cost into its constituent phases, so a real-hardware compute
        # bottleneck can be pinned to a specific phase instead of guessed at.
        # misc_ms is whatever this breakdown didn't explicitly bracket (arm-freeze
        # checks, attach/detach handling, contact-distance telemetry, tracking-
        # error compute) plus any genuine OS scheduling wait.
        _telemetry_ms = (time.perf_counter() - _telemetry_start) * 1000.0
        _accounted_ms = _kin_ms + _cbf_ms + _gov_ms + _solve_ms + _telemetry_ms
        _misc_ms = max(0.0, _tick_dt_ms - _accounted_ms)
        self.pub_loop_timing.publish(Float64MultiArray(data=[
            _tick_dt_ms, _kin_ms, _cbf_ms, _gov_ms, _solve_ms, _telemetry_ms, _misc_ms]))

    def _publish_telemetry(self, q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total,
                           J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                           d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
                           qdot_err_14, xdot_err_6, top_active_pairs):
        # Publish the full dashboard telemetry set (downsampled, off the hot path).
        # Slacks + shadow prices
        self.pub_slacks.publish(Float64MultiArray(data=[float(abs(slack_r)), float(abs(slack_l))]))
        # Two INDEPENDENT per-arm CBF shadow prices (lambda_cbf_R, lambda_cbf_L),
        # replacing the single combined value. Published together on
        # /qp_debug/lambda_cbf so the plotter can show both on the same axes.
        self.pub_lambda_cbf.publish(Float64MultiArray(
            data=[self.qp.last_lambda_cbf_right, self.qp.last_lambda_cbf_left]))
        if self.kin.idx_right and self.kin.idx_left:
            max_lambda_r = float(np.max(lambda_joints_total[self.kin.idx_right]))
            max_lambda_l = float(np.max(lambda_joints_total[self.kin.idx_left]))
        else:
            max_lambda_r = max_lambda_l = 0.0
        self.pub_lambda_joints.publish(Float64MultiArray(data=[max_lambda_r, max_lambda_l]))

        # End-effector state (pos + lin vel + RPY, 18 floats)
        if self.kin.ee_id_right is not None and self.kin.ee_id_left is not None:
            p_real_r = self.kin.data.oMf[self.kin.ee_id_right].translation
            p_real_l = self.kin.data.oMf[self.kin.ee_id_left].translation
            v_real_r = pin.getFrameVelocity(self.kin.model, self.kin.data, self.kin.ee_id_right, pin.LOCAL_WORLD_ALIGNED).linear
            v_real_l = pin.getFrameVelocity(self.kin.model, self.kin.data, self.kin.ee_id_left, pin.LOCAL_WORLD_ALIGNED).linear
            rpy_real_r = pin.rpy.matrixToRpy(self.kin.data.oMf[self.kin.ee_id_right].rotation)
            rpy_real_l = pin.rpy.matrixToRpy(self.kin.data.oMf[self.kin.ee_id_left].rotation)
            ee_data = []
            ee_data.extend(p_real_r.tolist()); ee_data.extend(v_real_r.tolist())
            ee_data.extend(p_real_l.tolist()); ee_data.extend(v_real_l.tolist())
            ee_data.extend(rpy_real_r.tolist()); ee_data.extend(rpy_real_l.tolist())
            self.pub_ee_state.publish(Float64MultiArray(data=ee_data))

        # Truthful per-arm effective reference the CLF actually tracks (see the
        # publisher's docstring): [x_r(3), rpy_r(3), x_l(3), rpy_l(3)]. Published
        # only once every arm's reference exists (avoids a half-populated frame).
        if (self.x_ref_right is not None and self.rpy_ref_right is not None
                and self.x_ref_left is not None and self.rpy_ref_left is not None):
            ref_eff = (list(self.x_ref_right) + list(self.rpy_ref_right)
                       + list(self.x_ref_left) + list(self.rpy_ref_left))
            self.pub_reference_effective.publish(
                Float64MultiArray(data=[float(v) for v in ref_eff]))

        # Loop frequency (measured)
        current_time = time.perf_counter()
        elapsed = current_time - self.last_freq_pub_time
        if elapsed > 0:
            self.pub_loop_freq.publish(Float64(data=self.publish_every_n / elapsed))
        self.last_freq_pub_time = current_time

        # Min distance + dynamic weights
        self.pub_min_dist.publish(Float64(data=abs_min_distance))
        # 7 floats: [weight_slack_r, weight_slack_l, gamma_mean, gamma_r, gamma_l,
        # posture_weight_r, posture_weight_l]. gamma_mean at index 2 for plotter.py's
        # live dashboard; posture weights APPENDED at 5,6 (existing consumers unaffected).
        gamma_mean = 0.5 * (self.qp.gamma_clf_r + self.qp.gamma_clf_l)
        self.pub_dynamic_weights.publish(Float64MultiArray(data=[
            float(self.qp.weight_slack_r), float(self.qp.weight_slack_l),
            float(gamma_mean), float(self.qp.gamma_clf_r), float(self.qp.gamma_clf_l),
            float(self.qp.posture_weight_r), float(self.qp.posture_weight_l)]))
        # Per-arm frozen/active ground truth for the RViz visualizer.
        self.pub_arm_frozen.publish(Float64MultiArray(
            data=[1.0 if self.right_frozen else 0.0, 1.0 if self.left_frozen else 0.0]))
        # [d_safe_R, d_safe_L]; each thickens only with its own arm's speed.
        self.pub_d_safe_dynamic.publish(Float64MultiArray(
            data=[float(d_safe_dynamic_r), float(d_safe_dynamic_l)]))
        # Soft-task cost decomposition [E_damp, E_posture, E_slack] for the
        # task-authority panel in the plotter (hard-constraint authority is the
        # shadow prices published above).
        self.pub_task_authority.publish(
            Float64MultiArray(data=[float(e) for e in self.qp.task_energies]))

        # Top-3 actually-enabled collision pairs (for the debug plot). Passed in
        # from the CbfResult (captured in the same CBF pass), not read off
        # self.col, so the real-hardware worker thread never shares it live.
        top = top_active_pairs
        pairs_str = ";".join(f"{n1}|{n2}|{d:.4f}" for (n1, n2, d) in top)
        self.pub_top_pairs.publish(String(data=pairs_str))

        # Tracking errors
        if qdot_err_14 is not None:
            self.pub_qdot_err.publish(Float64MultiArray(data=qdot_err_14.tolist()))
        if xdot_err_6 is not None:
            self.pub_xdot_err.publish(Float64MultiArray(data=xdot_err_6.tolist()))

        # Virtual wall marker
        self.viz.publish_wall_marker()

    def _publish_shared_collision(self, b_col_pair, J_soft_r, J_soft_l):
        # Cartesian collision gradient for shared autonomy, per arm: 14 floats
        # [b_col_r, b_col_l, J_c_cart_R(6), J_c_cart_L(6)]. Called every tick
        # (not downsampled) -- see call site in solve_and_publish for why.
        b_col_r, b_col_l = b_col_pair
        if self.kin.ee_id_right is not None and self.kin.ee_id_left is not None:
            J_EE_R_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_right, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_EE_L_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_left, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_c_cart_R = np.dot(J_soft_r, np.linalg.pinv(J_EE_R_6D))
            J_c_cart_L = np.dot(J_soft_l, np.linalg.pinv(J_EE_L_6D))
            self.pub_shared_col.publish(Float64MultiArray(
                data=[float(b_col_r), float(b_col_l)] + J_c_cart_R.tolist() + J_c_cart_L.tolist()))


def main():
    # NO signal handlers: rclpy's default SIGINT handler shuts down the whole
    # context BEFORE our except/finally below runs, so restore_controllers()
    # would find an already-dead context on Ctrl-C. Disabling it means Ctrl-C
    # just raises a plain KeyboardInterrupt and WE control shutdown ordering.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SafetyQPController()

    # Do NOT set use_sim_time here -- a sim-time timer can catch up in bursts
    # under executor load, firing several ticks back-to-back and lurching the arm.

    # --- PHASE 1: wait for TF, then verify controller state ---
    node.get_logger().info("[Main] Waiting for TF...")
    node.wait_for_tf()

    node.get_logger().info("[Main] Verifying Controller State...")
    if node.check_and_switch_controllers():
        print("------------------------------------------------")
        print("SAFETY CONTROLLER RUNNING (Velocity Mode)")
        print("------------------------------------------------")
    else:
        print("[Error] Could not switch controllers. Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- PHASE 2: visualization + diagnostics ---
    # Meshcat is suspended entirely on real hardware: its dedicated _run_viz
    # thread renders every 0.2s regardless of anything else, permanent CPU cost
    # for a web visualizer nobody watches on the robot itself (RViz + the
    # plotters remain fully available -- this only skips the Meshcat 3D view).
    if node.REAL_HARDWARE:
        node.get_logger().info(
            "\033[93m[Viz] REAL HARDWARE: Meshcat suspended (RViz/plotters unaffected).\033[0m")
    else:
        node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    # spin_once(timeout_sec=0.1) in a loop, NOT rclpy.spin(node): with the
    # signal handler disabled above, an indefinitely-blocking spin() has
    # nothing to wake it on Ctrl-C -- this returns to Python every 0.1s so
    # KeyboardInterrupt actually gets raised promptly (same pattern already
    # used by trajectory_generator.py's main()).
    node.start_control_loop()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_controllers()
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
