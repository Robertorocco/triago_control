#!/usr/bin/env python3
# main_qp_controller.py
"""
The Orchestrator (ROS 2 Node).

Wires together the five specialized modules and drives the safety-critical
control loop:

    RobotKinematics  -> FK / Jacobians / filtered velocity / digital twin
    CollisionManager -> SoftMin CBF gradient + dynamic safety margin
    SharedAutonomyHandler -> grasp / CBF-bypass / attachment commands
    QPFormulator     -> the CLF-CBF-QP that produces safe joint velocities
    VisualizationEngine -> Meshcat + RViz telemetry (off the critical path)

Responsibilities:
    * fetch the URDF and initialize every sub-module,
    * own the cartesian-reference and joint-state callbacks + motion watchdogs,
    * run `solve_and_publish` on a CONFIGURABLE-frequency timer (see
      `set_control_frequency`, which replaces the original hard-coded 1/300 dt),
    * keep the TSID JS velocity controllers active,
    * publish joint velocity commands and all dashboard telemetry.
"""

import rclpy
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

import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.qp_controller.collision_manager import CollisionManager
from triago_control.qp_controller.shared_autonomy_handler import SharedAutonomyHandler
from triago_control.qp_controller.qp_formulator import QPFormulator
from triago_control.qp_controller.visualization_engine import VisualizationEngine


class SafetyQPController(Node):
    """ROS 2 node orchestrating the bimanual QP-CLF-CBF safety controller."""

    def __init__(self):
        super().__init__('safety_qp_controller')

        # Configurable control frequency (replaces the hard-coded 1/300 target dt)
        self._control_freq = cfg.CONTROL_FREQ_DEFAULT
        self.loop_timer = None

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
        self.col.build_collision_model(right_offsets, left_offsets)
        self.col.define_collision_pairs()

        self.viz = VisualizationEngine(self, self.kin.model, self.col.cmodel, self.urdf_path)
        self.viz.add_gripper_visual_boxes(self.col)
        self.hri = SharedAutonomyHandler(self, self.col, self.kin, self.viz)
        self.qp = QPFormulator(self.kin.model)

        # --- CONTROL MODE / REFERENCES ---
        self.orientation_ctrl = cfg.ORIENTATION_CTRL
        self.x_ref_right = None; self.rpy_ref_right = None; self.xdot_ref_right = None; self.w_ref_right = None
        self.x_ref_left = None;  self.rpy_ref_left = None;  self.xdot_ref_left = None;  self.w_ref_left = None
        self.task_dim_right = 6.0
        self.task_dim_left = 6.0

        # --- WATCHDOGS ---
        self.right_imposed_motion = False
        self.left_imposed_motion = False
        self.last_right_msg_time = time.time()
        self.last_left_msg_time = time.time()

        # --- LOOP / SIM STATE ---
        self.active_controller_mode = False
        self.publish_counter = 0
        self.publish_every_n = cfg.PUBLISH_EVERY_N
        self.last_freq_pub_time = time.perf_counter()
        self.last_sim_time = None
        self.last_qdot_cmd_14 = np.zeros(14)

        # --- COMMAND PUBLISHERS ---
        self.pub_right = self.create_publisher(Float64MultiArray, f'/{cfg.RIGHT_CONTROLLER}/joint_velocity_cmd', 1)
        self.pub_left = self.create_publisher(Float64MultiArray, f'/{cfg.LEFT_CONTROLLER}/joint_velocity_cmd', 1)

        # --- TELEMETRY PUBLISHERS ---
        self.pub_qdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_err', 10)
        self.pub_xdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/xdot_err', 10)
        self.pub_slacks = self.create_publisher(Float64MultiArray, '/qp_debug/slacks', 10)
        self.pub_ee_state = self.create_publisher(Float64MultiArray, '/qp_debug/ee_real', 10)
        self.pub_debug_h = self.create_publisher(Float64, '/qp_debug/safety_margin', 10)
        self.pub_loop_freq = self.create_publisher(Float64, '/qp_debug/loop_freq', 10)
        self.pub_min_dist = self.create_publisher(Float64, '/qp_debug/min_distance', 10)
        self.pub_top_pairs = self.create_publisher(String, '/qp_debug/top_pairs', 10)
        self.pub_lambda_cbf = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_cbf', 10)
        self.pub_lambda_joints = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_joints', 10)
        self.pub_dynamic_weights = self.create_publisher(Float64MultiArray, '/qp_debug/dynamic_weights', 10)
        self.pub_d_safe_dynamic = self.create_publisher(Float64MultiArray, '/qp_debug/d_safe_dynamic', 10)
        self.pub_qdot_cmd = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_cmd', 10)
        self.pub_task_authority = self.create_publisher(Float64MultiArray, '/qp_debug/task_authority', 10)
        self.pub_shared_col = self.create_publisher(Float64MultiArray, '/collision_constraints', 10)

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
        # its current EE pose (held by a zero-velocity CLF) with double slack
        # weight, but is NOT zeroed — its QP-computed motion is ALWAYS sent to
        # TSID so it can bend to help the active arm avoid collisions.
        self.active_arm = 'right'
        self.right_frozen = False
        self.left_frozen = False
        self._refs_initialized = False
        self.create_subscription(String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        # Services for controller switching
        self.switch_srv = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_srv = self.create_client(ListControllers, '/controller_manager/list_controllers')

        # Low-rate RViz obstacle marker timer (matches original 0.5s cadence)
        self.timer_obs = self.create_timer(0.5, lambda: self.viz.publish_obstacle_marker(self.hri))

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
        self.loop_timer = self.create_timer(1.0 / self._control_freq, self.solve_and_publish)
        self.get_logger().info(f"[FREQ] Control loop set to {self._control_freq:.1f} Hz.")

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
            return True
        self.get_logger().error("Switch Service Failed.")
        return False

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def joint_callback(self, msg):
        # Parse physical joint positions and hand them to the kinematics filter.
        if self.kin.model is None:
            return
        q_physical = pin.neutral(self.kin.model)
        v_measured = np.zeros(self.kin.model.nv)  # direct velocity (real hardware only)
        for i, name in enumerate(msg.name):
            if self.kin.model.existJointName(name):
                jid = self.kin.model.getJointId(name)
                idx_q = self.kin.model.joints[jid].idx_q
                idx_v = self.kin.model.joints[jid].idx_v
                if idx_q >= 0:
                    q_physical[idx_q] = msg.position[i]
                if self.REAL_HARDWARE and idx_v >= 0 and i < len(msg.velocity):
                    v_measured[idx_v] = msg.velocity[i]
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.REAL_HARDWARE:
            self.kin.update_from_joint_state(q_physical, time_stamp, v_direct=v_measured)
        else:
            self.kin.update_from_joint_state(q_physical, time_stamp)

    def grasp_active_cb(self, msg):
        """Tracks whether shared autonomy is autonomously driving a grasp/lift."""
        self.grasp_active = bool(msg.data)

    def _freeze_arm(self, side):
        """Snapshot one arm's CURRENT EE pose as its held reference (zero velocity).

        Used when an arm becomes inactive (arm switch / stale teleop). The arm
        keeps imposed_motion=True so its CLF holds it at this pose; with the
        doubled inactive slack weight it stays put unless yielding helps the
        active arm. Requires up-to-date FK (call after kinematics update).
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
            self.last_right_msg_time = time.time()
            self.right_frozen = True
        else:
            self.x_ref_left = pos; self.rpy_ref_left = rpy
            self.xdot_ref_left = np.zeros(3); self.w_ref_left = np.zeros(3)
            self.left_imposed_motion = True
            self.last_left_msg_time = time.time()
            self.left_frozen = True

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
            self.last_right_msg_time = time.time()
        elif len(msg.data) >= 6:
            self.x_ref_right = np.array(msg.data[0:3])
            self.xdot_ref_right = np.array(msg.data[3:6])
            self.right_imposed_motion = True
            self.right_frozen = False
            self.last_right_msg_time = time.time()

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
            self.last_left_msg_time = time.time()
        elif len(msg.data) >= 6:
            self.x_ref_left = np.array(msg.data[0:3])
            self.xdot_ref_left = np.array(msg.data[3:6])
            self.left_imposed_motion = True
            self.left_frozen = False
            self.last_left_msg_time = time.time()

    # =====================================================================
    # TASK ERROR EXTRACTION (5D / 6D / 3D)
    # =====================================================================
    def _arm_task_error(self, ee_id, x_ref, rpy_ref, xdot_ref, w_ref, task_dim):
        # Compute the (weighted-ready) task error + feedforward velocity for one arm.
        if x_ref is None or ee_id is None:
            return np.zeros(3), np.zeros(3)
        x_real = self.kin.data.oMf[ee_id].translation
        e_pos = x_ref - x_real
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

    # =====================================================================
    # MAIN CONTROL LOOP
    # =====================================================================
    def solve_and_publish(self):
        # One control tick: kinematics -> SoftMin -> QP -> publish -> evolve twin.
        if self.kin.current_q is None:
            return

        # --- Watchdog: a stale-reference arm is FROZEN at its current pose (held
        # by a zero-velocity CLF) rather than going limp. Option B keeps it under
        # QP control so it can still bend to help the active arm avoid collisions.
        if self.right_imposed_motion and not self.right_frozen \
                and (time.time() - self.last_right_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self._freeze_arm('right')
            print("[Safety] Watchdog: Right reference stale -> frozen at current pose.")
        if self.left_imposed_motion and not self.left_frozen \
                and (time.time() - self.last_left_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self._freeze_arm('left')
            print("[Safety] Watchdog: Left reference stale -> frozen at current pose.")

        # --- 0. Kinematics + geometry refresh ---
        self.kin.update_kinematics()
        self.kin.debug_interrogate()
        self.col.update_geometry(self.kin.current_q)

        # One-time: freeze BOTH arms at their startup pose so every arm always has
        # a holding CLF (no limp/uncontrolled arm). Teleop overrides the active one.
        if not self._refs_initialized:
            self._freeze_arm('right')
            self._freeze_arm('left')
            self._refs_initialized = True

        # --- Deferred attachment (needs fresh oMi / oMg) ---
        if self.hri.pending_attach is not None:
            arm_side, color = self.hri.pending_attach
            self.hri.pending_attach = None
            try:
                self.hri.attach_object_visually(arm_side, color)
            except Exception as e:
                self.get_logger().warn(f"[TOPOLOGY] Attach failed: {e}")

        # --- Deferred detachment (needs fresh oMi / oMg to freeze release pose) ---
        if self.hri.pending_detach is not None:
            arm_side, color, world_pos = self.hri.pending_detach
            self.hri.pending_detach = None
            try:
                self.hri.detach_object_visually(arm_side, color, world_pos)
            except Exception as e:
                self.get_logger().warn(f"[TOPOLOGY] Detach failed: {e}")

        # --- Grasp contact distance telemetry ---
        self.hri.publish_contact_distances()

        # --- Low-level tracking error (commanded vs measured) ---
        qdot_err_14, xdot_err_6 = self.kin.compute_tracking_errors(self.last_qdot_cmd_14)

        # --- 1. SoftMin CBF aggregation (TWO independent per-arm barriers) ---
        J_soft_r, h_soft_r, J_soft_l, h_soft_l, d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance = \
            self.col.compute_softmin_jacobian(
                self.kin.current_v, self.kin.idx_right, self.kin.idx_left,
                self.hri.grasp_margin_targets, self.hri.attached_objects,
                self.hri.attached_adjacency, self.hri.ignored_targets, self.publish_counter,
                attach_ramp_shifts=self.hri.get_attach_ramp_shifts(),
                attached_object_arm=self.hri.attached_object_arm)

        # --- 2. Task errors ---
        e_r, v_r, e_l, v_l = self.extract_task_errors()

        # --- 3. Build + solve the CLF-CBF-QP ---
        dt = 1.0 / self._control_freq
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
        q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total = self.qp.build_and_solve(
            self.kin, J_soft_r, h_soft_r, J_soft_l, h_soft_l,
            d_safe_dynamic_r, d_safe_dynamic_l,
            self.right_imposed_motion, self.left_imposed_motion,
            self.xdot_ref_right, self.xdot_ref_left, e_r, v_r, e_l, v_l, dt,
            right_frozen=self.right_frozen, left_frozen=self.left_frozen,
            tracking_boost_arm=boost_arm)

        self.publish_counter += 1

        # --- 4. Downsampled telemetry publishing ---
        if self.publish_counter % self.publish_every_n == 0:
            self._publish_telemetry(q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total,
                                    J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                                    d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
                                    qdot_err_14, xdot_err_6)

        # --- 5. Command publishing ---
        # Option B: ALWAYS send the QP-computed velocity to TSID for BOTH arms.
        # The old per-arm zero-overwrite (when an arm had no fresh reference) is
        # removed: it discarded the QP's collision-avoidance motion for the
        # inactive arm, which let the two arms silently inter-penetrate. The
        # inactive arm is instead held by its frozen-pose CLF (+ doubled slack),
        # so its commanded motion is meaningful and safe.
        cmd_data_r = [0.0] * 7
        cmd_data_l = [0.0] * 7
        if self.active_controller_mode:
            if self.kin.idx_right:
                cmd_data_r = q_dot_safe[self.kin.idx_right].tolist()
                self.pub_right.publish(Float64MultiArray(data=cmd_data_r))
            if self.kin.idx_left:
                cmd_data_l = q_dot_safe[self.kin.idx_left].tolist()
                self.pub_left.publish(Float64MultiArray(data=cmd_data_l))

        # Save the exact command sent to hardware for next tick's tracking-error math
        self.last_qdot_cmd_14 = np.concatenate((cmd_data_r, cmd_data_l))
        if self.publish_counter % self.publish_every_n == 0:
            self.pub_qdot_cmd.publish(Float64MultiArray(data=self.last_qdot_cmd_14.tolist()))

        # --- 6. Evolve the digital twin (ideal kinematics) ---
        if cfg.SIMULATE_IDEAL_KINEMATICS:
            current_time = time.perf_counter()
            if self.last_sim_time is None:
                dt_sim = 0.001
            else:
                dt_sim = current_time - self.last_sim_time
            if dt_sim > 0.1:
                dt_sim = 0.001
            self.kin.integrate_simulated_state(q_dot_safe, dt_sim)
            self.last_sim_time = current_time

        # --- 7. External debug visualizer (optional tethers / overlays) ---
        if self.publish_counter % self.publish_every_n == 0:
            if not cfg.DISABLE_CBF:
                # Legacy single scalar: the WORSE (smaller) margin of the two arms.
                # Each arm now uses its OWN dynamic margin (see the coupling audit
                # in collision_manager.compute_softmin_jacobian).
                margin_r = h_soft_r - d_safe_dynamic_r
                margin_l = h_soft_l - d_safe_dynamic_l
                self.pub_debug_h.publish(Float64(data=float(min(margin_r, margin_l))))
            self.viz.publish_debug(
                self.kin.model, self.kin.data, self.col.cdata, self.kin.current_q,
                q_dot_safe, None, None, self.kin.ee_id_right, self.kin.ee_id_left,
                cfg.JOINT_LIMIT_BUFFER_BASE)
            self.viz.publish_teleop_tether()

        # --- Diagnostic brake tracker ---
        # --- Diagnostic brake tracker (disabled: console spam) ---
        # if self.publish_counter % 200 == 0:
        #     print("\n=== DECOUPLED QP BRAKES ===")
        #     print(f"Collision Brakes:  {self.qp.last_lambda_col:.4f}")
        #     print(f"Joint Brakes (R):  {self.qp.last_lambda_joints_right:.4f}")
        #     print(f"Joint Brakes (L):  {self.qp.last_lambda_joints_left:.4f}")
        #     print("===========================\n")

    def _publish_telemetry(self, q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total,
                           J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                           d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
                           qdot_err_14, xdot_err_6):
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

        # Loop frequency (measured)
        current_time = time.perf_counter()
        elapsed = current_time - self.last_freq_pub_time
        if elapsed > 0:
            self.pub_loop_freq.publish(Float64(data=self.publish_every_n / elapsed))
        self.last_freq_pub_time = current_time

        # Min distance + dynamic weights
        self.pub_min_dist.publish(Float64(data=abs_min_distance))
        self.pub_dynamic_weights.publish(Float64MultiArray(data=[float(self.qp.weight_slack), float(self.qp.gamma_clf)]))
        # Per-arm dynamic safety margins (2026-07-01 coupling fix): each arm's
        # margin now thickens only with ITS OWN speed. Published as a 2-element
        # array [d_safe_R, d_safe_L] (was a single shared Float64).
        self.pub_d_safe_dynamic.publish(Float64MultiArray(
            data=[float(d_safe_dynamic_r), float(d_safe_dynamic_l)]))
        # Soft-task cost decomposition [E_damp, E_posture, E_slack] for the
        # task-authority panel in the plotter (hard-constraint authority is the
        # shadow prices published above).
        self.pub_task_authority.publish(
            Float64MultiArray(data=[float(e) for e in self.qp.task_energies]))

        # Top-3 actually-enabled collision pairs (for the debug plot)
        top = getattr(self.col, 'top_active_pairs', [])
        pairs_str = ";".join(f"{n1}|{n2}|{d:.4f}" for (n1, n2, d) in top)
        self.pub_top_pairs.publish(String(data=pairs_str))

        # Tracking errors
        if qdot_err_14 is not None:
            self.pub_qdot_err.publish(Float64MultiArray(data=qdot_err_14.tolist()))
        if xdot_err_6 is not None:
            self.pub_xdot_err.publish(Float64MultiArray(data=xdot_err_6.tolist()))

        # Virtual wall marker
        self.viz.publish_wall_marker()

        # Cartesian projection of the collision gradient for shared autonomy.
        # Each arm's OWN cartesian gradient now comes from ITS OWN independent
        # SoftMin (J_soft_r for the right projection, J_soft_l for the left),
        # matching the per-arm CBF split -- previously both used the single
        # combined J_soft/b_col, which leaked the other arm's barrier into
        # whichever arm main_shared_autonomy currently treats as active.
        # NEW layout (14 floats): [b_col_r, b_col_l, J_c_cart_R(6), J_c_cart_L(6)]
        # (old layout was 13 floats: [b_col, J_c_cart_R(6), J_c_cart_L(6)] --
        # main_shared_autonomy.collision_data_callback is updated to match).
        b_col_r, b_col_l = b_col_pair
        if self.kin.ee_id_right is not None and self.kin.ee_id_left is not None:
            J_EE_R_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_right, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_EE_L_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_left, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_c_cart_R = np.dot(J_soft_r, np.linalg.pinv(J_EE_R_6D))
            J_c_cart_L = np.dot(J_soft_l, np.linalg.pinv(J_EE_L_6D))
            self.pub_shared_col.publish(Float64MultiArray(
                data=[float(b_col_r), float(b_col_l)] + J_c_cart_R.tolist() + J_c_cart_L.tolist()))


def main():
    rclpy.init()
    node = SafetyQPController()

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
    node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    node.start_control_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
