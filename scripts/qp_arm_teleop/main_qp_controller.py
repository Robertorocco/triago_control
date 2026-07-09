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
from triago_control.qp_controller.reference_governor import ReferenceGovernor
from triago_control.qp_controller.world_loader import load_world


class SafetyQPController(Node):
    """ROS 2 node orchestrating the bimanual QP-CLF-CBF safety controller."""

    def __init__(self):
        super().__init__('safety_qp_controller')

        # Configurable control frequency (replaces the hard-coded 1/300 target dt)
        self._control_freq = cfg.CONTROL_FREQ_DEFAULT
        self.loop_timer = None

        # --- WORLD SCENE (2026-07-04) -----------------------------------
        # Which obstacle layout (table + red/blue cylinders + optional extra
        # obstacles) to build into the CBF's collision model. Independent of
        # which Gazebo world was actually launched -- see world_loader.py's
        # module docstring: the Gazebo launch command is UNCHANGED, this
        # parameter only tells the QP/RViz/Meshcat side which YAML to mirror
        # it with. Override at runtime, e.g.:
        #   ros2 run triago_control main_qp_controller.py --ros-args \
        #        -p world_name:=no_obstacle
        self.declare_parameter('world_name', 'no_obstacle')
        world_name = self.get_parameter('world_name').get_parameter_value().string_value
        self.world_scene = load_world(world_name)
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
        # Head chain (2026-07-01): SAME hardware/geometry recipe as the arms
        # (calculate_offsets is reused verbatim), added as a quasi-static CBF
        # obstacle for the arms -- see cfg.HEAD_CHAIN's docstring. The head is
        # NEVER added to idx_right/idx_left (RobotKinematics never maps
        # HEAD_JOINTS into the QP's actuated velocity indices), so this adds
        # zero rows/columns to the QP decision vector.
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

        # --- REFERENCE GOVERNOR (2026-07-01, per-arm CLF-safety layer) ---
        # Bounds position/orientation error, reference velocity/acceleration so
        # the CLF row's demand is always bounded → QP feasibility preserved.
        # One instance per arm (each has its own velocity memory for accel limiting).
        self.gov_right = ReferenceGovernor('right', model=self.kin.model,
                                           ee_frame_name=cfg.RIGHT_TCP_FRAME)
        self.gov_left = ReferenceGovernor('left', model=self.kin.model,
                                          ee_frame_name=cfg.LEFT_TCP_FRAME)

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
        # The ACTUAL per-arm reference the CLF tracks (truthful for BOTH arms):
        # for the active arm it is the commanded reference, for a frozen/inactive
        # arm it is the held pose (main_qp_controller._freeze_arm) -- unlike
        # /arm_*/cartesian_reference, which goes stale for the inactive arm since
        # teleop/shared_autonomy only publish the active one. Layout (12 floats):
        # [x_r(3), rpy_r(3), x_l(3), rpy_l(3)] (raw reference, pre-governor).
        self.pub_reference_effective = self.create_publisher(
            Float64MultiArray, '/qp_debug/reference_effective', 10)
        self.pub_debug_h = self.create_publisher(Float64, '/qp_debug/safety_margin', 10)
        self.pub_loop_freq = self.create_publisher(Float64, '/qp_debug/loop_freq', 10)
        self.pub_min_dist = self.create_publisher(Float64, '/qp_debug/min_distance', 10)
        self.pub_top_pairs = self.create_publisher(String, '/qp_debug/top_pairs', 10)
        self.pub_lambda_cbf = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_cbf', 10)
        self.pub_lambda_joints = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_joints', 10)
        self.pub_dynamic_weights = self.create_publisher(Float64MultiArray, '/qp_debug/dynamic_weights', 10)
        self.pub_d_safe_dynamic = self.create_publisher(Float64MultiArray, '/qp_debug/d_safe_dynamic', 10)
        self.pub_qdot_cmd = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_cmd', 10)
        # Generic measured joint velocity (2026-07-04, for offline_plotter.py):
        # kin.current_v is ALREADY the environment-correct signal -- EMA-filtered
        # differentiated velocity in Gazebo (self.REAL_HARDWARE=False) or the
        # direct sensor reading on real hardware (self.REAL_HARDWARE=True); see
        # robot_kinematics.update_from_joint_state. No new logic is needed here,
        # only a publisher: this was previously computed every tick but never
        # put on a topic. Named generically ("measured", not "filtered") since
        # the SAME topic means different things in sim vs. real hardware.
        self.pub_qdot_measured = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_measured', 10)
        self.pub_task_authority = self.create_publisher(Float64MultiArray, '/qp_debug/task_authority', 10)
        self.pub_shared_col = self.create_publisher(Float64MultiArray, '/collision_constraints', 10)
        # Per-arm frozen/active ground truth for the RViz visualizer (2026-07-01):
        # [right_frozen, left_frozen] as 0.0/1.0. Lets qp_visualizer_tutorial draw
        # BOTH grippers blue when both arms are actively driven (e.g. by
        # trajectory_generator.py), not just whichever arm a stale single
        # "active_arm" notion pointed at.
        self.pub_arm_frozen = self.create_publisher(Float64MultiArray, '/qp_debug/arm_frozen', 10)
        # Live joint limits for the plotter's slider GUI (latched, published
        # once + on every late-subscriber via a slow timer): the REAL limits
        # from the Pinocchio model built from the live URDF -- the SAME
        # numbers the joint-limit CBF rows in qp_formulator enforce.
        self.pub_joint_limits = self.create_publisher(String, '/qp_debug/joint_limits', 10)
        self.timer_joint_limits = self.create_timer(2.0, self._publish_joint_limits)

        # Reference governor telemetry (2026-07-01): publishes the DIFFERENCE
        # between raw and governed references (6D each arm: [dx,dy,dz,droll,dpitch,dyaw])
        # so the dedicated plotter window can show when/where the governor clamps.
        # Layout: [pos_diff_R(3), ori_diff_R(3), vel_diff_R(3), wvel_diff_R(3),
        #          pos_diff_L(3), ori_diff_L(3), vel_diff_L(3), wvel_diff_L(3)] = 24 floats
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

    def _publish_joint_limits(self):
        """Publish [name:lower:upper;...] for every joint the slider GUI needs
        (arms + head + gripper fingers), read from the live Pinocchio model
        via RobotKinematics.get_joint_limits -- the SAME limits enforced by
        the joint-limit CBF rows in qp_formulator.build_and_solve. A plain
        String (semicolon/colon encoded) is used to avoid introducing a new
        custom message type for a low-rate (2s), non-critical debug topic.
        Runs on a slow timer (not the hot loop) and self-cancels after the
        first successful publish -- the URDF/model never changes at runtime.
        """
        if self.kin.model is None:
            return
        names = cfg.RIGHT_JOINTS + cfg.LEFT_JOINTS + cfg.HEAD_JOINTS + cfg.GRIPPER_FINGER_JOINTS
        lower, upper = self.kin.get_joint_limits(names)
        payload = ";".join(f"{n}:{lo:.4f}:{hi:.4f}" for n, lo, hi in zip(names, lower, upper))
        self.pub_joint_limits.publish(String(data=payload))
        self.timer_joint_limits.cancel()

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
            self.gov_right.reset()  # Clear velocity memory (arm is now stationary)
        else:
            self.x_ref_left = pos; self.rpy_ref_left = rpy
            self.xdot_ref_left = np.zeros(3); self.w_ref_left = np.zeros(3)
            self.left_imposed_motion = True
            self.last_left_msg_time = time.time()
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
        # task_dim == 3.0: POSITION-ONLY CLF, orientation fully relaxed. Used by
        # the local-minima escape (2026-07-01) to give up the orientation task
        # entirely during an obstacle-induced escape (see solve_and_publish's
        # task_dim_eff_right/left override). Checked BEFORE the normal
        # orientation_ctrl branch so it applies even when orientation_ctrl=True.
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
        # REFERENCE GOVERNOR (2026-07-01): apply velocity/error/acceleration
        # bounds BEFORE the CLF sees the reference. The RAW references
        # (self.x_ref_right, etc.) are PRESERVED for future plotting / consumers;
        # the governed versions are used ONLY for the CLF task-error computation
        # below (and passed to build_and_solve as the feedforward velocities).
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

        # --- LOCAL MINIMA ESCAPE (2026-07-01) ---
        # Detect a possible QP-CLF-CBF local minimum from the 3D position
        # error (per instruction, position-only for now) and the shadow
        # prices from the QP's PREVIOUS solve (self.qp.last_lambda_cbf_*
        # and last_lambda_joints_* are exactly last tick's values -- the
        # current tick hasn't solved yet). If detected, apply a smooth,
        # PER-ARM posture-weight correction (and, for an obstacle-induced
        # minimum, force a position-only task_dim) until the error recovers
        # or the max escape duration elapses.
        task_dim_eff_right = self.task_dim_right
        task_dim_eff_left = self.task_dim_left
        if cfg.ENABLE_LOCAL_MINIMA_ESCAPE and not cfg.BLENDING:
            # NOTE (2026-07-03): the local-minima escape is DISABLED in BLENDING
            # mode. The escape mechanism was designed for the policy-only case
            # (no human in the loop to redirect) — in BLENDING mode the HUMAN IS
            # the escape mechanism (reference catch-up + divergence override). The
            # persistent position gap created by the catch-up term can easily
            # trigger the stuck-detector false-positive (error > 0.15m for > 2s),
            # which then forces task_dim=3.0 (drops orientation tracking entirely),
            # causing the "orientation goes wild" report.
            if x_gov_r is not None and self.kin.ee_id_right is not None:
                err_norm_r = float(np.linalg.norm(
                    x_gov_r - np.array(self.kin.data.oMf[self.kin.ee_id_right].translation)))
                pscale_r, tdim_override_r = self.gov_right.update_local_minima_escape(
                    err_norm_r, self.qp.last_lambda_cbf_right, self.qp.last_lambda_joints_right,
                    dt, logger=print)
                self.qp.posture_scale_right = pscale_r
                if tdim_override_r is not None:
                    task_dim_eff_right = tdim_override_r
            if x_gov_l is not None and self.kin.ee_id_left is not None:
                err_norm_l = float(np.linalg.norm(
                    x_gov_l - np.array(self.kin.data.oMf[self.kin.ee_id_left].translation)))
                pscale_l, tdim_override_l = self.gov_left.update_local_minima_escape(
                    err_norm_l, self.qp.last_lambda_cbf_left, self.qp.last_lambda_joints_left,
                    dt, logger=print)
                self.qp.posture_scale_left = pscale_l
                if tdim_override_l is not None:
                    task_dim_eff_left = tdim_override_l
        else:
            self.qp.posture_scale_right = 1.0
            self.qp.posture_scale_left = 1.0

        # Compute CLF task errors from the GOVERNED references (not the raw ones).
        # task_dim_eff_{right,left} may be overridden to 3.0 (position-only) by
        # the local-minima escape above when an obstacle-induced minimum is
        # detected -- this is what "gives up orientation" during the escape.
        e_r, v_r = self._arm_task_error(self.kin.ee_id_right, x_gov_r, rpy_gov_r,
                                        v_gov_r, w_gov_r, task_dim_eff_right)
        e_l, v_l = self._arm_task_error(self.kin.ee_id_left, x_gov_l, rpy_gov_l,
                                        v_gov_l, w_gov_l, task_dim_eff_left)

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
        q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total = self.qp.build_and_solve(
            self.kin, J_soft_r, h_soft_r, J_soft_l, h_soft_l,
            d_safe_dynamic_r, d_safe_dynamic_l,
            self.right_imposed_motion, self.left_imposed_motion,
            self.xdot_ref_right, self.xdot_ref_left, e_r, v_r, e_l, v_l, dt,
            right_frozen=self.right_frozen, left_frozen=self.left_frozen,
            tracking_boost_arm=boost_arm, orient_boost_arms=orient_boost_arms)

        self.publish_counter += 1

        # --- 4. Downsampled telemetry publishing ---
        if self.publish_counter % self.publish_every_n == 0:
            self._publish_telemetry(q_dot_safe, slack_r, slack_l, b_col_pair, lambda_joints_total,
                                    J_soft_r, h_soft_r, J_soft_l, h_soft_l,
                                    d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance,
                                    qdot_err_14, xdot_err_6)
            # Generic measured joint velocity (14 floats: R7+L7), same
            # environment-dependent signal the QP itself consumes -- see the
            # publisher's own comment above.
            if self.kin.idx_right and self.kin.idx_left and self.kin.current_v is not None:
                meas_v_14 = np.concatenate(
                    (self.kin.current_v[self.kin.idx_right], self.kin.current_v[self.kin.idx_left]))
                self.pub_qdot_measured.publish(Float64MultiArray(data=meas_v_14.tolist()))

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

            # --- Reference governor telemetry (2026-07-01): publishes the RAW-minus-
            # GOVERNED difference so the plotter can show when/where the governor
            # clamps. Layout: [pos_diff_R(3), ori_diff_R(3), vel_diff_R(3),
            # wvel_diff_R(3), pos_diff_L(3), ori_diff_L(3), vel_diff_L(3),
            # wvel_diff_L(3)] = 24 floats. All zeros when the governor is off or
            # the raw reference is already within bounds (passthrough).
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
        # Per-arm slack weights (2026-07-01): weight_slack_r weights ONLY delta_r
        # and weight_slack_l weights ONLY delta_l in the QP Hessian (confirmed --
        # see qp_formulator.build_and_solve's slack block assembly). Previously
        # only their AVERAGE was published; now both are sent so the plotter can
        # show them independently, matching how the QP actually uses them.
        self.pub_dynamic_weights.publish(Float64MultiArray(
            data=[float(self.qp.weight_slack_r), float(self.qp.weight_slack_l), float(self.qp.gamma_clf)]))
        # Per-arm frozen/active ground truth for the RViz visualizer.
        self.pub_arm_frozen.publish(Float64MultiArray(
            data=[1.0 if self.right_frozen else 0.0, 1.0 if self.left_frozen else 0.0]))
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
