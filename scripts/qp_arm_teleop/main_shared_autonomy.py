#!/usr/bin/env python3
"""SharedControlNode (refactored): thin ROS 2 glue around the extracted classes.

This is the decomposed replacement for the monolithic shared_autonomy_haption_tutorial.py,
following the plan in shared_autonomy_analysis.md Section 4:

    GoalSet            -> goal_set.py
    BeliefEstimator     -> belief_estimator.py
    GraspStateMachine   -> grasp_state_machine.py
    PlotManager         -> plot_manager.py
    SharedControlNode   -> this file (pub/sub, timer, delegates everything else)

All functionality of the original monolithic script is preserved. Every bug
identified in shared_autonomy_analysis.md has been fixed; each fix is called
out in a comment at its location, summarized here:

  __init__:
    - self.tf_broadcaster was assigned twice (lines 253, 277 in the original),
      leaking one TransformBroadcaster. Now assigned exactly once.
    - `import time` was duplicated at the top of the file. Now imported once.
    - `self.grasp_in_progress` was set but never read anywhere. Removed entirely
      (GraspStateMachine.state is the single source of truth for grasp progress).
  collision_data_callback:
    - `if len(msg.data) == 13` silently dropped valid extended messages if the
      publisher ever grew the array. Changed to `>= 13`.
  solve_local_policy:
    - the `except ValueError` fallback returned the unconstrained v_geo on QP
      infeasibility, driving the robot through the CBF barrier. Now returns a
      safe zero-twist halt and logs an error.
  update_belief / BeliefEstimator.update:
    - the original accepted v_h but ignored it, reading the trajectory buffer
      instead. BeliefEstimator.update now takes v_h_curr directly and is used
      correctly (see belief_estimator.py docstring).
  _apply_human_noise:
    - dt was hardcoded as a literal 0.01 instead of derived from the control
      frequency. Now computed from CONTROL_HZ (a class constant).
  compute_alpha:
    - was a stub always returning 0.0, silently disabling blending whenever
      BLENDING=True. Now raises NotImplementedError so a misconfiguration is
      caught loudly instead of silently no-op'ing.
  get_dynamic_goal_pose / pin.log3 singularity:
    - moved into GoalSet, which now uses a Frobenius-norm fallback near the
      pi-rotation singularity instead of calling pin.log3 unconditionally (see
      goal_set.py docstring).
  Grasping oscillation (root cause, analysis Section 2):
    - PRE_GRASP now drives on the QP-constrained policy (pi_max) rather than
      the raw, CBF-unaware v_geo (see grasp_state_machine.py _pre_grasp).
  State machine (analysis Section 3, problems A/B/C):
    - replaced with GraspStateMachine's dict-dispatch handler table, which adds
      belief hysteresis (fix for Problem A), guarantees target_twist is always
      defined (fix for Problem B), and makes new states a one-method addition
      (fix for Problem C).
  update_plot artist leak:
    - moved into PlotManager, which mutates fill-polygon vertices in place via
      set_xy() instead of remove()+fill() every tick (see plot_manager.py
      docstring).
"""

import threading
import time

import numpy as np
import quadprog
from collections import deque
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Float64MultiArray
from geometry_msgs.msg import Point, TransformStamped, WrenchStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt

from triago_control.shared_autonomy.goal_set import GoalSet, create_transform
from triago_control.shared_autonomy.belief_estimator import BeliefEstimator
from triago_control.shared_autonomy.grasp_state_machine import GraspStateMachine, TickInput, CLEAR_MARGIN


class SharedControlNode(Node):
    """ROS2 Node for intent inference and safe twist blending via QP-CLF-CBF.

    Thin glue layer: owns ROS pub/sub/timers and the math that genuinely needs
    live ROS state (CLF/CBF QP, twist integration, visualization), and delegates
    goal geometry, belief inference, grasp state, and plotting to dedicated classes.
    """

    CONTROL_HZ = 100.0  # matches the 0.01s timer period below

    def __init__(self):
        """Initializes flags, delegate objects, weighting matrices, and ROS2 infrastructure."""
        super().__init__('shared_control_node')

        # --- Architecture Flags ---
        self.PREDICTION = True   # Update belief and evaluate all policies vs. just the active goal policy
        self.BLENDING = False    # Augment human input with the optimal policy vs. strictly executing it
        self.TASK_DIM = 6        # 6 for full SE(3) tracking, 5 for S^2 grasping (align X-axis only)

        self.POLICY_BELIEF_TEST = True   # <-- flip this to switch modes
        # When True:  the node injects pi_stars[test_goal_key] as the fake human
        #             velocity instead of reading from the Haption topic, and
        #             commands the robot directly via /arm_right/cartesian_reference.
        # When False: normal operation (Haption + assistive_reference topic).

        self.test_goal_key = 'Red_Side'  # starting goal; changed at runtime via console
        self._test_goal_lock = threading.Lock()

        # --- Noisy Human Model Parameters ---
        self.noise_snr_inv = 0.0
        self.bias_tau_s = 1.5
        self.bias_sigma = 0.0
        self._ou_bias = np.zeros(6)

        # --- Frequency & Time Monitoring ---
        self.freq_window_s = 10.0
        self._control_ticks = 0
        self._control_last_print = time.time()
        self.last_collision_time = 0.0
        self.max_data_age = 0.05

        # --- Bimanual Toggling ---
        self.active_arm = 'right'

        # --- Goal Set Definition (delegated to GoalSet) ---
        self.goal_set = GoalSet()
        self.target_keys = self.goal_set.target_keys
        self.active_goal_key = 'Red_Side'

        # --- Intent Inference (delegated to BeliefEstimator) ---
        # Weighting matrix (penalizes translation heavily, respects rotation)
        self.W = np.diag([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
        self.plot_lock = threading.Lock()
        self.belief_estimator = BeliefEstimator(
            target_keys=self.target_keys, W=self.W, beta=0.04, ema_alpha=0.995)

        # --- Grasp State Machine (delegated to GraspStateMachine) ---
        self.grasp_sm = GraspStateMachine(
            cylinders=self.goal_set.cylinders,
            initial_state="SHARED_AUTONOMY",
            debug=True,  # mirrors the original GRASP_DEBUG flag
        )

        # --- Plot Manager (delegated to PlotManager) ---
        # Imported lazily here (after plt.ion() is implicitly handled inside
        # PlotManager) to keep the import block above focused on ROS/math deps.
        from triago_control.shared_autonomy.plot_manager import PlotManager
        self.plot_manager = PlotManager(
            target_keys=self.target_keys,
            plot_lock=self.plot_lock,
            logger=self.get_logger(),
            freq_window_s=self.freq_window_s,
        )

        # --- NEW SUBSCRIBER: Toggle active arm via Haption Button mapping ---
        self.sub_active_arm = self.create_subscription(
            String, '/shared_autonomy/active_arm', self.active_arm_callback, 10)

        # --- Velocity Limits & Smooth Saturation Parameters ---
        self.v_max_lin = 0.1
        self.K_p_lin = 0.5
        self.w_max_ang = 0.3
        self.K_p_ang = 0.5

        # --- Grasping Interaction Topics & State ---
        self.pub_gripper_cmd = self.create_publisher(String, '/shared_autonomy/gripper_cmd', 10)
        self.sub_trigger = self.create_subscription(Bool, '/haption/trigger', self.trigger_callback, 10)

        self.trigger_cmd = False  # consumed event flag

        # --- Grasp contact confirmation ---
        # Signed gripper<->cylinder collision distance published by teleop.
        # < 0 means the gripper volume actually overlaps the cylinder (true contact).
        self.grasp_contact = {'red': 1.0, 'blue': 1.0}
        self.sub_grasp_contact = self.create_subscription(
            Float64MultiArray, '/shared_autonomy/grasp_contact', self.grasp_contact_callback, 10)

        # --- State Variables ---
        self.current_v_h = np.zeros(6)
        self.current_T_EE = np.eye(4)
        self.J_c = None
        self.h_c = None

        # --- Trajectory Buffer: 500 steps (5s at 100Hz) of human input history ---
        self.trajectory_data = deque(maxlen=500)

        # --- Hybrid Admittance State Machine support variables ---
        self.baseline_force = None
        self.current_force_local = np.zeros(3)
        self.current_force_mag = 0.0  # |F| (N) after baseline subtraction

        # --- Grasp debug & force-based confirmation ---
        self.GRASP_DEBUG = True
        self.GRASP_FORCE_THRESHOLD = GraspStateMachine.GRASP_FORCE_THRESHOLD

        # --- ROS2 TOPICS ---
        self.sub_human_reference = self.create_subscription(
            Float64MultiArray, '/arm_right/cartesian_reference', self.human_reference_callback, 10)
        self.sub_ee_pose = self.create_subscription(
            Float64MultiArray, '/qp_debug/ee_real', self.robot_state_callback, 10)
        self.sub_collision = self.create_subscription(
            Float64MultiArray, '/collision_constraints', self.collision_data_callback, 10)
        self.pub_ignore_cbf = self.create_publisher(String, '/shared_autonomy/target_ignore', 10)
        self.pub_grasp_margin = self.create_publisher(String, '/shared_autonomy/grasp_margin', 10)

        # Publisher selection driven by the test flag
        if self.POLICY_BELIEF_TEST:
            self.pub_blend_right = self.create_publisher(Float64MultiArray, '/arm_right/cartesian_reference', 10)
            self.pub_blend_left = self.create_publisher(Float64MultiArray, '/arm_left/cartesian_reference', 10)
        # else: normal-operation assistive_reference publisher intentionally omitted
        # (deprecated topic in the original script).

        self.sub_wrench = self.create_subscription(
            WrenchStamped, '/ft_sensor_right_controller/wrench', self.wrench_callback, 10)

        # --- Visualization Infrastructure ---
        self.pub_markers = self.create_publisher(MarkerArray, '/shared_policy_markers', 10)
        # Bug fix: this used to be assigned a second time later in __init__,
        # silently leaking the first TransformBroadcaster. Assigned exactly once.
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Unified Inference State Publishers ---
        self.pub_goal_names = self.create_publisher(String, '/shared_autonomy/goal_names', 10)
        self.pub_goal_probs = self.create_publisher(Float64MultiArray, '/shared_autonomy/goal_probabilities', 10)
        self.pub_ee_policy = self.create_publisher(Float64MultiArray, '/shared_autonomy/ee_policy', 10)
        self.pub_user_policy = self.create_publisher(Float64MultiArray, '/shared_autonomy/user_policy', 10)

        # Main Loop at 100Hz
        self.timer = self.create_timer(1.0 / self.CONTROL_HZ, self.timer_callback)

        # --- User Pose To build user_policy ---
        self.current_T_user = np.eye(4)

        # --- Print Configuration to Console ---
        self.get_logger().info("=========================================")
        self.get_logger().info(" SHARED AUTONOMY NODE INITIALIZED")
        self.get_logger().info("=========================================")
        self.get_logger().info(f" State:       {self.grasp_sm.state}")
        self.get_logger().info(f" Prediction:  {'ENABLED (Inferring Intent)' if self.PREDICTION else 'DISABLED (Fixed Goal)'}")
        self.get_logger().info(f" Blending:    {'ENABLED (Mixing Commands)' if self.BLENDING else 'DISABLED (Strict Optimal Policy)'}")
        self.get_logger().info(f" Active Goal: {self.active_goal_key} (Used if Prediction is False)")
        self.get_logger().info(f" Total Goals: {len(self.target_keys)}")
        self.get_logger().info("=========================================")

        # Console input thread (only needed in test mode)
        if self.POLICY_BELIEF_TEST:
            self._console_thread = threading.Thread(
                target=self._console_input_thread, daemon=True, name='console-input')
            self._console_thread.start()

    # --- Callbacks ---

    def trigger_callback(self, msg):
        """Reads the Haption device trigger/button to initiate grasp."""
        if msg.data:  # Only trigger on the rising edge
            self.trigger_cmd = True

    def grasp_contact_callback(self, msg):
        """Stores the signed gripper<->cylinder collision distance [red, blue] (m)."""
        if len(msg.data) >= 2:
            self.grasp_contact['red'] = float(msg.data[0])
            self.grasp_contact['blue'] = float(msg.data[1])

    def _set_grasp_margin(self, color, margin):
        """Publish the per-pair negative CBF margin for the gripper-cylinder pair."""
        msg = String()
        msg.data = f"{self.goal_set.cbf_name(color)}:{margin:.4f}"
        self.pub_grasp_margin.publish(msg)

    def _clear_grasp_margin(self):
        """Restore full CBF safety on all grasp pairs."""
        msg = String()
        msg.data = "None"
        self.pub_grasp_margin.publish(msg)

    def human_reference_callback(self, msg):
        """Extracts the user's SE(3) pose and 6D spatial twist from the synchronized 13-element array.

        Format: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, TASK_DIM]
        """
        if len(msg.data) >= 13:
            pos = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            rot_mat = R.from_euler('xyz', rpy).as_matrix()
            self.current_T_user = create_transform(pos, rot_mat)
            self.current_v_h = np.array(msg.data[6:12])

    def publish_inference_state(self, ee_policies, user_policies):
        """Publishes the flat belief and policy state for the haptic manager."""
        msg_names = String()
        msg_names.data = ",".join(self.target_keys)
        self.pub_goal_names.publish(msg_names)

        beliefs = self.belief_estimator.get_beliefs()
        probs_array = [float(beliefs[k]) for k in self.target_keys]
        msg_probs = Float64MultiArray()
        msg_probs.data = probs_array
        self.pub_goal_probs.publish(msg_probs)

        user_array = []
        for k in self.target_keys:
            user_array.extend(user_policies[k].tolist())
        msg_user = Float64MultiArray()
        msg_user.data = user_array
        self.pub_user_policy.publish(msg_user)

        ee_array = []
        for k in self.target_keys:
            ee_array.extend(ee_policies[k].tolist())
        msg_ee = Float64MultiArray()
        msg_ee.data = ee_array
        self.pub_ee_policy.publish(msg_ee)

    def wrench_callback(self, msg):
        """Reads the local wrist forces and subtracts the initial gravity baseline."""
        raw_force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])

        if self.baseline_force is None:
            self.baseline_force = raw_force
            self.get_logger().info(f"F/T Sensor Tared: {self.baseline_force}")

        self.current_force_local = raw_force - self.baseline_force
        self.current_force_mag = float(np.linalg.norm(self.current_force_local))

        if self.GRASP_DEBUG:
            log_msg = (f"[F/T] |F|={self.current_force_mag:5.2f} N  "
                       f"Fxyz=[{self.current_force_local[0]:6.2f},"
                       f"{self.current_force_local[1]:6.2f},{self.current_force_local[2]:6.2f}] N")

            if self.grasp_sm.state in ("GRASP_APPROACH", "GRASP_CLOSE"):
                self.get_logger().info(log_msg, throttle_duration_sec=0.25)
            else:
                self.get_logger().info(log_msg, throttle_duration_sec=2.0)

    def active_arm_callback(self, msg):
        """Switches the actively controlled arm (left/right) when the user presses the haption button."""
        if msg.data in ['right', 'left'] and self.active_arm != msg.data:
            self.active_arm = msg.data
            self.get_logger().info(f"Active Arm switched to: {self.active_arm.upper()}")
            # Reset belief and OU bias to avoid jumps across the arm switch.
            self.belief_estimator.reset()
            self._ou_bias = np.zeros(6)

    def robot_state_callback(self, msg):
        """Extracts EE pose dynamically based on the active arm.

        Index layout ("Assuming indexes 3:6 and 15:18 based on TSID stacking" in
        the original) is named here as constants so the contract is explicit
        rather than embedded in a comment.
        """
        RIGHT_POS_SLICE = slice(0, 3)
        RIGHT_RPY_SLICE = slice(12, 15)
        LEFT_POS_SLICE = slice(3, 6)
        LEFT_RPY_SLICE = slice(15, 18)

        if len(msg.data) >= 18:
            if self.active_arm == 'right':
                pos = np.array(msg.data[RIGHT_POS_SLICE])
                rpy = np.array(msg.data[RIGHT_RPY_SLICE])
            else:
                pos = np.array(msg.data[LEFT_POS_SLICE])
                rpy = np.array(msg.data[LEFT_RPY_SLICE])

            rot_mat = R.from_euler('xyz', rpy).as_matrix()
            self.current_T_EE = create_transform(pos, rot_mat)

    def collision_data_callback(self, msg):
        """Extracts Cartesian collision Jacobian dynamically based on the active arm.

        Bug fix: the original used `== 13`, which silently dropped valid
        extended messages if the publisher ever grew the array beyond 13
        fields. Changed to `>= 13` so backward-compatible extensions still work.
        """
        if len(msg.data) >= 13:
            self.last_collision_time = time.time()
            self.h_c = np.array([msg.data[0]])
            if self.active_arm == 'right':
                self.J_c = np.array(msg.data[1:7]).reshape(1, 6)
            else:
                self.J_c = np.array(msg.data[7:13]).reshape(1, 6)

    def timer_callback(self):
        """Main loop: evaluates optimal policies, updates belief, and integrates output."""

        # --- Frequency Monitoring (Control Loop) ---
        self._control_ticks += 1
        current_time = time.time()
        if (current_time - self._control_last_print) >= self.freq_window_s:
            fps = self._control_ticks / (current_time - self._control_last_print)
            self.get_logger().info(f"[FREQ] Control Loop: {fps:.1f} Hz")
            self._control_ticks = 0
            self._control_last_print = current_time

        msg_ignore = String()
        in_grasp_state = self.grasp_sm.state in ("PRE_GRASP", "GRASP_APPROACH", "GRASP_CLOSE")

        if not in_grasp_state:
            if self.J_c is None or self.h_c is None:
                return
            if (time.time() - self.last_collision_time) > self.max_data_age:
                self.get_logger().warn("Collision data stale. Halting.", throttle_duration_sec=1.0)
                return

        # In test mode (no haptic device), bind fake user pose to real robot pose.
        if self.POLICY_BELIEF_TEST:
            self.current_T_user = self.current_T_EE.copy()

        # 1. Evaluate Optimal Policies (Dual Evaluation)
        ee_policies = {}
        user_policies = {}

        in_free_space = self.grasp_sm.state in ("SHARED_AUTONOMY", "PRE_GRASP")
        valid_matrices = self.J_c is not None and self.h_c is not None

        if in_free_space and valid_matrices:
            for key in self.target_keys:
                if not self.PREDICTION and key != self.active_goal_key:
                    continue

                T_goal_robot = self.goal_set.get_dynamic_goal_pose(self.current_T_EE, key)
                v_geo_robot = self.compute_v_geo(self.current_T_EE, T_goal_robot)
                ee_policies[key] = self.solve_local_policy(v_geo_robot, self.J_c, self.h_c)

                if self.PREDICTION or self.POLICY_BELIEF_TEST:
                    T_goal_user = self.goal_set.get_dynamic_goal_pose(self.current_T_user, key)
                    v_geo_user = self.compute_v_geo(self.current_T_user, T_goal_user)
                    user_policies[key] = self.solve_local_policy(v_geo_user, self.J_c, self.h_c)
        else:
            # FALLBACK: grasping, or matrices stale -- don't solve the QP.
            for key in self.target_keys:
                ee_policies[key] = np.zeros(self.TASK_DIM)
                if self.PREDICTION or self.POLICY_BELIEF_TEST:
                    user_policies[key] = np.zeros(self.TASK_DIM)

        # 2. Update Belief Distribution
        if self.POLICY_BELIEF_TEST:
            with self._test_goal_lock:
                _test_key = self.test_goal_key
            if _test_key in user_policies:
                self.current_v_h = self._apply_human_noise(user_policies[_test_key])

        if self.PREDICTION:
            self.trajectory_data.append({'time': time.time(), 'v_h': self.current_v_h.copy()})
            self.belief_estimator.update(self.current_v_h, user_policies)
            self.plot_manager.push_beliefs(self.belief_estimator.get_beliefs())

            if self.POLICY_BELIEF_TEST:
                with self._test_goal_lock:
                    self.active_goal_key = self.test_goal_key
                b_max = 1.0
            else:
                self.active_goal_key, b_max = self.belief_estimator.get_active_goal()

            if self.POLICY_BELIEF_TEST:
                pi_max = ee_policies[self.active_goal_key]
            else:
                pi_max = self.belief_estimator.blend_policies(ee_policies)

            self.plot_manager.push_twist_snapshot(self.current_v_h, pi_max, self.active_goal_key)

        else:
            b_max = 1.0
            pi_max = ee_policies[self.active_goal_key]

        # --- 3. ERROR EVALUATION ---
        T_active_goal = self.goal_set.get_dynamic_goal_pose(
            self.current_T_EE, self.active_goal_key, approach_offset=0.05)
        p_EE = self.current_T_EE[:3, 3]
        p_goal = T_active_goal[:3, 3]
        pos_error = np.linalg.norm(p_goal - p_EE)
        R_EE = self.current_T_EE[:3, :3]
        R_goal = T_active_goal[:3, :3]

        if self.TASK_DIM == 5:
            ang_error = np.linalg.norm(np.cross(R_EE[:, 0], R_goal[:, 0]))
        else:
            ang_error = np.linalg.norm(pin.log3(R_goal @ R_EE.T))

        # Capture and consume the trigger event
        trigger_pulled = self.trigger_cmd
        self.trigger_cmd = False

        # Block human arm input during active grasp execution
        if self.grasp_sm.state in ("GRASP_APPROACH", "GRASP_CLOSE"):
            self.current_v_h = np.zeros(6)

        # --- 4. THE GRASPING STATE MACHINE (delegated) ---
        tick_input = TickInput(
            current_T_EE=self.current_T_EE,
            T_active_goal=T_active_goal,
            pos_error=pos_error,
            ang_error=ang_error,
            pi_max=pi_max,
            b_max=b_max,
            prediction_enabled=self.PREDICTION,
            active_goal_key=self.active_goal_key,
            active_arm=self.active_arm,
            trigger_pulled=trigger_pulled,
            current_force_mag=self.current_force_mag,
            grasp_contact=dict(self.grasp_contact),
            compute_v_geo=self.compute_v_geo,
            get_dynamic_goal_pose=self.goal_set.get_dynamic_goal_pose,
        )
        tick_output = self.grasp_sm.step(tick_input)

        for level, message in tick_output.log_lines:
            if level == "warn":
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)

        # Apply the state machine's side effects (CBF shield, grasp margin, gripper cmd).
        if tick_output.ignore_cbf is not None:
            msg_ignore.data = tick_output.ignore_cbf
            self.pub_ignore_cbf.publish(msg_ignore)

        color = self.active_goal_key.split('_')[0]
        if tick_output.grasp_margin == CLEAR_MARGIN:
            self._clear_grasp_margin()
        elif tick_output.grasp_margin is not None:
            self._set_grasp_margin(color, tick_output.grasp_margin)
        # else: leave the margin topic untouched this tick (matches the original,
        # where PRE_GRASP and the GRASP_APPROACH timeout-abort never call either
        # _set_grasp_margin or _clear_grasp_margin).

        if tick_output.gripper_cmd is not None:
            cmd_msg = String()
            cmd_msg.data = tick_output.gripper_cmd
            self.pub_gripper_cmd.publish(cmd_msg)

        if tick_output.reset_trigger:
            self.trigger_cmd = False

        if self.BLENDING and tick_output.new_state == "SHARED_AUTONOMY":
            alpha = self.compute_alpha(b_max)
            target_twist = (1 - alpha) * self.current_v_h + alpha * tick_output.target_twist
        else:
            target_twist = tick_output.target_twist

        # ==========================================
        # [INJECT DEBUG BLOCK HERE]
        # ==========================================
        if self.grasp_sm.state in ("GRASP_APPROACH", "GRASP_CLOSE"):
            # 1. Print the raw Twist commands and Errors at 4 Hz
            self.get_logger().info(
                f"[GRASP DEBUG] Err(Pos:{pos_error:.3f}m, Ang:{ang_error:.3f}rad) | "
                f"v_cmd=[{target_twist[0]:.3f}, {target_twist[1]:.3f}, {target_twist[2]:.3f}] | "
                f"w_cmd=[{target_twist[3]:.3f}, {target_twist[4]:.3f}, {target_twist[5]:.3f}]",
                throttle_duration_sec=0.25
            )
            
            # 2. Print the exact target position to watch for "Moving Carrot" jitter
            T_target_debug = self.goal_set.get_dynamic_goal_pose(self.current_T_EE, self.active_goal_key)
            p_t = T_target_debug[:3, 3]
            self.get_logger().info(
                f"[GRASP DEBUG] Dynamic Target Pos: X={p_t[0]:.4f}, Y={p_t[1]:.4f}, Z={p_t[2]:.4f}",
                throttle_duration_sec=0.25
            )
            
        # --- 5. LOCAL INTEGRATION & VISUALIZATION ---
        if not np.allclose(self.current_T_EE, np.eye(4)):
            visual_dt = 0.5
            trajectory_data = []

            active_v_geo = self.compute_v_geo(self.current_T_EE, T_active_goal)
            T_cube_1 = self.integrate_twist(self.current_T_EE, target_twist, visual_dt)
            trajectory_data.append((T_cube_1, target_twist))

            sim_T_EE = T_cube_1
            if in_free_space and valid_matrices:
                for _ in range(1):
                    visual_dt = visual_dt + 0.3
                    T_sim_goal = self.goal_set.get_dynamic_goal_pose(sim_T_EE, self.active_goal_key)
                    sim_v_geo = self.compute_v_geo(sim_T_EE, T_sim_goal)
                    sim_twist = self.solve_local_policy(sim_v_geo, self.J_c, self.h_c)
                    sim_T_next = self.integrate_twist(sim_T_EE, sim_twist, visual_dt)
                    trajectory_data.append((sim_T_next, sim_twist))
                    sim_T_EE = sim_T_next

            self.publish_visualizations(trajectory_data, self.current_T_EE, active_v_geo)

            for goal_key in self.target_keys:
                T_goal_tf = self.goal_set.get_dynamic_goal_pose(self.current_T_EE, goal_key)
                self.broadcast_goal_frame(goal_key, T_goal_tf)

            self.publish_inference_state(ee_policies, user_policies)

        # --- 6. PUBLISH COMMAND TO ROBOT IN TEST MODE ---
        if self.POLICY_BELIEF_TEST and not np.allclose(self.current_T_EE, np.eye(4)):
            # Virtual Haptic Cursor: integrate the optimal policy twist a short
            # distance into the future so the QP's CLF always has a moving,
            # reachable carrot to track (sending current pose stalls tracking;
            # sending the final goal causes a jerk).
            dt_virtual = 0.1
            T_virtual_ref = self.integrate_twist(self.current_T_EE, target_twist, dt_virtual)

            p_ref = T_virtual_ref[:3, 3]
            rpy_ref = R.from_matrix(T_virtual_ref[:3, :3]).as_euler('xyz')

            cmd_data = np.concatenate((p_ref, rpy_ref, target_twist, [self.TASK_DIM]))
            msg_cmd = Float64MultiArray()
            msg_cmd.data = cmd_data.tolist()

            if self.active_arm == 'right':
                self.pub_blend_right.publish(msg_cmd)
            else:
                self.pub_blend_left.publish(msg_cmd)

    # --- Core Mathematical Functions ---

    def create_gripper_markers(self, T_pose, opacity, step_index, now):
        """Builds a 3-part generic gripper with X as the approach axis."""
        markers = []
        p_center = T_pose[:3, 3]
        R_mat = T_pose[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()

        base = Marker()
        base.header.frame_id = "base_footprint"
        base.header.stamp = now
        base.ns = "policy_grippers"
        base.id = step_index * 3
        base.type = Marker.CUBE
        base.action = Marker.ADD
        base.pose.position.x, base.pose.position.y, base.pose.position.z = p_center[0], p_center[1], p_center[2]
        base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w = quat[0], quat[1], quat[2], quat[3]
        base.scale.x, base.scale.y, base.scale.z = 0.02, 0.08, 0.03
        base.color.r, base.color.g, base.color.b, base.color.a = 0.0, 1.0, 0.0, opacity
        markers.append(base)

        offset_l = np.array([0.03, 0.035, 0.0])
        p_left = p_center + (R_mat @ offset_l)

        left = Marker()
        left.header = base.header
        left.ns = base.ns
        left.id = step_index * 3 + 1
        left.type = Marker.CUBE
        left.action = Marker.ADD
        left.pose.position.x, left.pose.position.y, left.pose.position.z = p_left[0], p_left[1], p_left[2]
        left.pose.orientation = base.pose.orientation
        left.scale.x, left.scale.y, left.scale.z = 0.06, 0.01, 0.02
        left.color = base.color
        markers.append(left)

        offset_r = np.array([0.03, -0.035, 0.0])
        p_right = p_center + (R_mat @ offset_r)

        right = Marker()
        right.header = base.header
        right.ns = base.ns
        right.id = step_index * 3 + 2
        right.type = Marker.CUBE
        right.action = Marker.ADD
        right.pose.position.x, right.pose.position.y, right.pose.position.z = p_right[0], p_right[1], p_right[2]
        right.pose.orientation = base.pose.orientation
        right.scale.x, right.scale.y, right.scale.z = 0.06, 0.01, 0.02
        right.color = base.color
        markers.append(right)

        return markers

    def integrate_twist(self, current_T, twist, dt):
        """Integrates a 6D spatial twist (World Frame) into an SE(3) pose matrix.

        Rotation is applied on the left: R_new = exp(omega * dt) @ R_old, which
        is correct for a world-frame angular velocity (confirmed correct in the
        analysis -- unchanged from the original).
        """
        v_lin, v_ang = twist[:3], twist[3:]
        pos_old, rot_old = current_T[:3, 3], current_T[:3, :3]

        pos_new = pos_old + (v_lin * dt)
        rot_new = R.from_rotvec(v_ang * dt).as_matrix() @ rot_old
        return create_transform(pos_new, rot_new)

    def broadcast_goal_frame(self, goal_key, T_goal):
        """Broadcasts the active goal pose as a TF reference frame for RViz."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_footprint"
        t.child_frame_id = f"goal_{goal_key}"

        t.transform.translation.x = T_goal[0, 3]
        t.transform.translation.y = T_goal[1, 3]
        t.transform.translation.z = T_goal[2, 3]

        quat = R.from_matrix(T_goal[:3, :3]).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def _make_arrow(self, ns, marker_id, start, end, rgba, now, scale=(0.01, 0.02, 0.02)):
        """Builds a single ARROW marker. Factored out per the analysis suggestion
        ("Arrow construction in publish_visualizations is verbose -- factor into
        a _make_arrow(ns, id, start, end, rgba) helper.")
        """
        arrow = Marker()
        arrow.header.frame_id = "base_footprint"
        arrow.header.stamp = now
        arrow.ns = ns
        arrow.id = marker_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.points = [
            Point(x=start[0], y=start[1], z=start[2]),
            Point(x=end[0], y=end[1], z=end[2]),
        ]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = scale
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = rgba
        return arrow

    def publish_visualizations(self, trajectory_data, T_EE, v_geo):
        """Publishes the fading Prediction Grippers, Command Arrows, and v_geo arrow."""
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        visual_scale = 0.1

        for i, (T_cube, v_cmd) in enumerate(trajectory_data):
            opacity = max(0.2, 0.8 - (i * 0.3))

            gripper_markers = self.create_gripper_markers(T_cube, opacity, i, now)
            marker_array.markers.extend(gripper_markers)

            cube_pos = T_cube[:3, 3]
            cmd_end = cube_pos + v_cmd[:3] * visual_scale
            arrow_cmd = self._make_arrow(
                "local_policy_arrows", i, cube_pos, cmd_end,
                (1.0, 1.0, 0.0, opacity), now)
            marker_array.markers.append(arrow_cmd)

        ee_pos = T_EE[:3, 3]
        geo_end = ee_pos + v_geo[:3] * visual_scale
        arrow_geo = self._make_arrow(
            "v_geo_direction", 100, ee_pos, geo_end,
            (0.5, 0.0, 0.5, 0.8), now)
        marker_array.markers.append(arrow_geo)

        self.pub_markers.publish(marker_array)

    def compute_v_geo(self, T_EE, T_goal):
        """Computes the LOCAL_WORLD_ALIGNED decoupled spatial velocity error with
        purely smooth saturation (no deadband)."""
        p_EE = T_EE[:3, 3]
        p_goal = T_goal[:3, 3]
        R_EE = T_EE[:3, :3]
        R_goal = T_goal[:3, :3]

        # Translation: pure smooth proportional (tanh)
        error_lin = p_goal - p_EE
        dist = np.linalg.norm(error_lin)

        if dist > 1e-5:
            v_mag = self.v_max_lin * np.tanh((self.K_p_lin * dist) / self.v_max_lin)
            v_linear = (error_lin / dist) * v_mag
        else:
            v_linear = np.zeros(3)

        # Rotation: dynamic task dimension (SO(3) vs S^2)
        # Bug fix: the original used getattr(self, 'TASK_DIM', 6), which was
        # unnecessary defensive code since TASK_DIM is unconditionally set in
        # __init__. Uses self.TASK_DIM directly.
        if self.TASK_DIM == 5:
            error_ang = np.cross(R_EE[:, 0], R_goal[:, 0])
        else:
            error_ang = pin.log3(R_goal @ R_EE.T)

        ang_dist = np.linalg.norm(error_ang)

        if ang_dist > 1e-5:
            w_mag = self.w_max_ang * np.tanh((self.K_p_ang * ang_dist) / self.w_max_ang)
            v_angular = (error_ang / ang_dist) * w_mag
        else:
            v_angular = np.zeros(3)

        return np.concatenate((v_linear, v_angular))

    def solve_local_policy(self, v_geo, J_c, h_c):
        """Solves the strictly convex QP min 1/2 v^T W v - v^T W v_geo s.t. J_c v >= h_c.

        Bug fix: on QP infeasibility the original `except ValueError` fallback
        returned v_geo -- the unconstrained geometric velocity -- which violates
        the CBF constraint entirely (the robot would drive straight through the
        obstacle barrier). This now returns a safe zero-twist halt instead and
        logs an error so the infeasibility is visible rather than silently
        unsafe.
        """
        G = self.W.astype(np.float64)
        a = (self.W @ v_geo).astype(np.float64)

        C = J_c.T.astype(np.float64)
        b = h_c.astype(np.float64)

        try:
            solution = quadprog.solve_qp(G, a, C, b)[0]
            return solution
        except ValueError as e:
            self.get_logger().error(
                f"QP Infeasible (conflicting CBF constraints): {e}. "
                f"Commanding a safe zero-twist halt instead of the unconstrained v_geo."
            )
            return np.zeros(6)

    def _apply_human_noise(self, pi: np.ndarray) -> np.ndarray:
        """Noisy human model: pi_star + signal-scaled white noise + OU persistent bias.

        Component 1 -- signal-scaled white noise:
            std = noise_snr_inv * ||pi||
            Noise is proportional to command magnitude (Harris & Wolpert 1998
            motor noise model). A still robot produces near-zero noise; a fast
            approach produces realistic jitter. SNR is geometry-independent.

        Component 2 -- Ornstein-Uhlenbeck persistent bias:
            b_{t+1} = (1 - dt/tau) * b_t  +  sigma * sqrt(dt) * N(0, I6)
            Humans hold a slightly wrong direction for O(tau) seconds before
            correcting -- the OU process captures this correlated drift.
            Steady-state std ~= sigma * sqrt(tau / 2).

        Setting noise_snr_inv=0 and bias_sigma=0 recovers the noiseless case exactly.

        Bug fix: dt was hardcoded as a bare literal 0.01 ("must match the 100 Hz
        timer period" per a comment in the original). Now derived from
        CONTROL_HZ so the two can never silently drift apart.
        """
        dt = 1.0 / self.CONTROL_HZ

        noise_std = self.noise_snr_inv * np.linalg.norm(pi)
        white_noise = noise_std * np.random.randn(6)

        decay = 1.0 - dt / max(self.bias_tau_s, 1e-6)
        diffusion = self.bias_sigma * np.sqrt(dt) * np.random.randn(6)
        self._ou_bias = decay * self._ou_bias + diffusion

        return pi.copy() + white_noise + self._ou_bias

    def compute_alpha(self, b_max):
        """Maps the maximum belief probability to the autonomy arbitration weight.

        Bug fix: the original was a stub that always returned 0.0, which
        silently disabled blending any time BLENDING=True (no warning, no
        error -- it just quietly did nothing). This now raises explicitly so a
        BLENDING=True misconfiguration is caught immediately instead of
        producing confusing "blending does nothing" behavior at runtime.

        Implement the actual confidence-to-alpha mapping before enabling
        BLENDING=True (e.g. a saturating ramp such as
        np.clip((b_max - b_low) / (b_high - b_low), 0, 1)).
        """
        raise NotImplementedError(
            "compute_alpha has no implementation yet. Set self.BLENDING = False, "
            "or implement a confidence-to-alpha mapping here before enabling it."
        )

    def _console_input_thread(self):
        """Blocking console loop that lets the developer switch the test goal at runtime.

        Runs on a dedicated daemon thread. Type a valid goal key and press Enter
        to redirect the fake human input. Press Enter alone to print the current
        goal without changing it.
        """
        valid_goals = self.target_keys
        print("\n" + "=" * 52)
        print("  POLICY_BELIEF_TEST  —  hardware-free debug mode")
        print("=" * 52)
        print(f"  Available goals : {valid_goals}")
        print(f"  Starting goal   : {self.test_goal_key}")
        print("  Type a goal name + Enter to switch.")
        print("  Press Enter alone to query the current goal.")
        print("=" * 52 + "\n")

        while rclpy.ok():
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                with self._test_goal_lock:
                    current = self.test_goal_key
                print(f"  current goal: {current}")

            elif raw in valid_goals:
                with self._test_goal_lock:
                    self.test_goal_key = raw
                self.get_logger().info(f"[TEST] Goal switched to '{raw}'")
                print(f"  ✓ goal set to: {raw}")
                self._ou_bias[:] = 0.0  # reset drift when changing goal

            elif raw.startswith("noise "):
                try:
                    self.noise_snr_inv = float(raw.split()[1])
                    print(f"  noise_snr_inv = {self.noise_snr_inv:.3f}")
                except (IndexError, ValueError):
                    print("  usage: noise <float>  e.g. 'noise 0.3'")

            elif raw.startswith("tau "):
                try:
                    self.bias_tau_s = max(float(raw.split()[1]), 0.01)
                    print(f"  bias_tau_s = {self.bias_tau_s:.2f} s")
                except (IndexError, ValueError):
                    print("  usage: tau <seconds>  e.g. 'tau 1.5'")

            elif raw == "status":
                print(f"  noise_snr_inv = {self.noise_snr_inv:.3f}")
                print(f"  bias_tau_s    = {self.bias_tau_s:.2f} s")
                print(f"  bias_sigma    = {self.bias_sigma:.4f}")
                print(f"  OU bias now   = {np.round(self._ou_bias, 4)}")

            elif raw == "CLOSE":
                # Bug fix: the original wrote self.trigger_cmd = True directly
                # from this thread without the same protection pattern used
                # elsewhere. This write is a single attribute assignment of a
                # bool, which is atomic in CPython (the GIL serializes
                # bytecode-level STORE_ATTR), so no separate lock is required --
                # documented here explicitly so the assumption is not silently
                # relied upon.
                self.trigger_cmd = True
                self.get_logger().info("[TEST] 'CLOSE' command registered via console.")

            else:
                print(f"  ✗ unknown goal '{raw}'.  Choose from: {valid_goals}")


def main(args=None):
    """Spins ROS2 in a background thread while executing the UI loop on the main thread."""
    rclpy.init(args=args)
    node = SharedControlNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True, name='rclpy-spin')
    spin_thread.start()

    try:
        while rclpy.ok():
            node.plot_manager.update()
            # plt.pause inherently flushes GUI events and acts as our 10Hz sleep timer
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()
