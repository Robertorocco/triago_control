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
import warnings

import numpy as np
import quadprog
from collections import deque
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Float64MultiArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt

from triago_control.shared_autonomy.goal_set import GoalSet, create_transform
from triago_control.shared_autonomy.belief_estimator import BeliefEstimator
from triago_control.shared_autonomy.grasp_state_machine import GraspStateMachine, TickInput, CLEAR_MARGIN

# Gazebo IFRA_LinkAttacher plugin service (kinematic grasp in simulation)
try:
    from linkattacher_msgs.srv import AttachLink, DetachLink
    _HAS_LINKATTACHER = True
except Exception:
    _HAS_LINKATTACHER = False


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

        self.POLICY_BELIEF_TEST = False   # <-- flip this to switch modes
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

        # --- Visualization rate decoupling ---
        # The control loop runs at 100 Hz. Publish markers at EVERY tick (no
        # decimation) to guarantee the green policy gripper never disappears in
        # RViz — a lower rate combined with even brief stalls caused it to blink.
        # The marker traffic is small (a handful of CUBE markers per publish).
        self.VIZ_DECIM = 1            # 100 Hz / 1 = full rate marker refresh
        self._viz_counter = 0
        self._viz_miss_count = 0      # diagnostic: counts consecutive skipped viz ticks

        # --- Bimanual Toggling ---
        self.active_arm = 'right'

        # --- Goal Set Definition (delegated to GoalSet) ---
        self.goal_set = GoalSet()
        self.target_keys = self.goal_set.target_keys
        self.active_goal_key = 'Red_Side'

        # --- Intent Inference (delegated to BeliefEstimator) ---
        # Weighting matrix (penalizes translation heavily, respects rotation)
        self.W = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        self.plot_lock = threading.Lock()
        self.belief_estimator = BeliefEstimator(
            target_keys=self.target_keys, W=self.W, beta=0.04, ema_alpha=0.995)
        # The Platform placement goal only becomes demandable once something is
        # actually held — exclude it while the gripper is empty.
        self.belief_estimator.set_excluded_goals({self.goal_set.PLATFORM_KEY})
        # Color of the currently-held cylinder ('Red'/'Blue'), or None when empty.
        self.grasped_color = None

        # --- User-led belief acquisition ("listen more, lock less") ---
        # The belief update step (beta) is scaled each tick by engagement * warmup:
        #  - engagement: ~0 when the user twist is near zero (no evidence -> belief
        #    gently relaxes toward uniform instead of locking on noise), ramps to 1
        #    as the user actively moves.
        #  - warmup: a short exploration window after start / release / arm-switch /
        #    grasp-completion, during which learning is slowed so the autonomy does
        #    not instantly lock a goal and drag the device.
        self.BELIEF_V_LOW = 0.005     # m/s linear twist -> "still"
        self.BELIEF_V_HIGH = 0.030    # m/s linear twist -> fully engaged
        self.BELIEF_WARMUP_S = 2.5    # s exploration window after a reset
        self.BELIEF_WARMUP_FLOOR = 0.25  # min learning-rate scale during warm-up
        self._belief_warmup_start = time.time()
        self._prev_grasp_exec = False

        # --- Grasp State Machine (delegated to GraspStateMachine) ---
        self.grasp_sm = GraspStateMachine(
            cylinders=self.goal_set.cylinders,
            initial_state="SHARED_AUTONOMY",
            debug=True,  # mirrors the original GRASP_DEBUG flag
        )
        # Tracks the grasp SM state across ticks so the node can react to entries
        # (e.g. arming the placement goal) exactly once.
        self._prev_sm_state = self.grasp_sm.state

        # --- Per-arm state: TWO independent state machines -------------------
        # One GraspStateMachine + BeliefEstimator PER ARM. self.grasp_sm and
        # self.belief_estimator always POINT at the ACTIVE arm's instance, so the
        # rest of timer_callback is unchanged; the inactive arm's FSM/belief are
        # simply never stepped (their state is FROZEN) until that arm is
        # reactivated. Scalar context (grasped color, active goal, goal_set
        # placement bookkeeping) is saved/restored per arm on switch.
        self._sm = {
            'right': self.grasp_sm,
            'left': GraspStateMachine(
                cylinders=self.goal_set.cylinders,
                initial_state="SHARED_AUTONOMY", debug=True),
        }
        self._be = {
            'right': self.belief_estimator,
            'left': BeliefEstimator(
                target_keys=self.target_keys, W=self.W, beta=0.04, ema_alpha=0.995),
        }
        self._be['left'].set_excluded_goals({self.goal_set.PLATFORM_KEY})
        self._ctx_grasped = {'right': None, 'left': None}
        self._ctx_active_goal = {'right': self.active_goal_key, 'left': self.active_goal_key}
        self._ctx_goalset = {'right': (None, None, 0.0), 'left': (None, None, 0.0)}

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

        # NOTE: this node DRIVES the arm switch itself (double-click on the left
        # button → _switch_active_arm) and PUBLISHES /shared_autonomy/active_arm
        # for the other nodes (teleop, force manager, QP). It does NOT subscribe
        # to that topic (no self-echo loop, no duplicate switch logic).

        # --- Velocity Limits & Smooth Saturation Parameters ---
        self.v_max_lin = 0.1
        self.K_p_lin = 0.5
        self.w_max_ang = 0.3
        self.K_p_ang = 0.5

        # Teleoperation-aware limits for user_policies (published to haptic manager).
        # Lower than the robot-side limits because the operator + haptic device +
        # clutch-integration loop is slower and more compliant than direct robot
        # control. The robot-side ee_policies (green gripper, grasp execution)
        # still use the full v_max_lin / w_max_ang above.
        # NOTE: far from the goal the tanh saturates the policy to EXACTLY this
        # ceiling (constant cruise speed by design) — so this ceiling IS the
        # "drag-too-fast-when-far" speed. Lowered to tame it.
        self.v_max_lin_user = 0.04    # m/s   — comfortable hand tracking speed [was 0.07]
        self.w_max_ang_user = 0.10    # rad/s — comfortable hand rotation rate  [was 0.15]

        # --- Grasping Interaction Topics & State ---
        self.pub_gripper_cmd = self.create_publisher(String, '/shared_autonomy/gripper_cmd', 10)

        # --- Gazebo Link Attacher plugin (kinematic grasp in simulation) ---
        # Robot model name in Gazebo + per-arm gripper grasping links.
        # Overridable at runtime: --ros-args -p robot_model_name:=tiago_dual
        self.declare_parameter('robot_model_name', 'triago')
        self.robot_model_name = self.get_parameter('robot_model_name').value
        # NOTE: gripper_*_grasping_link is a TF-only frame (lumped into
        # arm_*_7_link via a fixed joint) and does NOT exist as a Gazebo link,
        # so LinkAttacher cannot use it. arm_*_7_link is the real, solid wrist
        # link that the whole gripper base is rigidly fused into — attaching the
        # cylinder there is stable regardless of finger open/close.
        self.declare_parameter('grasp_link_right', 'arm_right_7_link')
        self.declare_parameter('grasp_link_left', 'arm_left_7_link')
        self.gripper_link = {
            'right': self.get_parameter('grasp_link_right').value,
            'left':  self.get_parameter('grasp_link_left').value,
        }
        self.cylinder_model = {'red': 'red_cylinder', 'blue': 'blue_cylinder'}
        self.cylinder_link = 'link'
        # Track what is currently attached so we can detach/re-attach across retries.
        self.plugin_attached = {}  # {arm: (model2_name, link2_name)}
        self.attach_cli = None
        self.detach_cli = None
        if _HAS_LINKATTACHER:
            self.attach_cli = self.create_client(AttachLink, '/ATTACHLINK')
            self.detach_cli = self.create_client(DetachLink, '/DETACHLINK')
            self.get_logger().info(
                f"[INIT] LinkAttacher ready. robot_model_name='{self.robot_model_name}'. "
                f"Override with -p robot_model_name:=<name> if attach is REJECTED.")
        else:
            self.get_logger().warn(
                "[INIT] linkattacher_msgs not found — plugin grasp disabled.")
        self.sub_trigger = self.create_subscription(Bool, 'virtuose/button_left', self.trigger_callback, 10)

        self.trigger_cmd = False  # consumed event flag
        self._grasp_cue_phase = 0.0  # pulsing animation counter for PRE_GRASP sphere

        # Double-click arm switch state (left button: single=grasp, double=switch arm)
        self.DOUBLE_CLICK_WINDOW = 0.5  # seconds to detect the second press
        self._btn_first_press_time = None
        self._btn_timer = None
        self._btn_last_state = False  # for rising-edge detection (raw topic is level, not edge)

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

        # --- ROS2 TOPICS ---
        self.sub_human_reference_right = self.create_subscription(
            Float64MultiArray, '/arm_right/cartesian_reference', self.human_reference_callback_right, 10)
        self.sub_human_reference_left = self.create_subscription(
            Float64MultiArray, '/arm_left/cartesian_reference', self.human_reference_callback_left, 10)
        self.sub_ee_pose = self.create_subscription(
            Float64MultiArray, '/qp_debug/ee_real', self.robot_state_callback, 10)
        self.sub_collision = self.create_subscription(
            Float64MultiArray, '/collision_constraints', self.collision_data_callback, 10)
        self.pub_ignore_cbf = self.create_publisher(String, '/shared_autonomy/target_ignore', 10)
        self.pub_grasp_margin = self.create_publisher(String, '/shared_autonomy/grasp_margin', 10)

        # Cartesian-reference publishers. Created in BOTH modes now:
        #  - In test mode: the node is the sole reference source (virtual cursor).
        #  - In teleop mode: the node only publishes here DURING autonomous grasp
        #    execution (approach/close/lift), when it takes authority from the
        #    Haption clutch. See section 6 and /shared_autonomy/grasp_active.
        self.pub_blend_right = self.create_publisher(Float64MultiArray, '/arm_right/cartesian_reference', 10)
        self.pub_blend_left = self.create_publisher(Float64MultiArray, '/arm_left/cartesian_reference', 10)

        # Authority-handover flag: True while the node drives the arm autonomously
        # (grasp approach/close/lift). teleop_triago_clutch freezes and re-anchors
        # on the falling edge so teleop resumes cleanly from the post-grasp pose.
        self.pub_grasp_active = self.create_publisher(Bool, '/shared_autonomy/grasp_active', 10)
        # Publish the active arm name so teleop_triago_clutch and the force manager
        # can follow the arm switch (they need to know which /arm_*/cartesian_reference
        # to publish/subscribe and which EE slice to read from /qp_debug/ee_real).
        self.pub_active_arm = self.create_publisher(String, '/shared_autonomy/active_arm', 10)

        # Active goal pose + confidence for the haptic position virtual fixture.
        # [x, y, z, roll, pitch, yaw, confidence] in base_footprint.
        self.pub_active_goal_pose = self.create_publisher(
            Float64MultiArray, '/shared_autonomy/active_goal_pose', 10)

        self.sub_wrench_removed = None  # Force sensor removed — not used in this architecture.

        # --- Visualization Infrastructure ---
        self.pub_markers = self.create_publisher(MarkerArray, '/shared_policy_markers', 10)
        # Separate topic for the YELLOW guidance gripper so it can be toggled
        # independently in RViz. It is reference-anchored (current_T_user) and
        # shows the belief-weighted USER-policy blend — i.e. exactly the velocity
        # field that the haptic F_guide renders onto the operator's handle. This
        # is the human-side counterpart to the green robot-policy gripper.
        self.pub_guidance_marker = self.create_publisher(MarkerArray, '/guidance_policy_marker', 10)
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

        # Open gripper on startup and reset all CBF state
        self.pub_gripper_cmd.publish(String(data=f"CLOSE_RIGHT_0.7000"))
        self.pub_gripper_cmd.publish(String(data=f"CLOSE_LEFT_0.7000"))
        # Clear any leftover CBF bypasses and margins from previous runs
        self.pub_ignore_cbf.publish(String(data="CLEAR"))
        self._clear_grasp_margin()
        self.get_logger().info("[INIT] Grippers opened. CBF state reset.")

    # --- Callbacks ---

    def trigger_callback(self, msg):
        """Double-click detection on the Haption left button.

        Single press (no second press within 0.5 s) → grasp trigger (original).
        Double press (two presses within 0.5 s)     → arm switch (left ↔ right).

        The virtuose/button_left topic is a LEVEL signal (True while held, False
        while released), published at 150 Hz. We detect the RISING EDGE ourselves
        (transition False→True) so one physical press is registered exactly once.
        """
        # --- Rising-edge detection (level → edge) ---
        current = bool(msg.data)
        rising = current and not self._btn_last_state
        self._btn_last_state = current
        if not rising:
            return

        now = time.time()
        if self._btn_first_press_time is not None and (now - self._btn_first_press_time) < self.DOUBLE_CLICK_WINDOW:
            # --- DOUBLE PRESS: switch arm ---
            self._btn_first_press_time = None
            if self._btn_timer is not None:
                self._btn_timer.cancel()
                self._btn_timer = None
            new_arm = 'left' if self.active_arm == 'right' else 'right'
            self._switch_active_arm(new_arm)
        else:
            # --- FIRST PRESS: start the window ---
            self._btn_first_press_time = now
            # Fire a one-shot timer to commit the single-press (grasp trigger)
            # if no second press arrives within the window.
            if self._btn_timer is not None:
                self._btn_timer.cancel()
            self._btn_timer = self.create_timer(
                self.DOUBLE_CLICK_WINDOW, self._btn_single_press_commit, callback_group=None)

    def _btn_single_press_commit(self):
        """Called 0.5 s after the first press if no double-click occurred → grasp trigger."""
        if self._btn_timer is not None:
            self._btn_timer.cancel()
            self._btn_timer = None
        self._btn_first_press_time = None
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

    def _configure_post_grasp(self, color):
        """Arm the HOLDING phase right after a successful attach.

        - Records the grasped object + its symmetry axis in the goal set (so the
          Platform goal can enforce 'cylinder axis perpendicular to platform').
        - Excludes the grasped cylinder's own grasp goals from the belief
          estimator (they are no longer reachable) and enables the Platform goal.
        - In test mode, defaults the demanded goal to the Platform so the
          placement flow starts immediately (the user can still switch).
        """
        self.grasped_color = color.capitalize()
        self.goal_set.set_grasped(color, self.current_T_EE)
        self._update_goal_exclusions()
        if self.POLICY_BELIEF_TEST:
            with self._test_goal_lock:
                self.test_goal_key = self.goal_set.PLATFORM_KEY
        self.get_logger().info(
            f"[POST-GRASP] {self.grasped_color} held → its grasp goals excluded, "
            f"'{self.goal_set.PLATFORM_KEY}' enabled. Remaining goals stay demandable.")

    def _release_object(self):
        """Open the gripper, detach the payload and reset to the start phase.

        Per design: this is NOT a new dedicated phase — the system simply returns
        to SHARED_AUTONOMY as if freshly started, now accounting for the updated
        world (one cylinder already placed). Re-excludes the Platform goal and
        makes every cylinder demandable again.
        """
        arm = self.active_arm
        # Open fingers fully.
        self.pub_gripper_cmd.publish(String(data=f"CLOSE_{arm.upper()}_0.7000"))
        # Detach the Gazebo plugin weld.
        self._plugin_detach(arm)
        # World building + QP-side detach. The cylinder is NOT back at its spawn:
        # under the perfect-fall assumption it now rests UPRIGHT on the placement
        # surface at the XY where the EE released it. We update the goal set so the
        # re-enabled red goals point at the new location, and pass that same pose
        # to the QP collision world via the DETACH command so the obstacle is
        # placed there (with the smooth barrier ramp). Needs the held color, so do
        # this BEFORE clearing grasped_color below.
        if self.grasped_color is not None:
            ee_xy = self.current_T_EE[:2, 3]
            fallen = np.array([float(ee_xy[0]), float(ee_xy[1]),
                               self.goal_set.platform_rest_z(self.grasped_color)])
            self.goal_set.relocate_cylinder(self.grasped_color, fallen)
            self.pub_gripper_cmd.publish(String(
                data=f"DETACH_{arm.upper()}_{self.grasped_color.upper()}_"
                     f"{fallen[0]:.4f}_{fallen[1]:.4f}_{fallen[2]:.4f}"))
            self.get_logger().info(
                f"[WORLD] {self.grasped_color} cylinder placed at "
                f"[{fallen[0]:.3f}, {fallen[1]:.3f}, {fallen[2]:.3f}] (perfect-fall model).")

        # Reset the grasp state machine into the post-OPEN lift phase: it will
        # drive a short vertical lift to clear the just-placed object, then return
        # to SHARED_AUTONOMY on its own (mirrors the post-CLOSE LIFT -> HOLDING).
        self.grasp_sm._transition("RELEASE_LIFT")
        self.grasp_sm._release_lift_start = None
        self.grasp_sm.grip_position = 0.7
        self.grasp_sm.grip_contact_detected = False
        self.grasp_sm.grip_force_stable_since = None
        self.grasp_sm._lift_start_time = None
        self.grasp_sm._holding_entered = False

        # Reset goal availability via the UNION rule (accounts for the OTHER arm
        # possibly still holding a cylinder): clear THIS arm's grasp, then re-derive
        # exclusions for both arms.
        prev = self.grasped_color
        self.goal_set.clear_grasped()
        self.grasped_color = None
        self._update_goal_exclusions()
        # Restart the belief warm-up so the system re-acquires intent gently
        # (looks for the user's twist) instead of instantly locking a goal and
        # yanking the device right after release.
        self._belief_warmup_start = time.time()

        # In test mode, default the demand to the other (still-on-table) cylinder
        # so the sequential pick-and-place demo keeps flowing.
        if self.POLICY_BELIEF_TEST:
            next_goal = 'Blue_Side' if prev == 'Red' else 'Red_Side'
            with self._test_goal_lock:
                self.test_goal_key = next_goal

        self.get_logger().info(
            "[RELEASE] Object placed. Gripper opened, payload detached, "
            "back to SHARED_AUTONOMY.")

    def _plugin_attach(self, arm, color):
        """Weld the cylinder to the gripper link via the Gazebo LinkAttacher plugin.

        The plugin captures the CURRENT relative transform between the two links
        at attach time and freezes it as a fixed joint — so attachment works
        identically whether the gripper grabbed the cylinder from the side (at
        any height) or from the top. No manual relative-pose bookkeeping needed.
        """
        if self.attach_cli is None:
            self.get_logger().error("[ATTACH] LinkAttacher msgs not found — cannot attach.")
            return
        if not self.attach_cli.service_is_ready():
            if not self.attach_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("[ATTACH] /ATTACHLINK service NOT available. "
                                        "Is the LinkAttacher plugin loaded in the world?")
                return
        req = AttachLink.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self.gripper_link[arm]
        req.model2_name = self.cylinder_model[color]
        req.link2_name = self.cylinder_link
        desc = (f"{self.robot_model_name}/{self.gripper_link[arm]} <-> "
                f"{self.cylinder_model[color]}/{self.cylinder_link}")
        future = self.attach_cli.call_async(req)
        future.add_done_callback(
            lambda f: self._attach_done(f, arm, color, desc))

    def _attach_done(self, future, arm, color, desc):
        """Verify the attach service actually succeeded."""
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[ATTACH] Service call FAILED ({desc}): {e}")
            return
        success = getattr(res, 'success', None)
        message = getattr(res, 'message', '')
        if success is None:
            # Response has no 'success' field — log raw response once
            self.get_logger().warn(f"[ATTACH] Response (no success field): {res}")
            self.plugin_attached[arm] = (self.cylinder_model[color], self.cylinder_link)
        elif success:
            self.plugin_attached[arm] = (self.cylinder_model[color], self.cylinder_link)
            self.get_logger().info(f"[ATTACH] OK: {desc}  ({message})")
        else:
            self.get_logger().error(f"[ATTACH] REJECTED: {desc}  ({message})")

    def _plugin_detach(self, arm):
        """Release a previously plugin-attached object for the given arm (retry support)."""
        if self.detach_cli is None or arm not in self.plugin_attached:
            return
        model2, link2 = self.plugin_attached[arm]
        if not self.detach_cli.service_is_ready():
            if not self.detach_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("[DETACH] /DETACHLINK service NOT available.")
                return
        req = DetachLink.Request()
        req.model1_name = self.robot_model_name
        req.link1_name = self.gripper_link[arm]
        req.model2_name = model2
        req.link2_name = link2
        desc = f"{self.gripper_link[arm]} <-> {model2}"
        future = self.detach_cli.call_async(req)
        future.add_done_callback(lambda f: self._detach_done(f, arm, desc))

    def _detach_done(self, future, arm, desc):
        """Verify the detach service actually succeeded."""
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[DETACH] Service call FAILED ({desc}): {e}")
            return
        success = getattr(res, 'success', None)
        message = getattr(res, 'message', '')
        self.plugin_attached.pop(arm, None)
        if success is None:
            self.get_logger().warn(f"[DETACH] Response (no success field): {res}")
        elif success:
            self.get_logger().info(f"[DETACH] OK: {desc}  ({message})")
        else:
            self.get_logger().error(f"[DETACH] REJECTED: {desc}  ({message})")

    def human_reference_callback_right(self, msg):
        """Right-arm reference → current_T_user + current_v_h (only when active)."""
        if self.active_arm != 'right':
            return
        if len(msg.data) >= 13:
            pos = np.array(msg.data[0:3])
            rpy = np.array(msg.data[3:6])
            rot_mat = R.from_euler('xyz', rpy).as_matrix()
            self.current_T_user = create_transform(pos, rot_mat)
            self.current_v_h = np.array(msg.data[6:12])

    def human_reference_callback_left(self, msg):
        """Left-arm reference → current_T_user + current_v_h (only when active)."""
        if self.active_arm != 'left':
            return
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

    def _arm_grasped(self, arm):
        """Held color for an arm ('Red'/'Blue'/None) — active arm reads live, the
        other reads its saved context."""
        return self.grasped_color if arm == self.active_arm else self._ctx_grasped[arm]

    def _update_goal_exclusions(self):
        """UNION goal exclusions applied to BOTH arms' belief estimators.

        A cylinder held by EITHER arm is un-graspable by both (its Top/Side goals
        excluded everywhere). The Platform placement goal is demandable for an arm
        only if THAT arm is currently holding something.
        """
        held = set()
        for a in ('right', 'left'):
            c = self._arm_grasped(a)
            if c:
                held.add(c)
        for arm in ('right', 'left'):
            excl = set()
            for color in held:
                excl.add(f"{color}_Top")
                excl.add(f"{color}_Side")
            if not self._arm_grasped(arm):
                excl.add(self.goal_set.PLATFORM_KEY)
            self._be[arm].set_excluded_goals(excl)

    def _switch_active_arm(self, new_arm):
        """Swap the active arm's context so each hand is an INDEPENDENT state
        machine; the leaving arm's FSM/belief/grasped state is frozen and the
        entering arm's is restored exactly where it was left.
        """
        if new_arm not in ('right', 'left') or new_arm == self.active_arm:
            return
        old = self.active_arm
        # 1. Save the leaving arm's scalar + goal_set placement context.
        self._ctx_grasped[old] = self.grasped_color
        self._ctx_active_goal[old] = self.active_goal_key
        self._ctx_goalset[old] = (self.goal_set.grasped_color,
                                   self.goal_set.grasped_axis_local,
                                   self.goal_set.grasped_z_offset)
        # 2. Activate the new arm: re-point at its FSM/belief, restore its context.
        self.active_arm = new_arm
        self.grasp_sm = self._sm[new_arm]
        self.belief_estimator = self._be[new_arm]
        self.grasped_color = self._ctx_grasped[new_arm]
        self.active_goal_key = self._ctx_active_goal[new_arm]
        gc, ga, gz = self._ctx_goalset[new_arm]
        self.goal_set.grasped_color = gc
        self.goal_set.grasped_axis_local = ga
        self.goal_set.grasped_z_offset = gz
        self._prev_sm_state = self.grasp_sm.state
        self._belief_warmup_start = time.time()
        # 3. Notify the other nodes + the dual belief plot. Belief is NOT reset:
        # each arm keeps (and resumes from) its own frozen belief.
        self.pub_active_arm.publish(String(data=new_arm))
        self.plot_manager.push_arm_switch(new_arm)
        self.get_logger().info(
            f"\033[95m[ARM SWITCH] → {new_arm.upper()} | state={self.grasp_sm.state} "
            f"| holding={self.grasped_color}\033[0m")

    def robot_state_callback(self, msg):
        """Extracts EE pose dynamically based on the active arm.

        Index layout ("Assuming indexes 3:6 and 15:18 based on TSID stacking" in
        the original) is named here as constants so the contract is explicit
        rather than embedded in a comment.
        """
        RIGHT_POS_SLICE = slice(0, 3)
        RIGHT_RPY_SLICE = slice(12, 15)
        LEFT_POS_SLICE = slice(6, 9)
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
            if fps < 50.0:
                self.get_logger().warn(f"[FREQ] Control Loop DROPPED: {fps:.1f} Hz")
            self._control_ticks = 0
            self._control_last_print = current_time

        msg_ignore = String()
        # States that run blind (no goal tracking, no collision-data dependency).
        # HOLDING is intentionally NOT here: it drives toward goals via the QP and
        # therefore needs fresh collision data, like SHARED_AUTONOMY / PRE_GRASP.
        in_grasp_state = self.grasp_sm.state in (
            "PRE_GRASP", "GRASP_ALIGN", "GRASP_APPROACH", "GRASP_CLOSE", "LIFT", "RELEASE_LIFT", "ABORT_RETREAT")

        if not in_grasp_state:
            if self.J_c is None or self.h_c is None:
                # No collision data has EVER arrived yet — nothing to draw/solve.
                return
            # Bug fix (green gripper marker disappearing): the old code did a hard
            # `return` here whenever the collision data was older than max_data_age,
            # which halted the WHOLE callback — including marker publishing. With
            # the 500 ms marker lifetime, any jitter in the QP's /collision_constraints
            # rate then blinked the green policy gripper out of RViz. In teleop
            # SHARED_AUTONOMY this node does not even command the arm, so halting
            # viz is needless. Instead we just WARN here and let the loop continue:
            # staleness is folded into `valid_matrices` below, so the policy QP is
            # skipped (policies -> 0, a safe halt that also stops the test-mode
            # command), but visualization keeps publishing and stays alive.
            if (time.time() - self.last_collision_time) > self.max_data_age:
                self.get_logger().warn(
                    "Collision data stale — skipping policy solve (viz kept alive).",
                    throttle_duration_sec=1.0)

        # In test mode (no haptic device), bind fake user pose to real robot pose.
        if self.POLICY_BELIEF_TEST:
            self.current_T_user = self.current_T_EE.copy()

        # 1. Evaluate Optimal Policies (Dual Evaluation)
        ee_policies = {}
        user_policies = {}

        in_free_space = self.grasp_sm.state in ("SHARED_AUTONOMY", "PRE_GRASP", "HOLDING")
        # valid_matrices now also requires the collision data to be FRESH: stale
        # data -> policies solved to zero (safe halt) but visualization continues.
        valid_matrices = (self.J_c is not None and self.h_c is not None
                          and (time.time() - self.last_collision_time) <= self.max_data_age)
        excluded = self.belief_estimator.get_excluded_goals()

        if in_free_space and valid_matrices:
            for key in self.target_keys:
                # Excluded goals (already-grasped cylinder, or Platform while
                # empty) are NOT evaluated: their policy is a zero placeholder so
                # downstream consumers (belief plot, inference publishers) still
                # see every key, but the QP is never solved for them.
                if key in excluded:
                    ee_policies[key] = np.zeros(self.TASK_DIM)
                    if self.PREDICTION or self.POLICY_BELIEF_TEST:
                        user_policies[key] = np.zeros(self.TASK_DIM)
                    continue

                if not self.PREDICTION and key != self.active_goal_key:
                    continue

                # Goal-manifold point resolved w.r.t. the REAL ROBOT POSE
                # (current_T_EE). The EE is always physically valid (constrained by
                # the QP-CLF-CBF) and moves slowly — unlike the reference, which can
                # fly through obstacles and pull the manifold point into unreachable
                # configurations. Anchoring the goal at the EE means the policy
                # always asks for a physically reachable next step, and the haptic
                # guidance (F_guide) renders a force that corresponds to an
                # achievable EE motion. This single resolution owns the sticky
                # orientation/azimuth memory (update_memory=True) and is shared by
                # BOTH policies below, so the robot and the haptic guidance always
                # aim at the SAME point.
                # Test mode: current_T_user == current_T_EE → identical to before.
                T_goal = self.goal_set.get_dynamic_goal_pose(self.current_T_EE, key)

                # EE policy: velocity FROM the real EE toward that goal (commands
                # the robot in test/grasp mode, feeds pi_max and the green gripper).
                v_geo_robot = self.compute_v_geo(self.current_T_EE, T_goal)
                ee_policies[key] = self.solve_local_policy(v_geo_robot, self.J_c, self.h_c)

                if self.PREDICTION or self.POLICY_BELIEF_TEST:
                    # User policy: velocity FROM the reference toward the SAME
                    # goal — this is the policy F_guide renders onto the handle.
                    # Uses the lower teleop-aware velocity limits so the guidance
                    # field doesn't demand speeds the hand can't comfortably track.
                    v_geo_user = self.compute_v_geo(
                        self.current_T_user, T_goal,
                        v_max_lin=self.v_max_lin_user,
                        w_max_ang=self.w_max_ang_user)
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

            # Suspend belief learning entirely during autonomous grasp execution
            # (ALIGN/APPROACH/CLOSE/LIFT): the arm is driven by the SM, the user
            # twist is zeroed, and the belief should stay FROZEN until the grasp
            # ends — otherwise the distribution evolves on meaningless data.
            grasp_exec_now = self.grasp_sm.state in (
                "GRASP_ALIGN", "GRASP_APPROACH", "GRASP_CLOSE", "LIFT", "RELEASE_LIFT", "ABORT_RETREAT")
            # Falling edge (grasp just finished) -> restart the warm-up so the
            # post-grasp navigation does not instantly lock onto a goal.
            if self._prev_grasp_exec and not grasp_exec_now:
                self._belief_warmup_start = time.time()
            self._prev_grasp_exec = grasp_exec_now

            if not grasp_exec_now:
                # engagement: how actively the user is moving (linear speed).
                # When moving strongly -> full learning rate (beta, ema_alpha as tuned).
                # When nearly still -> learning slows down but does NOT stop: the
                # proximity/direction signal is weaker but still informative (the
                # nearest goal slowly climbs). This fixes the "50/50 deadlock near
                # aligned goals" where the user is stationary but clearly closer to
                # one goal than the other.
                speed = float(np.linalg.norm(self.current_v_h[0:3]))
                engagement = float(np.clip(
                    (speed - self.BELIEF_V_LOW) / max(self.BELIEF_V_HIGH - self.BELIEF_V_LOW, 1e-6),
                    0.0, 1.0))
                # Floor the engagement at a small but nonzero value so the belief
                # never fully freezes — a slow trickle of evidence keeps flowing.
                engagement = max(engagement, 0.05)
                warmup = float(np.clip(
                    (time.time() - self._belief_warmup_start) / self.BELIEF_WARMUP_S,
                    self.BELIEF_WARMUP_FLOOR, 1.0))
                # Position-distance cost for each active goal, anchored at the
                # REFERENCE pose (current_T_user) to match the reference-anchored
                # user policy and the reference twist (current_v_h). Breaks the
                # degeneracy when policies are aligned (e.g. Red_Side / Blue_Side):
                # the goal nearest the reference has the lower cost and its belief
                # slowly climbs even when the user is still. (Test mode:
                # current_T_user == current_T_EE, so this is unchanged there.)
                pos_costs = {}
                for key in self.target_keys:
                    if key in excluded:
                        continue
                    T_g = self.goal_set.get_dynamic_goal_pose(
                        self.current_T_EE, key, update_memory=False)
                    pos_costs[key] = float(np.sum(
                        (self.current_T_EE[:3, 3] - T_g[:3, 3]) ** 2))
                self.belief_estimator.update(
                    self.current_v_h, user_policies,
                    gain=engagement * warmup, pos_costs=pos_costs)
                # else: user still -> FREEZE belief (hold history, no decay)

            self.plot_manager.push_beliefs(
                self.belief_estimator.get_beliefs(),
                self.belief_estimator.get_excluded_goals())

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

        # --- 3. ERROR EVALUATION (grasp condition) ---
        # Two deliberately DIFFERENT anchors here, do not "unify" them:
        #   * The GOAL is defined w.r.t. the REFERENCE (current_T_user) — it is the
        #     user's intention. Defining it from the robot EE instead would let the
        #     goal chase the lagging robot and the user's intent would not be met.
        #   * The CONDITION is checked on the REAL ROBOT pose (current_T_EE) vs that
        #     goal. So the operator can ONLY trigger a grasp once the real robot has
        #     actually been steered (via the reference) into the target config —
        #     i.e. they must hold the reference so the robot converges to it.
        # update_memory=False: the policy loop already committed the sticky choice
        # for this key this tick, so this only reads it.
        T_active_goal = self.goal_set.get_dynamic_goal_pose(
            self.current_T_EE, self.active_goal_key, approach_offset=0.05,
            update_memory=False)
        p_EE = self.current_T_EE[:3, 3]        # REAL robot — grasp condition anchor
        p_goal = T_active_goal[:3, 3]
        pos_error = np.linalg.norm(p_goal - p_EE)
        R_EE = self.current_T_EE[:3, :3]       # REAL robot — grasp condition anchor
        R_goal = T_active_goal[:3, :3]

        if self.TASK_DIM == 5:
            ang_error = np.linalg.norm(np.cross(R_EE[:, 0], R_goal[:, 0]))
        else:
            R_rel = R_goal @ R_EE.T
            trace = np.trace(R_rel)
            if trace <= -1.0 + 1e-4:
                ang_error = np.pi
            else:
                ang_error = np.linalg.norm(pin.log3(R_rel))

        # Capture and consume the trigger event
        trigger_pulled = self.trigger_cmd
        self.trigger_cmd = False

        # Block human arm input during active grasp execution + lift.
        # The Haption input is disconnected: the grasp state machine drives the
        # arm autonomously through approach, close and lift. Teleoperation
        # resumes automatically once HOLDING is reached.
        if self.grasp_sm.state in ("GRASP_ALIGN", "GRASP_APPROACH", "GRASP_CLOSE", "LIFT", "RELEASE_LIFT", "ABORT_RETREAT"):
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
            current_force_mag=0.0,
            current_force_local=np.zeros(3),
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
            # On ATTACH command, also weld the cylinder via the Gazebo plugin and
            # reconfigure the shared-autonomy goal set for the HOLDING phase.
            if tick_output.gripper_cmd.startswith("ATTACH_"):
                parts = tick_output.gripper_cmd.split('_')
                arm = parts[1].lower()
                grasped = parts[2].lower()
                self._plugin_attach(arm, grasped)
                self._configure_post_grasp(grasped)

        if tick_output.reset_trigger:
            self.trigger_cmd = False

        # Release / placement: open the gripper, detach the payload and fall back
        # to SHARED_AUTONOMY (the system behaves as if freshly started, now aware
        # the world has one cylinder already placed).
        if tick_output.release_object:
            self._release_object()

        if self.BLENDING and tick_output.new_state == "SHARED_AUTONOMY":
            alpha = self.compute_alpha(b_max)
            target_twist = (1 - alpha) * self.current_v_h + alpha * tick_output.target_twist
        else:
            target_twist = tick_output.target_twist

        # --- AUTHORITY HANDOVER + HAPTIC FIXTURE STATE ---
        # During autonomous grasp execution (approach/close/lift) the node DRIVES
        # the arm directly (see section 6) and the Haption teleop must yield.
        grasp_exec = self.grasp_sm.state in (
            "GRASP_ALIGN", "GRASP_APPROACH", "GRASP_CLOSE", "LIFT", "RELEASE_LIFT", "ABORT_RETREAT")
        self.pub_grasp_active.publish(Bool(data=grasp_exec))
        # Latched active-arm state: late-joining nodes (teleop, force manager)
        # always know which arm is being controlled.
        self.pub_active_arm.publish(String(data=self.active_arm))

        # Active goal pose + confidence for the haptic position virtual fixture.
        # Confidence is forced to 0 during grasp execution so the fixture releases
        # while the arm is being driven autonomously.
        fix_conf = 0.0 if grasp_exec else float(b_max)
        gp = T_active_goal[:3, 3]
        # Suppress scipy's "Gimbal lock detected" UserWarning: at pitch = ±90°
        # (e.g. the Top-grasp / Platform poses where the gripper points straight
        # down) the xyz-Euler decomposition is non-unique, but it still round-trips
        # correctly through from_euler on the haptic side, so the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grpy = R.from_matrix(T_active_goal[:3, :3]).as_euler('xyz')
        self.pub_active_goal_pose.publish(Float64MultiArray(
            data=[float(gp[0]), float(gp[1]), float(gp[2]),
                  float(grpy[0]), float(grpy[1]), float(grpy[2]), fix_conf]))

        # --- 5. LOCAL INTEGRATION & VISUALIZATION ---
        viz_active = not np.allclose(self.current_T_EE, np.eye(4))
        if not viz_active:
            self._viz_miss_count += 1
            if self._viz_miss_count == 50:  # print once after 0.5s of missed viz
                self.get_logger().warn(
                    "[VIZ] Markers NOT publishing: EE pose is still identity "
                    "(robot_state_callback not yet received?).")
        else:
            if self._viz_miss_count > 0:
                self.get_logger().info(
                    f"[VIZ] Resumed after {self._viz_miss_count} missed ticks.")
                self._viz_miss_count = 0

            self._viz_counter += 1
            if self._viz_counter >= self.VIZ_DECIM:
                self._viz_counter = 0

                visual_dt = 0.5
                trajectory_data = []
                active_v_geo = self.compute_v_geo(self.current_T_EE, T_active_goal)
                T_cube_1 = self.integrate_twist(self.current_T_EE, target_twist, visual_dt)
                trajectory_data.append((T_cube_1, target_twist))

                sim_T_EE = T_cube_1
                if in_free_space and valid_matrices:
                    for _ in range(1):
                        visual_dt = visual_dt + 0.3
                        T_sim_goal = self.goal_set.get_dynamic_goal_pose(
                            sim_T_EE, self.active_goal_key, update_memory=False)
                        sim_v_geo = self.compute_v_geo(sim_T_EE, T_sim_goal)
                        sim_twist = self.solve_local_policy(sim_v_geo, self.J_c, self.h_c)
                        sim_T_next = self.integrate_twist(sim_T_EE, sim_twist, visual_dt)
                        trajectory_data.append((sim_T_next, sim_twist))
                        sim_T_EE = sim_T_next

                # (Diagnostic removed: offset < 5mm warning was spamming the console
                # without actionable info — the green gripper simply overlaps the
                # robot mesh when the policy twist is small/rotational-only.)

                # --- SINGLE CONSOLIDATED MARKER PUBLISH ---
                # All marker types are collected into ONE MarkerArray and published
                # in a single call. This halves the message rate on the topic
                # (was ~100 msg/s with separate publishes) and eliminates RViz
                # subscriber queue starvation that caused the green gripper to
                # appear non-updating while goal markers refreshed fine.
                combined_markers = MarkerArray()

                # Green policy grippers (predictive trajectory)
                combined_markers.markers.extend(
                    self._build_policy_gripper_markers(trajectory_data))

                # Goal poses as belief-opacity gripper markers
                beliefs = self.belief_estimator.get_beliefs()
                combined_markers.markers.extend(
                    self._build_goal_pose_markers(beliefs))

                # PRE_GRASP visual cue: pulsing green sphere + one-shot console msg
                if self.grasp_sm.state == "PRE_GRASP":
                    combined_markers.markers.extend(
                        self._build_grasp_ready_cue(self.current_T_EE))
                    if not getattr(self, '_pregrasp_cue_logged', False):
                        self._pregrasp_cue_logged = True
                        self.get_logger().info(
                            "=== [PRE-GRASP READY] Aligned! Press LEFT BUTTON on Haption to execute grasp. ===")
                else:
                    if getattr(self, '_pregrasp_cue_logged', False):
                        self._pregrasp_cue_logged = False
                        combined_markers.markers.extend(self._build_clear_grasp_ready_cue())

                # Grasp-guidance arrows: how to move/rotate to satisfy the grasp
                # condition. Only for cylinder grasp goals (not the Platform), and
                # only while still free (SHARED_AUTONOMY / PRE_GRASP).
                is_cylinder_goal = self.active_goal_key.split('_')[0] in self.goal_set.cylinders
                if is_cylinder_goal and self.grasp_sm.state in ("SHARED_AUTONOMY", "PRE_GRASP"):
                    combined_markers.markers.extend(
                        self._build_grasp_guidance(self.current_T_EE, T_active_goal, pos_error, ang_error))
                    self._guidance_shown = True
                elif getattr(self, '_guidance_shown', False):
                    self._guidance_shown = False
                    combined_markers.markers.extend(self._build_clear_grasp_guidance())

                # ONE publish per tick — all markers in a single message.
                self.pub_markers.publish(combined_markers)

                # --- YELLOW GUIDANCE GRIPPER (separate topic, toggleable) ------ #
                # Reference-anchored counterpart to the green robot-policy gripper.
                # It integrates the belief-weighted USER-policy blend (pi_blend) from
                # the REFERENCE pose (current_T_user) — i.e. the exact velocity field
                # that the haptic F_guide renders onto the handle. Published on its
                # own topic (/guidance_policy_marker) so it can be shown/hidden in
                # RViz independently of the robot-policy markers.
                if user_policies and all(k in user_policies for k in self.target_keys):
                    try:
                        pi_blend_user = self.belief_estimator.blend_policies(user_policies)
                    except KeyError:
                        pi_blend_user = None
                    if pi_blend_user is not None:
                        guidance_markers = MarkerArray()
                        guid_now = self.get_clock().now().to_msg()
                        T_guid_1 = self.integrate_twist(self.current_T_user, pi_blend_user, 0.5)
                        guidance_markers.markers.extend(
                            self.create_gripper_markers(
                                T_guid_1, 0.85, 0, guid_now,
                                ns="guidance_policy", rgb=(1.0, 0.85, 0.0)))
                        self.pub_guidance_marker.publish(guidance_markers)

            # Inference state (consumed by the haptic force manager) is lightweight
            # Float64MultiArray traffic — keep it at the full control rate.
            self.publish_inference_state(ee_policies, user_policies)

        # --- 6. PUBLISH COMMAND TO ROBOT ---
        # Test mode: the node is the sole reference source (always publishes).
        # Teleop mode: the node ONLY publishes during autonomous grasp execution,
        # taking authority from the Haption clutch (which freezes on grasp_active).
        publish_cmd = self.POLICY_BELIEF_TEST or grasp_exec
        if publish_cmd and not np.allclose(self.current_T_EE, np.eye(4)):
            # Virtual Haptic Cursor: integrate the optimal policy twist a short
            # distance into the future so the QP's CLF always has a moving,
            # reachable carrot to track (sending current pose stalls tracking;
            # sending the final goal causes a jerk).
            # NOTE: 0.02s is conservative. Larger values (e.g. 0.1) cause the
            # reference to fling far when the twist reverses (CBF repulsion), which
            # made the green gripper fly away from the goal. 0.02s = 2mm lead at
            # 0.1 m/s — just enough for the CLF to track smoothly.
            dt_virtual = 0.02
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

    def create_gripper_markers(self, T_pose, opacity, step_index, now, ns="policy_grippers", rgb=(0.0, 1.0, 0.0)):
        """Builds a 3-part generic gripper with X as the approach axis."""
        markers = []
        p_center = T_pose[:3, 3]
        R_mat = T_pose[:3, :3]
        quat = R.from_matrix(R_mat).as_quat()
        cr, cg, cb = rgb

        # 1.5 s auto-expiry: long enough to ride out jitter in the publish rate
        # (e.g. a brief /collision_constraints stall) so the policy gripper does
        # NOT blink out of RViz, but still auto-clears if the node truly stops.
        lifetime_sec = 1
        lifetime_nsec = 500000000  # -> 1.5 s total

        base = Marker()
        base.header.frame_id = "base_footprint"
        base.header.stamp = now
        base.ns = ns
        base.id = step_index * 3
        base.type = Marker.CUBE
        base.action = Marker.ADD
        base.lifetime.sec = lifetime_sec
        base.lifetime.nanosec = lifetime_nsec
        base.pose.position.x, base.pose.position.y, base.pose.position.z = p_center[0], p_center[1], p_center[2]
        base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w = quat[0], quat[1], quat[2], quat[3]
        base.scale.x, base.scale.y, base.scale.z = 0.02, 0.08, 0.03
        base.color.r, base.color.g, base.color.b, base.color.a = cr, cg, cb, opacity
        markers.append(base)

        offset_l = np.array([0.03, 0.035, 0.0])
        p_left = p_center + (R_mat @ offset_l)

        left = Marker()
        left.header = base.header
        left.ns = base.ns
        left.id = step_index * 3 + 1
        left.type = Marker.CUBE
        left.action = Marker.ADD
        left.lifetime = base.lifetime
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
        right.lifetime = base.lifetime
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
        arrow.lifetime.sec = 0
        arrow.lifetime.nanosec = 500000000  # 500 ms auto-expiry
        arrow.points = [
            Point(x=start[0], y=start[1], z=start[2]),
            Point(x=end[0], y=end[1], z=end[2]),
        ]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = scale
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = rgba
        return arrow

    def _build_policy_gripper_markers(self, trajectory_data):
        """Builds the fading Prediction Grippers (green) — returns markers list."""
        markers = []
        now = self.get_clock().now().to_msg()

        for i, (T_cube, v_cmd) in enumerate(trajectory_data):
            opacity = max(0.2, 0.8 - (i * 0.3))
            gripper_markers = self.create_gripper_markers(T_cube, opacity, i, now)
            markers.extend(gripper_markers)

        return markers

    def publish_visualizations(self, trajectory_data, T_EE, v_geo):
        """Publishes the fading Prediction Grippers (green) only — arrows removed for bandwidth."""
        marker_array = MarkerArray()
        marker_array.markers.extend(self._build_policy_gripper_markers(trajectory_data))
        self.pub_markers.publish(marker_array)

    # Per-goal-family color (matches the world colors) for the belief-opacity markers.
    GOAL_FAMILY_RGB = {
        'Red': (1.0, 0.2, 0.2),
        'Blue': (0.2, 0.4, 1.0),
        'Platform': (1.0, 0.85, 0.0),  # yellow placement disk
    }

    @staticmethod
    def _belief_to_opacity(belief):
        """Continuous opacity ramp: 0.2 at belief 0 → 0.8 at belief 1 (clamped)."""
        b = max(0.0, min(1.0, float(belief)))
        return 0.2 + 0.6 * b

    def _build_goal_pose_markers(self, beliefs, now=None):
        """Build every goal pose as a gripper marker whose opacity tracks its belief.

        Returns a list of Marker objects (not a MarkerArray).
        """
        if now is None:
            now = self.get_clock().now().to_msg()
        markers = []
        for i, goal_key in enumerate(self.target_keys):
            family = goal_key.split('_')[0]
            rgb = self.GOAL_FAMILY_RGB.get(family, (0.0, 1.0, 0.0))
            opacity = self._belief_to_opacity(beliefs.get(goal_key, 0.0))
            # Draw each goal at the manifold point anchored at the REAL EE pose
            # (same anchor as the policy loop), so the RViz goal grippers match
            # where the system is actually steering. update_memory=False (read-only).
            T_goal = self.goal_set.get_dynamic_goal_pose(
                self.current_T_EE, goal_key, update_memory=False)
            markers.extend(
                self.create_gripper_markers(T_goal, opacity, i, now,
                                            ns="goal_poses", rgb=rgb))
        return markers

    def publish_goal_pose_markers(self, beliefs, now=None):
        """Draw every goal pose as a gripper marker whose opacity tracks its belief.

        Replaces the per-goal TF frames (which cannot fade and went stale when a
        goal was excluded). Every goal is drawn every tick, so low/zero-belief
        goals (e.g. the just-grasped cylinder, or the Platform while empty) simply
        fade toward 0.2 opacity instead of cluttering RViz — no state-machine viz
        logic, just a continuous function of the belief estimator's output.
        """
        marker_array = MarkerArray()
        marker_array.markers.extend(self._build_goal_pose_markers(beliefs, now))
        self.pub_markers.publish(marker_array)

    def _build_grasp_ready_cue(self, T_EE):
        """Build the pulsing green sphere marker — returns list of Marker."""
        import math
        self._grasp_cue_phase += 0.05
        pulse = 0.55 + 0.25 * math.sin(self._grasp_cue_phase * 2.0 * math.pi)

        # Place the sphere at the top of the cylinder being grasped.
        color = self.active_goal_key.split('_')[0]
        if color in self.goal_set.cylinders:
            cyl = self.goal_set.cylinders[color]
            cue_pos = cyl['pos'] + np.array([0.0, 0.0, cyl['height'] / 2.0 + 0.08])
        else:
            cue_pos = T_EE[:3, 3]

        m = Marker()
        m.header.frame_id = "base_footprint"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "grasp_ready_cue"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(cue_pos[0])
        m.pose.position.y = float(cue_pos[1])
        m.pose.position.z = float(cue_pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.10
        m.scale.y = 0.10
        m.scale.z = 0.10
        m.color.r = 0.1
        m.color.g = 1.0
        m.color.b = 0.1
        m.color.a = float(pulse)
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500000000  # 500 ms — auto-expires if we stop publishing
        return [m]

    def _build_clear_grasp_ready_cue(self):
        """Build a DELETE marker for the pulsing sphere — returns list of Marker."""
        m = Marker()
        m.header.frame_id = "base_footprint"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "grasp_ready_cue"
        m.id = 0
        m.action = Marker.DELETE
        return [m]

    def publish_grasp_ready_cue(self, T_EE):
        """Publish a pulsing green sphere above the active-goal cylinder when PRE_GRASP is active.

        Positioned at the TOP of the target cylinder (not the EE), so it's clearly
        visible and not occluded by the gripper mesh. Pulses between opacity 0.3
        and 0.8 at ~1 Hz.
        """
        ma = MarkerArray()
        ma.markers.extend(self._build_grasp_ready_cue(T_EE))
        self.pub_markers.publish(ma)

    def clear_grasp_ready_cue(self):
        """Remove the pulsing sphere when leaving PRE_GRASP."""
        ma = MarkerArray()
        ma.markers.extend(self._build_clear_grasp_ready_cue())
        self.pub_markers.publish(ma)

    def publish_grasp_guidance(self, T_EE, T_goal, pos_error, ang_error, now=None):
        """Draw 'how to move to satisfy the grasp condition' cues on the gripper (legacy wrapper)."""
        ma = MarkerArray()
        if now is None:
            now = self.get_clock().now().to_msg()
        ma.markers.extend(self._build_grasp_guidance_markers(T_EE, T_goal, pos_error, ang_error, now))
        self.pub_markers.publish(ma)

    def clear_grasp_guidance(self):
        """Remove all grasp-guidance arrows (legacy wrapper)."""
        ma = MarkerArray()
        ma.markers.extend(self._build_clear_grasp_guidance())
        self.pub_markers.publish(ma)

    def _build_grasp_guidance_markers(self, T_EE, T_goal, pos_error, ang_error, now=None):
        """Core builder for grasp guidance markers — returns list of Marker.

        Appears when within 2x the PRE_GRASP-ready range (and the target is a
        cylinder grasp goal):
          - a GREEN straight arrow from the EE to the standoff goal position
            (where to translate), shown while position is not yet within range;
          - an ORANGE curved arrow (arc) around the rotation-error axis at the EE
            (which way to rotate), shown while the approach axis is not aligned.
        Each cue turns to a brighter colour once its own sub-condition is met, and
        is DELETEd individually so the operator gets unambiguous per-axis feedback
        (no POV ambiguity like the static red ghost gripper).
        """
        if now is None:
            now = self.get_clock().now().to_msg()
        markers = []

        pos_enter = self.grasp_sm.POS_ERR_ENTER
        ang_enter = self.grasp_sm.ANG_ERR_ENTER
        p_ee = T_EE[:3, 3]
        p_goal = T_goal[:3, 3]

        # --- POSITION ARROW (EE -> standoff) ---
        show_pos = (pos_error < 2.0 * pos_enter) and (pos_error > 0.012)
        a = Marker()
        a.header.frame_id = "base_footprint"
        a.header.stamp = now
        a.ns = "grasp_guidance_pos"
        a.id = 0
        a.type = Marker.ARROW
        if show_pos:
            a.action = Marker.ADD
            a.points = [
                Point(x=float(p_ee[0]), y=float(p_ee[1]), z=float(p_ee[2])),
                Point(x=float(p_goal[0]), y=float(p_goal[1]), z=float(p_goal[2])),
            ]
            a.scale.x, a.scale.y, a.scale.z = 0.008, 0.02, 0.025
            within = pos_error < pos_enter
            a.color.r, a.color.g, a.color.b, a.color.a = (
                (0.2, 1.0, 0.2, 0.95) if within else (1.0, 0.9, 0.1, 0.95))
        else:
            a.action = Marker.DELETE
        markers.append(a)

        # --- ORIENTATION ARC (curved arrow around the rotation-error axis) ---
        R_ee = T_EE[:3, :3]
        R_goal = T_goal[:3, :3]
        err_rotvec = R.from_matrix(R_goal @ R_ee.T).as_rotvec()
        ang = float(np.linalg.norm(err_rotvec))
        show_ang = (ang_error < 2.0 * ang_enter) and (ang > 0.06)

        arc = Marker()
        arc.header.frame_id = "base_footprint"
        arc.header.stamp = now
        arc.ns = "grasp_guidance_rot"
        arc.id = 0
        arc.type = Marker.LINE_STRIP
        head = Marker()
        head.header.frame_id = "base_footprint"
        head.header.stamp = now
        head.ns = "grasp_guidance_rot"
        head.id = 1
        head.type = Marker.ARROW

        if show_ang and ang > 1e-6:
            axis = err_rotvec / ang
            # A reference vector perpendicular to the rotation axis.
            seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ref = np.cross(axis, seed)
            ref = ref / np.linalg.norm(ref)
            radius = 0.07
            n_seg = 18
            arc.points = []
            for i in range(n_seg + 1):
                th = ang * (i / n_seg)
                v = R.from_rotvec(axis * th).apply(ref)
                pt = p_ee + radius * v
                arc.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
            arc.scale.x = 0.006  # line width
            within_a = ang_error < ang_enter
            arc.color.r, arc.color.g, arc.color.b, arc.color.a = (
                (0.2, 1.0, 0.2, 0.95) if within_a else (1.0, 0.55, 0.0, 0.95))
            arc.action = Marker.ADD

            # Arrowhead: a short cone from the second-last to the last arc point.
            p_tip = p_ee + radius * R.from_rotvec(axis * ang).apply(ref)
            v_prev = R.from_rotvec(axis * (ang * (n_seg - 1) / n_seg)).apply(ref)
            p_prev = p_ee + radius * v_prev
            tangent = p_tip - p_prev
            tn = np.linalg.norm(tangent)
            tangent = tangent / tn if tn > 1e-9 else axis
            head.action = Marker.ADD
            head.points = [
                Point(x=float(p_tip[0]), y=float(p_tip[1]), z=float(p_tip[2])),
                Point(x=float(p_tip[0] + 0.025 * tangent[0]),
                      y=float(p_tip[1] + 0.025 * tangent[1]),
                      z=float(p_tip[2] + 0.025 * tangent[2])),
            ]
            head.scale.x, head.scale.y, head.scale.z = 0.006, 0.02, 0.02
            head.color = arc.color
        else:
            arc.action = Marker.DELETE
            head.action = Marker.DELETE

        markers.append(arc)
        markers.append(head)
        return markers

    def _build_grasp_guidance(self, T_EE, T_goal, pos_error, ang_error, now=None):
        """Build grasp guidance markers — returns list of Marker."""
        if now is None:
            now = self.get_clock().now().to_msg()
        return self._build_grasp_guidance_markers(T_EE, T_goal, pos_error, ang_error, now)

    def _build_clear_grasp_guidance(self):
        """Build DELETE markers for grasp guidance — returns list of Marker."""
        markers = []
        for ns, mid in (("grasp_guidance_pos", 0), ("grasp_guidance_rot", 0), ("grasp_guidance_rot", 1)):
            m = Marker()
            m.header.frame_id = "base_footprint"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            markers.append(m)
        return markers

    def publish_grasp_guidance(self, T_EE, T_goal, pos_error, ang_error, now=None):
        """Draw 'how to move to satisfy the grasp condition' cues on the gripper (legacy wrapper)."""
        ma = MarkerArray()
        if now is None:
            now = self.get_clock().now().to_msg()
        ma.markers.extend(self._build_grasp_guidance_markers(T_EE, T_goal, pos_error, ang_error, now))
        self.pub_markers.publish(ma)

    def clear_grasp_guidance(self):
        """Remove all grasp-guidance arrows (legacy wrapper)."""
        ma = MarkerArray()
        ma.markers.extend(self._build_clear_grasp_guidance())
        self.pub_markers.publish(ma)

    def compute_v_geo(self, T_EE, T_goal, v_max_lin=None, w_max_ang=None):
        """Computes the LOCAL_WORLD_ALIGNED decoupled spatial velocity error with
        purely smooth saturation (no deadband).

        Optional v_max_lin / w_max_ang override the instance defaults, allowing
        the user-policy path to use lower (teleop-friendly) velocity limits
        without affecting the robot-policy path.
        """
        if v_max_lin is None:
            v_max_lin = self.v_max_lin
        if w_max_ang is None:
            w_max_ang = self.w_max_ang

        p_EE = T_EE[:3, 3]
        p_goal = T_goal[:3, 3]
        R_EE = T_EE[:3, :3]
        R_goal = T_goal[:3, :3]

        # Translation: pure smooth proportional (tanh)
        error_lin = p_goal - p_EE
        dist = np.linalg.norm(error_lin)

        if dist > 1e-5:
            v_mag = v_max_lin * np.tanh((self.K_p_lin * dist) / v_max_lin)
            v_linear = (error_lin / dist) * v_mag
        else:
            v_linear = np.zeros(3)

        # Rotation: dynamic task dimension (SO(3) vs S^2)
        if self.TASK_DIM == 5:
            error_ang = np.cross(R_EE[:, 0], R_goal[:, 0])
        else:
            # pin.log3 is undefined (NaN) when the relative rotation is exactly π
            # (trace(R_rel) = -1). Detect this from the trace and fall back to a
            # safe approximation (the cross-product of an arbitrary column pair,
            # which gives the correct axis and is bounded).
            R_rel = R_goal @ R_EE.T
            trace = np.trace(R_rel)
            if trace <= -1.0 + 1e-4:
                # Near-π singularity: use the first non-degenerate column cross
                error_ang = np.cross(R_EE[:, 0], R_goal[:, 0])
                n = np.linalg.norm(error_ang)
                if n < 1e-6:
                    error_ang = np.cross(R_EE[:, 1], R_goal[:, 1])
                    n = np.linalg.norm(error_ang)
                if n > 1e-6:
                    error_ang = error_ang / n * np.pi  # magnitude = π
                else:
                    error_ang = np.zeros(3)
            else:
                error_ang = pin.log3(R_rel)

        ang_dist = np.linalg.norm(error_ang)

        if ang_dist > 1e-5:
            w_mag = w_max_ang * np.tanh((self.K_p_ang * ang_dist) / w_max_ang)
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
                if raw in self.belief_estimator.get_excluded_goals():
                    print(f"  ✗ '{raw}' is not demandable right now (excluded).")
                else:
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
                self.trigger_cmd = True
                self.get_logger().info("[TEST] 'CLOSE' command registered via console.")

            elif raw == "OPEN":
                # Open gripper, detach the payload and reset to the start phase
                # (delegates to the shared release routine used by the trigger).
                self._release_object()
                print("  ✓ released: gripper opened, object detached, reset to SHARED_AUTONOMY.")

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
