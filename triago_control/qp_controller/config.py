# config.py
"""Centralized configuration for the TRIAGo bimanual QP-CLF-CBF controller: every flag, gain and name lives here."""

import numpy as np

# =============================================================================
# 1. BOOLEAN FEATURE FLAGS
# =============================================================================
# WALL_COLLIDER applies to the legacy path only (world_scene=None); YAML worlds set their own wall.
WALL_COLLIDER = False          # virtual collision wall (XZ plane)
DEBUG = False                   # verbose timing / kinematics console tracing
GRASP_DEBUG = True              # verbose grasp / CBF-bypass interaction tracing
DISABLE_CBF = False             # mathematically delete the collision barrier
DYNAMIC_CBF = False             # dynamically remove pairs for interaction
DYNAMIC_SLACK_WEIGHT = True    # raise slack weight in free space, drop it near obstacles
COMPARISON_CLF = True           # normalized (unit-error) scalar CLF formulation
DYNAMIC_GAMMA_CLF = False       # vary CLF convergence rate with the safety margin
DYNAMIC_POSTURE_WEIGHT = False  # per-joint posture weight rises with that joint's own limit lambda; also drops lambda_jl from the slack/gamma driver (OFF = today)
SIMULATE_IDEAL_KINEMATICS = False  # True = pure-math digital twin instead of measured state
ORIENTATION_CTRL = True         # True = 6DOF (pos+ori), False = 3DOF (pos only)

# =============================================================================
# 1b. EXPERIMENT CONDITION SELECTOR (2x2x2 user-study factorial)
# =============================================================================
# Three orthogonal factors select one of 8 study cells:
#   CONTROL_MODE    -- CLUTCH (position control) vs JOYSTICK (velocity control)
#   ASSIST_FEEDBACK -- channel F: assistive guidance forces on the handle
#   ASSIST_BLENDING -- channel B: reference-level user<->policy blending (robot side)
#
#   MODE      F      B      condition          force manager
#   CLUTCH    False  False  Sync only          haptic_force_manager_C
#   CLUTCH    True   False  Guided feedback    haptic_force_manager_CF
#   CLUTCH    False  True   Guided blending    haptic_force_manager_CB
#   CLUTCH    True   True   Full guidance      haptic_force_manager_CFB
#   JOYSTICK  False  False  Sync only          haptic_force_manager_J
#   JOYSTICK  True   False  Guided feedback    haptic_force_manager_JF
#   JOYSTICK  False  True   Guided blending    haptic_force_manager_JB
#   JOYSTICK  True   True   Full guidance      haptic_force_manager_JFB
#
# Every teleop/force-manager node calls validate_condition() at startup and hard-errors
# on mismatch, so a mis-launched experiment fails loudly instead of running the wrong cell.
CLUTCH = "CLUTCH"
JOYSTICK = "JOYSTICK"

# Active experiment condition: edit these three to select a study cell.
CONTROL_MODE   = CLUTCH      # CLUTCH (position control) | JOYSTICK (velocity control)
ASSIST_FEEDBACK = True          # channel F: assistive guidance forces on the handle
ASSIST_BLENDING = True      # channel B: reference-level user<->policy blending

# Backward-compat alias: legacy code reads BLENDING to decide reference-topic ownership.
BLENDING = ASSIST_BLENDING


def validate_condition(node_name, control_mode=None, feedback=None, blending=None):
    """Startup guard: raises RuntimeError unless the launched node matches the selected study cell."""
    errors = []
    if control_mode is not None and CONTROL_MODE != control_mode:
        errors.append(f"requires CONTROL_MODE == {control_mode!r}, but config has {CONTROL_MODE!r}")
    if feedback is not None and ASSIST_FEEDBACK != feedback:
        errors.append(f"requires ASSIST_FEEDBACK == {feedback}, but config has {ASSIST_FEEDBACK}")
    if blending is not None and ASSIST_BLENDING != blending:
        errors.append(f"requires ASSIST_BLENDING == {blending}, but config has {ASSIST_BLENDING}")
    if errors:
        raise RuntimeError(
            f"[CONDITION GUARD] Node '{node_name}' cannot run under the current "
            f"config.py experiment condition "
            f"(CONTROL_MODE={CONTROL_MODE!r}, ASSIST_FEEDBACK={ASSIST_FEEDBACK}, "
            f"ASSIST_BLENDING={ASSIST_BLENDING}):\n  - "
            + "\n  - ".join(errors)
            + "\nEdit the condition selector in config.py (section 1b) to match, "
            "or launch the correct node for this condition.")

# --- Joystick home pose (Haption base frame) -------------------------------
# Fixed home position; home ORIENTATION tracks the gripper via a per-arm reference delta
# captured once at first activation, angle-compressed by JOYSTICK_ROT_HOME_SCALE, never re-anchored.
# Device workspace is x[0.14,0.36] y[-0.24,0.22] z[-0.18,0.18] m; this operator rest pose keeps at
# least 9.7 cm of handle travel on every axis, the tightest direction being +x.
JOYSTICK_NEUTRAL_POSITION_M = [0.2624, 0.0103, -0.0729]
# Neutral handle quaternion, measured on the device at the operator's comfortable rest orientation.
JOYSTICK_NEUTRAL_ORIENTATION_XYZW = [-0.015140674076974392, 0.8170770406723022,
                                     0.06124841421842575, 0.5730659365653992]

# --- Displacement -> twist mapping (teleop_triago_joystick.py) --------------
# Twist magnitude is proportional to the handle's distance from home past a radial deadband.
# Each deadband must stay above the centering spring's settle precision, or the residual
# settle-oscillation of a released handle is read as spurious user input.
JOYSTICK_DEADBAND_LIN = 0.03456      # m   (3.46 cm)
JOYSTICK_DEADBAND_ANG = 0.23328      # rad (~13.37 deg)
JOYSTICK_K_TRANS = 1.6               # (m/s) per m of handle linear displacement
JOYSTICK_K_ROT = 1.5                 # (rad/s) per rad of handle angular displacement
JOYSTICK_V_MAX_LIN = 0.10            # m/s   hard safety clamp on the commanded linear twist
JOYSTICK_V_MAX_ANG = 0.50            # rad/s hard safety clamp on the commanded angular twist

# --- Viscous damping on the commanded twist ---------------------------------
# Dimensionless fraction of handle velocity subtracted from the command, applied only outside
# the deadband (a still/settling handle still commands exactly zero). Keep well below 1.0.
JOYSTICK_DAMP_LIN = 0.225            # unitless fraction of handle linear velocity
JOYSTICK_DAMP_ANG = 0.075            # unitless fraction of handle angular velocity

# --- Home-orientation rebasing scale ---------------------------------------
# Gripper rotation is divided by this when building the home orientation (gripper 90 deg ->
# handle 60 deg), fitting the Haption's narrower rotational workspace; never scales the twist.
JOYSTICK_ROT_HOME_SCALE = 1.5

# --- Restorative centering spring (joystick force managers) -----------------
# The homing spring-damper toward the dynamic home pose, unified across all joystick cells.
JOYSTICK_SPRING_KP_LIN = 30.0        # N/m
JOYSTICK_SPRING_KD_LIN = 0.5         # N/(m/s)
JOYSTICK_SPRING_KP_ANG = 0.75        # Nm/rad
JOYSTICK_SPRING_KD_ANG = 0.075       # Nm/(rad/s)

# Live home pose broadcast (teleop -> force manager), layout [pos(3), quat_xyzw(4)];
# the teleop node owns it so the spring target and the twist zero-point are identical.
JOYSTICK_HOME_POSE_TOPIC = "/joystick/home_pose"

# --- Twist arbitration (main_shared_autonomy.compute_alpha) -----------------
# Authority alpha on the policy in v_blend = (1-alpha)*v_user + alpha*pi_policy, driven only by
# the alignment s in [-1,1] between user and policy twists (two-sided ramp, continuous at s=0):
#   s >= 0: alpha = MIN + (MAX-MIN)*s   |   s < 0: alpha = MIN*(1+s)  ->  0 at full opposition.
# A still user (twist inside the deadband) uses the reduced IDLE authority (gentle crawl).
ALIGN_ALPHA_MIN = 0.2                # policy weight at perpendicular alignment (s=0)
ALIGN_ALPHA_MAX = 0.8                # policy weight at full active agreement (s=1)
ALIGN_ALPHA_IDLE = 0.35             # policy weight when the user is still
ALIGN_ALPHA_LPF_COEFF = 0.1          # low-pass on alpha for C0-continuity of the blend

# =============================================================================
# 2. SAFETY + CONTROL HYPERPARAMETERS
# =============================================================================
ALPHA_SOFTMIN = 50.0            # sharpness of the SoftMin collision aggregation
GAMMA_CBF = 0.70               # CBF class-K gain
D_SAFE_BASE = 0.015            # base safety distance for the collision barrier
K_V_SAFE = 0.1                 # predictive velocity horizon (brake earlier at high speed)

# Local-minima stall escape: the reactive QP has no planner, so a goal behind a flat obstacle
# can settle at a saddle point (goal gradient anti-parallel to the barrier gradient).
ENABLE_STALL_ESCAPE = False
STALL_SPEED_FILTER_ALPHA = 0.97 # EMA on the raw EE-speed estimate before the stall check
STALL_SPEED_THRESH = 0.01       # [m/s] "almost zero" measured EE speed
STALL_HOLD_S = 3.0              # [s] sustained stall before escaping
STALL_ERR_POS_THRESH = 0.10     # [m] position error still counting as "not there yet"
STALL_ERR_ANG_THRESH = np.deg2rad(30.0)  # [rad] orientation error, same idea
STALL_RESUME_SPEED = 0.03       # [m/s] speed that counts as "moving again" -> drop escape
STALL_TANGENT_MIN = 0.15        # below this tangential component, use the fallback tangent
STALL_ESCAPE_STEP_M = 0.05      # [m] waypoint offset along the tangent
STALL_ESCAPE_SPEED = 0.03       # [m/s] feedforward speed along the tangent

# Near-goal policy shaping (main_shared_autonomy only; the QP controller is untouched).
POLICY_NEAR_GOAL_POS_M = 0.10                 # [m]
POLICY_NEAR_GOAL_ANG_RAD = np.deg2rad(15.0)   # [rad]
POLICY_NEAR_GOAL_MIN_LEAD_M = 0.05            # [m] stretched carrot-lead floor
POLICY_NEAR_GOAL_DT_MAX_S = 2.0               # [s] stretched dt_virtual cap

# When True, an inter-arm pair's margin also reflects the OTHER arm's speed (cost-neutral option).
ENABLE_INTER_ARM_CLOSING_MARGIN = False
ALPHA_FILTER = 0.5            # EMA coefficient for joint-velocity reconstruction (~20 ms window)
DAMP = 12.0                    # joint-velocity regularization (lambda) in the QP cost
P_GAIN_LIMITS = 1.8            # joint-limit CBF gamma (braking aggressiveness)
JOINT_LIMIT_BUFFER_BASE = 0.15  # base joint-limit braking buffer
JOINT_LIMIT_K_V = 0.1          # joint-limit velocity horizon (seconds of look-ahead)
# Never add a hard slew-rate box on q_dot: it cannot relieve the CBF row and breaks feasibility.

# Posture / joint-limit avoidance: negative gradient of a barrier diverging at each joint's
# limits, on the NORMALIZED joint position; near-zero mid-range, sharp (clamped) near a limit.
K_GRADIENT = 0.07              # gain on the negative potential gradient
V_MAX_POSTURE = 1.0            # rad/s hard clamp on the posture reference (solver safety)
W_CENTER = 0.96                # posture-task weight in the QP cost (used only when DYNAMIC_POSTURE_WEIGHT is off)
POSTURE_GRASP_SCALE = 0.05     # posture weight scale during autonomous precision phases
POSTURE_SCALE_TAU = 0.2        # s first-order ramp for the posture-scale switch

# Dynamic per-joint posture weight (DYNAMIC_POSTURE_WEIGHT). Same exp(-beta*lambda^2)
# tolerance kernel as the slack schedule, blended in the OPPOSITE direction: slack
# decays MAX->BASE as lambda grows, posture climbs BASE->MAX with that joint's own
# joint-limit shadow price, concentrating reconfiguration authority on the joint
# actually fighting its limit.
BASE_WEIGHT_POSTURE = 0.8      # posture weight when the joint is idle (lambda_jl ~ 0)
MAX_WEIGHT_POSTURE = 1.5       # posture weight when the joint is hard against its limit
BETA_POSTURE = 0.05            # kernel sensitivity (own knee; lambda_jl range differs from lambda_cbf)
POSTURE_WEIGHT_FILTER_TAU = 0.2  # 2nd-stage LPF on the posture weight (symmetric with WEIGHT_SLACK_FILTER_TAU)

# Rate damping: + RATE_WEIGHT * ||dq - dq_measured||^2 on the arm joints anchors the
# task-null wrist DOFs to the arm's real state each tick (kills qdot oscillation).
# Anchoring on the QP's own last output instead is a self-referential low-pass: rejected.
ENABLE_RATE_DAMPING = True
RATE_DAMPING_VS_MEASURED = True
RATE_WEIGHT = 50.0          # sim default; real-hw tuned value (500) lives on the real-hw branch
# During fine grasp centering the measured velocity is near zero, so full RATE_WEIGHT resists
# the tiny corrective qdot needed; relaxed to this on the tracking-boosted arm only.
RATE_WEIGHT_GRASP = 20.0    # sim default; real-hw tuned value (200) lives on the real-hw branch

# =============================================================================
# 3. DYNAMIC SCALING BOUNDARIES
# =============================================================================
# Decoupled dynamic slack weighting.
BASE_WEIGHT_SLACK = 62.5        # slack weight when active against an obstacle
MAX_WEIGHT_SLACK = 150.0       # free-space slack weight; also pinned on a frozen arm
BETA = 0.4                     # how fast slack weight returns to baseline as lambda grows
SLACK_FILTER_TAU = 0.15        # LPF on the shadow prices feeding the scheduler
WEIGHT_SLACK_FILTER_TAU = 0.2  # 2nd-stage LPF directly on the slack weight

# Dynamic gamma (CLF) scheduling.
GAMMA_CLF_DEFAULT = 0.75       # static / initial CLF convergence rate
GAMMA_MIN = 0.5                # lower bound of the scheduled CLF gamma
GAMMA_MAX = 1.0                # upper bound of the scheduled CLF gamma
BETA_GAMMA = 5.0               # how quickly gamma drops as the collision lambda grows
GAMMA_FILTER_TAU = 0.125       # LPF time constant for the gamma scheduler

# =============================================================================
# 3b. REFERENCE GOVERNOR (pre-CLF reference shaping)
# =============================================================================
# Bounds reference error/velocity/acceleration so the CLF always sees an admissible target,
# preserving QP feasibility under aggressive or discontinuous commands.
ENABLE_REFERENCE_GOVERNOR = True       # master switch (False = raw passthrough)

GOV_V_MAX_LIN = 0.20                  # [m/s] max linear reference velocity
GOV_V_MAX_ANG = 0.3                   # [rad/s] max angular reference velocity

GOV_E_MAX_POS = 0.30                  # [m] max allowed position error norm
GOV_E_MAX_ORI = 1.0                   # [rad] max allowed orientation error norm

GOV_A_MAX_LIN = 0.75                  # [m/s^2] max linear reference acceleration
GOV_A_MAX_ANG = 3.0                   # [rad/s^2] max angular reference acceleration

# =============================================================================
# 4. LOOP / TELEMETRY SETTINGS
# =============================================================================
CONTROL_FREQ_DEFAULT = 150.0   # control loop frequency [Hz]
PUBLISH_EVERY_N = 2            # publish 1 of every N iterations to the dashboard
WATCHDOG_TIMEOUT = 0.5         # seconds without a reference before motion is frozen
DISTANCE_FILTER_THRESHOLD = 0.15  # ignore collision pairs farther than this [m]
K_MAX_PAIRS = 60               # max number of closest pairs fed into the SoftMin

# --- Real-hardware async execution (main_qp_controller_real.py only) ---------
# The real-hardware subclass moves the SoftMin CBF and RViz overlays onto worker threads;
# a staleness watchdog freezes both arms if the CBF result is unrefreshed too long.
REAL_ASYNC_CBF = True          # run the SoftMin CBF on its own worker thread
REAL_ASYNC_VIZ = True          # run the RViz debug overlays on their own thread
CBF_STALENESS_MAX_TICKS = 3    # freeze both arms after this many stale ticks (auto-resume)

# Task weights [Px,Py,Pz,R,P,Y]: position-dominant 25:1, so orientation yields first near obstacles.
TASK_WEIGHTS_6D = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05]) * 10.0

# Precision-phase task weights (5:1): applied per arm during autonomous grasp/release execution
# and to any arm carrying an attached object, so approach/placement orientation converges tightly.
TASK_WEIGHTS_6D_GRASP = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2]) * 10.0

MESH_PATHS = ["/opt/pal/alum/share", "/opt/ros/humble/share", "/opt/pal/ferrum/share", "."]

# =============================================================================
# 5. ROBOT + CONTROLLER NAMES
# =============================================================================
RIGHT_CONTROLLER = "arm_right_joint_space_controller_vel"
LEFT_CONTROLLER  = "arm_left_joint_space_controller_vel"
CONFLICTING_CONTROLLERS = [
    "arm_right_controller", "arm_left_controller", "arm_head_controller",
    "arm_right_joint_trajectory_controller", "arm_left_joint_trajectory_controller",
]

# Kinematic chains (must match URDF link names).
RIGHT_CHAIN = ['arm_right_1_link', 'arm_right_2_link', 'arm_right_3_link',
               'arm_right_4_link', 'arm_right_5_link', 'arm_right_6_link', 'arm_right_7_link']
LEFT_CHAIN  = ['arm_left_1_link', 'arm_left_2_link', 'arm_left_3_link',
               'arm_left_4_link', 'arm_left_5_link', 'arm_left_6_link', 'arm_left_7_link']

# Active joint names (map the QP output to ROS messages).
RIGHT_JOINTS = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']
LEFT_JOINTS  = ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']
# Head + gripper-finger joints: slider-GUI limit lookup only, never in the arm QP decision vector.
HEAD_JOINTS = ['arm_head_1_joint', 'arm_head_2_joint', 'arm_head_3_joint',
               'arm_head_4_joint', 'arm_head_5_joint', 'arm_head_6_joint', 'arm_head_7_joint']
GRIPPER_FINGER_JOINTS = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

RIGHT_TCP_FRAME = 'gripper_right_grasping_link'
LEFT_TCP_FRAME  = 'gripper_left_grasping_link'
REF_FRAME = 'base_footprint'

# Head chain: quasi-static CBF obstacle only -- live FK capsules, no head joint in the QP.
HEAD_CHAIN = ['arm_head_1_link', 'arm_head_2_link', 'arm_head_3_link',
              'arm_head_4_link', 'arm_head_5_link', 'arm_head_6_link', 'arm_head_7_link']
HEAD_TOOL_LINK = 'arm_head_tool_link'

# Joint-slider GUI layout (shared by the controller's limit publisher and the plotter's grid).
SLIDER_LAYOUT = [
    ['arm_left_1_joint', 'arm_head_1_joint', 'arm_right_1_joint'],
    ['arm_left_2_joint', 'arm_head_2_joint', 'arm_right_2_joint'],
    ['arm_left_3_joint', 'arm_head_3_joint', 'arm_right_3_joint'],
    ['arm_left_4_joint', 'arm_head_4_joint', 'arm_right_4_joint'],
    ['arm_left_5_joint', 'arm_head_5_joint', 'arm_right_5_joint'],
    ['arm_left_6_joint', 'arm_head_6_joint', 'arm_right_6_joint'],
    ['arm_left_7_joint', 'arm_head_7_joint', 'arm_right_7_joint'],
]
GRIPPER_SLIDER_ROW = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

# =============================================================================
# 6. GEOMETRY + WORKSPACE
# =============================================================================
CAPSULE_RADIUS = 0.06                 # default arm-capsule radius (links without an override below)

# =============================================================================
# 6b. PER-LINK CAPSULE ALIGNMENT OVERRIDES
# =============================================================================
# Per-link correction where the CAD mesh isn't collinear with the joint-to-joint capsule
# (mm, joint-local frame). Link 6 intentionally absent: grasp tuning depends on its raw geometry.
_CAPSULE_FIX = {
    1: {'lateral_offset': [-22.46, 9.11, 0.00], 'radius': 70.42, 'proximal_extension': 0.00, 'distal_extension': 0.20},
    2: {'lateral_offset': [8.82, -0.00, -8.67], 'radius': 70.07, 'proximal_extension': 0.00, 'distal_extension': 0.00},
    3: {'lateral_offset': [11.49, 25.66, 0.00], 'radius': 70.97, 'proximal_extension': 0.00, 'distal_extension': 0.20},
    4: {'lateral_offset': [-8.82, 0.00, -8.67], 'radius': 70.07, 'proximal_extension': 0.00, 'distal_extension': 0.00},
    5: {'lateral_offset': [0.72, -32.79, 0.00], 'radius': 72.29, 'proximal_extension': 13.52, 'distal_extension': 0.00},
}
CAPSULE_OFFSET_OVERRIDES = {}
for _n, _fix in _CAPSULE_FIX.items():
    for _side in ('arm_right', 'arm_left', 'arm_head'):
        CAPSULE_OFFSET_OVERRIDES[f'{_side}_{_n}_link'] = _fix
del _n, _fix, _side, _CAPSULE_FIX

# Legacy fallback only, superseded by config/worlds/<world_name>.yaml -- add new obstacles there.
CYLINDER_SIZE = [0.02, 0.15]          # [radius, length] of the workspace cylinders
RED_CYLINDER_POS = [0.800, -0.20, 0.775]
BLUE_CYLINDER_POS = [0.800, 0.20, 0.775]
TABLE_POS = [1.0, 0.0, 0.35]
TABLE_SIZE = [0.6, 0.5, 0.7]

WALL_SIZE = [1.0, 0.02, 1.0]          # virtual wall [length_x, thickness_y, height_z]
WALL_POS = [0.5, 0.0, 0.5]            # virtual wall position relative to base_link

# =============================================================================
# 7. OFFLINE PLOTTER (static, publication-quality figures)
# =============================================================================
# Recording-trigger contract (std_msgs/Bool): True = start/continue, False = motion concluded.
OFFLINE_RECORD_TRIGGER_TOPIC = "/offline_plotter/record_trigger"

# Root directory; each recorded trial gets its own timestamped subfolder.
OFFLINE_PLOT_ROOT_DIR = "~/exchange/ros2-ws/triago_offline_plots"

# Seconds of settling recorded after the trigger goes False, before figures are finalized.
OFFLINE_PLOT_POST_TRIGGER_S = 15.0

# The trigger edge is a one-shot VOLATILE message: the generator holds until a recorder
# subscribes, up to this many seconds (0.0 = no wait), so cross-host discovery can't miss it.
OFFLINE_RECORD_WAIT_TIMEOUT_S = 10.0

# A `ros2 bag record` of the trial window is written next to the figures for offline replay.
OFFLINE_BAG_ENABLE = True
OFFLINE_BAG_STORAGE_ID = "sqlite3"

# Superset allowlist (QP telemetry + teleoperation + shared autonomy + head perception);
# topics nothing publishes are harmless -- ros2 bag record captures only what is live.
OFFLINE_BAG_TOPICS = [
    "/joint_states",
    "/tf",
    "/tf_static",
    "/qp_debug/ee_real",
    "/qp_debug/xdot_err",
    "/qp_debug/qdot_err",
    "/qp_debug/qdot_cmd",
    "/qp_debug/qdot_measured",
    "/qp_debug/slacks",
    "/qp_debug/min_distance",
    "/qp_debug/safety_margin",
    "/qp_debug/lambda_cbf",
    "/qp_debug/lambda_joints",
    "/qp_debug/joint_limits",
    "/qp_debug/d_safe_dynamic",
    "/qp_debug/dynamic_weights",
    "/qp_debug/task_authority",
    "/qp_debug/governor",
    "/qp_debug/arm_frozen",
    "/qp_debug/loop_freq",
    "/collision_constraints",
    "/arm_right/cartesian_reference",
    "/arm_left/cartesian_reference",
    "/qp_debug/reference_effective",
    OFFLINE_RECORD_TRIGGER_TOPIC,
    # Head perception + active vision (optional per trial).
    "/head_perception/telemetry",
    "/head_perception/markers",
    "/head_perception/qdot_cmd",
    "/head_perception/qdot_measured",
    "/head_perception/debug_json",
    "/perceived_world/snapshot",
    "/real_perception",
    "/head_active_tracking/telemetry",
    "/head_active_tracking/qdot",
    # Teleoperation device (optional per trial).
    "/arm_right/user_cartesian_reference",
    "/arm_left/user_cartesian_reference",
    "/virtuose/pose",
    "/virtuose/velocity",
    "/virtuose/button_right",
    "/virtuose/button_left",
    "/virtuose/deadman",
    "/virtuose/articular_position",
    "/virtuose/force_cmd",
    "/joystick/home_pose",
    # Shared autonomy (optional per trial).
    "/shared_autonomy/blend_debug",
    "/shared_autonomy/goal_names",
    "/shared_autonomy/goal_probabilities",
    "/shared_autonomy/ee_policy",
    "/shared_autonomy/user_policy",
    "/shared_autonomy/active_goal_pose",
    "/shared_autonomy/grasp_active",
    "/shared_autonomy/active_arm",
    "/shared_autonomy/gripper_cmd",
    "/shared_autonomy/target_ignore",
    "/shared_autonomy/grasp_margin",
]
