# config.py
"""
Centralized configuration for the TRIAGo Bimanual QP-CLF-CBF Controller.

Single source of truth for every boolean flag, safety buffer, hyperparameter
and hardware-specific workaround that governs the controller. Nothing in the
control law is hard-coded elsewhere: tuning behaviour should only ever
require editing values in this file.

Sections:
    1. Boolean feature flags        (enable / disable whole subsystems)
    2. Safety + control hyperparams (CBF / CLF / filtering gains)
    3. Dynamic scaling boundaries   (adaptive slack + gamma scheduling)
    4. Loop / telemetry settings    (frequency, publish downsampling)
    5. Robot + controller names      (URDF chains, ROS controller IDs)
    6. Geometry + workspace          (capsule radius, obstacles, walls)
"""

import numpy as np

# =============================================================================
# 1. BOOLEAN FEATURE FLAGS
# =============================================================================
# WALL_COLLIDER: legacy-path only (world_scene=None). For a YAML-loaded world,
# set that world's `virtual_wall` obstacle's `collision:` field instead.
WALL_COLLIDER = False          # Enable the virtual collision wall (XZ plane) [legacy path only]
FLYING_OBSTACLE = False         # Enable the flying obstacle marker / collider
PINHOLE_TASK = True             # [DEAD FLAG, unused] superseded by the `world_name` ROS parameter
DEBUG = False                   # Verbose timing / kinematics console tracing
GRASP_DEBUG = True              # Verbose grasp / CBF-bypass interaction tracing
DISABLE_CBF = False             # Mathematically delete the collision barrier
DYNAMIC_CBF = False             # Dynamically remove pairs for interaction
DYNAMIC_SLACK_WEIGHT = True    # Increase slack weights in free space, drop near obstacles
COMPARISON_CLF = True           # Use the normalized (unit-error) scalar CLF formulation
DYNAMIC_GAMMA_CLF = False       # Vary CLF convergence rate with the safety margin
SIMULATE_IDEAL_KINEMATICS = False  # True = pure math digital twin, False = real hardware
ORIENTATION_CTRL = True         # True = control Pos+Ori (6DOF), False = Pos only (3DOF)

# =============================================================================
# 1b. SHARED-AUTONOMY BLENDING -- "JOYSTICK MODE"
# =============================================================================
# Master switch. False = legacy Virtual-Fixture teleop: teleop_triago_clutch.py
# drives /arm_*/cartesian_reference directly; main_shared_autonomy.py only takes
# over during autonomous grasp execution / POLICY_BELIEF_TEST. True = JOYSTICK
# MODE: teleop_triago_joystick.py maps the Haption handle's displacement-from-home
# into a pure Cartesian twist on /arm_*/user_cartesian_reference, and
# main_shared_autonomy.py arbitrates that user twist against the belief-weighted
# optimal policy (computed from the TRUE EE pose) and is the sole writer of
# /arm_*/cartesian_reference. This replaces the deprecated "combine F_sync force
# feedback with a direct user/policy twist blend" design, whose force-into-handle
# path created an unstable feedback loop. See teleop_triago_joystick.py's module
# docstring for the full rationale.
BLENDING = True

# --- Joystick home pose (Haption base frame) -------------------------------
# The handle is spring-centered to this pose; its displacement from it IS the
# command. Position is FIXED. Orientation is re-based at joystick startup to the
# robot gripper's INITIAL orientation (see JOYSTICK_ROT_HOME_SCALE) so "handle at
# rest" always means "hold the current gripper orientation", however the gripper
# is posed (e.g. a top-down grasp approach).
JOYSTICK_NEUTRAL_POSITION_M = [0.5, -0.03, -0.03]
# Measured on the device (virtuose/pose) with the handle at the operator's
# comfortable rest orientation — the pose the spring controller settles at when
# the user releases the handle. This is the handle orientation that maps to "hold
# the gripper's current orientation" (zero rotational command) the first time each
# arm becomes active.
JOYSTICK_NEUTRAL_ORIENTATION_XYZW = [-0.015140674076974392, 0.8170770406723022,
                                     0.06124841421842575, 0.5730659365653992]

# --- Displacement -> twist mapping (teleop_triago_joystick.py) --------------
# The commanded twist magnitude is STRICTLY PROPORTIONAL to the handle's distance
# from home, past a deadband. Deadband: displacement below these -> zero twist
# (and, in the arbitration, a zero user twist is treated as perfectly ALIGNED so
# the autonomy leads the motion). Reduced 20% from the original 12 cm / ~17 deg
# (0.12 m / 0.30 rad) initial values for a more responsive handle -- to be tuned.
JOYSTICK_DEADBAND_LIN = 0.096        # m   (9.6 cm) -- large: the spring can't settle to mm precision
JOYSTICK_DEADBAND_ANG = 0.24         # rad (~13.7 deg) -- large: residual oscillation around home
JOYSTICK_K_TRANS = 1.6               # (m/s) per m of handle linear displacement
JOYSTICK_K_ROT = 1.5                 # (rad/s) per rad of handle angular displacement
JOYSTICK_V_MAX_LIN = 0.10            # m/s   hard safety clamp on the commanded linear twist
JOYSTICK_V_MAX_ANG = 0.50            # rad/s hard safety clamp on the commanded angular twist

# --- Viscous damping on the commanded twist (_compute_user_twist) ----------
# v_haption = K_TRANS * eff_lin - DAMP_LIN * handle_vel_lin   (mirrored for angular).
# Dimensionless (fraction of the handle's own instantaneous velocity subtracted
# from the command), applied ONLY outside the deadband so a still/settling handle
# still commands exactly zero (the deadband guarantee is untouched -- see the
# "inside deadband -> zero (no damping)" branches in _compute_user_twist).
# Purpose: attenuate fast, jittery handle motion (tremor, spring-settle ringing)
# without fighting deliberate, sustained motion. Keep well below 1.0 -- this
# directly subtracts from the commanded velocity, not a physical damper.
JOYSTICK_DAMP_LIN = 0.225            # unitless fraction of handle linear velocity
JOYSTICK_DAMP_ANG = 0.075            # unitless fraction of handle angular velocity

# --- Home-orientation rebasing scale ---------------------------------------
# The gripper's rotation away from its startup reference is scaled DOWN by this
# factor when building the handle's home orientation (gripper 90 deg -> handle
# ~69 deg), so the handle's rest orientation tracks the gripper closely (feels
# synchronized) while still staying within the Haption's more restrictive
# rotational workspace. Applies ONLY to the home-pose construction, NOT to the
# commanded twist. LOWER -> stronger sync (1.0 = 1:1); higher -> looser.
JOYSTICK_ROT_HOME_SCALE = 1.3

# --- Restorative centering spring (haptic_force_manager_blending_tutorial.py) --
# The ONLY haptic force rendered in joystick mode: a virtual spring-damper pulling
# the handle back to the (dynamic) home pose. No F_guide / F_fixture / F_sync.
JOYSTICK_SPRING_KP_LIN = 60.0        # N/m        (stiffer: resists driving far from home)
JOYSTICK_SPRING_KD_LIN = 1.0         # N/(m/s)    damping raised with the stiffer spring
JOYSTICK_SPRING_KP_ANG = 1.5         # Nm/rad
JOYSTICK_SPRING_KD_ANG = 0.15        # Nm/(rad/s)

# Topic carrying the live joystick home pose (teleop_triago_joystick.py ->
# haptic_force_manager_blending_tutorial.py) so both nodes agree on the SAME
# dynamic home (single source of truth -- the teleop node owns it). Layout:
#   [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w] in the Haption base frame.
JOYSTICK_HOME_POSE_TOPIC = "/joystick/home_pose"

# --- Twist arbitration (main_shared_autonomy.compute_alpha) -----------------
# The autonomy authority alpha (weight on the optimal policy in the blend
#   v_blend = (1 - alpha) * v_user + alpha * pi_policy)
# is a function ONLY of the ALIGNMENT s in [-1,1] between the user twist and the
# policy twist (mean per-channel cosine over whichever channels the user is
# actively commanding):
#   alpha = ALIGN_ALPHA_MIN + (ALIGN_ALPHA_MAX - ALIGN_ALPHA_MIN) * clip(s, 0, 1)
# Misaligned (s <= 0)  -> alpha = ALIGN_ALPHA_MIN  (policy 20%, USER 80%).
# Aligned    (s  = 1)  -> alpha = ALIGN_ALPHA_MAX  (policy 80%, user 20%) -- fast,
#                        but ONLY while the user is actively moving in agreement.
# No user input (handle inside the joystick deadband, v_user = 0) uses a REDUCED
# idle authority ALIGN_ALPHA_IDLE, so the robot only GENTLY crawls toward the
# inferred goal when the user is still; the instant the user pushes in a direction
# the policy agrees with, alpha ramps up to ALIGN_ALPHA_MAX (fast).
ALIGN_ALPHA_MIN = 0.2                # policy weight when fully misaligned (user prioritised)
ALIGN_ALPHA_MAX = 0.8                # policy weight when aligned AND the user is moving in agreement
ALIGN_ALPHA_IDLE = 0.35             # policy weight when the user is still (gentle autonomous crawl)
ALIGN_ALPHA_LPF_COEFF = 0.1          # low-pass on alpha for C0-continuity of the blend

# =============================================================================
# 2. SAFETY + CONTROL HYPERPARAMETERS
# =============================================================================
ALPHA_SOFTMIN = 50.0            # Sharpness of the SoftMin collision aggregation
GAMMA_CBF = 0.75               # CBF class-K gain
D_SAFE_BASE = 0.015            # Base safety distance for the collision barrier
K_V_SAFE = 0.1                 # Predictive velocity horizon (brake earlier at high speed)
ALPHA_FILTER = 0.5            # EMA coefficient for hardware velocity filtering (~20ms window)
DAMP = 10.0                    # Joint velocity regularization (Lambda) in the QP cost
P_GAIN_LIMITS = 2.5            # Joint-limit CBF gamma (braking aggressiveness)
JOINT_LIMIT_BUFFER_BASE = 0.15  # Base joint-limit braking buffer
JOINT_LIMIT_K_V = 0.1          # Joint-limit velocity horizon (seconds to look ahead)
LOCK_THRESHOLD = 0.001         # Below this input energy, the posture lock engages

# Posture / joint-limit avoidance: repulsive potential field, negative gradient of a
# barrier that diverges at each joint's limits, evaluated on the normalized joint
# position so every joint is defended at the same fraction of its travel. Near-zero
# in mid-range (CLF keeps tracking priority), grows sharply (clamped) near a limit.
K_GRADIENT = 0.05              # gain on the negative potential gradient
V_MAX_POSTURE = 1.0            # rad/s hard clamp on the posture reference (solver safety)
W_CENTER = 1.0                 # posture-task weight in the QP cost (vs DAMP=10)
POSTURE_GRASP_SCALE = 0.05     # posture weight scaled to this (x W_CENTER) during autonomous
                               #   precision phases (grasp/align/approach/close/lift)
POSTURE_SCALE_TAU = 0.2        # s -- first-order ramp time-constant for the posture-scale switch

# =============================================================================
# 3. DYNAMIC SCALING BOUNDARIES
# =============================================================================
# Decoupled dynamic slack weighting
BASE_WEIGHT_SLACK = 25.0        # Standard slack weight (active against an obstacle)
MAX_WEIGHT_SLACK = 100.0       # Max slack weight (free space); also the fixed slack
                               #   weight pinned on an inactive (frozen) arm to decouple it
BETA = 0.4                     # How fast slack weights return to baseline as lambda grows
SLACK_FILTER_TAU = 0.15        # LPF time constant on the shadow prices feeding the scheduler

# Dynamic gamma (CLF) scheduling
GAMMA_CLF_DEFAULT = 0.75       # Static / initial CLF convergence rate (Vdot <= -gamma*V)
GAMMA_MIN = 0.5                # Lower bound of the scheduled CLF gamma
GAMMA_MAX = 1.0                # Upper bound of the scheduled CLF gamma
BETA_GAMMA = 5.0               # How quickly gamma drops as the collision lambda grows
GAMMA_FILTER_TAU = 0.125       # Low-pass time constant for the gamma scheduler

# =============================================================================
# 3b. REFERENCE GOVERNOR (intermediate CLF-safety layer)
# =============================================================================
# Intermediate filter between the raw cartesian reference and what the CLF actually
# sees: bounds position/orientation error and reference velocity/acceleration,
# preserving QP feasibility under aggressive/discontinuous commands. See
# reference_governor.py for the full design.
ENABLE_REFERENCE_GOVERNOR = True       # Master switch (False = raw passthrough, no filtering)

GOV_V_MAX_LIN = 0.20                  # [m/s] max linear reference velocity passed to the CLF
GOV_V_MAX_ANG = 1.2                   # [rad/s] max angular reference velocity passed to the CLF

GOV_E_MAX_POS = 0.30                  # [m] max allowed position error norm (30 cm)
GOV_E_MAX_ORI = 0.524                 # [rad] max allowed orientation error norm (~30 deg)

GOV_A_MAX_LIN = 2.0                   # [m/s^2] max linear acceleration of the governed reference
GOV_A_MAX_ANG = 8.0                   # [rad/s^2] max angular acceleration of the governed reference

# =============================================================================
# 3c. LOCAL MINIMA ESCAPE (governor extension)
# =============================================================================
# Detects a possible QP-CLF-CBF local minimum (large, near-constant 3D position
# tracking error) and applies a temporary, per-arm posture-weight correction to
# help escape it. See reference_governor.ReferenceGovernor.update_local_minima.
# Tuned against the CURRENT gains in sections 2/3 -- revisit if those change.
ENABLE_LOCAL_MINIMA_ESCAPE = False    # Master switch (independent of ENABLE_REFERENCE_GOVERNOR)
# The only corrective action is the posture-weight + task_dim nudge below (no
# planner/strategy selector -- an earlier RRT-Connect fallback attempt was removed).

LME_ERROR_TRIGGER = 0.15             # [m] error norm above which "stuck" is considered
LME_ERROR_STUCK_WINDOW = 2.0         # [s] time window checked for a near-constant error
LME_ERROR_STUCK_TOLERANCE = 0.02     # [m] max variation within the window to call it "stuck"
LME_ERROR_RECOVERED = 0.05           # [m] error norm below which the escape ends (success)
LME_MAX_ESCAPE_DURATION = 10.0       # [s] max time the escape correction is held before giving up

# Categorization uses shadow prices from the previous QP solve. Obstacle takes
# priority if both conditions are met simultaneously.
LME_LAMBDA_CBF_THRESHOLD = 10.0      # lambda_cbf > this -> obstacle-induced minimum
LME_LAMBDA_JOINT_THRESHOLD = 1.0     # lambda_joints > this -> joint-limit-induced minimum

LME_POSTURE_SCALE_OBSTACLE = 0.2     # x1/5 posture weight (more redundancy to slip past)
LME_POSTURE_SCALE_JOINT = 5.0        # x5 posture weight (push harder away from the limit)
LME_TASK_DIM_OBSTACLE = 3.0          # force position-only CLF during obstacle escape
LME_RAMP_TAU = 0.3                   # [s] smooth ramp time-constant for the posture-scale change

LME_CONSOLE_PERIOD = 3.0             # [s] throttle period for the non-spam status print

# =============================================================================
# 4. LOOP / TELEMETRY SETTINGS
# =============================================================================
CONTROL_FREQ_DEFAULT = 300.0   # Default control loop frequency [Hz]
PUBLISH_EVERY_N = 2            # Publish 1 of every N iterations to the dashboard
WATCHDOG_TIMEOUT = 0.5         # Seconds without a reference before motion is frozen
DISTANCE_FILTER_THRESHOLD = 0.15  # Ignore collision pairs farther than this [m]
K_MAX_PAIRS = 60               # Max number of closest pairs fed into the SoftMin

# Diagonal task weights [Px,Py,Pz,Roll,Pitch,Yaw]: heavily penalize position, barely
# penalize orientation (25:1 ratio), so the QP prioritizes closing position error
# over matching orientation when the two conflict near an obstacle.
TASK_WEIGHTS_6D = np.array([1.0, 1.0, 1.0, 0.04, 0.04, 0.04]) * 10.0

# Mesh package search paths used to build the Meshcat visual model from the URDF.
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

# Kinematic chains (must match URDF link names)
RIGHT_CHAIN = ['arm_right_1_link', 'arm_right_2_link', 'arm_right_3_link',
               'arm_right_4_link', 'arm_right_5_link', 'arm_right_6_link', 'arm_right_7_link']
LEFT_CHAIN  = ['arm_left_1_link', 'arm_left_2_link', 'arm_left_3_link',
               'arm_left_4_link', 'arm_left_5_link', 'arm_left_6_link', 'arm_left_7_link']

# Active joint names (for mapping QP output to ROS messages)
RIGHT_JOINTS = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint',
                'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']
LEFT_JOINTS  = ['arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint',
                'arm_left_4_joint', 'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint']
# Head + gripper-finger joint names, for the slider GUI's live limit lookup only --
# NOT part of the arm QP's decision vector.
HEAD_JOINTS = ['arm_head_1_joint', 'arm_head_2_joint', 'arm_head_3_joint',
               'arm_head_4_joint', 'arm_head_5_joint', 'arm_head_6_joint', 'arm_head_7_joint']
GRIPPER_FINGER_JOINTS = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

RIGHT_TCP_FRAME = 'gripper_right_grasping_link'
LEFT_TCP_FRAME  = 'gripper_left_grasping_link'
REF_FRAME = 'base_footprint'

# Head chain: quasi-static CBF obstacle for the arms ONLY. Same 7-DOF hardware as the
# L/R arms but NOT part of this QP's decision vector -- modeled purely as geometry
# (capsules from live FK) so the arms avoid it without adding head velocity to the QP.
# HEAD_TOOL_LINK mirrors gripper_{side}_base_link's role for the arms.
HEAD_CHAIN = ['arm_head_1_link', 'arm_head_2_link', 'arm_head_3_link',
              'arm_head_4_link', 'arm_head_5_link', 'arm_head_6_link', 'arm_head_7_link']
HEAD_TOOL_LINK = 'arm_head_tool_link'

# Live joint-position slider GUI layout (plotter.py "Joint Positions" window): shared
# source of truth for main_qp_controller.py (publishes live limits) and plotter.py
# (lays out the grid). col 0 = left arm, col 1 = head, col 2 = right arm. Gripper
# fingers live in their own row below (GRIPPER_SLIDER_ROW), not column-aligned.
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
CAPSULE_RADIUS = 0.06                 # Default radius of the arm collision capsules
                                       # (used for any link WITHOUT an entry in
                                       # CAPSULE_OFFSET_OVERRIDES below)

# =============================================================================
# 6b. PER-LINK CAPSULE ALIGNMENT OVERRIDES
# =============================================================================
# calculate_offsets builds each link's capsule as a straight joint-to-joint segment
# with the single global CAPSULE_RADIUS above, but the real CAD mesh isn't always
# collinear with that line and several links are physically thicker than the default
# radius -- so the raw capsule can let the visual mesh poke outside it. Rather than
# growing CAPSULE_RADIUS globally (which fattens every link, loosening CBF margins
# everywhere), each entry here corrects ONE named link from its real mesh vertex data
# (see scripts/qp_arm_teleop/capsule_alignment_audit.py --suggest-fix). A link absent
# from this dict is completely unaffected -- byte-identical to before this feature.
#
# Fields (mm, joint-local frame -- same frame calculate_offsets' placement/length
# live in): lateral_offset [x,y,z] re-centers the capsule's core line perpendicular
# to its axis; radius replaces CAPSULE_RADIUS for this link only; proximal_extension/
# distal_extension extend the segment's ends along its axis to cover mesh that
# reaches past the raw joint-to-joint segment.
#
# Radii below are reduced ~5mm from their tight (audit-computed) fit as an
# exploratory step to loosen the CBF's felt conservatism -- this deliberately
# reintroduces a small (~4mm) accepted mesh protrusion on links 1-5, expected when
# re-running the audit. Link 6 is intentionally ABSENT (grasp tuning depends on its
# original, un-overridden geometry). Applied by
# CollisionManager._apply_capsule_override as a pure additive correction after the
# dominant-axis snap. The same fix applies across arm_right/arm_left/arm_head for a
# given link number (measurements came out identical across all three chains).
_CAPSULE_FIX = {
    1: {'lateral_offset': [-22.46, 9.11, 0.00], 'radius': 70.42, 'proximal_extension': 0.00, 'distal_extension': 0.20},
    2: {'lateral_offset': [8.82, -0.00, -8.67], 'radius': 70.07, 'proximal_extension': 0.00, 'distal_extension': 0.00},
    3: {'lateral_offset': [11.49, 25.66, 0.00], 'radius': 70.97, 'proximal_extension': 0.00, 'distal_extension': 0.20},
    4: {'lateral_offset': [-8.82, 0.00, -8.67], 'radius': 70.07, 'proximal_extension': 0.00, 'distal_extension': 0.00},
    5: {'lateral_offset': [0.72, -32.79, 0.00], 'radius': 72.29, 'proximal_extension': 13.52, 'distal_extension': 0.00},
    # 6: intentionally removed -- see note above.
}
CAPSULE_OFFSET_OVERRIDES = {}
for _n, _fix in _CAPSULE_FIX.items():
    for _side in ('arm_right', 'arm_left', 'arm_head'):
        CAPSULE_OFFSET_OVERRIDES[f'{_side}_{_n}_link'] = _fix
del _n, _fix, _side, _CAPSULE_FIX

# Legacy fallback only: superseded by config/worlds/<world_name>.yaml (see
# world_loader.py). Any caller that constructs CollisionManager/VisualizationEngine
# WITHOUT passing a world_scene still reads these values. Do not add new obstacles
# here -- add them to a world YAML instead.
CYLINDER_SIZE = [0.02, 0.15]          # [Radius, Length] of the workspace cylinders
RED_CYLINDER_POS = [0.800, -0.20, 0.775]
BLUE_CYLINDER_POS = [0.800, 0.20, 0.775]
TABLE_POS = [1.0, 0.0, 0.35]
TABLE_SIZE = [0.6, 0.5, 0.7]

WALL_SIZE = [1.0, 0.02, 1.0]          # Virtual wall [length_x, thickness_y, height_z]
WALL_POS = [0.5, 0.0, 0.5]            # Virtual wall position relative to base_link

# =============================================================================
# 7. OFFLINE PLOTTER (static, publication-quality figures)
# =============================================================================
# See scripts/qp_arm_teleop/offline_plotter.py for the full design.

# Generic recording-trigger contract (std_msgs/Bool): True = start/continue
# recording, False = commanded motion concluded. offline_plotter.py owns all
# post-trigger behaviour; the publisher only reports the raw on/off signal.
OFFLINE_RECORD_TRIGGER_TOPIC = "/offline_plotter/record_trigger"

# Root directory under which each recorded trial gets its own timestamped subfolder.
OFFLINE_PLOT_ROOT_DIR = "~/exchange/ros2-ws/triago_offline_plots"

# How long (seconds) offline_plotter.py keeps recording after the trigger above goes
# False, before finalizing and saving the figures (captures the settling phase on the
# same time axis as the tracking motion).
OFFLINE_PLOT_POST_TRIGGER_S = 10.0
