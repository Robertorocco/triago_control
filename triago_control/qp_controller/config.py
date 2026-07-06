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

# Shared-autonomy TWIST BLENDING master switch. False = legacy: teleop_triago_clutch.py
# drives the robot directly, main_shared_autonomy.py only takes over during autonomous
# grasp execution or POLICY_BELIEF_TEST. True = main_shared_autonomy.py persistently
# blends v_blend = (1-alpha)*v_user + alpha*pi_policy and is the sole writer of
# /arm_*/cartesian_reference even in normal teleop.
BLENDING = True

# alpha(belief) shaping -- see main_shared_autonomy.compute_alpha.
#   alpha = ALPHA_MAX * belief_norm**ALPHA_GAMMA * dist_gate(||user_ref - EE||)
# belief_norm is the belief confidence normalised into [0,1]; dist_gate goes
# 1 (EE matched to the user reference, d <= SYNC_D_MIN) -> 0 (d >= SYNC_D_MAX),
# so blending only switches on once F_sync has pulled EE and reference together.
ALPHA_MAX = 0.80          # Hard cap on autonomy authority (user retains >= 20%)
ALPHA_GAMMA = 0.5         # <1 = alpha ramps toward ALPHA_MAX quickly once belief is high
ALPHA_LPF_COEFF = 0.08    # Low-pass filter coefficient on alpha

# --- F_sync distance-decaying stiffness + alpha distance gate (2026-07-06) ---
# d = ||EE - user_reference|| (position divergence). F_sync's linear magnitude
# follows a 1/d-shaped law: MAXIMUM near the reference (lock/settle zone) and
# MINIMUM far away, so the user can move the handle FREELY when far without the
# tether fighting them / injecting a fake twist into the loop. The SAME two
# distances gate alpha smoothly to 0 when far. Read by both
# main_shared_autonomy.py and haptic_force_manager_blending_tutorial.py.
SYNC_D_MIN = 0.03         # m -- at/below this: F_sync = SYNC_F_MAX and dist_gate = 1
SYNC_D_MAX = 0.30         # m -- at/beyond this: F_sync = SYNC_F_MIN and dist_gate = 0
SYNC_F_MAX = 8.0          # N -- linear sync force magnitude at SYNC_D_MIN (close)
SYNC_F_MIN = 1.0          # N -- linear sync force magnitude at SYNC_D_MAX (far)

# Distance-based assistance-intensity boost: compensates pi_policy's natural falloff
# near the goal so the approach can actually conclude, capped so full autonomy is
# never reached even at the goal. Smoothstep-shaped in EE-to-goal distance.
ALPHA_PROXIMITY_NEAR = 0.05      # m -- full boost gain at/inside this distance
ALPHA_PROXIMITY_FAR = 0.20       # m -- no boost beyond this distance (gain = 1.0)
ALPHA_PROXIMITY_MAX_GAIN = 1.5   # multiplier applied to alpha at ALPHA_PROXIMITY_NEAR
ALPHA_PROXIMITY_CAP = 0.90       # hard ceiling on the boosted alpha

# User-effort authority gating: scales alpha down by how briskly the user is moving
# the handle (linear OR angular), so a fast hand/wrist always takes precedence over
# belief-driven assistance. effort = max(lin_effort, ang_effort), both clipped [0,1].
ALPHA_EFFORT_THRESHOLD = 0.4      # m/s -- linear twist norm saturating effort to 1.0
ALPHA_EFFORT_ANG_THRESHOLD = 1.0  # rad/s -- angular twist norm saturating effort to 1.0
ALPHA_EFFORT_OVERRIDE = 0.5       # fraction of alpha displaced by full user effort
ALPHA_EFFORT_LPF_COEFF = 0.15     # low-pass filter on the effort signal

# Position/orientation-divergence authority override + reference catch-up: unlike the
# effort gate above, this reacts to a SUSTAINED pose gap (doesn't decay when the user
# stops moving), plus a bounded P-control pull toward the user's held pose so the
# robot actually follows through -- gated by a deadband, capped, and always routed
# through the same reference the QP-CBF tracks (never bypasses safety).
ALPHA_DIVERGENCE_NEAR = 0.05      # m -- no override below this position gap
ALPHA_DIVERGENCE_FAR = 0.20       # m -- full override at/beyond this position gap
ALPHA_DIVERGENCE_ANG_NEAR = 0.15  # rad (~8.6 deg) -- no override below this orientation gap
ALPHA_DIVERGENCE_ANG_FAR = 0.60   # rad (~34 deg) -- full override at/beyond this orientation gap
ALPHA_DIVERGENCE_OVERRIDE = 0.6   # fraction of alpha displaced at full divergence
ALPHA_DIVERGENCE_LPF_COEFF = 0.15  # LPF on the divergence-override signal

CATCHUP_DEADBAND_POS = 0.03       # m -- below this position gap, catch-up contributes nothing
CATCHUP_FULL_POS = 0.15           # m -- full catch-up gain reached at/beyond this gap
K_CATCHUP_LIN = 1.5               # P-gain (1/s) on the linear position gap (pre-clip)
V_CATCHUP_MAX_LIN = 0.06          # m/s -- hard cap on the catch-up linear velocity

CATCHUP_DEADBAND_ANG = 0.15       # rad (~8.6 deg) -- below this, no orientation catch-up
CATCHUP_FULL_ANG = 0.6            # rad (~34 deg) -- full orientation catch-up gain
K_CATCHUP_ANG = 1.0               # P-gain (1/s) on the orientation gap (pre-clip)
V_CATCHUP_MAX_ANG = 0.15          # rad/s -- hard cap on the catch-up angular velocity

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
