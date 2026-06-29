# config.py
"""
Centralized configuration for the TRIAGo Bimanual QP-CLF-CBF Controller.

This module is the SINGLE source of truth for every boolean flag, safety buffer,
hyperparameter and hardware-specific workaround that governs the controller.
Nothing in the control law is hard-coded elsewhere: tuning the robot's behaviour
should only ever require editing values in this file.

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
WALL_COLLIDER = False          # Enable the virtual collision wall (XZ plane)
FLYING_OBSTACLE = False         # Enable the flying obstacle marker / collider
PINHOLE_TASK = True             # Load the obstacle set from the PINHOLE world
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
# 2. SAFETY + CONTROL HYPERPARAMETERS
# =============================================================================
ALPHA_SOFTMIN = 50.0            # Sharpness of the SoftMin collision aggregation
GAMMA_CBF = 0.75               # CBF class-K gain [scaled down from 1.5 for 300Hz]
D_SAFE_BASE = 0.015            # Base safety distance for the collision barrier
K_V_SAFE = 0.1                 # Predictive velocity horizon [0.1 -> 0.2: brake earlier
                               #   at high speed so fast unsafe motion cannot penetrate]
ALPHA_FILTER = 0.5            # EMA coefficient for hardware velocity filtering (~20ms window)
DAMP = 10.0                    # Joint velocity regularization (Lambda) in the QP cost
P_GAIN_LIMITS = 2.5            # Joint-limit CBF gamma (braking aggressiveness) [scaled from 5.0]
JOINT_LIMIT_BUFFER_BASE = 0.15  # Base joint-limit braking buffer [expanded from 0.1]
JOINT_LIMIT_K_V = 0.1          # Joint-limit velocity horizon (seconds to look ahead)
LOCK_THRESHOLD = 0.001         # Below this input energy, the posture lock engages

# --- Posture / joint-limit avoidance: repulsive potential field ---
# The posture reference velocity is the NEGATIVE GRADIENT of a barrier potential
# that diverges at each joint's limits. It is evaluated on the NORMALIZED joint
# position p = 2*(q - mid) / range  in [-1, 1], so EVERY joint is defended equally
# at the same FRACTION of its travel (range-independent):
#     H(p)       = 1/(1 - p)^2 + 1/(1 + p)^2
#     dH/dp      = 2/(1 - p)^3 - 2/(1 + p)^3
#     q_dot_post = -K_GRADIENT * dH/dp           (clamped to +/- V_MAX_POSTURE)
# Near-zero in the comfortable mid-range (so the CLF keeps tracking priority) and
# grows sharply (clamped) only as a joint nears a limit, using the arm redundancy
# to reconfigure away from it. Replaces the old q_neutral spring (KP_POSTURE) and
# the Chan & Dubey ramp (KP_LIMIT_AVOID / LIMIT_AVOID_THRESH / JOINT_LIMIT_AVOID).
K_GRADIENT = 0.05              # gain on the negative potential gradient
V_MAX_POSTURE = 1.0            # rad/s hard clamp on the posture reference (solver safety)
W_CENTER = 1.0                 # posture-task weight in the QP cost (vs DAMP=10): ~0 authority
                               #   in mid-range (v_ref ~ 0), meaningful near limits, never
                               #   overrides the CLF (which is a hard slack-penalised constraint)
POSTURE_GRASP_SCALE = 0.05     # posture weight is scaled to this (×W_CENTER) during autonomous
                               #   precision phases (grasp/align/approach/close/lift) so the QP
                               #   spends the redundancy on precise tracking, not posture
POSTURE_SCALE_TAU = 0.2        # s — first-order ramp time-constant for the posture-scale switch

# =============================================================================
# 3. DYNAMIC SCALING BOUNDARIES
# =============================================================================
# --- Decoupled dynamic slack weighting ---
BASE_WEIGHT_SLACK = 15.0        # Standard slack weight (active against an obstacle)
MAX_WEIGHT_SLACK = 50.0        # Maximum slack weight (in free space)
BETA = 0.4                     # How fast slack weights return to baseline as lambda grows
                               #   [1.0 -> 0.4: gentler curve, less abrupt swing near lambda~1]
SLACK_FILTER_TAU = 0.15        # LPF time constant on the shadow prices feeding the slack
                               #   scheduler (smooths the noisy raw lambda -> no weight jumps)
# --- Dynamic gamma (CLF) scheduling ---
GAMMA_CLF_DEFAULT = 1.5        # Static / initial CLF convergence rate (Vdot <= -gamma*V)
GAMMA_MIN = 0.5                # Lower bound of the scheduled CLF gamma
GAMMA_MAX = 1.0                # Upper bound of the scheduled CLF gamma
BETA_GAMMA = 5.0               # How quickly gamma drops as the collision lambda grows
GAMMA_FILTER_TAU = 0.125       # Low-pass time constant for the gamma scheduler

# =============================================================================
# 4. LOOP / TELEMETRY SETTINGS
# =============================================================================
CONTROL_FREQ_DEFAULT = 300.0   # Default control loop frequency [Hz] (was hard-coded 1/300)
PUBLISH_EVERY_N = 2            # Publish 1 of every N iterations to the dashboard
WATCHDOG_TIMEOUT = 0.5         # Seconds without a reference before motion is frozen
DISTANCE_FILTER_THRESHOLD = 0.15  # Ignore collision pairs farther than this [m]
K_MAX_PAIRS = 60               # Max number of closest pairs fed into the SoftMin

# Diagonal task weights [Px, Py, Pz, Roll, Pitch, Yaw]: heavily penalize position,
# barely penalize orientation. Orientation lowered from 0.1 -> 0.04 (relative to
# the 1.0 position weight, i.e. a 25:1 position:orientation ratio) so the QP
# prioritizes CLOSING POSITION ERROR over matching orientation when the two
# conflict near an obstacle — fixes the "parked at the wrong position but right
# orientation" behaviour the operator reported. The CLF math is unchanged (still
# a positive-definite diagonal-weighted scalar CLF); only the weighting ratio.
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

RIGHT_TCP_FRAME = 'gripper_right_grasping_link'
LEFT_TCP_FRAME  = 'gripper_left_grasping_link'
REF_FRAME = 'base_footprint'

# =============================================================================
# 6. GEOMETRY + WORKSPACE
# =============================================================================
CAPSULE_RADIUS = 0.06                 # Radius of the arm collision capsules

CYLINDER_SIZE = [0.02, 0.15]          # [Radius, Length] of the workspace cylinders
RED_CYLINDER_POS = [0.800, -0.20, 0.775]
BLUE_CYLINDER_POS = [0.800, 0.20, 0.775]
TABLE_POS = [1.0, 0.0, 0.35]
TABLE_SIZE = [0.6, 0.5, 0.7]

WALL_SIZE = [1.0, 0.02, 1.0]          # Virtual wall [length_x, thickness_y, height_z]
WALL_POS = [0.5, 0.0, 0.5]            # Virtual wall position relative to base_link
