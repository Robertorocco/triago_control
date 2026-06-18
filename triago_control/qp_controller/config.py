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
DYNAMIC_SLACK_WEIGHT = False    # Increase slack weights in free space, drop near obstacles
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
K_V_SAFE = 0.1                 # Predictive velocity horizon for margin expansion
ALPHA_FILTER = 0.15            # EMA coefficient for hardware velocity filtering (~60ms window)
DAMP = 10.0                    # Joint velocity regularization (Lambda) in the QP cost
KP_POSTURE = 0.1               # Stiffness of the posture-hold "virtual spring"
P_GAIN_LIMITS = 2.5            # Joint-limit CBF gamma (braking aggressiveness) [scaled from 5.0]
JOINT_LIMIT_BUFFER_BASE = 0.15  # Base joint-limit braking buffer [expanded from 0.1]
JOINT_LIMIT_K_V = 0.1          # Joint-limit velocity horizon (seconds to look ahead)
LOCK_THRESHOLD = 0.001         # Below this input energy, the posture lock engages
W_CENTER = 0.01                # Posture centering weight in the QP cost

# =============================================================================
# 3. DYNAMIC SCALING BOUNDARIES
# =============================================================================
# --- Decoupled dynamic slack weighting ---
BASE_WEIGHT_SLACK = 5.0        # Standard slack weight (active against an obstacle)
MAX_WEIGHT_SLACK = 50.0        # Maximum slack weight (in free space)
BETA = 1.0                     # How fast slack weights return to baseline as lambda grows
# --- Dynamic gamma (CLF) scheduling ---
GAMMA_CLF_DEFAULT = 1.0        # Static / initial CLF convergence rate (Vdot <= -gamma*V)
GAMMA_MIN = 0.5                # Lower bound of the scheduled CLF gamma
GAMMA_MAX = 1.0                # Upper bound of the scheduled CLF gamma
BETA_GAMMA = 5.0               # How quickly gamma drops as the collision lambda grows
GAMMA_FILTER_TAU = 0.125       # Low-pass time constant for the gamma scheduler

# =============================================================================
# 4. LOOP / TELEMETRY SETTINGS
# =============================================================================
CONTROL_FREQ_DEFAULT = 300.0   # Default control loop frequency [Hz] (was hard-coded 1/300)
PUBLISH_EVERY_N = 10           # Publish 1 of every N iterations to the dashboard
WATCHDOG_TIMEOUT = 0.5         # Seconds without a reference before motion is frozen
DISTANCE_FILTER_THRESHOLD = 0.15  # Ignore collision pairs farther than this [m]
K_MAX_PAIRS = 60               # Max number of closest pairs fed into the SoftMin

# Diagonal task weights [Px, Py, Pz, Roll, Pitch, Yaw]: heavily penalize position,
# barely penalize orientation. Equivalent to np.array([1,1,1,0.1,0.1,0.1]) * 10.
TASK_WEIGHTS_6D = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]) * 10.0

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
