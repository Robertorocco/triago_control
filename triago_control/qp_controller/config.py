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

# --- Shared-autonomy TWIST BLENDING (2026-07-03) ---
# Master switch for the shared-autonomy blending architecture. This is the
# SINGLE source of truth consumed by BOTH:
#   - main_shared_autonomy.py  (applies the blend formula + becomes the sole
#                                publisher of /arm_*/cartesian_reference)
#   - teleop_triago_clutch.py  (redirects its OWN publisher to
#                                /arm_*/user_cartesian_reference instead of
#                                /arm_*/cartesian_reference, so the two nodes
#                                never fight over the same topic)
# False = legacy behavior: teleop_triago_clutch.py drives the robot directly,
#         main_shared_autonomy.py only takes over during autonomous grasp
#         execution or POLICY_BELIEF_TEST (unchanged from before).
# True  = shared-autonomy blending: main_shared_autonomy.py persistently
#         integrates v_blend = (1-alpha)*v_user + alpha*pi_policy every tick
#         and is the SOLE writer of /arm_*/cartesian_reference even in normal
#         teleop (not just during grasp execution).
BLENDING = False

# alpha(belief) shaping -- see main_shared_autonomy.compute_alpha for the
# exact formula. x = normalised belief in [0,1] (0 at uniform, 1 at certainty).
#   alpha = ALPHA_MAX * x**ALPHA_GAMMA
ALPHA_MAX = 0.60          # Hard cap on autonomy authority away from the goal
                          #   (user retains >= 40%). [was 0.80/20% -- operator
                          #   reported that once a goal was picked, autonomy had
                          #   too much authority and "almost anything can be
                          #   done" without the user being able to meaningfully
                          #   override; lowered by 20 percentage points, i.e.
                          #   the user's guaranteed floor rises from 20% -> 40%.]
ALPHA_GAMMA = 0.5         # <1 = alpha ramps toward ALPHA_MAX quickly once belief
                          #   is "sufficiently high"; 1.0 = linear ramp
ALPHA_LPF_COEFF = 0.08    # Low-pass filter coefficient on alpha (lower = smoother,
                          #   avoids any discontinuity in the blended reference
                          #   when the belief distribution shifts)

# --- Distance-based assistance-intensity boost (task completion, 2026-07-03) ---
# The QP-constrained policy twist (pi_policy) naturally shrinks toward zero as
# the EE nears the goal (proportional/CLF-style convergence -- by design, so it
# doesn't overshoot). Combined with ALPHA_MAX capping the blend weight, the
# operator reported the robot sometimes couldn't actually CONCLUDE the
# approach: near the goal the assistive contribution became too weak to close
# the last few centimeters. This SEPARATE, smooth proximity gain temporarily
# raises the EFFECTIVE alpha as the EE nears the active goal -- compensating
# for pi_policy's own natural falloff -- capped by ALPHA_PROXIMITY_CAP so full
# autonomy is still never reached even at the goal. Smoothstep-shaped in the
# EE-to-goal distance (see compute_alpha), so it introduces no discontinuity.
ALPHA_PROXIMITY_NEAR = 0.05      # m -- full boost gain at/inside this distance
ALPHA_PROXIMITY_FAR = 0.20       # m -- no boost beyond this distance (gain = 1.0)
ALPHA_PROXIMITY_MAX_GAIN = 1.5   # multiplier applied to alpha at ALPHA_PROXIMITY_NEAR
ALPHA_PROXIMITY_CAP = 0.90       # hard ceiling on the BOOSTED alpha (user retains >= 10%
                                 #   even at the moment of task completion)

# --- User-effort authority gating (2026-07-03) ---
# Operator report: with a fixed alpha(belief), the arm is almost "blind" to the
# user's own hand twist -- pi_policy is a large, tanh-saturated velocity while
# comfortable hand motion is much smaller, so even at a moderate/low alpha the
# policy's contribution to v_blend dominates the user's. The user could only
# ever "pick a different goal" (via belief), never meaningfully resist/steer
# once a goal was inferred.
#
# Fix: scale alpha DOWN by how much the user is ACTIVELY moving the handle,
# not just by belief. When the user is (near-)still, alpha is unaffected (full
# belief-driven assistance -- helpful when intent changes or near an obstacle,
# where the user naturally isn't pushing). When the user is moving briskly,
# alpha drops -- their own effort takes precedence.
#
#   effort = clip(||v_user|| / ALPHA_EFFORT_THRESHOLD, 0, 1)
#   alpha_effective = alpha_belief * (1 - effort * ALPHA_EFFORT_OVERRIDE)
#
# Deliberately NOT implemented on the force/haptic side (no Lagrangian
# multipliers, no discontinuous/filtered shadow-price feedback) -- this is a
# smooth function of the user's OWN commanded twist norm, always continuous
# and independent of any QP dual variable.
ALPHA_EFFORT_THRESHOLD = 0.4    # m/s -- user linear twist norm considered "fast hand
                                 #   movement"; effort saturates to 1.0 at/above this
ALPHA_EFFORT_OVERRIDE = 0.5     # fraction of alpha_belief displaced by full user effort.
                                 #   0.5: at max effort, alpha is halved -- the rest of
                                 #   the reduction in autonomy "following" already comes
                                 #   from fast motion naturally lowering the belief
                                 #   estimate itself (see BeliefEstimator/engagement).
ALPHA_EFFORT_LPF_COEFF = 0.15    # low-pass filter on the effort signal (smooths the
                                 #   ||v_user|| norm before it gates alpha; independent
                                 #   of ALPHA_LPF_COEFF, which smooths the final alpha)

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
BASE_WEIGHT_SLACK = 25.0        # Standard slack weight (active against an obstacle)
MAX_WEIGHT_SLACK = 100.0       # Maximum slack weight (in free space); ALSO the fixed slack
                               #   weight pinned on an INACTIVE (frozen) arm to decouple it.
                               #   [60 -> 100, 2026-07-01: operator-tested value, tighter
                               #   free-space tracking, kept for their tuned configs]
BETA = 0.4                     # How fast slack weights return to baseline as lambda grows
                               #   [1.0 -> 0.4: gentler curve, less abrupt swing near lambda~1]
SLACK_FILTER_TAU = 0.15        # LPF time constant on the shadow prices feeding the slack
                               #   scheduler (smooths the noisy raw lambda -> no weight jumps)
# --- Dynamic gamma (CLF) scheduling ---
GAMMA_CLF_DEFAULT = 0.75       # Static / initial CLF convergence rate (Vdot <= -gamma*V).
                               #   Set to (GAMMA_MIN+GAMMA_MAX)/2; the old 1.5 was above the
                               #   scheduled GAMMA_MAX, which was inconsistent.
GAMMA_MIN = 0.5                # Lower bound of the scheduled CLF gamma
GAMMA_MAX = 1.0                # Upper bound of the scheduled CLF gamma
BETA_GAMMA = 5.0               # How quickly gamma drops as the collision lambda grows
GAMMA_FILTER_TAU = 0.125       # Low-pass time constant for the gamma scheduler

# =============================================================================
# 3b. REFERENCE GOVERNOR (intermediate CLF-safety layer, 2026-07-01)
# =============================================================================
# An intermediate filter between the raw cartesian reference topic and the
# CLF's actual perceived reference. Bounds the position/orientation error and
# reference velocity/acceleration that the CLF must handle, preserving QP
# feasibility guarantees even under aggressive/discontinuous commands.
# See triago_control/qp_controller/reference_governor.py for the full design.
ENABLE_REFERENCE_GOVERNOR = False      # Master switch (False = raw passthrough, no filtering)

# --- Velocity shaping: clamp the reference velocity magnitude (direction preserved) ---
GOV_V_MAX_LIN = 0.20                  # [m/s] max linear reference velocity passed to the CLF
GOV_V_MAX_ANG = 1.2                   # [rad/s] max angular reference velocity passed to the CLF

# --- Position/orientation error bounding: the CLF never sees an error larger than this ---
GOV_E_MAX_POS = 0.30                  # [m] max allowed position error norm (30 cm)
GOV_E_MAX_ORI = 0.524                 # [rad] max allowed orientation error norm (~30 deg)

# --- Acceleration limiting: rate-limit the velocity change between ticks ---
GOV_A_MAX_LIN = 2.0                   # [m/s²] max linear acceleration of the governed reference
GOV_A_MAX_ANG = 8.0                   # [rad/s²] max angular acceleration of the governed reference

# =============================================================================
# 3c. LOCAL MINIMA ESCAPE (governor extension, 2026-07-01)
# =============================================================================
# Detects a possible QP-CLF-CBF local minimum (a large, near-constant 3D
# position tracking error) and applies a temporary, PER-ARM posture-weight
# correction to help escape it. See reference_governor.ReferenceGovernor.
# update_local_minima for the full state machine.
#
# NOTE: the lambda thresholds below are tuned by the operator for the
# CURRENT parameter set (GAMMA_CBF, D_SAFE_BASE, P_GAIN_LIMITS, etc. in
# section 2, and MAX_WEIGHT_SLACK etc. in section 3). If those are retuned,
# these thresholds may need to be revisited.
ENABLE_LOCAL_MINIMA_ESCAPE = True    # Master switch (independent of ENABLE_REFERENCE_GOVERNOR)

# --- Escape STRATEGY selector (2026-07-03) ---
# Detection (the "stuck" check below) and categorization ALWAYS run the same
# way regardless of this flag -- it only decides what CORRECTIVE ACTION is
# taken once a local minimum is detected:
#   'posture' : legacy behaviour. Immediately applies the posture-weight
#               nudge (+ task_dim=3 for an obstacle-induced minimum) while
#               ALSO letting the RRT planner fire in the background as a
#               fallback (RRT_TRIGGER_DELAY_S after the posture nudge started)
#               if the cheap heuristic alone doesn't resolve the error.
#   'rrt'     : the posture-weight nudge is NEVER applied (posture_scale stays
#               at 1.0 the whole time) -- the arm is only allowed to move out
#               of the local minimum once the RRT planner finds a validated,
#               collision-checked path and that path is actively being
#               executed. Until then the arm simply holds (QP naturally
#               commands ~0 velocity against the binding CBF/joint-limit row).
#               task_dim=3 (position-only) is applied ONLY while a planned
#               path is being tracked (see ReferenceGovernor.is_executing_path
#               in reference_governor.py) -- NOT tied to the raw detection
#               state anymore. This fixes a real bug: previously task_dim=3
#               was tied to "detection state == escaping", but consuming a
#               waypoint can itself reduce the position error enough to flip
#               detection back to 'normal' mid-path, snapping task_dim back to
#               6D while the arm was still only following a position-only
#               plan -- fighting itself. Now it is tied EXACTLY to "a
#               validated RRT path is being executed", restored to 6D the
#               instant execution ends (success / timeout / user override).
LME_ESCAPE_STRATEGY = 'rrt'          # 'posture' or 'rrt'

# --- Trigger: "stuck" detection (3D position error only, per instruction) ---
LME_ERROR_TRIGGER = 0.15             # [m] error norm above which "stuck" is considered
LME_ERROR_STUCK_WINDOW = 2.0         # [s] time window checked for a near-constant error
LME_ERROR_STUCK_TOLERANCE = 0.02     # [m] max variation within the window to call it "stuck"
LME_ERROR_RECOVERED = 0.05           # [m] error norm below which the escape ends (success)
LME_MAX_ESCAPE_DURATION = 10.0       # [s] max time the escape correction is held before giving up

# --- Categorization (shadow prices from the PREVIOUS QP solve, same
# convention as the slack/gamma scheduler). Obstacle takes priority if BOTH
# conditions are met simultaneously. ---
LME_LAMBDA_CBF_THRESHOLD = 10.0      # lambda_cbf > this -> obstacle-induced minimum
LME_LAMBDA_JOINT_THRESHOLD = 1.0     # lambda_joints > this -> joint-limit-induced minimum

# --- Escape corrections (posture task ONLY -- no other weight touched) ---
LME_POSTURE_SCALE_OBSTACLE = 0.2     # x1/5 posture weight (more redundancy to slip past the obstacle)
LME_POSTURE_SCALE_JOINT = 5.0        # x5 posture weight (push harder away from the limit)
LME_TASK_DIM_OBSTACLE = 3.0          # force position-only CLF (give up orientation) during obstacle escape
LME_RAMP_TAU = 0.3                   # [s] smooth ramp time-constant for the posture-scale change
                                     #   (same first-order technique as the grasp-phase POSTURE_SCALE_TAU ramp)

# --- Console reporting ---
LME_CONSOLE_PERIOD = 3.0             # [s] throttle period for the non-spam status print while escaping

# =============================================================================
# 3d. RRT-CONNECT JOINT-SPACE PLANNER (2026-07-01)
# =============================================================================
# A bidirectional RRT-Connect that plans in the FULL 7D joint-space of one arm,
# using the SAME hppfcl collision model the CBF uses — so the planner and the
# real-time controller agree on what's safe. Runs asynchronously in a background
# thread, triggered by the local-minima escape when the posture correction alone
# fails AND the reference is approximately still. Output: a sequence of Cartesian
# waypoints (derived from the planned collision-free joint configs via FK) fed to
# the governor's waypoint queue.
ENABLE_RRT_PLANNER = True             # Master switch (requires ENABLE_LOCAL_MINIMA_ESCAPE too)
# NOTE (2026-07-02): there is NO millisecond requirement on any of the planner's
# budgets below. While the planner thread runs (possibly for several seconds),
# the QP-CLF-CBF keeps driving the arm with the local-minima escape's posture
# correction (lowered posture weight + task_dim=3 for an obstacle-induced
# minimum) -- see reference_governor.update_local_minima_escape. Budgets were
# previously tuned for sub-second returns, which starved the goal-IK search
# (see §9.10/§9.11 in context.md) and are now widened.
RRT_PLANNING_BUDGET_S = 5.0           # [s] max wall-clock time RRT-Connect itself is allowed
RRT_STEP_SIZE = 0.15                  # [rad] max step per RRT extend in joint-space
RRT_GOAL_BIAS = 0.10                  # probability of sampling the goal directly (vs random)
RRT_GOAL_POS_TOLERANCE = 0.03         # [m] EE position tolerance for "goal reached" in the tree
RRT_PROGRESS_LOG_PERIOD_S = 1.0       # [s] throttle period for RRT-Connect's "still growing" console line
# --- Goal-configuration IK (damped least-squares, position-only) ---
# The goal config is found by DLS IK (FK(q).translation -> x_goal), NOT by
# uniform random rejection sampling (which cannot hit a 3cm Cartesian ball in
# 7D and always failed with samples=0). Restarts from random seeds if a solve
# converges into collision or stalls.
RRT_IK_DAMPING = 0.05                 # DLS damping (rad·s/m-ish): larger = more stable near singularities
RRT_IK_MAX_STEP = 0.20                # [rad] per-iteration joint step cap for the IK integrator
RRT_IK_TIME_BUDGET_S = 2.0            # [s] max wall-clock time the goal-IK search is allowed
RRT_IK_MAX_RESTARTS = 40              # max number of random-seed restarts within that time budget
RRT_IK_ITERS_PER_RESTART = 150        # DLS iterations per restart before declaring it stuck
# --- Goal-IK null-space obstacle avoidance (2026-07-02) ---
# Position is only 3 constraints on a 7-DOF arm (4 redundant DOF). Pure random
# restarts left those 4 DOF wherever chance put them, so the IK converged to the
# EXACT target position every time but almost always with a colliding elbow/
# wrist posture (a narrow obstacle-free band is unlikely to be hit by chance).
# Fix: once close to the target AND under-clear, add a SECONDARY task -- climb
# the (finite-differenced) collision-clearance gradient, PROJECTED into the
# primary task's null space so it never fights position convergence.
RRT_IK_CLEARANCE_MARGIN = 0.02        # [m] extra clearance beyond D_SAFE_BASE+RRT_COLLISION_MARGIN
                                      #   the goal config must achieve (buffer above the CBF's own
                                      #   margin so the accepted goal isn't immediately re-triggering)
RRT_IK_CLEARANCE_ACTIVATION_RADIUS = 0.08  # [m] position-error radius below which the secondary
                                           #   (clearance) task engages -- skips the extra distance
                                           #   queries while still far from the target (cheap)
RRT_IK_CLEARANCE_GAIN = 0.15          # [rad] per-iteration null-space step size along the
                                      #   normalized clearance gradient
RRT_IK_CLEARANCE_FD_EPS = 0.01        # [rad] finite-difference step for the clearance gradient
RRT_COLLISION_MARGIN = 0.005          # [m] extra safety margin ON TOP of D_SAFE_BASE for the planner
                                      #   (so the planned path is more conservative than the CBF)
RRT_MAX_SAMPLES = 8000                # absolute sample cap (fallback if wall-clock budget is generous)
RRT_SHORTCUT_ITERS = 50              # smoothing passes after a raw path is found
RRT_WAYPOINT_ADVANCE_THRESH = 0.04    # [m] ||x_real - x_wp|| below which the governor advances
                                      #   to the next waypoint in the queue
RRT_EXECUTION_ERROR_EXIT = 0.03       # [m] error below which the planned-path execution ends (success)
RRT_EXECUTION_TIMEOUT = 20.0          # [s] max time following the planned waypoints before giving up
RRT_REF_STILL_VELOCITY = 0.02         # [m/s] reference velocity norm below which "still" is declared
                                      #   (allows for small hand oscillation on the haptic device)
RRT_TRIGGER_DELAY_S = 4.0             # [s] time the local-minima escape must be active (posture
                                      #   correction already applied) before the RRT planner kicks in
                                      #   (gives the cheap heuristic a chance to work first)

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
# Head + gripper-finger joint names (for the slider GUI's live limit lookup
# only -- these are NOT part of the arm QP's decision vector).
HEAD_JOINTS = ['arm_head_1_joint', 'arm_head_2_joint', 'arm_head_3_joint',
               'arm_head_4_joint', 'arm_head_5_joint', 'arm_head_6_joint', 'arm_head_7_joint']
GRIPPER_FINGER_JOINTS = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

RIGHT_TCP_FRAME = 'gripper_right_grasping_link'
LEFT_TCP_FRAME  = 'gripper_left_grasping_link'
REF_FRAME = 'base_footprint'

# --- Head chain (2026-07-01): quasi-static CBF obstacle for the arms ONLY ---
# The head is mechanically identical hardware to the L/R arms (7-DOF), but is
# NOT part of THIS QP's decision vector -- it is driven by its own, separate
# vision-based controller (qp_head_visual_servo.py, future work). Here it is
# modeled purely as GEOMETRY: capsules built from the live FK of its own
# joints every tick (see CollisionManager.calculate_offsets / build_collision_model),
# so the arms can avoid it, without ever appearing in idx_right/idx_left (the
# QP's actuated joints) and without adding a single row/column of head
# velocity to the QP. `HEAD_TOOL_LINK` mirrors gripper_{side}_base_link's role
# for the arms: the fixed frame past arm_head_7_link used to size that link's
# capsule length.
HEAD_CHAIN = ['arm_head_1_link', 'arm_head_2_link', 'arm_head_3_link',
              'arm_head_4_link', 'arm_head_5_link', 'arm_head_6_link', 'arm_head_7_link']
HEAD_TOOL_LINK = 'arm_head_tool_link'

# --- Live joint-position slider GUI layout (plotter.py "Joint Positions" window) ---
# Shared, single source of truth for BOTH main_qp_controller.py (which computes
# and publishes the real limits from the live Pinocchio model via
# RobotKinematics.get_joint_limits) and plotter.py (which lays out the grid).
# Mirrors the reference control-panel image's column/row arrangement:
#   col 0 = left arm (7), col 1 = head (7), col 2 = right arm (7).
# (2026-07-01) The 4th "gripper" column was REMOVED -- with only 2 joints it
# left 5 empty cells and looked unbalanced. The two gripper fingers now live
# in their OWN dedicated row below this grid (see GRIPPER_SLIDER_ROW),
# intentionally laid out with a visual gap and NOT column-aligned with the
# arm/head grid above (see plotter.py Window 6). Rows 0-4 of the old 4th
# column, and the reference image's torso_lift_joint slider + joystick
# widget, remain INTENTIONALLY not encoded here per instruction.
SLIDER_LAYOUT = [
    ['arm_left_1_joint', 'arm_head_1_joint', 'arm_right_1_joint'],
    ['arm_left_2_joint', 'arm_head_2_joint', 'arm_right_2_joint'],
    ['arm_left_3_joint', 'arm_head_3_joint', 'arm_right_3_joint'],
    ['arm_left_4_joint', 'arm_head_4_joint', 'arm_right_4_joint'],
    ['arm_left_5_joint', 'arm_head_5_joint', 'arm_right_5_joint'],
    ['arm_left_6_joint', 'arm_head_6_joint', 'arm_right_6_joint'],
    ['arm_left_7_joint', 'arm_head_7_joint', 'arm_right_7_joint'],
]
# Dedicated gripper row (2026-07-01): rendered below SLIDER_LAYOUT in its own
# visually-separated section, deliberately NOT aligned to the 3 columns above.
GRIPPER_SLIDER_ROW = ['gripper_left_finger_joint', 'gripper_right_finger_joint']

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
