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
# WALL_COLLIDER: DEPRECATED (2026-07-04) -- only consulted by
# collision_manager.py's LEGACY (world_scene=None) fallback path. To enable/
# disable the wall in a world loaded via world_loader.py, set that world's
# YAML `virtual_wall` obstacle's `collision:` field instead (see
# config/worlds/no_obstacle.yaml, where it is False, matching this).
WALL_COLLIDER = False          # Enable the virtual collision wall (XZ plane) [legacy path only]
FLYING_OBSTACLE = False         # Enable the flying obstacle marker / collider
# PINHOLE_TASK: dead flag, never wired to any code path (kept only so any
# external reference to it doesn't raise AttributeError). Superseded by the
# `world_name` ROS parameter on main_qp_controller.py / main_shared_autonomy.py
# -- see world_loader.py.
PINHOLE_TASK = True             # Load the obstacle set from the PINHOLE world [DEAD FLAG]
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
BLENDING = True

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
#   lin_effort = clip(||v_user_lin|| / ALPHA_EFFORT_THRESHOLD, 0, 1)
#   ang_effort = clip(||v_user_ang|| / ALPHA_EFFORT_ANG_THRESHOLD, 0, 1)
#   effort = max(lin_effort, ang_effort)
#   alpha_effective = alpha_belief * (1 - effort * ALPHA_EFFORT_OVERRIDE)
#
# ORIENTATION SYMMETRY FIX (2026-07-03): the first version of this gate only
# read ||v_user[0:3]|| (linear), so spinning the handle produced ZERO effort
# and alpha never backed off for pure rotation -- the operator reported the
# gripper orientation stayed almost frozen while position was clearly
# steerable. ang_effort (from v_user[3:6]) now participates via max() with
# lin_effort, so EITHER a fast hand OR a fast wrist rotation hands the user
# authority, exactly mirroring how position is already treated. Per operator
# instruction, TASK_WEIGHTS_6D (the CLF's own position:orientation cost ratio)
# is intentionally NOT touched -- this fix operates purely at the alpha/blend
# level, upstream of the CLF.
#
# Deliberately NOT implemented on the force/haptic side (no Lagrangian
# multipliers, no discontinuous/filtered shadow-price feedback) -- this is a
# smooth function of the user's OWN commanded twist norm, always continuous
# and independent of any QP dual variable.
ALPHA_EFFORT_THRESHOLD = 0.4    # m/s -- user linear twist norm considered "fast hand
                                 #   movement"; effort saturates to 1.0 at/above this
ALPHA_EFFORT_ANG_THRESHOLD = 1.0  # rad/s -- user ANGULAR twist norm considered "fast wrist
                                 #   rotation" (2026-07-03, orientation symmetry fix). Scaled
                                 #   the same way as the linear threshold: 10x the comfortable
                                 #   teleop rate (w_max_ang_user=0.10 rad/s in main_shared_
                                 #   autonomy.py), mirroring how ALPHA_EFFORT_THRESHOLD=0.4 is
                                 #   10x the comfortable LINEAR rate (v_max_lin_user=0.04 m/s).
                                 #   effort = max(lin_effort, ang_effort) -- either a fast hand
                                 #   OR a fast wrist twist hands the user authority.
ALPHA_EFFORT_OVERRIDE = 0.5     # fraction of alpha_belief displaced by full user effort.
                                 #   0.5: at max effort, alpha is halved -- the rest of
                                 #   the reduction in autonomy "following" already comes
                                 #   from fast motion naturally lowering the belief
                                 #   estimate itself (see BeliefEstimator/engagement).
ALPHA_EFFORT_LPF_COEFF = 0.15    # low-pass filter on the effort signal (smooths the
                                 #   ||v_user|| norm before it gates alpha; independent
                                 #   of ALPHA_LPF_COEFF, which smooths the final alpha)

# --- Position-divergence authority override + reference catch-up (2026-07-03) ---
# Operator report: the velocity-effort gate above only reacts while the hand is
# ACTIVELY MOVING. The instant the user decelerates and holds their hand at a
# displaced position, ||v_user|| -> 0, the effort gate relaxes, and pi_policy
# (still large, belief-driven) dominates v_blend again -- so the robot barely
# moves toward where the user is HOLDING their hand, and the user ends up
# fighting F_sync (whose restoring force is proportional to exactly that
# unclosed gap) without the robot ever following through. This is architectural:
# the twist blend has no memory of the user's PERSISTENT REFERENCE POSE
# (current_T_user), only their instantaneous derivative (current_v_h).
#
# Two complementary, purely-geometric mechanisms (built only from
# current_T_user vs current_T_EE -- NOT from any QP Lagrangian/shadow-price,
# per explicit operator constraint that those are discontinuous or need
# lag-inducing filtering):
#
# 1. DIVERGENCE ALPHA OVERRIDE (same shape as the velocity-effort gate, but
#    driven by SUSTAINED position/orientation gap instead of instantaneous
#    twist -- does NOT decay when the user stops moving):
#      pos_div_t = smoothstep(||pos_user - pos_EE||, ALPHA_DIVERGENCE_NEAR, FAR)
#      ang_div_t = smoothstep(||log3(R_user @ R_EE^T)||, ALPHA_DIVERGENCE_ANG_NEAR, ANG_FAR)
#      div_effort = max(pos_div_t, ang_div_t)
#      alpha *= (1 - div_effort * ALPHA_DIVERGENCE_OVERRIDE)
#    ORIENTATION SYMMETRY FIX (2026-07-03): the first version only read the
#    position gap, so a user holding their reference ROTATED away from the
#    gripper (but at the same position) got zero override -- alpha stayed
#    belief-driven and the policy's angular twist dominated v_blend
#    unopposed, which is why orientation looked "almost frozen" while
#    position was clearly steerable. ang_div_t (via max()) now closes this
#    gap exactly the way pos_div_t already does for position.
#
# 2. BOUNDED REFERENCE CATCH-UP (the actual fix for "robot doesn't follow
#    through to where the hand is held"): a gentle, CAPPED P-control pull
#    ADDED directly onto the blended twist, toward the user's persistent
#    pose. Gated by a deadband so normal small tracking gaps are untouched.
#    It never bypasses the downstream QP CLF-CBF -- it only ever contributes
#    a bounded velocity INTO the same reference the QP tracks, so collision/
#    joint-limit safety is fully preserved; a genuinely blocked path still
#    can't be forced through.
ALPHA_DIVERGENCE_NEAR = 0.05      # m -- no override below this position gap
ALPHA_DIVERGENCE_FAR = 0.20       # m -- full override at/beyond this position gap
ALPHA_DIVERGENCE_ANG_NEAR = 0.15  # rad (~8.6 deg) -- no override below this orientation gap
                                  #   (2026-07-03, orientation symmetry fix: mirrors the SAME
                                  #   ratio as the linear NEAR/FAR pair -- see below)
ALPHA_DIVERGENCE_ANG_FAR = 0.60   # rad (~34 deg) -- full override at/beyond this orientation
                                  #   gap. Chosen to match CATCHUP_DEADBAND_ANG/CATCHUP_FULL_ANG
                                  #   exactly (same physical gap triggers both the alpha override
                                  #   AND the catch-up pull, so the two mechanisms agree on what
                                  #   counts as "the user has rotated the reference away").
ALPHA_DIVERGENCE_OVERRIDE = 0.6   # fraction of alpha displaced at full divergence (position
                                  #   OR orientation -- whichever is more diverged, see
                                  #   compute_alpha). Stronger than ALPHA_EFFORT_OVERRIDE=0.5:
                                  #   this is meant to be the DOMINANT, sustained signal.
ALPHA_DIVERGENCE_LPF_COEFF = 0.15  # LPF on the divergence-override signal (independent
                                  #   of ALPHA_LPF_COEFF / ALPHA_EFFORT_LPF_COEFF)

CATCHUP_DEADBAND_POS = 0.03       # m -- below this position gap, catch-up contributes nothing
CATCHUP_FULL_POS = 0.15           # m -- full catch-up gain reached at/beyond this gap
K_CATCHUP_LIN = 1.5               # P-gain (1/s) on the linear position gap (pre-clip)
V_CATCHUP_MAX_LIN = 0.06          # m/s -- hard cap on the catch-up linear velocity
                                  #   (deliberately gentle -- a slow, steady pull, not a snap)

CATCHUP_DEADBAND_ANG = 0.15       # rad (~8.6 deg) -- below this, no orientation catch-up
CATCHUP_FULL_ANG = 0.6            # rad (~34 deg) -- full orientation catch-up gain
K_CATCHUP_ANG = 1.0               # P-gain (1/s) on the orientation gap (pre-clip)
V_CATCHUP_MAX_ANG = 0.15          # rad/s -- hard cap on the catch-up angular velocity

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
ENABLE_REFERENCE_GOVERNOR = True       # Master switch (False = raw passthrough, no filtering)
                                       # [2026-07-04, operator-enabled for this test pass]

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
ENABLE_LOCAL_MINIMA_ESCAPE = False    # Master switch (independent of ENABLE_REFERENCE_GOVERNOR)
# NOTE (2026-07-03): an RRT-Connect joint-space planner was attempted as a
# fallback escape strategy for this mechanism (background-thread planning,
# Cartesian waypoint queue tracked by the reference governor). The attempt was
# UNSUCCESSFUL and has been fully removed from the codebase (rrt_planner.py
# deleted; reference_governor.py and main_qp_controller.py stripped of all RRT
# integration). The ONLY corrective action this flag can now enable is the
# posture-weight + task_dim nudge below (no planner, no strategy selector).

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
# 3d. [REMOVED] RRT-Connect joint-space planner (2026-07-01 -- 2026-07-03)
# =============================================================================
# A bidirectional RRT-Connect local-minima-escape fallback was attempted here
# (full 7D joint-space planning, damped-least-squares goal-IK with null-space
# obstacle avoidance, background-thread execution, Cartesian waypoint queue).
# The attempt was UNSUCCESSFUL and has been fully removed (2026-07-03):
# rrt_planner.py deleted; reference_governor.py, main_qp_controller.py, and
# plotter.py stripped of all RRT integration (subscriptions, telemetry,
# markers, trigger logic). ENABLE_LOCAL_MINIMA_ESCAPE (see §3c above) now
# offers ONLY the original posture-weight + task_dim correction.

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
CAPSULE_RADIUS = 0.06                 # DEFAULT radius of the arm collision capsules
                                       # (used for any link WITHOUT an entry in
                                       # CAPSULE_OFFSET_OVERRIDES below)

# =============================================================================
# 6b. PER-LINK CAPSULE ALIGNMENT OVERRIDES (2026-07-04)
# =============================================================================
# CollisionManager.calculate_offsets builds each arm/head link's capsule as a
# STRAIGHT segment directly between joint(i) and joint(i+1) (dominant-axis
# snapped) with the single global CAPSULE_RADIUS above. The real CAD mesh's
# true centerline is not always collinear with that line, and several links
# are physically thicker at their widest point than CAPSULE_RADIUS -- so a
# capsule built purely from the joint geometry can let the visual mesh poke
# outside it (verified via scripts/qp_arm_teleop/capsule_alignment_audit.py:
# every arm_*_{1..6}_link showed a real, measurable protrusion at the default
# 60mm radius, ranging ~2mm to ~33mm depending on the link).
#
# Rather than growing CAPSULE_RADIUS globally (which would fatten EVERY link
# uniformly -- including ones already fine -- silently loosening the tuned
# CBF margins, D_SAFE_BASE/ALPHA_SOFTMIN/etc., everywhere instead of just
# where needed), each entry here corrects ONE named link, computed directly
# from that link's real mesh vertex data (closed-form: see
# capsule_alignment_audit.py's `_compute_capsule_fix` for the exact
# derivation) -- NOT hand-guessed. A link absent from this dict is completely
# unaffected: CAPSULE_RADIUS + calculate_offsets' original placement/length,
# byte-identical to before this feature existed.
#
# Fields per entry (all in mm, joint-local frame -- the SAME frame
# calculate_offsets' placement/length already live in):
#   lateral_offset       : [x, y, z] shift of the capsule's core line,
#                           PERPENDICULAR to its own axis (re-centers onto
#                           the mesh's true centroid; by construction this
#                           vector has zero component along the axis).
#   radius                : per-link capsule radius, REPLACING CAPSULE_RADIUS
#                           for this link only.
#
# 2026-07-04 UPDATE (operator-requested reduction, exploratory pass): the
# tight-fit radii (raw "radius_needed" + 1mm margin) made the SoftMin CBF
# feel overly conservative once visualized in Meshcat/tested live -- every
# link visibly "puffed out" ~25% vs. the original 60mm. Per operator
# instruction, EVERY overridden radius below has been reduced by 5mm from
# its tight-fit value (i.e. now ~4mm BELOW the raw radius_needed, not above
# it) as an exploratory step to loosen the CBF's felt conservatism before
# deciding on D_SAFE_BASE / final tuning. This DELIBERATELY reintroduces a
# small (~4mm), KNOWN, ACCEPTED mesh protrusion on links 1-5 -- re-running
# capsule_alignment_audit.py (no --suggest-fix) is EXPECTED to show
# protrusion_mm ~= +4.00 on those links now; that is not a regression, it is
# the direct, deliberate consequence of this change, kept for easy
# before/after comparison. If a reduced-radius config is ever locked in
# permanently, the proximal/distal extensions below should be RE-DERIVED
# for the new (smaller) radius via a fresh --suggest-fix pass -- they are
# currently still the values computed for the ORIGINAL tight-fit radius
# (the axial hemisphere-cap reach shrinks with radius too, so these are a
# reasonable approximation for a small reduction like this, not an exact
# re-fit).
#   proximal_extension    : extends the segment's PROXIMAL end (t=0, at the
#                           parent joint) outward along its own axis, to
#                           cover mesh that reaches past the raw joint-to-
#                           joint segment (accounts for the capsule's rounded
#                           end caps not otherwise reaching that far).
#   distal_extension      : same, for the DISTAL end (t=1).
#
# Applied by CollisionManager._apply_capsule_override, called from
# build_collision_model's add_arm_geoms -- AFTER the existing dominant-axis
# snap, as a pure additive correction (never rewrites calculate_offsets'
# own straight-line-segment logic).
#
# Computed 2026-07-04 from the repo's own triago_extracted.urdf via:
#   ros2 run triago_control capsule_alignment_audit.py \
#        --urdf triago_extracted.urdf --suggest-fix
# The SAME override applies to arm_right_*/arm_left_*/arm_head_* for a given
# link NUMBER: the audit's real per-mesh measurement came out numerically
# identical across all three chains for each link number (consistent with
# config.py's own documented convention that the head reuses the arms'
# exact hardware/geometry recipe, and that this URDF defines both arms with
# consistent local joint-frame conventions). If this dict is ever
# regenerated against a DIFFERENT URDF and right/left no longer match,
# split them into separate 'arm_right_N_link'/'arm_left_N_link' entries.
#
# VERIFY after any change here by re-running the audit WITHOUT --suggest-fix.
# NOTE (2026-07-04, see the radius comment above): with the operator-
# requested -5mm reduction below, links 1-5 are now EXPECTED to show
# protrusion_mm ~= +4.00 (not <= 0) -- this is the deliberate, accepted
# result of this exploratory pass, not a bug.
#
# Link 6 REMOVED entirely (2026-07-04, operator instruction): the grasping
# phase was tuned against link 6's ORIGINAL geometry (CAPSULE_RADIUS=60mm,
# no offset/extension) -- restoring it here means link 6 is COMPLETELY
# ABSENT from CAPSULE_OFFSET_OVERRIDES below, which by this file's own
# contract ("A link absent from this dict is completely unaffected") makes
# it byte-identical to before this whole feature existed: pure
# calculate_offsets() placement/length + the global CAPSULE_RADIUS. Its
# ORIGINAL measured misalignment (33.37mm protrusion at 60mm radius, per
# the very first capsule_alignment_audit.py run) is therefore KNOWINGLY
# reintroduced -- an explicit operator trade-off (grasp-tuning fidelity
# over collision-geometry tightness for this one link), not an oversight.
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

# --- DEPRECATED (2026-07-04): world scene obstacles now live in YAML --------
# TABLE_POS/TABLE_SIZE/RED_CYLINDER_POS/BLUE_CYLINDER_POS/CYLINDER_SIZE/
# WALL_POS/WALL_SIZE/WALL_COLLIDER (section 1) are SUPERSEDED by
# config/worlds/<world_name>.yaml, loaded via world_loader.load_world() and
# passed as `world_scene` into CollisionManager.build_collision_model() /
# VisualizationEngine(...). See world_loader.py's module docstring for the
# full schema and how to author a new world (different table/cylinder pose or
# size, extra obstacles for a harder task, etc.) -- and main_qp_controller.py /
# main_shared_autonomy.py's `world_name` ROS parameter to select one at
# runtime, e.g.:
#   ros2 run triago_control main_qp_controller.py --ros-args -p world_name:=no_obstacle
#
# These constants are KEPT (not deleted) purely as the LEGACY FALLBACK path:
# any call site that constructs CollisionManager/VisualizationEngine WITHOUT
# passing a world_scene (world_scene=None) still reads these exact values, so
# behavior for such a caller is byte-for-byte unchanged. Do not add new
# obstacles here -- add them to a world YAML instead.
CYLINDER_SIZE = [0.02, 0.15]          # [Radius, Length] of the workspace cylinders
RED_CYLINDER_POS = [0.800, -0.20, 0.775]
BLUE_CYLINDER_POS = [0.800, 0.20, 0.775]
TABLE_POS = [1.0, 0.0, 0.35]
TABLE_SIZE = [0.6, 0.5, 0.7]

WALL_SIZE = [1.0, 0.02, 1.0]          # Virtual wall [length_x, thickness_y, height_z]
WALL_POS = [0.5, 0.0, 0.5]            # Virtual wall position relative to base_link

# =============================================================================
# 7. OFFLINE PLOTTER (static, publication-quality figures, 2026-07-04)
# =============================================================================
# See scripts/qp_arm_teleop/offline_plotter.py for the full design. Two
# concerns live here so every current/future publisher of the recording
# trigger agrees on the exact same values without importing each other.

# Generic recording-trigger contract (std_msgs/Bool): True = "the active
# motion source is now commanding real motion, start/continue recording";
# False = "the commanded motion has concluded". offline_plotter.py owns ALL
# post-trigger behaviour (see OFFLINE_PLOT_POST_TRIGGER_S below) -- the
# publisher only ever reports the raw on/off signal and knows nothing about
# how it is used downstream. trajectory_generator.py drives this today (True
# on WAITING->TRACKING, False on TRACKING->REGULATION); a future
# teleoperation-side trigger (e.g. "handle grasped, clutch released") can
# drive the exact same topic later without any change to offline_plotter.py.
OFFLINE_RECORD_TRIGGER_TOPIC = "/offline_plotter/record_trigger"

# Root directory under which each recorded trial gets its own timestamped
# subfolder (see offline_plotter.py's _finalize_and_save). Change if
# ~/exchange isn't the mount point on a given machine.
OFFLINE_PLOT_ROOT_DIR = "~/exchange/ros2-ws/triago_offline_plots"

# How long (seconds) offline_plotter.py keeps recording AFTER the trigger
# above goes False, before finalizing and saving the figures. This is what
# captures the REGULATION/settling phase on the SAME time axis as the
# tracking motion; a vertical dashed grey line is drawn at the exact instant
# the trigger went False on every time-series subplot (unlabeled -- no
# legend entry needed, per instruction).
OFFLINE_PLOT_POST_TRIGGER_S = 10.0
