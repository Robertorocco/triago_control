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
DEBUG = False                   # Verbose timing / kinematics console tracing
GRASP_DEBUG = True              # Verbose grasp / CBF-bypass interaction tracing
DISABLE_CBF = False             # Mathematically delete the collision barrier
DYNAMIC_CBF = False             # Dynamically remove pairs for interaction
DYNAMIC_SLACK_WEIGHT = True    # Increase slack weights in free space, drop near obstacles
                               # (keep True: disabling costs tracking, gains nothing -- context.md)
COMPARISON_CLF = True           # Use the normalized (unit-error) scalar CLF formulation
DYNAMIC_GAMMA_CLF = False       # Vary CLF convergence rate with the safety margin
SIMULATE_IDEAL_KINEMATICS = False  # True = pure math digital twin, False = real hardware
ORIENTATION_CTRL = True         # True = control Pos+Ori (6DOF), False = Pos only (3DOF)

# =============================================================================
# 1b. EXPERIMENT CONDITION SELECTOR (fair 2x3 teleoperation study design)
# =============================================================================
# The user study compares TWO control modes across the full factorial of the two
# orthogonal assistance channels below (four F/B combinations per mode).
# Historically this was governed by the single boolean `BLENDING`, which
# conflated three independent things (which teleop node, which haptic force
# manager, and whether main_shared_autonomy blends the reference). That made the
# clutch vs joystick comparison UNFAIR: the clutch condition only ever closed an
# assistive-FEEDBACK loop (guidance forces on the handle) while the joystick
# condition only ever closed an assistive-ACTION loop (reference blending).
#
# The design now exposes TWO ORTHOGONAL assistance channels so any combination
# can be selected independently of the control mode:
#   * ASSIST_FEEDBACK  (channel F) -- assistive haptic FORCES on the handle
#                                      (F_guide velocity field + F_fixture funnel),
#                                      on top of the always-present F_sync tether.
#   * ASSIST_BLENDING  (channel B) -- reference-level ACTION arbitration of the
#                                      user twist with the belief-weighted policy
#                                      in main_shared_autonomy (sole writer of
#                                      /arm_*/cartesian_reference).
#
# The FULL 2x2x2 factorial (see .kiro/context.md for the authoritative table):
#
#   CONTROL_MODE  ASSIST_FEEDBACK  ASSIST_BLENDING   condition              force manager
#   ------------  ---------------  ---------------   --------------------   ----------------------------------------
#   CLUTCH        False            False             Sync only              haptic_force_manager_C
#   CLUTCH        True             False             Guided feedback (VF)   haptic_force_manager_CF
#   CLUTCH        False            True              Guided blending        haptic_force_manager_CB
#   CLUTCH        True             True              Full guidance          haptic_force_manager_CFB
#   JOYSTICK      False            False             Sync only              haptic_force_manager_J
#   JOYSTICK      True             False             Guided feedback        haptic_force_manager_JF   
#   JOYSTICK      False            True              Guided blending        haptic_force_manager_JB
#   JOYSTICK      True             True              Full guidance          haptic_force_manager_JFB
#
# CB and JF are the two off-diagonal cells the original "fair 2x3" omitted; they
# complete the factorial for the paper even though each pairs a control mode with
# its NON-native assist channel (clutch is a position/Virtual-Fixture framework
# not meant to run on blending alone; the velocity joystick was conceived as the
# smart auto-blending solution). Included for a complete user-study comparison.
#
# Force-manager naming: every file is haptic_force_manager_<CELL>, where <CELL>
# encodes the active study condition -- C=CLUTCH / J=JOYSTICK (control mode,
# always first), then F if ASSIST_FEEDBACK and B if ASSIST_BLENDING. Baseline
# (no assist) is just the mode letter. So C/CF/CB/CFB (clutch) and J/JF/JB/JFB
# (joystick) -- eight cells total.
#
# Each teleop / force-manager node calls cfg.validate_condition(...) at startup
# and HARD-ERRORS if the launched node does not match the selected condition, so
# a mis-launched experiment fails loudly instead of silently running the wrong
# strategy.
CLUTCH = "CLUTCH"
JOYSTICK = "JOYSTICK"

# --- Active experiment condition (edit these three to select a study cell) ---
CONTROL_MODE   = CLUTCH      # CLUTCH (position control) | JOYSTICK (velocity control)
ASSIST_FEEDBACK = True      # channel F: assistive guidance forces on the handle
ASSIST_BLENDING = False      # channel B: reference-level user<->policy blending

# --- Backward-compatibility alias -------------------------------------------
# Legacy code (teleop topic routing, main_shared_autonomy) reads cfg.BLENDING to
# decide whether main_shared_autonomy owns /arm_*/cartesian_reference. Blending
# is ON in exactly the ASSIST_BLENDING conditions, so BLENDING mirrors it.
BLENDING = ASSIST_BLENDING


def validate_condition(node_name, control_mode=None, feedback=None, blending=None):
    """Startup guard: assert the launched node matches the selected study cell.

    Each node passes ONLY the constraints it cares about (a teleop node fixes the
    control mode; a force manager fixes the full triple). Any provided constraint
    that disagrees with the active config raises RuntimeError with a clear message
    so a mis-launched condition fails loudly. Returns None on success.
    """
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
# The handle is spring-centered to this pose; its displacement from it IS the
# command. Position is FIXED. Orientation TRACKS the gripper via a per-arm
# reference delta (see teleop_triago_joystick.py._update_home_orientation):
#   1. Capture the gripper's orientation ONCE, the first time each arm becomes
#      active -- that arm's reference (its home then == neutral).
#   2. Each tick, take the gripper's DELTA from that reference (base frame).
#   3. Scale the delta ANGLE by 1/JOYSTICK_ROT_HOME_SCALE to compress it into the
#      handle's more limited rotational workspace, and map its axis to the Haption
#      frame (180°-Z frame flip).
#   4. Apply the scaled/mapped delta to JOYSTICK_NEUTRAL_ORIENTATION_XYZW → home_rot.
# So at the reference pose home_rot = neutral_rot (zero displacement = zero twist);
# as the gripper rotates (grasp approach, post-grasp, etc.) the home follows
# smoothly. The reference is saved/restored per arm and NEVER re-anchored after
# first capture, so the home stays continuous (no jump-to-neutral) across grasps
# and arm switches.
JOYSTICK_NEUTRAL_POSITION_M = [0.5, -0.03, -0.03]
# Measured on the device (virtuose/pose) with the handle at the operator's
# comfortable rest orientation — the pose the spring controller settles at when
# the user releases the handle. This is the neutral handle quaternion that each
# arm's captured gripper reference maps to: at first activation home == this quat,
# and any subsequent gripper rotation away from the reference produces a
# proportional (scaled) deviation of the home from it.
JOYSTICK_NEUTRAL_ORIENTATION_XYZW = [-0.015140674076974392, 0.8170770406723022,
                                     0.06124841421842575, 0.5730659365653992]

# --- Displacement -> twist mapping (teleop_triago_joystick.py) --------------
# The commanded twist magnitude is STRICTLY PROPORTIONAL to the handle's distance
# from home, past a deadband. Deadband: displacement below these -> zero twist
# (and, in the arbitration, a zero user twist is treated as perfectly ALIGNED so
# the autonomy leads the motion). Shrunk further for a more responsive handle:
# LIN -20% (0.0864 -> 0.06912), ANG -10% (0.2592 -> 0.23328) -- still large enough
# to reject the centering spring's residual settle-oscillation near home.
JOYSTICK_DEADBAND_LIN = 0.06912      # m   (6.91 cm) -- large: the spring can't settle to mm precision
JOYSTICK_DEADBAND_ANG = 0.23328      # rad (~13.37 deg) -- large: residual oscillation around home
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

# --- Restorative centering spring (haptic_force_manager_JB.py) --
# The ONLY haptic force rendered in joystick mode: a virtual spring-damper pulling
# the handle back to the (dynamic) home pose. No F_guide / F_fixture / F_sync.
JOYSTICK_SPRING_KP_LIN = 60.0        # N/m        (stiffer: resists driving far from home)
JOYSTICK_SPRING_KD_LIN = 1.0         # N/(m/s)    damping raised with the stiffer spring
JOYSTICK_SPRING_KP_ANG = 1.5         # Nm/rad
JOYSTICK_SPRING_KD_ANG = 0.15        # Nm/(rad/s)

# Topic carrying the live joystick home pose (teleop_triago_joystick.py ->
# haptic_force_manager_JB.py) so both nodes agree on the SAME
# dynamic home (single source of truth -- the teleop node owns it). Layout:
#   [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w] in the Haption base frame.
JOYSTICK_HOME_POSE_TOPIC = "/joystick/home_pose"

# --- Twist arbitration (main_shared_autonomy.compute_alpha) -----------------
# The autonomy authority alpha (weight on the optimal policy in the blend
#   v_blend = (1 - alpha) * v_user + alpha * pi_policy)
# is a function ONLY of the ALIGNMENT s in [-1,1] between the user twist and the
# policy twist (mean per-channel cosine over whichever channels the user is
# actively commanding). Two-sided ramp, continuous at s = 0 (alpha = ALIGN_ALPHA_MIN):
#   s >= 0:  alpha = ALIGN_ALPHA_MIN + (ALIGN_ALPHA_MAX - ALIGN_ALPHA_MIN) * s
#   s <  0:  alpha = ALIGN_ALPHA_MIN * (1 + s)
# Aligned      (s =  1) -> ALIGN_ALPHA_MAX  (policy 80%, user 20%) -- fast, but ONLY
#                          while the user actively moves in agreement.
# Perpendicular(s =  0) -> ALIGN_ALPHA_MIN  (policy 20%, USER 80%).
# Opposed  (s -> -1)    -> alpha -> 0: the autonomy authority ramps DOWN as the user
#                          drives against the policy (e.g. s=-0.5 -> ALIGN_ALPHA_MIN/2,
#                          s=-1 -> 0), so a deliberate counter-motion meets ZERO
#                          resistance instead of a permanent ALIGN_ALPHA_MIN counter-pull.
# No user input (handle inside the joystick deadband, v_user = 0) uses a REDUCED
# idle authority ALIGN_ALPHA_IDLE, so the robot only GENTLY crawls toward the
# inferred goal when the user is still; the instant the user pushes in a direction
# the policy agrees with, alpha ramps up to ALIGN_ALPHA_MAX (fast).
ALIGN_ALPHA_MIN = 0.2                # policy weight at perpendicular alignment (s=0); ramps to 0 as s->-1
ALIGN_ALPHA_MAX = 0.8                # policy weight when aligned AND the user is moving in agreement
ALIGN_ALPHA_IDLE = 0.35             # policy weight when the user is still (gentle autonomous crawl)
ALIGN_ALPHA_LPF_COEFF = 0.1          # low-pass on alpha for C0-continuity of the blend

# =============================================================================
# 2. SAFETY + CONTROL HYPERPARAMETERS
# =============================================================================
ALPHA_SOFTMIN = 50.0            # Sharpness of the SoftMin collision aggregation
GAMMA_CBF = 0.70               # CBF class-K gain
D_SAFE_BASE = 0.015            # Base safety distance for the collision barrier
K_V_SAFE = 0.1                 # Predictive velocity horizon (brake earlier at high speed)

# Local-minima stall escape (main_qp_controller.py::_apply_stall_escape): the
# reactive CLF-CBF QP has no planner, so a goal on the far side of a flat
# obstacle (e.g. reaching up and over a table from underneath) can settle at a
# genuine saddle point -- goal-gradient and barrier-gradient anti-parallel.
ENABLE_STALL_ESCAPE = False
STALL_SPEED_FILTER_ALPHA = 0.97 # EMA coeff smoothing the raw EE-speed estimate
                                 # before the stall check (real hardware's raw
                                 # current_v is jittery and rarely reads near-zero)
STALL_SPEED_THRESH = 0.01       # [m/s] "almost zero" measured EE speed
STALL_HOLD_S = 3.0              # [s] sustained stall before escaping
STALL_ERR_POS_THRESH = 0.10     # [m] position error still counts as "not there yet"
STALL_ERR_ANG_THRESH = np.deg2rad(30.0)  # [rad] orientation error, same idea
STALL_RESUME_SPEED = 0.03       # [m/s] speed that counts as "moving again" -> drop escape
STALL_TANGENT_MIN = 0.15        # goal direction's component tangential to the
                                 # blocking normal below this -> use the fallback tangent
STALL_ESCAPE_STEP_M = 0.05      # [m] waypoint offset along the tangent
STALL_ESCAPE_SPEED = 0.03       # [m/s] feedforward speed along the tangent

# Near-goal policy strength (main_shared_autonomy.py, real hardware + unattended
# policy-test runs ONLY -- see its own use of these next to the POLICY_BELIEF_TEST
# virtual-cursor dt-stretch. The QP controller itself is NOT touched: this only
# shapes the REFERENCE main_shared_autonomy.py publishes, same as it always has.
POLICY_NEAR_GOAL_POS_M = 0.10                 # [m]
POLICY_NEAR_GOAL_ANG_RAD = np.deg2rad(15.0)   # [rad]
POLICY_NEAR_GOAL_MIN_LEAD_M = 0.05            # [m] stretched carrot-lead floor
POLICY_NEAR_GOAL_DT_MAX_S = 2.0               # [s] stretched dt_virtual cap

# In compute_softmin_jacobian, an inter-arm pair's margin normally reflects only
# the row-owning arm's own speed (d_safe_dynamic_r/l below); when True, it's
# widened by the OTHER arm's speed too, for pairs that touch both arms (see
# .kiro/context.md §8.3). Off by default -- didn't fix the ripple it was tested
# against, kept as an optional, cost-neutral margin refinement.
ENABLE_INTER_ARM_CLOSING_MARGIN = False
ALPHA_FILTER = 0.5            # EMA coefficient for hardware velocity filtering (~20ms window)
DAMP = 12.0                    # Joint velocity regularization (Lambda) in the QP cost.
                               # 12 is tuned: 10 left a bimanual-convergence ripple, 15 bought
                               # nothing more and cost tracking (see .kiro/context.md).
P_GAIN_LIMITS = 1.8            # Joint-limit CBF gamma (braking aggressiveness)
JOINT_LIMIT_BUFFER_BASE = 0.15  # Base joint-limit braking buffer
JOINT_LIMIT_K_V = 0.1          # Joint-limit velocity horizon (seconds to look ahead)
# Do not add a hard slew-rate box on q_dot: it can't relieve the CBF row and
# breaks QP feasibility when the CBF's required direction changes quickly.

# Posture / joint-limit avoidance: repulsive potential field, negative gradient of a
# barrier that diverges at each joint's limits, evaluated on the normalized joint
# position so every joint is defended at the same fraction of its travel. Near-zero
# in mid-range (CLF keeps tracking priority), grows sharply (clamped) near a limit.
K_GRADIENT = 0.07              # gain on the negative potential gradient
V_MAX_POSTURE = 1.0            # rad/s hard clamp on the posture reference (solver safety)
W_CENTER = 0.96                # posture-task weight in the QP cost (vs DAMP=10)
POSTURE_GRASP_SCALE = 0.05     # posture weight scaled to this (x W_CENTER) during autonomous
                               #   precision phases (grasp/align/approach/close/lift)
POSTURE_SCALE_TAU = 0.2        # s -- first-order ramp time-constant for the posture-scale switch

# Rate damping: + RATE_WEIGHT * ||dq - dq_measured||^2 on the arm joints, anchoring
# the weakly-pinned wrist DOFs to the arm's real state each tick (kills qdot
# oscillation). RATE_DAMPING_VS_MEASURED=False anchors on the QP's own last
# output instead -- rejected: a self-referential low-pass, worse real motion.
ENABLE_RATE_DAMPING = True
RATE_DAMPING_VS_MEASURED = True
RATE_WEIGHT =1700.0           # hand-tuned on real hardware -- adopted, current solution

# =============================================================================
# 3. DYNAMIC SCALING BOUNDARIES
# =============================================================================
# Decoupled dynamic slack weighting
BASE_WEIGHT_SLACK = 25.0        # Standard slack weight (active against an obstacle)
MAX_WEIGHT_SLACK = 120.0       # Max slack weight (free space); also the fixed slack
                               #   weight pinned on an inactive (frozen) arm to decouple it
BETA = 0.4                     # How fast slack weights return to baseline as lambda grows
SLACK_FILTER_TAU = 0.15        # LPF time constant on the shadow prices feeding the scheduler
WEIGHT_SLACK_FILTER_TAU = 0.2  # 2nd-stage LPF on the slack weight itself (smooths the jump)

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
GOV_V_MAX_ANG = 0.3                   # [rad/s] max angular reference velocity passed to the CLF

GOV_E_MAX_POS = 0.30                  # [m] max allowed position error norm (30 cm)
GOV_E_MAX_ORI = 1.0                   # [rad] max allowed orientation error norm

GOV_A_MAX_LIN = 1.0                   # [m/s^2] max linear acceleration of the governed reference
GOV_A_MAX_ANG = 4.0                   # [rad/s^2] max angular acceleration of the governed reference

# =============================================================================
# 4. LOOP / TELEMETRY SETTINGS
# =============================================================================
CONTROL_FREQ_DEFAULT = 150.0   # Default control loop frequency [Hz]
PUBLISH_EVERY_N = 2            # Publish 1 of every N iterations to the dashboard
WATCHDOG_TIMEOUT = 0.5         # Seconds without a reference before motion is frozen
DISTANCE_FILTER_THRESHOLD = 0.15  # Ignore collision pairs farther than this [m]
K_MAX_PAIRS = 60               # Max number of closest pairs fed into the SoftMin

# --- Real-hardware async execution (main_qp_controller_real.py only) ---------
# These take effect ONLY in the real-hardware subclass (main_qp_controller_real.py)
# AND only when REAL_HARDWARE is actually detected from the URDF; the base
# main_qp_controller.py and every simulation run ignore them entirely. The real
# subclass moves the SoftMin CBF (the dominant per-tick cost on the robot) onto a
# worker thread and the RViz debug overlays onto a second thread, so the control
# tick no longer blocks on either. A staleness watchdog freezes BOTH arms (zero
# velocity command, auto-resume) if the CBF result is unrefreshed for
# CBF_STALENESS_MAX_TICKS consecutive ticks -- so async never lets the QP act on
# an unboundedly-old barrier. Set either flag False to run the real controller
# synchronously (identical to the base) for a direct A/B comparison, without a
# code edit.
REAL_ASYNC_CBF = True          # Run the SoftMin CBF on its own worker thread
REAL_ASYNC_VIZ = True          # Run the RViz debug overlays on their own thread
CBF_STALENESS_MAX_TICKS = 3    # Freeze both arms if the CBF is unrefreshed this
                               # many consecutive ticks (auto-resume on refresh)

# Diagonal task weights [Px,Py,Pz,Roll,Pitch,Yaw]: heavily penalize position, barely
# penalize orientation (25:1 ratio), so the QP prioritizes closing position error
# over matching orientation when the two conflict near an obstacle.
TASK_WEIGHTS_6D = np.array([1.0, 1.0, 1.0, 0.05, 0.05, 0.05]) * 10.0

# Task weights used by an arm doing PRECISION work with the object (qp_formulator
# selects this per arm from orient_boost_arms). Applies to: the active arm during
# autonomous grasp/release execution (grasp_active -> GRASP_ALIGN/APPROACH/CLOSE,
# LIFT, RELEASE_LIFT) AND any arm currently CARRYING an attached object (the whole
# HOLDING / placement-approach phase, so the release pose is aligned too). Same
# POSITION weights as TASK_WEIGHTS_6D but ORIENTATION is raised 0.4 -> 2.0 (25:1 ->
# 5:1), so the gripper's approach-axis / placement orientation converges tightly.
# The nominal 25:1 ratio is ideal for free-space approach (position-first,
# orientation yields near obstacles) but leaves too much residual orientation error
# once the gripper is on / holding the object. STATIC per-phase swap (one discrete
# vector when boosted, the nominal vector otherwise) -- NOT a continuous /
# shadow-price-driven weight update.
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
# Per-link correction where the real CAD mesh isn't collinear with the
# straight joint-to-joint capsule (see capsule_alignment_audit.py --suggest-fix).
# mm, joint-local frame: lateral_offset re-centers the core line; radius
# overrides CAPSULE_RADIUS; proximal/distal_extension extend the segment ends.
# Link 6 intentionally absent (grasp tuning depends on its un-overridden geometry).
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
OFFLINE_PLOT_POST_TRIGGER_S = 15.0

# The WAITING->TRACKING trigger is a single VOLATILE Bool; a late-joining
# subscriber (e.g. cross-network) can miss it. The generator holds in WAITING
# until the recorder subscribes, up to this many seconds (0.0 = no wait).
OFFLINE_RECORD_WAIT_TIMEOUT_S = 10.0

# offline_plotter.py spawns `ros2 bag record` for the trial window into
# <trial_dir>/bag, so any trial can be replayed offline later.
OFFLINE_BAG_ENABLE = True
OFFLINE_BAG_STORAGE_ID = "sqlite3"

# The OFFICIAL recorder: superset of the QP-controller telemetry this file
# always had PLUS scripts/analysis/study_config.py's curated shared_autonomy/*
# + virtuose/* (teleoperation) allowlist PLUS the head-perception topics that
# allowlist never carried. One list, so a bag from either workflow captures
# the same data; harmless to list topics nothing publishes this run (ros2 bag
# record only captures what's actually publishing).
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
    # Head perception + active-vision (main_head.py) -- optional per trial,
    # harmless to always list; ros2 bag record only captures what's publishing.
    "/head_perception/telemetry",
    "/head_perception/markers",
    "/head_perception/qdot_cmd",
    "/head_perception/qdot_measured",
    "/head_perception/debug_json",
    "/perceived_world/snapshot",
    "/real_perception",
    "/head_active_tracking/telemetry",
    "/head_active_tracking/qdot",
    # Teleoperation device (haption_teleoperation) -- optional per trial.
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
    # Shared autonomy (main_shared_autonomy.py) -- optional per trial.
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
