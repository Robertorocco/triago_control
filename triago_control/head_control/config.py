"""
Single source of truth for the head perception + look-at subsystem.

WHY a dedicated config module:
    Mirrors the project convention (``qp_controller/config.py``). Every tunable
    value — topic names, geometric thresholds, control gains — lives here so the
    rest of the package never hard-codes a magic number. When something does not
    work, this is the ONE file you touch to retune.

FRAME CONVENTIONS (read this before editing anything geometric):
    * Camera *optical* frame (REP-103): +Z points FORWARD out of the lens,
      +X points RIGHT in the image, +Y points DOWN in the image. This is the
      frame in which raw deprojected points live.
    * ``base_footprint``: +X forward, +Y left, +Z up. Located on the ground at
      the centre of the mobile base. We express the *known* table pose and all
      detection *outputs* in this frame.
    * ASSUMPTION: the robot is spawned at the world origin, so
      ``base_footprint`` coincides with the Gazebo world frame. If that is ever
      false, only ``BASE_POSE_IN_WORLD`` below needs changing — the camera pose
      itself is always computed *relative to base_footprint* via Pinocchio FK,
      so it is robust regardless of the URDF root link.
"""

import numpy as np

# =============================================================================
# 1. CAMERA TOPICS  (override at runtime with ROS params of the same lowercase
#    name, e.g.  --ros-args -p color_topic:=/my/color)
# =============================================================================
# Real TRIAGo head camera topics (RealSense D455, PAL-configured).
# NOTE: depth is NOT aligned to color — it uses its own intrinsics/resolution.
# We subscribe to the DEPTH camera_info for deprojection (not color).
COLOR_TOPIC = "/gripper_head_camera_rgbd/color/image_raw"
DEPTH_TOPIC = "/gripper_head_camera_rgbd/depth/image_raw"
CAMERA_INFO_TOPIC = "/gripper_head_camera_rgbd/depth/camera_info"

# Optical frame of the head camera (must match the URDF / existing servo script).
CAMERA_OPTICAL_FRAME = "gripper_head_camera_rgbd_color_optical_frame"
# Reference frame for control targets and detection outputs.
BASE_FRAME = "base_footprint"

# =============================================================================
# 2. KNOWN PRIOR KNOWLEDGE  (the ONLY thing we tell the algorithm in advance)
# =============================================================================
# Table pose in the WORLD frame, taken from the Gazebo SDF:
#   <model name="work_table"> <pose>1.000 0.0 0.35 0 0 0</pose>
#   <box><size>0.6 0.5 0.7</size></box>
# The box is centred at z=0.35 with height 0.7, so its TOP surface is at z=0.70.
TABLE_CENTER_WORLD = np.array([1.000, 0.0, 0.35])   # geometric centre of the box
TABLE_SIZE = np.array([0.6, 0.5, 0.7])              # x, y, z extents [m]
TABLE_TOP_Z_WORLD = TABLE_CENTER_WORLD[2] + TABLE_SIZE[2] / 2.0   # = 0.70 m

# Robot base pose in world. Identity == robot spawned at world origin.
# (x, y, z, roll, pitch, yaw). Only edit if the robot is NOT at the origin.
BASE_POSE_IN_WORLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Convenience: table centre expressed in base_footprint (derived from the two
# poses above). With identity base pose this equals the world coordinates.
TABLE_CENTER_BASE = TABLE_CENTER_WORLD - BASE_POSE_IN_WORLD[:3]
TABLE_TOP_CENTER_BASE = np.array(
    [TABLE_CENTER_BASE[0], TABLE_CENTER_BASE[1], TABLE_TOP_Z_WORLD - BASE_POSE_IN_WORLD[2]]
)

# =============================================================================
# 3. HEAD KINEMATICS  (identical hardware to the L/R arms — 7-DOF)
# =============================================================================
HEAD_CONTROLLER = "arm_head_joint_space_controller_vel"
HEAD_CONFLICTING_CONTROLLER = "arm_head_controller"   # default trajectory ctrl

HEAD_JOINTS = [
    "arm_head_1_joint", "arm_head_2_joint", "arm_head_3_joint",
    "arm_head_4_joint", "arm_head_5_joint", "arm_head_6_joint", "arm_head_7_joint",
]

# =============================================================================
# 4. LOOK-AT CONTROL  (point the camera optical +Z axis at the table)
# =============================================================================
LOOKAT_LAMBDA = 2.0          # proportional gain on the angular look-at error
MAX_HEAD_VELOCITY = 0.25     # rad/s per joint (moderate, allows tracking the scan)
# Per-joint velocity-regularisation weights: heavier on proximal joints so the
# coarse pointing is done by the wrist, keeping motion smooth and predictable.
HEAD_JOINT_WEIGHTS = np.array([50.0, 40.0, 30.0, 10.0, 5.0, 1.0, 1.0])
LOOKAT_SLACK_WEIGHT = 500.0  # HIGH penalty: the look-at task dominates over posture

# Posture target for the REDUNDANT DOF. The head has 7 DOF but look-at only
# constrains 2 (pointing direction); the remaining 5 are resolved by pulling
# toward this posture. We use a KNOWN-GOOD table-observation config instead of
# joint mid-range, so the camera settles at a forward, top-down viewpoint with
# good table coverage — NOT the grazing "camera above the base" pose that
# mid-range produces.
#
# UPDATED 2026-07-02 (accuracy pass): moved from ~0.81m to ~0.63m camera-to-
# table-centre distance (verified via Pinocchio FK against the real URDF —
# reachable with >0.5 rad margin from every joint limit, similar elevation
# angle/"character" to the previous target, i.e. not a wild alternate branch).
# Stereo depth noise scales roughly with distance^2, so this alone is an
# estimated ~40% reduction in depth-noise VARIANCE (~23% in std) for free.
# Kept a comfortable margin above the RealSense D455's rated minimum usable
# depth range (0.4m) — closer than this would trade accuracy for missing/
# invalid depth returns, which is a worse trade. This is a SECONDARY fix
# relative to the rim-extraction fit correction in object_detector.py (which
# addresses a ~5mm systematic bias vs. this ~1-2mm noise-floor improvement),
# but it is free (no other tradeoff) so we take it.
HEAD_POSTURE_TARGET = np.array([-0.35, -0.25, -0.60, -1.15, -1.00, -1.25, 0.00])
POSTURE_GAIN = 0.50          # acts in the look-at null space (slack weight is high)
# Velocity-aware joint-limit CBF.
JOINT_LIMIT_GAMMA = 2.0
JOINT_LIMIT_BUFFER = 0.15    # rad safety buffer from the hard limit

# Consider the head "aligned" (pointing at the table) below this angular error.
LOOKAT_ALIGNED_DEG = 4.0

# =============================================================================
# 4b. ACTIVE PERCEPTION  (adaptive standoff / next-best-view distance control)
# =============================================================================
# WHY (paper framing): a FIXED posture target (HEAD_POSTURE_TARGET, §4) fixes
# the camera-to-table distance at design time. That is fragile: too far and the
# cylinder rim is resolved by too few pixels (radius under-estimated); too close
# and the table clips out of the field of view (partial scene, lost context).
# The right distance depends on the scene, which we do NOT want to hard-code.
#
# Instead we close a control loop around two ONLINE, SCENE-AGNOSTIC signals
# derived purely from what the camera actually observes (never from any known
# object pose / size — see the head-perception "no hard-coded ground truth"
# rule):
#
#   (1) FRAMING / CONTAINMENT.  We reproject the observed table-region cloud
#       back into the image and measure how much of it piles up in the OUTER
#       BORDER band of the frame, per edge. High border occupancy on an edge =>
#       the region of interest is running off that side of the image (clipping)
#       => the camera must BACK AWAY (or re-aim). Low occupancy on ALL edges =>
#       the whole ROI is comfortably contained => there is framing slack to
#       spend on resolution. This needs no object model: it only asks "does
#       what I see reach the edge of what I can see?".
#
#   (2) RESOLUTION SUFFICIENCY.  For each tracked object we compute its apparent
#       radius in PIXELS (r_px = fx * r / range) and its rim-fit RMS — both
#       already available. Small r_px / high RMS => the object is under-resolved
#       => if (and only if) framing has slack, MOVE CLOSER to gain detail.
#
# The loop maintains a desired camera STANDOFF distance d* along the viewing ray
# to the look-at target and adjusts it every perception tick with the priority
#       FRAMING (contain the ROI)  >  RESOLUTION (resolve detail)
# d* is then regulated by a dedicated soft "range" task inside the look-at QP
# (see look_at_controller.py), sitting BELOW the pointing task and ABOVE the
# posture spring in the QP's priority hierarchy. Net effect: the head always
# keeps the table centred, and slides along the ray to the closest distance at
# which the whole table still fits — exactly the behaviour we would otherwise
# have to hand-tune per scene.
ENABLE_ACTIVE_VIEW = True

# --- Standoff (range) QP task -------------------------------------------
STANDOFF_LAMBDA = 1.0        # proportional gain: range-rate = -lambda*(r - d*)
STANDOFF_SLACK_WEIGHT = 80.0 # QP slack weight: < LOOKAT_SLACK_WEIGHT (500) so
                             # pointing always wins; >> posture so distance is
                             # actively regulated, not left to the posture spring.

# --- Desired-standoff update law ----------------------------------------
# Absolute clamps on d* [m]. Lower bound keeps a margin above the RealSense
# rated min range (DEPTH_MIN=0.35) so we never drive the table inside the
# invalid-depth zone; upper bound keeps the objects resolvable at all.
VIEW_D_STAR_MIN = 0.45
VIEW_D_STAR_MAX = 1.10
VIEW_STEP_IN = 0.012         # [m] approach increment per perception tick
                             # (TUNED 2026-07-04: was 0.030, too fast — d* slewed
                             # faster than the head could physically track,
                             # causing oscillation vs. the scan waypoints.)
VIEW_STEP_OUT = 0.020        # [m] retreat increment per tick (was 0.050; same
                             # fix — the asymmetry retreat>approach is preserved
                             # but both are much slower than the QP's own rate.)
VIEW_STANDOFF_LPF = 0.15     # LPF coeff on d* (0..1, higher = more responsive);
                             # (TUNED 2026-07-04: was 0.30 — halved to further
                             # damp the oscillation cycle; guarantees a smooth,
                             # C0 setpoint for the QP task.)

# --- Framing / containment signal ---------------------------------------
VIEW_BORDER_MARGIN_FRAC = 0.06   # outer border band width = frac * min(W, H)
VIEW_BORDER_HIGH = 0.080         # per-edge occupancy above this => that edge is
                                 # clipping the ROI => RETREAT
                                 # (TUNED 2026-07-04: was 0.040.  The camera
                                 # views the table OBLIQUELY, so the far table
                                 # edge projects near the bottom border at any
                                 # reasonable range.  4% was chronic at 0.9m →
                                 # the planner retreated to 1.1m even though the
                                 # entire table+objects were visible.  8% filters
                                 # out this expected geometric bottom-band and
                                 # only triggers on genuine clipping.)
VIEW_BORDER_LOW = 0.020          # occupancy below this on ALL edges => ROI fully
                                 # contained => framing slack available
VIEW_MIN_PROJ_POINTS = 200       # need at least this many reprojected points to
                                 # trust the framing signal (else HOLD)
VIEW_CONTAIN_HYSTERESIS = 3      # [ticks] require this many consecutive
                                 # "contained" ticks before allowing APPROACH
                                 # (avoids scan-induced oscillation: one waypoint
                                 # clips -> RETREAT, next doesn't -> instant
                                 # APPROACH — the head never settles. Hysteresis
                                 # demands a persistent "contained" verdict
                                 # before moving closer again.)

# --- Resolution-sufficiency signal --------------------------------------
VIEW_RES_RADIUS_PX_TARGET = 12.0 # desired apparent object radius [px]; below
                                 # this an object is "under-resolved" (move
                                 # closer if framing allows).
                                 # (TUNED 2026-07-04: was 28.  r_px=fx*r/range =
                                 # 640*0.02/range.  28px needs range=0.46m, which
                                 # is AT the sensor's min depth — unreachable in
                                 # practice.  12px is achieved at ~1.07m, matches
                                 # the natural scan-range, and is sufficient for
                                 # the rim fit: 72 angular bins need ~12 boundary
                                 # points to populate each bin.)
VIEW_RES_FIT_RMS_OK = 0.0020     # [m] rim-fit RMS at/below which detail is
                                 # already good enough (don't push closer on
                                 # fit-quality grounds).

# =============================================================================
# 5. SCAN MOTION  (gentle sweep around the look-at target to improve coverage)
# =============================================================================
# RE-EVALUATED 2026-07-02 (accuracy pass): the scan was ORIGINALLY motivated by
# the belief that more viewpoints would average out the radius/height bias.
# Verified numerically that this was NOT the mechanism at fault — the bias was
# the circle fit running on the disk INTERIOR, not insufficient viewpoints
# (see object_detector.py's module docstring). A single, closer, rim-corrected
# view now gets within ~1mm of ground truth in simulation, i.e. the scan is no
# longer required to reach the target accuracy.
#
# Kept ENABLED anyway, because it is still legitimately useful for a DIFFERENT
# reason: angular COVERAGE. The rim extraction can only recover the boundary
# of what the camera actually saw — a single top-down-ish view still only
# shows ~150-180 deg of the side wall (the far side is self-occluded). The
# scan's cumulative arc-coverage tracking (object_tracker.py) still closes
# that gap over a few seconds, and per-frame estimates are no longer biased,
# so there's no longer a "more views = re-confirm the same bias" risk (see the
# updated fusion policy in object_tracker.py, now a per-frame EMA rather than
# grow-only-max). If startup latency matters more than full coverage in a
# given scenario, this can safely be set False now — accuracy will not
# meaningfully suffer, only the (already less important) full-circumference
# arc-coverage stat will stay lower.
ENABLE_SCAN = True
SCAN_DWELL_S = 4.0           # [s] time parked at each waypoint (settle + fuse)
SCAN_WAYPOINTS = [
    (0.00, 0.00),
    (0.08, 0.12),
    (0.08, -0.12),
    (-0.05, 0.12),
    (-0.05, -0.12),
]

# =============================================================================
# 5b. MULTI-VIEW ACCUMULATION  (DISABLED — see note)
# =============================================================================
# DISABLED by default. The single-frame pipeline is the proven, reliable path
# (~2cm, stable). Naive voxel accumulation regressed it: at a FIXED camera
# position it adds no parallax, and on the real depth stream it stacks
# frame-to-frame depth variation into a multi-layer band that breaks plane
# RANSAC (the "different heights / NO TABLE" failure). Genuine multi-view gain
# requires either (a) orbiting the camera POSITION around the table, or (b)
# ICP-registering each frame to the map before fusing — both deliberate future
# features. The VoxelMap code is kept for that work.
ENABLE_ACCUMULATION = False
VOXEL_MAP_LEAF = 0.008       # [m] fusion voxel size (8mm)
VOXEL_MAP_DECAY = 0.90       # per integrated frame
VOXEL_MAP_W_MIN = 0.40       # prune voxels whose weight decays below this
VOXEL_MAP_W_MAX = 25.0       # cap weight so the map stays responsive
VOXEL_MAP_QUERY_W = 1.0      # min weight for a voxel to be used in detection
INTEGRATE_VEL_THRESH = 0.04  # [rad/s] only fuse when head settled (if enabled)

# =============================================================================
# 5c. MANUAL OPTICAL-FRAME TF OVERRIDE  (EXPERIMENT, OFF by default)
# =============================================================================
# CONTEXT (2026-07-04): the head-camera XY position bias was proven, via a
# bias-vs-range regression on RAW (pre-tracker) detections, to be a CONSTANT
# translational offset (~2.5cm X, ~1.2cm Y for BOTH cylinders, near-zero
# slope vs. range) -- ruling out intrinsics and rotational/angular error.
# TF and Pinocchio FK agree with each other (both read the same live URDF),
# so a bug in the URDF's mount_link -> depth_optical_frame JOINT (rather than
# anywhere upstream in the head kinematic chain) would be INVISIBLE to the
# existing TF-vs-FK cross-check -- both would agree while both being
# consistently wrong.
#
# A colleague's independent PCL/C++ tabletop-perception node (different robot
# config, same class of problem) works around exactly this by NOT trusting
# whatever robot_state_publisher/URDF provides for that one joint -- it
# hardcodes a static_transform_publisher for camera_link -> depth_optical_
# frame instead. Their transform, decomposed numerically, is EXACTLY the
# generic REP-103 camera_link -> optical_frame convention (X-fwd/Y-left/Z-up
# -> X-right/Y-down/Z-fwd), zero translation -- not a scene-specific
# calibration number.
#
# This flag reproduces that SAME workaround structurally in our pipeline, as
# a toggleable A/B EXPERIMENT (not a silent default): when enabled, the
# camera pose used for perception is computed as
#     T_base_optical = T_base_mount  (TF, live)  @  T_mount_optical (MANUAL, fixed)
# instead of the normal path (TF lookup of base<-depth_optical_frame
# directly, which trusts the URDF's own joint for that last hop). If this
# changes/fixes the bias-vs-range regression's intercept, the URDF's fixed
# joint for MANUAL_OPTICAL_MOUNT_LINK -> depth optical frame is the
# confirmed root cause and should be corrected there (not permanently
# patched here). If it does NOT change the bias, this is definitively ruled
# out and the flag should be turned back off.
ENABLE_MANUAL_OPTICAL_TF = False   # RESULT: no measurable change vs. the
# standard pipeline (bias stayed ~-2.2cm X / -1.2cm Y either way) -- this
# hop is DEFINITIVELY RULED OUT. Turned off in favour of §5d below, which
# tests a different, still-open hypothesis. Kept in the code (not deleted)
# as a documented negative result / in case it's ever useful again.
# The rigid mount link that is the DIRECT TF parent of the depth optical
# frame (confirmed via `ros2 run tf2_tools view_frames`:
# "gripper_head_camera_rgbd_depth_optical_frame: parent: 'gripper_head_camera_link'").
MANUAL_OPTICAL_MOUNT_LINK = "gripper_head_camera_link"
# REP-103 camera_link -> optical_frame rotation (verified numerically to
# exactly match the colleague's static_transform_publisher args
# `0 0 0 -1.5708 0 -1.5708` decomposed as extrinsic yaw-pitch-roll):
#   X-forward/Y-left/Z-up  ->  X-right/Y-down/Z-forward
MANUAL_OPTICAL_R = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])
MANUAL_OPTICAL_T = np.array([0.0, 0.0, 0.0])   # zero translation (colleague's value)

# =============================================================================
# 5d. MANUAL MOUNT-TRANSLATION TF OVERRIDE  (EXPERIMENT, ON by default)
# =============================================================================
# CONTEXT (2026-07-04, from a colleague's qp_controller_node ROS params for
# their OWN camera-extrinsics pipeline on a related robot config):
#
#     camera_mount_parent_frame: arm_head_tool_link
#     camera_mount_translation: [0.0, 0.0, 0.0]      <-- ZERO
#     camera_mount_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]   <-- IDENTITY
#     camera_link_name: camera_head_camera_rgbd_link
#     optical_translation: [0.0, 0.0, 0.0]
#     optical_rpy: [-1.570, 0.0, -1.570]
#
# Verified numerically: their `optical_rpy` decomposes to EXACTLY the same
# rotation matrix already tested (and RULED OUT, see §5c/ENABLE_MANUAL_
# OPTICAL_TF above) -- so the rotation is NOT new information. What IS new:
# their config asserts ZERO translation between `arm_head_tool_link` and the
# camera's own link, whereas OUR live URDF's actual joint for that exact hop
# (`gripper_head_camera_joint`, parent=arm_head_tool_link,
# child=gripper_head_camera_link) has origin `xyz="-0.0406 0 -0.003"` -- a
# real 4.06cm/0.3cm offset, not zero. Our previous experiment (§5c) never
# tested this: it anchored at TF's LIVE pose of gripper_head_camera_link,
# which already has this -4.06cm baked in, then only replaced the FINAL hop
# (camera_link -> optical) with a rotation-only override starting FROM that
# already-offset point. This is a genuinely different, still-open
# hypothesis: what if the true physical mount offset is zero, not -4.06cm?
#
# When enabled, the camera pose used for perception is composed as:
#     T_base_camera_link = T_base_mountparent (TF, live)
#                           @ T_mountparent_cameralink (MANUAL: identity, zero -- per the colleague's config)
#     T_base_optical      = T_base_camera_link @ T_cameralink_optical (TF, live -- UNCHANGED,
#                                                                       the camera_link->optical
#                                                                       rotation is not in question)
# i.e. ONLY the arm_head_tool_link -> camera_link translation is overridden;
# everything else (pointing direction, the optical frame's own rotation
# convention) still comes from the live URDF/TF, unchanged.
#
# Mutually exclusive with ENABLE_MANUAL_OPTICAL_TF by construction (only one
# experiment is meant to be active at a time, to keep results unambiguous --
# main_head.py will warn loudly and prioritise this one if both are ever set
# True together). If this changes/fixes the bias-vs-GT diagnostic, the
# `gripper_head_camera_joint` origin in the URDF/xacro is the confirmed root
# cause and should be corrected there. If it does NOT change the bias, this
# is ALSO ruled out and the flag should be turned back off.
#
# BUGFIX (2026-07-04, same day): the FIRST version of this experiment set
# MANUAL_MOUNT_R = identity, copying the colleague's quaternion_xyzw=[0,0,0,1]
# literally. That was wrong: their chain puts the -90 deg pointing rotation
# on a DIFFERENT hop (their `optical_rpy`) than ours does. OUR URDF's actual
# `arm_head_tool_link -> gripper_head_camera_link` joint has rpy=
# "0 -1.5708 0" (a real -90 deg pitch) -- overriding that hop's ROTATION to
# identity deleted the pointing rotation entirely, sending the camera
# aiming ~90 deg away from the table (confirmed from the operator's log:
# TF rpy flipped from ~-134 deg to +136 deg roll -- not a small perturbation,
# a completely different orientation). The colleague's genuinely NEW claim
# was about TRANSLATION only (zero, vs. our -4.06cm/0/-0.3cm) -- their
# rotation convention doesn't transplant onto our joint layout. Fixed by
# overriding ONLY the translation and keeping OUR OWN live rotation for
# this hop (read via TF, exactly as it already was) -- this is now a
# translation-ONLY test, isolating exactly the one new variable.
ENABLE_MANUAL_MOUNT_TF = True
MANUAL_MOUNT_PARENT_FRAME = "arm_head_tool_link"
MANUAL_MOUNT_CAMERA_LINK = "gripper_head_camera_link"   # our chain's camera_link equivalent
MANUAL_MOUNT_T = np.array([0.0, 0.0, 0.0])       # zero (colleague's camera_mount_translation)
# MANUAL_MOUNT_R intentionally REMOVED -- rotation for this hop is taken
# live from TF/URDF, unchanged (see bugfix note above). Only translation
# is overridden.

# =============================================================================
# 6. POINT-CLOUD DEPROJECTION  (depth image -> 3D points)
# =============================================================================
# We subsample the depth image on a pixel grid to keep the cloud small enough
# for pure-numpy/scipy processing on a CPU. Stride 4 over 1280x720 -> ~57k pts.
PIXEL_STRIDE = 2
# DEPTH_MIN raised 0.20 -> 0.35 (2026-07-02): the RealSense D455 is only rated
# accurate from ~0.4m; points closer than that are frequently invalid/noisy
# stereo-matching artefacts, not genuine near-field returns. 0.35m keeps a
# small margin below the rated floor (so we don't clip legitimately-valid
# points right at the boundary) while rejecting the worst near-range noise.
# With the new, closer HEAD_POSTURE_TARGET (~0.63m to the table centre, see
# §4), the working range now sits safely above this floor with headroom.
DEPTH_MIN = 0.35             # [m] ignore points closer than this (noise/self)
DEPTH_MAX = 2.50             # [m] ignore points beyond this (background/walls)

# =============================================================================
# 7. WORKSPACE CROP  (in base_footprint, around the known table location)
# =============================================================================
# After transforming the cloud into base_footprint we keep only a box around
# the table. This removes the floor, far walls, and the robot's own body BEFORE
# any expensive processing — the single biggest speed & robustness win.
CROP_MARGIN_XY = 0.25        # [m] padding around the table footprint
CROP_Z_MIN = 0.20            # [m] floor cutoff (table body starts ~here)
CROP_Z_MAX = TABLE_TOP_Z_WORLD + 0.45   # well above the tallest expected object

# =============================================================================
# 8. RANSAC PLANE DETECTION  (find the table TOP surface)
# =============================================================================
PLANE_RANSAC_ITERS = 150
PLANE_DIST_THRESH = 0.010    # [m] inlier band half-thickness
# Only accept planes whose normal is within this of vertical (|n . up| >= ...).
PLANE_MIN_VERTICAL_DOT = 0.90
PLANE_MIN_INLIERS = 120      # below this, "no table found"
# Gate the plane height: the detected top must lie within this band of the
# known table top. Prevents locking onto the floor or a wall ledge.
PLANE_Z_TOLERANCE = 0.15     # [m] around TABLE_TOP_Z_WORLD

# =============================================================================
# 9. EUCLIDEAN CLUSTERING  (group above-plane points into candidate objects)
# =============================================================================
# VOXEL_SIZE lowered 10mm -> 3mm (2026-07-02 accuracy pass): a 10mm leaf is
# roughly HALF the cylinder's diameter (r=2cm) -- it was destroying most of
# the rim-extraction gain in object_detector.py by collapsing near-boundary
# points together before the rim fit ever sees them (verified numerically:
# with 10mm voxels the end-to-end radius bias was ~-3.4mm; with 3mm voxels,
# ~-1.4mm, matching the bias measured on the un-downsampled cluster). 3mm was
# chosen as the point where further shrinking gives diminishing returns (2mm
# barely improves on 3mm) while keeping the downsampled cluster small enough
# for the O(n log n) KD-tree clustering to stay comfortably real-time at
# PERCEPTION_RATE_HZ.
VOXEL_SIZE = 0.003           # [m] downsample leaf before clustering
CLUSTER_TOLERANCE = 0.030    # [m] max gap within one cluster
CLUSTER_MIN_POINTS = 25      # reject specks / noise
CLUSTER_MAX_POINTS = 200000
# Only look for objects in the slab just above the detected plane.
OBJECT_MIN_HEIGHT_ABOVE_PLANE = 0.010   # [m] start a hair above the surface
OBJECT_MAX_HEIGHT_ABOVE_PLANE = 0.40    # [m] tallest object we expect

# =============================================================================
# 10. CYLINDER FIT  (upright cylinder == axis aligned with table normal)
# =============================================================================
CYL_RADIUS_PERCENTILE = 95   # last-resort fallback only (see CYL_RIM_* below)
CYL_MIN_RADIUS = 0.010       # [m] plausibility gate
CYL_MAX_RADIUS = 0.080
CYL_MIN_HEIGHT = 0.030       # [m]
CYL_MAX_HEIGHT = 0.400

# --- Rim extraction (2026-07-02 accuracy pass) --------------------------
# Fitting a circle directly to a cluster that contains the cylinder's solid
# TOP FACE (a filled disk) is systematically biased toward a SMALLER radius
# -- interior points outnumber and sit closer to centre than the true
# boundary. Verified numerically: this alone explained roughly -3 to -5mm of
# the reported radius error, and was NOT fixed by scanning (more views just
# re-confirm the same biased fit). `_extract_rim` in object_detector.py bins
# the cluster by angle and keeps only the points near the LOCAL
# CYL_RIM_PERCENTILE-th percentile radius per bin, collapsing the disk
# interior away before the circle fit runs.
CYL_RIM_BINS = 72            # angular sectors (~5 deg each) for rim extraction
CYL_RIM_PERCENTILE = 93      # per-bin radius percentile defining the rim
# Percentile (not max) per bin -- verified numerically to be far more robust
# to RGB-D "flying pixel" outliers (stereo-matching smear at depth
# discontinuities can scatter a few points beyond the true rim; taking the
# raw max chases them back outward, taking the local percentile does not).
# 93/1.5mm was swept numerically as a good balance: ~-1.3mm bias on a clean
# synthetic cluster, ~+0.3mm with 10% simulated flying-pixel contamination
# (both comfortably sub-cm; pushing the percentile higher trades one for the
# other rather than improving both).
CYL_RIM_BAND = 0.0015        # [m] band width around the percentile radius
                              # averaged to form each bin's rim point

# Top-slice: used for BOTH the height estimate (median of the top slice,
# not z_max -- see object_detector.py, z_max is a biased-high max-statistic)
# and, historically, an alternative XY estimate (now superseded by rim
# extraction + Hyper fit above; kept only for the height use).
CYL_TOP_SLICE = 0.020        # [m] take points within this of the cluster's z_max

# Conservative radius inflation for collision use (0 = report raw estimate).
CYL_RADIUS_INFLATION = 0.000 # [m]

# Empirical head-camera bias correction (CALIBRATION). The arm chain grasps the
# cylinders correctly at their base_footprint config positions (x=0.80), but the
# HEAD-camera chain (base -> 7 head joints -> optical frame -> depth) is a
# separate, unvalidated chain that exhibits a systematic ~3cm offset on this
# setup. This is a legitimate one-time extrinsic calibration: measure
# (perceived_centre - true_centre) for a known object and put the NEGATIVE of it
# here to compensate. Default zero = raw, honest perception (the ~3cm then
# stands as genuine real-world sensing uncertainty).
PERCEPTION_XYZ_OFFSET = np.array([0.0, 0.0, 0.0])   # [m] added to every object centre

# =============================================================================
# 11. COLOUR CLASSIFICATION  (red vs blue from the aligned RGB)
# =============================================================================
# Hue is in [0, 1] (matplotlib.colors convention). Red wraps around 0/1.
COLOR_SAT_MIN = 0.35         # below this the cluster is "greyish" -> unknown
COLOR_VAL_MIN = 0.15         # below this it's too dark to classify
RED_HUE_LOW = 0.95           # hue >= this  (near 1.0) ...
RED_HUE_HIGH = 0.05          # ... OR hue <= this (near 0.0)  -> RED
BLUE_HUE_LOW = 0.55          # hue in [0.55, 0.75] -> BLUE
BLUE_HUE_HIGH = 0.75

# =============================================================================
# 12. TEMPORAL SMOOTHING + LOOP RATES
# =============================================================================
# Object-level tracker (EMA dims + persistence) — the robust alternative to
# point-cloud fusion. Fuses DERIVED object quantities across viewpoints, so
# head motion still helps (more arc -> more coverage, noise averages down)
# without the point-registration smear that broke voxel accumulation.
# NOTE (2026-07-02): dims (radius/height) switched from grow-only to EMA —
# see object_tracker.py's module docstring. TRACK_DIM_DECAY (grow-only's
# shrink-back rate) is removed as dead config along with it.
TRACK_MATCH_DIST = 0.15      # [m] associate a detection to a track within this
TRACK_MAX_UNSEEN = 15        # frames an unmatched track survives (~3s @5Hz)
TRACK_POS_ALPHA = 0.30       # EMA on position AND dimensions (0..1, higher = more responsive)

# (legacy single-frame EMA association — no longer used, kept for reference)
DETECTION_EMA_ALPHA = 0.40
DETECTION_MATCH_DIST = 0.10

# Velocity EMA filter (mirrors qp_controller's corrupted-encoder workaround):
# TRIAGo joint_states velocity field is unreliable. Derive velocity from
# position differences and filter with a first-order EMA.
ALPHA_VELOCITY_FILTER = 0.15  # ~60ms window, same as arm controller

CONTROL_RATE_HZ = 50.0       # head velocity command rate
PERCEPTION_RATE_HZ = 5.0     # perception pipeline rate (objects move slowly)
CONSOLE_SUMMARY_PERIOD_S = 5.0   # low-frequency console report (no spam!)

# =============================================================================
# 13. GROUND TRUTH (SIMULATION-ONLY — DIAGNOSTIC USE ONLY)
# =============================================================================
# Hard-coded from the Gazebo "tutorial" world SDF. The perception algorithm
# itself NEVER reads this section — it exists purely so diagnostic/plotting
# tools (head_plotter.py, calibration_audit.py) can compare estimates
# against a known answer, without duplicating the same numbers in multiple
# scripts (this project's single-source-of-truth convention). If the world
# SDF changes, this is the ONE place to update.
GT_RED_CENTER = np.array([0.800, -0.20, 0.775])    # [m] base_footprint
GT_RED_RADIUS = 0.02                               # [m]
GT_RED_HEIGHT = 0.15                               # [m]
GT_BLUE_CENTER = np.array([0.800, 0.20, 0.775])    # [m] base_footprint
GT_BLUE_RADIUS = 0.02                              # [m]
GT_BLUE_HEIGHT = 0.15                              # [m]
# GT_TABLE_TOP_Z intentionally NOT duplicated here — use TABLE_TOP_Z_WORLD
# (§2 above), which is the same 0.70m derived from the same SDF pose.
