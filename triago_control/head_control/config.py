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

# --- True optical-CENTRE frame for the DEPTH pipeline (main_head.py) ------
# Same principle as APRILTAG_OPTICAL_CENTER_FRAME (§15): Gazebo renders from
# the camera-link origin but tags the depth image as the depth-optical frame,
# which the URDF offsets from the link by ~2.5 cm. The depth pipeline transforms
# the point cloud via (R_optical, t_CENTRE), so in sim the POSITION must come
# from the link, not the optical frame, else every 3D point shifts by the
# static link→optical offset (measured [-0.011,-0.009,-0.021]).
# On REAL hardware set this to "" (empty) to disable the correction and use the
# depth image's own frame_id for both orientation AND position.
# Override at runtime: --ros-args -p depth_center_frame:=<frame>
DEPTH_OPTICAL_CENTER_FRAME = "gripper_head_camera_rgbd_link"   # SIM default

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

# These values match this repo's movement_tutorial.world table pose only.
# Other worlds need different numbers -- override via head_look_at_table.py's
# table_x/table_y/table_z ROS params rather than editing this file.

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
LOOKAT_LAMBDA = 1.0          # proportional gain on the angular look-at error
MAX_HEAD_VELOCITY = 0.10     # rad/s per joint (slow & smooth — gives the perception
                              # pipeline time to settle and fuse clean frames; fast
                              # motion kept the velocity gate permanently closed)
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
# ~0.63m camera-to-table distance, reachable with >0.5 rad joint-limit margin.
# Stereo depth noise scales with distance^2, so this stays well clear of the
# RealSense D455's 0.4m minimum usable range while cutting noise variance.
HEAD_POSTURE_TARGET = np.array([-0.35, -0.25, -0.60, -1.15, -1.00, -1.25, 0.00])
POSTURE_GAIN = 0.50          # acts in the look-at null space (slack weight is high)
# Velocity-aware joint-limit CBF.
JOINT_LIMIT_GAMMA = 2.0
JOINT_LIMIT_BUFFER = 0.15    # rad safety buffer from the hard limit

# Consider the head "aligned" (pointing at the table) below this angular error.
LOOKAT_ALIGNED_DEG = 4.0

# =============================================================================
# 5. SCAN MOTION  (gentle sweep around the look-at target to improve coverage)
# =============================================================================
# Scanning improves angular coverage (the rim fit only sees ~150-180 deg from
# one viewpoint), but requires a still head for accurate TF-based cloud
# transforms -- moving TF lags and causes ~5-10cm pose oscillation.
ENABLE_SCAN = False           # single viewpoint already covers the table
SCAN_DWELL_S = 8.0            # [s] parked per waypoint, settle below INTEGRATE_VEL_THRESH
SCAN_WAYPOINTS = [
    (0.00, 0.00),
    (0.05, 0.08),
    (0.05, -0.08),
    (-0.03, 0.08),
    (-0.03, -0.08),
]

# =============================================================================
# 5c. ACTIVE VISION  (perception-driven head motion — supersedes the fixed scan)
# =============================================================================
# When ENABLE_ACTIVE_VISION is True, the head is no longer driven by wall-clock
# time (ENABLE_SCAN / SCAN_DWELL_S). Instead look_at_controller.active_vision_target
# runs a two-phase policy that is LED BY WHAT THE CAMERA SEES:
#   Phase 0 SWEEP  : cycle the SCAN_WAYPOINTS offsets to discover the plane + every
#                    expected object, advancing off a waypoint once its incremental
#                    per-object arc-coverage gain PLATEAUS (not a fixed dwell).
#   Phase 1 REFINE : re-aim the fixation point at the WEAKEST object (lowest
#                    arc_coverage / confidence), nudged toward the azimuth of its
#                    unseen arc sectors, until its coverage plateaus or clears the
#                    floor; then re-pick the next weakest.
#   Phase 2 HOLD   : stop re-aiming; hold still so the rate-based convergence check
#                    (world_convergence.py) can run on settled frames and freeze.
# The move->settle->fuse discipline is UNCHANGED: fusion + convergence still only
# advance on settled frames (INTEGRATE_VEL_THRESH gate), so a moving head never
# corrupts the estimate — the policy only changes WHERE we look, never bypasses
# that gate. Bounded by ACTIVE_VISION_TIME_BUDGET_S so hardware can't scan forever.
ENABLE_ACTIVE_VISION = True   # False -> fall back to the fixed ENABLE_SCAN path
# "Plateau": a settled-frame arc-coverage gain below this (fraction, 0..1) counts
# as no-progress; PLATEAU_PATIENCE such consecutive frames ends a waypoint/refine.
ACTIVE_VISION_PLATEAU_EPS = 0.01
ACTIVE_VISION_PLATEAU_PATIENCE = 6      # settled frames of no-progress => advance
# Phase 1 re-aim: how far (base-frame XY, m) to offset the fixation point toward the
# weakest object's unseen-sector azimuth. Small — a fixed-base neck gives limited
# parallax; the main gain is centering the weak object in the FOV (see out-of-scope
# note in the plan re: orbiting the camera POSITION for true arc gain).
ACTIVE_VISION_REFINE_OFFSET = 0.06
ACTIVE_VISION_TIME_BUDGET_S = 45.0      # hard cap on total sweep+refine before HOLD

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
# 6. POINT-CLOUD DEPROJECTION  (depth image -> 3D points)
# =============================================================================
# We subsample the depth image on a pixel grid to keep the cloud small enough
# for pure-numpy/scipy processing on a CPU. Stride 4 over 1280x720 -> ~57k pts.
PIXEL_STRIDE = 2
# 0.35m: just below the D455's ~0.4m rated-accurate range, rejecting the
# worst near-range stereo noise without clipping valid near-boundary points.
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
# 3mm: coarser leaves collapse near-boundary points before the rim fit sees
# them, biasing the radius estimate; 3mm balances that against cluster cost.
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

# Fitting a circle to the cylinder's solid top face biases the radius small
# (interior points outnumber the boundary). `_extract_rim` in object_detector.py
# bins by angle and keeps only the local CYL_RIM_PERCENTILE-th-percentile
# points per bin, collapsing the disk interior away before the fit runs.
CYL_RIM_BINS = 72            # angular sectors (~5 deg each) for rim extraction
CYL_RIM_PERCENTILE = 93      # per-bin radius percentile defining the rim
# Percentile (not max) is robust to RGB-D flying-pixel outliers at depth
# discontinuities, which a raw max would chase outward.
CYL_RIM_BAND = 0.0015        # [m] band width around the percentile radius
                              # averaged to form each bin's rim point

# Height uses the top slice's median (z_max alone is a biased-high statistic).
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
# 13. TABLETOP PERCEPTION NODE (generic OBB + memory-tracked pipeline)
# =============================================================================
# Ported from the sibling `triago_perception` ROS 2 package (repository
# chiarapanagrosso/Triago_Project, src/tabletop_perception_node.cpp) as
# scripts/head_controller/tabletop_perception_node.py. Deliberately INDEPENDENT
# from ObjectDetector/ObjectTracker/TableSegmenter above:
#   * no colour classification, no upright-cylinder assumption — fits a
#     Z-aligned oriented bounding box (2D PCA) to ANY above-plane cluster.
#   * table-plane RANSAC here is UNBIASED (no verticality filter, no known-
#     table-height gate) — matching the original node, which has no such
#     scene-specific prior. TableSegmenter's gating is specific to the known
#     tutorial-table scenario and must not be reused here.
#   * temporal fusion is GROW-ONLY (dimensions can only increase) + hard
#     persistence deletion, NOT the EMA policy in object_tracker.py — this is
#     an intentional 1:1 port of the original node's own memory-tracking
#     design, not an endorsement of grow-only over EMA (see object_tracker.py's
#     module docstring for why EMA was chosen for the cylinder pipeline).
# The upstream node subscribes to `/filtered_cloud`, a topic produced OUTSIDE
# triago_control (a throttled RealSense cloud in the full stack) — there is no
# such publisher in this repo. The topic name is kept for interface parity and
# is overridable via the `cloud_topic` ROS parameter.
TP_CLOUD_TOPIC = "/filtered_cloud"
TP_TARGET_FRAME = BASE_FRAME

# RANSAC table-plane segmentation (unbiased — see note above).
TP_PLANE_RANSAC_ITERS = 100
TP_PLANE_DIST_THRESH = 0.02   # [m]

# Euclidean cluster extraction (above-plane points -> candidate objects).
TP_CLUSTER_TOLERANCE = 0.03   # [m]
TP_CLUSTER_MIN_POINTS = 50
TP_CLUSTER_MAX_POINTS = 25000

# Memory tracking (grow-only dimensions + hysteresis persistence).
TP_MAX_UNSEEN_FRAMES = 10     # frames an unmatched track survives
TP_MATCH_DIST_THRESH = 0.15   # [m] centroid distance to associate a cluster

# Marker cosmetics.
TP_TABLE_BBOX_PADDING = 0.005     # [m]
TP_OBSTACLE_PADDING = 0.005       # [m]
TP_TARGET_PADDING = 0.005         # [m]
TP_MIN_MARKER_SCALE = 0.01        # [m] floor so zero-thickness boxes stay visible

# TF lookup tolerance at the cloud's own timestamp (no "latest" fallback, to
# match the original node's exact-time-only lookup).
TP_TF_TIMEOUT_S = 0.05

# cloud_relay_node.py: nothing publishes /filtered_cloud upstream, so relay
# this project's real camera cloud to TP_CLOUD_TOPIC, throttled.
TP_RELAY_SOURCE_TOPIC = "/gripper_head_camera_rgbd/depth/color/points"
TP_RELAY_TARGET_TOPIC = TP_CLOUD_TOPIC
TP_RELAY_RATE_HZ = 10.0

# =============================================================================
# 14. GROUND TRUTH (SIMULATION-ONLY — DIAGNOSTIC USE ONLY)
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



# =============================================================================
# 15. APRILTAG-BASED PERCEPTION  (head_april_main.py)
# =============================================================================
# An alternative head perception path that reconstructs every object pose from
# a SINGLE camera-estimated fiducial pose, instead of the geometric RGB-D
# cylinder fit (main_head.py). The AprilTag is a fiducial of known geometry
# rigidly placed in the scene; the ONLY thing estimated from the image is the
# camera->tag transform (c_M_tag), produced by the external `visp_apriltag`
# ROS 2 node (ViSP). Each object pose is then reconstructed as
#
#     base_M_obj = base_M_cam . c_M_tag . tag_M_obj
#
# where base_M_cam comes from TF (robot_state_publisher), c_M_tag from the
# detection, and tag_M_obj is the KNOWN rigid transform of the object relative
# to the tag.
#
# WHY THIS IS NOT "GROUND-TRUTH INJECTION" (cf. the project rule): the algorithm
# never reads where anything actually IS in the world. It knows only the scene
# STRUCTURE relative to a marker it must localize visually — the defining
# paradigm of fiducial-based perception, and exactly what was requested
# ("reconstruct each object pose knowing only its pose relative to the tag").
# The absolute world coordinates in APRILTAG_SCENE below are the AUTHORING
# SOURCE for the tag-relative transforms (they mirror config/worlds/
# apriltag_world.world one-to-one) and the tag-relative transforms are derived
# from them at startup; they are never used as a runtime input to the
# perception output. This is the fiducial MODEL, not an empirical offset that
# masks a calibration error. (The separate §14 GT_* constants remain
# diagnostic-only, consumed by head_plotter.py to SCORE the reconstruction.)

# --- visp_apriltag node interface -----------------------------------------
# Topic on which the visp_apriltag node (node name "tracker_apriltag")
# publishes visp_tracker_common/msg/AprilTagDetectionArray. Each detection
# carries `id` and `pose` (geometry_msgs/Pose == c_M_tag, camera-optical->tag).
APRILTAG_DETECTIONS_TOPIC = "/tracker_apriltag/tags_info"
APRILTAG_ID = 0                 # tag id used as the scene reference fiducial
APRILTAG_FAMILY = "36h11"       # must match the printed / spawned tag
APRILTAG_SIZE = 0.12            # [m] black-border edge = ViSP tag_size (see model)
APRILTAG_POSE_METHOD = "homography_virtual_vs"

# The visp_apriltag node consumes the COLOR image + COLOR camera_info (AprilTag
# detection runs on RGB, NOT the depth stream used by main_head.py).
APRILTAG_IMAGE_TOPIC = "/gripper_head_camera_rgbd/color/image_raw"
APRILTAG_CAMERA_INFO_TOPIC = "/gripper_head_camera_rgbd/color/camera_info"
# Optical frame the color image (and therefore c_M_tag) lives in. TF base<-this
# frame gives the camera ORIENTATION. (Same frame already used as
# CAMERA_OPTICAL_FRAME, §1.)
APRILTAG_CAMERA_OPTICAL_FRAME = CAMERA_OPTICAL_FRAME

# --- True optical-CENTRE frame (SIM vs REAL — important) ------------------
# ViSP's c_M_tag is always in optical convention (X-right, Y-down, Z-forward);
# the only question is WHERE the optical centre physically sits. We take the
# camera ORIENTATION from APRILTAG_CAMERA_OPTICAL_FRAME but the POSITION from
# this frame.
#
#   * REAL RealSense hardware: the driver tags the image with the color optical
#     frame whose origin IS the true optical centre -> set this EQUAL to
#     APRILTAG_CAMERA_OPTICAL_FRAME (i.e. no position correction).
#   * Gazebo (sim): the camera plugin RENDERS from the camera-link origin but
#     still tags the image as the color optical frame, which the URDF offsets
#     from the link by ~2.5 cm (measured: camera_link->color_optical_frame
#     translation [-0.011,-0.009,-0.021]). ViSP's pose is therefore relative to
#     the LINK, so the true optical centre is the link origin. Using the offset
#     optical frame's position caused a constant ~2.7 cm reconstruction bias on
#     every object (root cause, not maskable with a fudge offset). Set this to
#     the camera-link frame in sim.
# Override at runtime with -p optical_center_frame:=<frame>.
APRILTAG_OPTICAL_CENTER_FRAME = "gripper_head_camera_rgbd_link"   # SIM default

# --- ViSP tag frame orientation in the world ------------------------------
# The tag lies flat, face up (surface normal = world +Z). ViSP's DEFAULT frame
# (align_z=false) has Z out of the tag toward the camera, X=image-right,
# Y=image-up (verified: cMo == Rot_x(180deg) for a fronto-parallel view).
#
# Measured tag-frame yaw in base_footprint is -90 deg (Gazebo's UV mapping),
# not the naive rpy=[0,0,0]. Re-measure if the physical tag is re-oriented.
APRILTAG_CENTER_WORLD = np.array([1.000, 0.0, 0.701])   # tag plate centre (world)
APRILTAG_RPY_WORLD = np.array([0.0, 0.0, -np.pi / 2])   # ViSP tag frame RPY in world

# --- Scene layout (authoring source for the tag-relative model) -----------
# One entry per reconstructed object. `center`/`rpy` are the object frame pose
# in the WORLD, matching config/worlds/apriltag_world.world exactly. At startup
# head_april_main.py derives tag_M_obj = world_M_tag^-1 . world_M_obj from these
# and thereafter uses ONLY tag_M_obj (never the absolute world pose).
#   kind   : "cylinder" (upright, radius+height) or "table" (top-surface box)
#   color  : "red"/"blue"/"table" -> marker colour + head_plotter classification
APRILTAG_SCENE = {
    "red_cylinder": {
        "kind": "cylinder", "color": "red",
        "center": np.array([0.800, -0.20, 0.775]),   # cylinder mid-height centre
        "rpy": np.array([0.0, 0.0, 0.0]),
        "radius": 0.02, "height": 0.15,
    },
    "blue_cylinder": {
        "kind": "cylinder", "color": "blue",
        "center": np.array([0.800, 0.20, 0.775]),
        "rpy": np.array([0.0, 0.0, 0.0]),
        "radius": 0.02, "height": 0.15,
    },
    "work_table": {
        "kind": "table", "color": "table",
        "center": np.array([1.000, 0.0, TABLE_TOP_Z_WORLD]),  # table TOP-surface centre
        "rpy": np.array([0.0, 0.0, 0.0]),
        "size_xy": np.array([0.6, 0.5]),   # table footprint (x, y) for the marker
    },
}

# Consider a detection stale (drop its markers) if no fresh tag has been seen
# for this long — so a lost fiducial doesn't leave a ghost pose on screen.
APRILTAG_DETECTION_TIMEOUT_S = 1.0


# =============================================================================
# 16. PERCEIVED-WORLD SNAPSHOT  (camera estimate -> QP-CLF-CBF collision world)
# =============================================================================
# The QP-CLF-CBF safety stack (qp_controller/) normally builds its collision
# world from a STATIC YAML (config/worlds/<name>.yaml). The camera-driven variant
# main_qp_controller_perceived.py instead builds it from THIS perception node's
# estimate: once the fused estimate is judged "confident" (stable + high-
# confidence over several settled frames — see world_convergence.py), main_head
# publishes ONE latched snapshot describing the table + cylinders, and the QP node
# builds its collision model from it ONCE, statically (no per-tick CBF updates —
# a deliberate design choice; a moving obstacle set would make the barrier
# non-stationary). This section holds the convergence criterion + snapshot config.
#
# The "confident concept": rather than thresholding the raw per-object soft
# confidence alone (arc_coverage x fit-quality, which climbs monotonically), we
# require BOTH (a) every expected cylinder above WORLD_CONF_MIN AND (b) the fused
# geometry to have STOPPED MOVING (per-tick drift below the tolerances below) for
# WORLD_STABLE_FRAMES consecutive SETTLED frames. Stability is the honest "the
# estimate has converged" signal; confidence alone can be high while the pose is
# still drifting in early frames.
PERCEIVED_WORLD_TOPIC = "/perceived_world/snapshot"   # latched MarkerArray, TRANSIENT_LOCAL
PERCEIVED_WORLD_RESCAN_TOPIC = "/perceived_world/rescan"  # std_msgs/Empty -> re-arm + re-publish

WORLD_EXPECTED_CYLINDERS = 2      # red + blue on the table
WORLD_CONF_MIN = 0.85             # each cylinder's tracker confidence must reach this

# Convergence gate: every object must clear the arc-coverage floor, then hold
# a tight oscillation-amplitude band (not a per-frame drift count, which can
# stay "stable" while consistently wrong) for a full window. See world_convergence.py.
WORLD_CONVERGE_POS_AMP = 0.001    # [m] max object-centre oscillation (peak dev from window mean)
WORLD_CONVERGE_DIM_AMP = 0.0015   # [m] max radius/height oscillation over the window
WORLD_CONVERGE_WINDOW_S = 1.0     # [s]   the amplitude band must hold this long
WORLD_MIN_ARC_COVERAGE = 0.55     # [0..1] every object must reach this before the lock can arm
WORLD_CONVERGE_POS_RATE = 0.005   # [m/s] drift-rate reference (telemetry/plot only, not the gate)
WORLD_CONVERGE_DIM_RATE = 0.003   # [m/s] (telemetry only)

# DEPRECATED (kept for the ENABLE_SCAN fallback / reference only — the rate-based
# criterion above is what WorldConvergenceMonitor now uses):
WORLD_STABLE_POS_TOL = 0.005      # [m] max per-settled-frame centre drift to count as "stable"
WORLD_STABLE_DIM_TOL = 0.003      # [m] max per-settled-frame radius/height drift
WORLD_STABLE_FRAMES = 15          # consecutive stable settled frames required (~3 s @ 5 Hz)

# Table box (fully camera-derived, decision 3): XY footprint from the RANSAC plane
# inliers, top-Z from the plane. The box spans from PERCEIVED_TABLE_BOTTOM_Z up to
# the perceived top (conservative solid volume — the arm works above the table, so
# blocking the full column below the top is the safe choice and needs no prior).
PERCEIVED_TABLE_BBOX_PCT = 2.0    # percentile trim on inlier X/Y (rejects flying-pixel outliers)
# The percentile trim above rejects OUTLIERS (which would over-size the box) but
# also shaves ~PCT% of the true span off each edge on clean data — the UNSAFE
# direction for a collision obstacle. A small outward margin offsets that and
# keeps the perceived table conservatively sized (safe) by default; occlusion can
# only ever make the raw footprint smaller, so a conservative bias is desirable.
PERCEIVED_TABLE_XY_MARGIN = 0.02  # [m] outward inflation of the perceived footprint (per side)
PERCEIVED_TABLE_BOTTOM_Z = 0.0    # [m] base-frame Z of the table box bottom (ground)
