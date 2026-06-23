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
LOOKAT_SLACK_WEIGHT = 1.0    # penalty on the look-at task slack
POSTURE_GAIN = 0.05          # null-space spring toward joint mid-range
# Velocity-aware joint-limit CBF.
JOINT_LIMIT_GAMMA = 2.0
JOINT_LIMIT_BUFFER = 0.15    # rad safety buffer from the hard limit

# Consider the head "aligned" (pointing at the table) below this angular error.
LOOKAT_ALIGNED_DEG = 4.0

# =============================================================================
# 5. SCAN MOTION  (gentle sweep around the look-at target to improve coverage)
# =============================================================================
# A single viewpoint sees the table top fine, but a slow sweep fills occluded
# regions and lets temporal smoothing average out depth noise. Disable if you
# want a static head.
ENABLE_SCAN = True
SCAN_AMPLITUDE_X = 0.06      # [m] sweep half-extent along table X (forward)
SCAN_AMPLITUDE_Y = 0.10      # [m] sweep half-extent along table Y (left/right)
SCAN_PERIOD_X = 14.0         # [s] period of the X oscillation (slower = easier to track)
SCAN_PERIOD_Y = 9.0          # [s] period of the Y oscillation (coprime-ish ->
                             #     Lissajous coverage of the surface)

# =============================================================================
# 6. POINT-CLOUD DEPROJECTION  (depth image -> 3D points)
# =============================================================================
# We subsample the depth image on a pixel grid to keep the cloud small enough
# for pure-numpy/scipy processing on a CPU. Stride 4 over 1280x720 -> ~57k pts.
PIXEL_STRIDE = 4
DEPTH_MIN = 0.20             # [m] ignore points closer than this (noise/self)
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
PLANE_MIN_INLIERS = 400      # below this, "no table found"
# Gate the plane height: the detected top must lie within this band of the
# known table top. Prevents locking onto the floor or a wall ledge.
PLANE_Z_TOLERANCE = 0.15     # [m] around TABLE_TOP_Z_WORLD

# =============================================================================
# 9. EUCLIDEAN CLUSTERING  (group above-plane points into candidate objects)
# =============================================================================
VOXEL_SIZE = 0.010           # [m] downsample leaf before clustering
CLUSTER_TOLERANCE = 0.030    # [m] max gap within one cluster
CLUSTER_MIN_POINTS = 25      # reject specks / noise
CLUSTER_MAX_POINTS = 200000
# Only look for objects in the slab just above the detected plane.
OBJECT_MIN_HEIGHT_ABOVE_PLANE = 0.010   # [m] start a hair above the surface
OBJECT_MAX_HEIGHT_ABOVE_PLANE = 0.40    # [m] tallest object we expect

# =============================================================================
# 10. CYLINDER FIT  (upright cylinder == axis aligned with table normal)
# =============================================================================
CYL_RADIUS_PERCENTILE = 95   # robust radius estimate from radial spread
CYL_MIN_RADIUS = 0.010       # [m] plausibility gate
CYL_MAX_RADIUS = 0.080
CYL_MIN_HEIGHT = 0.030       # [m]
CYL_MAX_HEIGHT = 0.400

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
# EMA on matched detections across frames -> stable poses despite depth noise.
DETECTION_EMA_ALPHA = 0.40   # 0..1, higher = more responsive / less smooth
# Max distance to associate a detection with the previous frame's object.
DETECTION_MATCH_DIST = 0.10  # [m]

CONTROL_RATE_HZ = 50.0       # head velocity command rate
PERCEPTION_RATE_HZ = 5.0     # perception pipeline rate (objects move slowly)
CONSOLE_SUMMARY_PERIOD_S = 5.0   # low-frequency console report (no spam!)
