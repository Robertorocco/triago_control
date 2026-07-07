"""Configuration for the user-study data-capture & analysis subsystem.

This module is the single source of truth for the *experiment / data-management*
settings used by everything under ``scripts/analysis/`` (the recorder and the
offline analysis scripts).

It is deliberately kept SEPARATE from ``triago_control.qp_controller.config``
(the controller's ``cfg``): the values here are participant identity, storage
paths, rosbag topic allowlists, resampling rates and offline-metric thresholds
-- none of them are controller gains, and controller behaviour is never tuned
from this file. See ``.kiro/context.md`` §15.

Nothing in this module imports ROS, so the offline analysis tools can use it
on a machine without a ROS installation.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Participant identity
# ---------------------------------------------------------------------------
# Set ONCE at the start of each participant's session. It stays constant across
# every trial that person runs. The recorder is relaunched per trial (§15.2) and
# reads this value each time, so there is no in-memory state to keep in sync.
#
# You do NOT have to edit this file mid-session: it can be overridden, in order
# of precedence, by
#   1. the recorder's  --ros-args -p participant:=<id>
#   2. the environment variable  TRIAGO_PARTICIPANT_ID
#   3. this constant (the fallback default)
PARTICIPANT_ID = "P00"


def resolve_participant_id(ros_param_value: str | None = None) -> str:
    """Return the effective participant id honouring the precedence above."""
    if ros_param_value:
        return str(ros_param_value).strip()
    env = os.environ.get("TRIAGO_PARTICIPANT_ID")
    if env:
        return env.strip()
    return PARTICIPANT_ID


# ---------------------------------------------------------------------------
# Independent variable: feedback conditions
# ---------------------------------------------------------------------------
# Canonical labels used in trial-folder names and the master table. The recorder
# validates the declared condition against ``cfg.BLENDING`` where it can.
#
#   virtual_fixture : cfg.BLENDING=False -- clutch teleop + full haptic guidance
#                     (F_guide + F_fixture + F_sync + F_cbf ...)
#   blending        : cfg.BLENDING=True  -- reference-level blending; the handle
#                     feels only the centering spring
#   no_assist       : cfg.BLENDING=False -- clutch teleop, guidance OFF
#                     (baseline; optionally only F_sync kept)
#
# NOTE: 'virtual_fixture' and 'no_assist' BOTH run at cfg.BLENDING=False and are
# indistinguishable at the flag level -> the experimenter's declaration is the
# source of truth for those two; only 'blending' can be auto-verified.
CONDITIONS = ("virtual_fixture", "blending", "no_assist")

# Which conditions imply cfg.BLENDING == True (used only for the sanity-check
# warning in the recorder; never to change behaviour).
BLENDING_CONDITIONS = ("blending",)


# ---------------------------------------------------------------------------
# Data storage  (LOCAL ONLY -- never committed to GitHub)
# ---------------------------------------------------------------------------
# Heavy artefacts (rosbags, per-trial timeseries, figures) live here, OUTSIDE
# the git repository. Override with the environment variable
# TRIAGO_STUDY_DATA_ROOT.
DATA_ROOT = os.path.expanduser(
    os.environ.get("TRIAGO_STUDY_DATA_ROOT", "~/exchange/triago_study_data")
)

# Master summary table (one row per trial), written by build_master_table.py
# into DATA_ROOT.
MASTER_TABLE_NAME = "trials_summary.csv"

# Per-trial artefact filenames (inside each trial's timestamped folder).
BAG_DIRNAME = "trial.bag"
TIMESERIES_NAME = "timeseries"          # extension added per TIMESERIES_FORMAT
METADATA_NAME = "metadata.json"


def trial_folder_name(participant: str, condition: str, world: str,
                      repetition: int, timestamp: str) -> str:
    """Canonical timestamped folder name for a single trial."""
    return f"{participant}_{condition}_{world}_r{repetition:02d}_{timestamp}"


# ---------------------------------------------------------------------------
# rosbag capture
# ---------------------------------------------------------------------------
# Curated allowlist (NOT `-a`): everything needed to replay the trial and
# re-derive every metric, WITHOUT the large head-camera point clouds.
# /tf, /tf_static and /joint_states are included so a bag is independently
# replayable. Topics that do not exist in a given run (e.g. the
# *user_cartesian_reference* topics outside Joystick mode) are simply skipped
# by `ros2 bag record` -- listing them here is harmless.
BAG_TOPICS = [
    # --- robot state (for standalone replay) ---
    "/joint_states",
    "/tf",
    "/tf_static",

    # --- QP controller telemetry (main_qp_controller.py) ---
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
    "/qp_debug/d_safe_dynamic",
    "/qp_debug/dynamic_weights",
    "/qp_debug/task_authority",
    "/qp_debug/loop_freq",
    "/collision_constraints",

    # --- Cartesian references (both arms, both modes) ---
    "/arm_right/cartesian_reference",
    "/arm_left/cartesian_reference",
    "/arm_right/user_cartesian_reference",   # Joystick mode only
    "/arm_left/user_cartesian_reference",    # Joystick mode only

    # --- shared autonomy / assistance (main_shared_autonomy.py) ---
    "/shared_autonomy/blend_debug",
    "/shared_autonomy/goal_names",
    "/shared_autonomy/goal_probabilities",
    "/shared_autonomy/ee_policy",
    "/shared_autonomy/user_policy",
    "/shared_autonomy/active_goal_pose",
    "/shared_autonomy/grasp_active",
    "/shared_autonomy/active_arm",
    "/shared_autonomy/gripper_cmd",

    # --- Haption device (haption_teleoperation) ---
    # virtuose_server_node has no namespace, so its relative topics resolve to
    # the absolute names below. The right button is the clutch (effort metric
    # for virtual_fixture / no_assist); the left button is the grasp trigger.
    "/virtuose/pose",
    "/virtuose/velocity",
    "/virtuose/button_right",
    "/virtuose/button_left",
    "/virtuose/articular_position",
    "/virtuose/force_cmd",
    "/joystick/home_pose",                   # Joystick mode only
]

# rosbag2 storage backend. "sqlite3" is the ROS 2 Humble default and always
# available (produces a .db3). "mcap" is more portable/robust for long-term
# archival but requires the rosbag2_storage_mcap package.
BAG_STORAGE_ID = "sqlite3"


# ---------------------------------------------------------------------------
# Tidy time-series (analysis layer)
# ---------------------------------------------------------------------------
RESAMPLE_HZ = 100.0             # common clock the recorder resamples topics onto
TIMESERIES_FORMAT = "parquet"   # "parquet" (preferred) or "csv"


# ---------------------------------------------------------------------------
# Offline metric thresholds  (consumed by study_metrics.py, not the recorder)
# ---------------------------------------------------------------------------
NEAR_MISS_DISTANCE_M = 0.05     # /qp_debug/min_distance below this = near-miss
CBF_ACTIVE_LAMBDA = 1.0         # /qp_debug/lambda_cbf above this = barrier active
BELIEF_CONFIDENCE = 0.80        # goal probability above this = "intent locked"
