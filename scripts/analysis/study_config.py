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
# Independent variable: study cell (AUTO-DERIVED from the controller config)
# ---------------------------------------------------------------------------
# The active study cell is identified AUTOMATICALLY from the three orthogonal
# flags in triago_control.qp_controller.config -- CONTROL_MODE, ASSIST_FEEDBACK,
# ASSIST_BLENDING -- so the experimenter never declares it. The cell CODE matches
# the force-manager naming convention (haptic_force_manager_<CODE>):
#   C = CLUTCH / J = JOYSTICK (control mode, always first),
#   + F if ASSIST_FEEDBACK, + B if ASSIST_BLENDING.
#
#   CONTROL_MODE  F  B   code   condition
#   CLUTCH        0  0   C      Clutch / Sync-only (baseline)
#   CLUTCH        1  0   CF     Clutch / Guided feedback (VF)
#   CLUTCH        1  1   CFB    Clutch / Full guidance
#   JOYSTICK      0  0   J      Joystick / Sync-only (baseline)
#   JOYSTICK      0  1   JB     Joystick / Guided blending
#   JOYSTICK      1  1   JFB    Joystick / Full guidance
# (CLUTCH+B-only and JOYSTICK+F-only are NOT part of the fair 2x3 design.)
VALID_CELLS = {
    ("CLUTCH",   False, False): "C",
    ("CLUTCH",   True,  False): "CF",
    ("CLUTCH",   True,  True):  "CFB",
    ("JOYSTICK", False, False): "J",
    ("JOYSTICK", False, True):  "JB",
    ("JOYSTICK", True,  True):  "JFB",
}
CELL_LABELS = {
    "C":   "Clutch / Sync-only (baseline)",
    "CF":  "Clutch / Guided feedback (VF)",
    "CFB": "Clutch / Full guidance",
    "J":   "Joystick / Sync-only (baseline)",
    "JB":  "Joystick / Guided blending",
    "JFB": "Joystick / Full guidance",
}


def derive_cell(cfg) -> dict:
    """Identify the active study cell from the controller config flags.

    Returns {code, label, control_mode, assist_feedback, assist_blending, valid,
    warning}. ``code`` is the C/CF/CFB/J/JB/JFB tag used in trial-folder names and
    the master table. ``valid`` is False for a flag combination outside the fair
    2x3 design (the trial is still recorded, just flagged). Never raises: a missing
    cfg yields a clearly-marked 'unknown' cell so the recorder can still capture
    the bag.
    """
    if cfg is None:
        return {"code": "unknown",
                "label": "cfg unavailable (cell not auto-detected)",
                "control_mode": None, "assist_feedback": None,
                "assist_blending": None, "valid": False,
                "warning": "triago_control.qp_controller.config not importable"}
    mode = str(getattr(cfg, "CONTROL_MODE", "?"))
    fb = bool(getattr(cfg, "ASSIST_FEEDBACK", False))
    bl = bool(getattr(cfg, "ASSIST_BLENDING", False))
    code = VALID_CELLS.get((mode, fb, bl))
    if code is not None:
        return {"code": code, "label": CELL_LABELS[code], "control_mode": mode,
                "assist_feedback": fb, "assist_blending": bl, "valid": True,
                "warning": None}
    # Outside the 2x3 design -- still build a descriptive code so nothing crashes.
    letter = "C" if mode == "CLUTCH" else ("J" if mode == "JOYSTICK" else "X")
    code = letter + ("F" if fb else "") + ("B" if bl else "")
    return {"code": code, "label": f"NON-STUDY cell ({mode}, F={fb}, B={bl})",
            "control_mode": mode, "assist_feedback": fb, "assist_blending": bl,
            "valid": False,
            "warning": f"({mode}, F={fb}, B={bl}) is not one of the 6 fair 2x3 cells"}


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

# --- Trial folder layout -------------------------------------------------
# DATA_ROOT/<PARTICIPANT_ID>/<world_shortcut>_<cell_code>/   (e.g. shield_JB)
#   The innermost folder IS the rosbag output directory (`ros2 bag record -o`),
#   so it also holds rosbag2's own metadata.yaml and the .db3/.mcap file. Our
#   metadata.json (METADATA_NAME) and the offline-cached timeseries
#   (TIMESERIES_NAME) are written alongside, inside that same folder.
# Re-recording the same (participant, world, condition) OVERWRITES the folder.
METADATA_NAME = "metadata.json"
TIMESERIES_NAME = "timeseries"          # extension added per TIMESERIES_FORMAT

def participant_dir(participant: str) -> str:
    """Absolute path of a participant's folder under DATA_ROOT."""
    return os.path.join(DATA_ROOT, participant)


def bag_folder_name(world_shortcut: str, cell_code: str) -> str:
    """Bag folder name '<world_shortcut>_<cell_code>' (e.g. 'shield_JB'). The
    cell_code is the auto-derived C/CF/CFB/J/JB/JFB tag (see derive_cell)."""
    return f"{world_shortcut}_{cell_code}"


def bag_path(participant: str, world_shortcut: str, cell_code: str) -> str:
    """Absolute rosbag output directory for one trial (overwritten on re-record)."""
    return os.path.join(participant_dir(participant),
                        bag_folder_name(world_shortcut, cell_code))


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
    "/qp_debug/governor",        # reference-governor clamp telemetry (raw - governed)
    "/qp_debug/arm_frozen",      # [right_frozen, left_frozen] -- ground-truth active arm
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


# ---------------------------------------------------------------------------
# cfg provenance snapshot
# ---------------------------------------------------------------------------
# Attribute names read from qp_controller.config and stored into each trial's
# metadata.json for reproducibility. Missing attributes are silently skipped,
# so this list can safely name more than any single cfg version defines.
CFG_SNAPSHOT_KEYS = (
    # The condition selector itself (identifies the study cell) -- captured first.
    "CONTROL_MODE", "ASSIST_FEEDBACK", "ASSIST_BLENDING",
    "BLENDING",
    "ALIGN_ALPHA_IDLE", "ALIGN_ALPHA_MIN", "ALIGN_ALPHA_MAX", "ALIGN_ALPHA_LPF_COEFF",
    "JOYSTICK_DEADBAND_LIN", "JOYSTICK_DEADBAND_ANG",
    "JOYSTICK_K_TRANS", "JOYSTICK_K_ROT",
    "D_SAFE_BASE", "K_V_SAFE",
    # Collision-pair cutoff: /qp_debug/min_distance reads a 1.0 sentinel when no
    # pair is within this range -- the analysis masks samples >= it (see
    # analyze_trial's obstacle-clearance panel).
    "DISTANCE_FILTER_THRESHOLD",
)
