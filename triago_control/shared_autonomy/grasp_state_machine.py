#!/usr/bin/env python3
"""GraspStateMachine: 4-state dict-dispatch grasp state machine.

Extracted from the procedural if/elif/elif/else chain inside the original
timer_callback, per shared_autonomy_analysis.md Section 3 (state machine
critique) and Section 4 (class decomposition table).

Design notes / fixes applied relative to the original:

Problem A (conflated identity and transition): the original PRE_GRASP branch
was an `elif self.PREDICTION and b_max > 0.90 and is_aligned:` condition, so
staying in PRE_GRASP required b_max to remain above 0.90 on every single tick.
A single noisy dip in a noisy EMA belief would drop the robot straight back to
SHARED_AUTONOMY even though `is_aligned` already has its own hysteresis. This
version adds the same kind of hysteresis to the belief gate: once *inside*
PRE_GRASP, the belief threshold for *staying* is relaxed (BELIEF_ENTER vs
BELIEF_STAY), exactly mirroring the existing is_aligned hysteresis pattern.

Problem B (target_twist multiple definition sites / possible UnboundLocalError):
every handler below is required to return a TickOutput with a concrete
target_twist; there is no implicit fallthrough, so target_twist can never be
left undefined.

Problem C (hard to extend): new states are added by writing one handler method
and one dict entry in _build_handlers; no existing handler needs to change.

This module has no ROS or matplotlib dependencies and operates purely on the
TickInput / TickOutput dataclasses, making each handler unit-testable in
isolation.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class TickInput:
    """Read-only snapshot of everything a state handler needs for one control tick."""

    # Geometry
    current_T_EE: np.ndarray
    T_active_goal: np.ndarray          # standoff (approach_offset=0.05) goal pose
    pos_error: float
    ang_error: float

    # Policy / belief
    pi_max: np.ndarray                 # commanded twist if no grasp logic intervenes
    b_max: float
    prediction_enabled: bool
    active_goal_key: str               # e.g. "Red_Side"
    active_arm: str                    # "right" / "left"

    # Event / sensor inputs
    trigger_pulled: bool
    current_force_mag: float
    current_force_local: np.ndarray     # [Fx, Fy, Fz] in wrist sensor frame
    grasp_contact: Dict[str, float]    # {'red': d, 'blue': d}

    # Compute helpers injected from the node (kept as callables so the state
    # machine never needs to import GoalSet / compute_v_geo itself).
    compute_v_geo: Callable[[np.ndarray, np.ndarray], np.ndarray]
    get_dynamic_goal_pose: Callable[..., np.ndarray]

    # Mutable scratch carried across ticks within a grasp sequence (locked pose,
    # grasp timer). Stored on the state machine itself, not here -- TickInput is
    # immutable per tick.


@dataclass
class TickOutput:
    """Everything a state handler produces for one control tick."""

    target_twist: np.ndarray
    new_state: str
    # CBF / margin side effects the node must publish this tick.
    ignore_cbf: Optional[str] = None          # None -> don't publish this tick
    # grasp_margin: float -> set that margin; "CLEAR" -> explicit _clear_grasp_margin();
    # None -> leave the margin topic untouched this tick (matches the original,
    # where PRE_GRASP never calls either _set_grasp_margin or _clear_grasp_margin).
    grasp_margin: Optional[object] = None
    gripper_cmd: Optional[str] = None         # e.g. "ORANGE_RIGHT_RED", "CLOSE_RIGHT_0.0150"
    reset_trigger: bool = False               # True -> node must force trigger_cmd back to False
    log_lines: list = field(default_factory=list)   # (level, message) tuples for the node to log


CLEAR_MARGIN = "CLEAR"  # sentinel for TickOutput.grasp_margin meaning "explicit clear"


class GraspStateMachine:
    """Dict-dispatch state machine: SHARED_AUTONOMY -> PRE_GRASP -> GRASP_APPROACH -> GRASP_CLOSE."""

    # --- Tunables (moved out of the node, kept as class-level constants so they
    # can be overridden per-instance without touching the handler logic) ---
    POS_ERR_ENTER = 0.04
    ANG_ERR_ENTER = 0.15
    POS_ERR_STAY = 0.06
    ANG_ERR_STAY = 0.20

    BELIEF_ENTER = 0.90   # belief threshold required to *enter* PRE_GRASP
    BELIEF_STAY = 0.75    # relaxed belief threshold required to *stay* in PRE_GRASP
    #   ^ Fix for Problem A: without this, a single noisy EMA dip below 0.90
    #     would kick the robot out of PRE_GRASP even while perfectly aligned.

    GRASP_CBF_MARGIN = -0.08
    GRASP_CONTACT_DEPTH = -0.025
    GRASP_FORCE_THRESHOLD = 2.0
    GRASP_CLOSE_HOLD_S = 4.0
    GRASP_APPROACH_TIMEOUT_S = 20.0

    # Force-controlled closure parameters
    GRIP_CLOSE_VELOCITY = 0.00001   # rad/s — very slow closure (~13s to close)
    GRIP_FINAL_POSITION = 0.035   # rad — target closed position (less tight)
    GRIP_FORCE_TARGET = 4.0       # N — target grip force on Fx axis
    GRIP_FORCE_CONTACT = 1.5      # N — threshold to detect first contact
    GRIP_FORCE_MAX = 8.0          # N — safety limit (stop closing)
    GRIP_K_FORCE = 0.005          # rad/s per N — force proportional gain
    GRIP_CONFIRM_DURATION = 1.0   # s — hold force above threshold to confirm

    def __init__(self, cylinders, initial_state="SHARED_AUTONOMY", debug=False):
        """Initializes the state machine.

        Args:
            cylinders: dict {color: {'radius': ..., 'cbf_name': ..., ...}} -- same
                       table owned by GoalSet, needed here for grip width and CBF
                       pair naming during the grasp sequence.
            initial_state: starting state name.
            debug: if True, handlers populate verbose log_lines on every tick
                   (mirrors the original GRASP_DEBUG flag).
        """
        self._state = initial_state
        self.debug = debug
        self.cylinders = cylinders

        # Scratch state carried across ticks within a grasp sequence.
        self.locked_grasp_pose = None
        self.grasp_timer = 0.0

        # Force-controlled closure state
        self.grip_position = 0.7   # Start fully open
        self.grip_contact_detected = False
        self.grip_force_stable_since = None

        self._last_state_logged = None

        self._handlers = self._build_handlers()

    @property
    def state(self):
        """Current state name."""
        return self._state

    def _build_handlers(self):
        return {
            "SHARED_AUTONOMY": self._shared_autonomy,
            "PRE_GRASP": self._pre_grasp,
            "GRASP_APPROACH": self._grasp_approach,
            "GRASP_CLOSE": self._grasp_close,
        }

    def _transition(self, new_state):
        """Centralized state transition (single place where self._state changes)."""
        self._state = new_state

    def _is_aligned(self, inp: TickInput) -> bool:
        """Hysteresis-aware alignment check (tighter to enter PRE_GRASP, looser to stay)."""
        if self._state == "PRE_GRASP":
            return inp.pos_error < self.POS_ERR_STAY and inp.ang_error < self.ANG_ERR_STAY
        return inp.pos_error < self.POS_ERR_ENTER and inp.ang_error < self.ANG_ERR_ENTER

    def _belief_ok(self, inp: TickInput) -> bool:
        """Hysteresis-aware belief gate (Fix for Problem A)."""
        threshold = self.BELIEF_STAY if self._state == "PRE_GRASP" else self.BELIEF_ENTER
        return inp.prediction_enabled and inp.b_max > threshold

    def step(self, inp: TickInput) -> TickOutput:
        """Evaluates the active goal's transition guard, then dispatches to the handler.

        Mirrors the original if/elif/elif/else ordering of priority: GRASP_CLOSE
        and GRASP_APPROACH are "sticky" (handled purely by their own internal
        logic since they must run to completion / timeout), whereas the choice
        between PRE_GRASP and SHARED_AUTONOMY is re-evaluated every tick based on
        belief + alignment.
        """
        if self._state in ("GRASP_CLOSE", "GRASP_APPROACH"):
            out = self._handlers[self._state](inp)
        elif self._belief_ok(inp) and self._is_aligned(inp):
            out = self._pre_grasp(inp)
        else:
            out = self._shared_autonomy(inp)

        if self.debug and self._state != self._last_state_logged:
            out.log_lines.append(
                ("info", f"[GRASP-DBG] STATE -> {self._state} "
                         f"(goal={inp.active_goal_key}, arm={inp.active_arm})"))
            self._last_state_logged = self._state

        return out

    # ------------------------------------------------------------------
    # PHASE 0: SHARED_AUTONOMY
    # ------------------------------------------------------------------
    def _shared_autonomy(self, inp: TickInput) -> TickOutput:
        self._transition("SHARED_AUTONOMY")
        log_lines = []
        if inp.trigger_pulled:
            log_lines.append(("warn", "GRASP REFUSED: Robot is not aligned in the safe PRE_GRASP zone."))

        return TickOutput(
            target_twist=inp.pi_max,
            new_state=self._state,
            ignore_cbf="None",
            grasp_margin=CLEAR_MARGIN,   # matches original's explicit _clear_grasp_margin()
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 1: PRE_GRASP
    # ------------------------------------------------------------------
    def _pre_grasp(self, inp: TickInput) -> TickOutput:
        log_lines = []
        entering = self._state != "PRE_GRASP"
        self._transition("PRE_GRASP")

        if entering:
            # Logged once, on the SHARED_AUTONOMY -> PRE_GRASP transition only.
            # (Previously there was a second, unconditional copy of this same
            # message logged on every tick while debug=True, which spammed the
            # console at the full ~100 Hz control-loop rate. The original
            # monolithic script throttled that second copy to 0.5s; here we
            # remove it entirely instead, since per the request only state
            # changes / unexpected events should be logged, not a periodic
            # heartbeat.)
            log_lines.append(("info", "=== [PRE-GRASP READY] Alignment perfect! Type 'CLOSE' to execute. ==="))
            log_lines.append(
                ("info", f"[GRASP-DBG] PRE_GRASP goal={inp.active_goal_key} b_max={inp.b_max:.3f} "
                         f"pos_err={inp.pos_error:.4f}m ang_err={inp.ang_error:.4f} (waiting for CLOSE)"))

        # Fix applied here (root cause of the grasping oscillation, per
        # shared_autonomy_analysis.md Section 2): the original computed
        #     target_twist = self.compute_v_geo(self.current_T_EE, T_active_goal)
        # i.e. the RAW geometric velocity, which has no knowledge of the CBF
        # barrier. At the standoff (which sits right at the edge of the
        # cylinder's safe zone) v_geo keeps pushing toward the goal, the CLF
        # tracks it into the barrier, the CBF pushes back, and the cycle
        # repeats. The fix is to use the QP-constrained policy (pi_max) here
        # instead, since the QP already incorporates the CBF constraint and
        # decelerates smoothly to zero as the barrier activates.
        target_twist = inp.pi_max

        color = inp.active_goal_key.split('_')[0]

        if inp.trigger_pulled:
            log_lines.append(
                ("info", f"[GRASP] CLOSE received in PRE_GRASP. Committing to grasp of "
                         f"{color} cylinder with {inp.active_arm} arm."))

            # Fix applied here: the radial-projection goal (GoalSet's Side/Top
            # grasp computation) is a function of its anchor pose, and its
            # curvature blows up as the anchor approaches the cylinder surface
            # -- any microscopic noise in the anchor (J_EE(q)*qdot jitter) turns
            # into a large jump in the projected target. The previous version
            # anchored this one-time freeze on inp.current_T_EE, the raw,
            # instantaneous, noisy EE pose sampled at the exact tick CLOSE
            # arrives -- baking that single noisy sample permanently into
            # locked_grasp_pose for the whole approach phase. inp.T_active_goal
            # (the standoff pose PRE_GRASP has already been smoothly converging
            # to and hovering at via the QP-constrained policy) is the stable
            # quantity instead: anchoring on it means the frozen envelop target
            # descends from a pose the control loop has already settled near,
            # not a single instant of measurement noise. xdot_ref for this
            # target is now effectively zero from the moment it's created.
            r = self.cylinders[color]['radius']
            self.locked_grasp_pose = inp.get_dynamic_goal_pose(
                inp.T_active_goal, inp.active_goal_key, approach_offset=-r - 0.06)

            self._transition("GRASP_APPROACH")
            self.grasp_timer = time.time()
            log_lines.append(
                ("info", f"[GRASP] Gripper-{color} CBF margin relaxed to "
                         f"{self.GRASP_CBF_MARGIN:+.3f} m (barrier still active). "
                         f"Approaching controlled contact..."))

            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=None,
                gripper_cmd=f"ORANGE_{inp.active_arm.upper()}_{color.upper()}",
                log_lines=log_lines,
            )

        return TickOutput(
            target_twist=target_twist,
            new_state=self._state,
            ignore_cbf="None",   # Shield stays UP while we hover
            grasp_margin=None,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 2: GRASP_APPROACH (decelerating blind insertion)
    # ------------------------------------------------------------------
    def _grasp_approach(self, inp: TickInput) -> TickOutput:
        self._transition("GRASP_APPROACH")
        log_lines = []
        color = inp.active_goal_key.split('_')[0]

        # Track the latched pose (not the moving carrot): as position error -> 0,
        # velocity -> 0, letting the dynamic CBF margin shrink smoothly down to
        # the contact depth instead of jerking into it.
        target_twist = inp.compute_v_geo(inp.current_T_EE, self.locked_grasp_pose)

        ang_fwd_err = np.linalg.norm(
            np.cross(inp.current_T_EE[:3, :3][:, 0], self.locked_grasp_pose[:3, :3][:, 0]))
        ang_ok = ang_fwd_err < 0.15

        contact_d = inp.grasp_contact.get(color.lower(), 1.0)
        contact_ok = contact_d <= self.GRASP_CONTACT_DEPTH
        # Per-tick [GRASP-DBG] heartbeat removed (was unconditional under
        # self.debug -> spammed at ~100 Hz). Only the transition/timeout
        # events below are logged now.

        if ang_ok and contact_ok:
            self._transition("GRASP_CLOSE")
            self.grasp_timer = time.time()
            # Reset force-control state for the new closure attempt
            self.grip_position = 0.04  # Start from near-cylinder (fingers already around it)
            self.grip_contact_detected = False
            self.grip_force_stable_since = None
            log_lines.append(
                ("info", f"[GRASP] Controlled contact reached (contact_d={contact_d:.4f}m). "
                         f"Freezing arm. Starting force-controlled finger closure."))

            cyl_radius = self.cylinders[color]['radius']
            # Start slightly wider than target — closure will slowly bring to GRIP_FINAL_POSITION
            self.grip_position = self.GRIP_FINAL_POSITION + 0.02

            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=f"+{self.cylinders[color]['cbf_name']}",
                grasp_margin=self.GRASP_CBF_MARGIN,
                gripper_cmd=None,  # Force loop will send incremental commands
                log_lines=log_lines,
            )

        if time.time() - self.grasp_timer > self.GRASP_APPROACH_TIMEOUT_S:
            log_lines.append(
                ("warn", f"[GRASP] Approach timeout — contact_d={contact_d:.4f}m "
                         f"(never reached {self.GRASP_CONTACT_DEPTH}m). Aborting grasp."))
            self._transition("SHARED_AUTONOMY")
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=None,    # original does not touch the margin topic on timeout abort
                reset_trigger=True,   # mirrors the original's self.trigger_cmd = False
                log_lines=log_lines,
            )

        return TickOutput(
            target_twist=target_twist,
            new_state=self._state,
            ignore_cbf=f"+{self.cylinders[color]['cbf_name']}",
            grasp_margin=self.GRASP_CBF_MARGIN,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 3: GRASP_CLOSE (slow closure then vertical lift)
    # ------------------------------------------------------------------
    LIFT_VELOCITY = 0.02  # m/s upward lift speed
    LIFT_DURATION = 5.0   # seconds to lift (0.02*5=10cm)

    def _grasp_close(self, inp: TickInput) -> TickOutput:
        self._transition("GRASP_CLOSE")
        log_lines = []
        color = inp.active_goal_key.split('_')[0]

        elapsed = time.time() - self.grasp_timer
        dt = 0.01  # 100 Hz tick

        # --- Phase A: Close fingers slowly until final position ---
        if not self.grip_contact_detected:
            self.grip_position -= self.GRIP_CLOSE_VELOCITY * dt
            self.grip_position = max(self.GRIP_FINAL_POSITION, self.grip_position)
            gripper_cmd = f"CLOSE_{inp.active_arm.upper()}_{self.grip_position:.4f}"

            # Fingers reached target → switch to lift phase
            if self.grip_position <= self.GRIP_FINAL_POSITION:
                self.grip_contact_detected = True  # Reuse flag as "closure done"
                self.grip_force_stable_since = time.time()  # Reuse as lift start time
                log_lines.append(("info", f"[GRASP] Fingers at target ({self.GRIP_FINAL_POSITION:.4f} rad). "
                                          f"Holding grip. Starting vertical lift."))

            # Arm stays frozen during closure
            target_twist = np.zeros(6)
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=self.GRASP_CBF_MARGIN,
                gripper_cmd=gripper_cmd,
                log_lines=log_lines,
            )

        # --- Phase B: Lift vertically (grip held at GRIP_FINAL_POSITION) ---
        lift_elapsed = time.time() - self.grip_force_stable_since

        # Command a pure vertical twist (Z-up in base_footprint)
        target_twist = np.array([0.0, 0.0, self.LIFT_VELOCITY, 0.0, 0.0, 0.0])

        # Keep sending the same grip position to hold fingers in place
        gripper_cmd = f"CLOSE_{inp.active_arm.upper()}_{self.GRIP_FINAL_POSITION:.4f}"

        if lift_elapsed >= self.LIFT_DURATION:
            log_lines.append(("info", f"[GRASP] Lift complete ({self.LIFT_DURATION:.1f}s). Attaching."))
            self._transition("SHARED_AUTONOMY")
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=CLEAR_MARGIN,
                gripper_cmd=f"ATTACH_{inp.active_arm.upper()}_{color.upper()}",
                log_lines=log_lines,
            )

        # Timeout fallback
        if elapsed > 20.0:
            log_lines.append(("warn", f"[GRASP] Timeout ({elapsed:.1f}s). Attaching anyway."))
            self._transition("SHARED_AUTONOMY")
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=CLEAR_MARGIN,
                gripper_cmd=f"ATTACH_{inp.active_arm.upper()}_{color.upper()}",
                log_lines=log_lines,
            )

        return TickOutput(
            target_twist=target_twist,
            new_state=self._state,
            ignore_cbf="None",
            grasp_margin=self.GRASP_CBF_MARGIN,
            gripper_cmd=gripper_cmd,
            log_lines=log_lines,
        )