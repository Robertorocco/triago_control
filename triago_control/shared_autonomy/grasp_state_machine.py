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
    release_object: bool = False              # True -> node must open gripper, detach payload, reset post-grasp state
    log_lines: list = field(default_factory=list)   # (level, message) tuples for the node to log


CLEAR_MARGIN = "CLEAR"  # sentinel for TickOutput.grasp_margin meaning "explicit clear"


class GraspStateMachine:
    """Dict-dispatch state machine: SHARED_AUTONOMY -> PRE_GRASP -> GRASP_APPROACH -> GRASP_CLOSE."""

    # --- Tunables (moved out of the node, kept as class-level constants so they
    # can be overridden per-instance without touching the handler logic) ---
    # NOTE: alignment is measured against the STANDOFF goal (approach_offset=0.05).
    # The guidance can drive the gripper right up to the cylinder (past the
    # standoff), which pushes pos_error back up toward ~0.05; thresholds are kept
    # forgiving enough (ENTER > standoff) that being at/near the surface still
    # counts as "aligned" so the grasp stays committable.
    POS_ERR_ENTER = 0.06
    ANG_ERR_ENTER = 0.20
    POS_ERR_STAY = 0.09
    ANG_ERR_STAY = 0.28

    BELIEF_ENTER = 0.90   # belief threshold required to *enter* PRE_GRASP
    BELIEF_STAY = 0.75    # relaxed belief threshold required to *stay* in PRE_GRASP
    #   ^ Fix for Problem A: without this, a single noisy EMA dip below 0.90
    #     would kick the robot out of PRE_GRASP even while perfectly aligned.

    GRASP_CBF_MARGIN = -0.08
    GRASP_CONTACT_DEPTH = -0.038     # gripper-box↔cylinder overlap to trigger close (slightly relaxed from -0.04)
    GRASP_INSERTION_TRAVEL = 0.09    # m, straight-line advance from standoff along approach axis (DEPTH knob)
    GRASP_FORCE_THRESHOLD = 2.0
    GRASP_CLOSE_HOLD_S = 4.0
    GRASP_APPROACH_TIMEOUT_S = 30.0  # increased from 20s — approach can be slow with relaxed CBF

    # Force-controlled closure parameters
    GRIP_CLOSE_VELOCITY = 0.01   # rad/s — very slow closure (~13s to close)
    GRIP_FINAL_POSITION = 0.045
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
        self._lift_start_time = None  # reset for LIFT phase
        self._release_lift_start = None  # reset for RELEASE_LIFT (post-OPEN) phase
        self._abort_lift_start = None  # reset for ABORT_LIFT (failed grasp retreat) phase
        self._abort_lift_color = None  # color of the cylinder being retreated from
        self._align_start = None  # reset for GRASP_ALIGN timeout
        self._holding_entered = False  # latch so the HOLDING banner prints once per grasp

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
            "GRASP_ALIGN": self._grasp_align,
            "GRASP_APPROACH": self._grasp_approach,
            "GRASP_CLOSE": self._grasp_close,
            "LIFT": self._lift,
            "HOLDING": self._holding,
            "RELEASE_LIFT": self._release_lift,
            "ABORT_RETREAT": self._abort_lift,
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
        if self._state in ("GRASP_CLOSE", "GRASP_APPROACH", "LIFT", "HOLDING", "GRASP_ALIGN", "RELEASE_LIFT", "ABORT_RETREAT"):
            # GRASP_* and LIFT are pure run-to-completion phases. HOLDING is
            # special: while an object is in the gripper the PRE_GRASP branch is
            # deliberately UNREACHABLE (you cannot commit a second grasp with the
            # same gripper). The user can still drive toward — and the belief
            # estimator can still predict — any remaining goal; that motion is
            # produced by the outer loop's policy (pi_max), which _holding passes
            # straight through. Only an explicit release (trigger) leaves HOLDING.
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
        self._holding_entered = False  # re-arm the HOLDING banner for the next grasp
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
                ("info", f"[GRASP] CLOSE received in PRE_GRASP. Aligning precisely before approach "
                         f"({color} cylinder, {inp.active_arm} arm)."))

            # Before committing the blind straight-line insertion, first drive to
            # the EXACT standoff goal (T_active_goal) with a tighter tolerance
            # than the PRE_GRASP entry condition. This re-centers the gripper
            # perfectly on the cylinder axis so the approach doesn't nudge the
            # object sideways and knock it over. The locking step uses v_geo
            # toward the standoff — but since we're already very close (~4-6 cm,
            # ~0.15 rad) this converges in <1 s. Once pos < 0.015 m and ang <
            # 0.08 rad, the approach starts.
            self._align_target = np.asarray(inp.T_active_goal, dtype=float).copy()
            self._align_start = None  # reset align timer
            self._transition("GRASP_ALIGN")
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
    # PHASE 1.5: GRASP_ALIGN (precise centering before the blind insertion)
    # ------------------------------------------------------------------
    ALIGN_POS_TOL = 0.022   # m — within ~2.2 cm of the standoff (relaxed ~10%: grasp
                            #   succeeds even from slightly looser alignment, per operator)
    ALIGN_ANG_TOL = 0.143   # rad — approach-axis within ~8.2° of the goal (relaxed ~10%)
    ALIGN_TIMEOUT_S = 12.0  # if alignment doesn't converge in this time, abort
                            #   (raised 6 -> 12s: the precise centring before the blind
                            #    insertion can be slow, and a premature abort/approach is
                            #    what was nudging the cylinder over with the fingertips)

    def _grasp_align(self, inp: TickInput) -> TickOutput:
        """Drive precisely to the standoff pose before committing the blind insertion.

        The PRE_GRASP entry tolerances are generous (6 cm / 0.20 rad) so the user
        can trigger comfortably; but the straight-line insertion needs near-perfect
        centering on the cylinder axis, or the fingers shove the object sideways.

        IMPORTANT: the gripper<->cylinder CBF is RELAXED here (same as in
        GRASP_APPROACH). With the nominal barrier the gripper is held a few cm
        short of the standoff, the position error can never reach ALIGN_POS_TOL,
        and alignment always times out (the bug the operator hit). Relaxed, the
        gripper can seat exactly on the standoff and converge.
        """
        self._transition("GRASP_ALIGN")
        log_lines = []
        color = inp.active_goal_key.split('_')[0]
        cbf_name = self.cylinders[color]['cbf_name']

        # Drive toward the exact standoff with the raw geometric velocity.
        target_twist = inp.compute_v_geo(inp.current_T_EE, self._align_target)

        pos_err = np.linalg.norm(inp.current_T_EE[:3, 3] - self._align_target[:3, 3])
        ang_err = np.linalg.norm(
            np.cross(inp.current_T_EE[:3, :3][:, 0], self._align_target[:3, :3][:, 0]))
        pos_ok = pos_err < self.ALIGN_POS_TOL
        ang_ok = ang_err < self.ALIGN_ANG_TOL

        if pos_ok and ang_ok:
            log_lines.append(
                ("info", f"[GRASP] Alignment converged (pos={pos_err:.4f}m, ang={ang_err:.4f}). "
                         f"Starting straight-line approach."))
            T_base = self._align_target.copy()
            R_base = T_base[:3, :3]
            approach_axis = R_base[:, 0]
            locked = np.eye(4)
            locked[:3, :3] = R_base
            locked[:3, 3] = T_base[:3, 3] + approach_axis * self.GRASP_INSERTION_TRAVEL
            self.locked_grasp_pose = locked
            self._transition("GRASP_APPROACH")
            self.grasp_timer = time.time()
            self._align_start = None
            log_lines.append(
                ("info", f"[GRASP] Gripper-{color} CBF margin relaxed to "
                         f"{self.GRASP_CBF_MARGIN:+.3f} m. Approaching controlled contact..."))
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=f"+{cbf_name}",
                grasp_margin=self.GRASP_CBF_MARGIN,
                log_lines=log_lines,
            )

        # Timeout guard with an EXPLICIT reason (which gate failed and by how much).
        if self._align_start is None:
            self._align_start = time.time()
        if time.time() - self._align_start > self.ALIGN_TIMEOUT_S:
            reasons = []
            if not pos_ok:
                reasons.append(
                    f"POSITION not centred (pos_err={pos_err:.4f} m, need < {self.ALIGN_POS_TOL} m)")
            if not ang_ok:
                reasons.append(
                    f"APPROACH-AXIS not aligned (ang_err={ang_err:.4f}, need < {self.ALIGN_ANG_TOL})")
            log_lines.append(
                ("warn", f"[GRASP FAILED] Alignment did not converge within {self.ALIGN_TIMEOUT_S:.0f}s — "
                         f"aborting. Reason(s): {'; '.join(reasons)}. Backing out along the reverse "
                         f"approach axis and restoring CBF."))
            self._abort_lift_start = None
            self._abort_lift_color = color
            self._transition("ABORT_RETREAT")
            self._align_start = None
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf=f"+{cbf_name}",  # keep bypass active during retreat
                grasp_margin=CLEAR_MARGIN,
                gripper_cmd=f"OPEN_{inp.active_arm.upper()}",
                reset_trigger=True,
                log_lines=log_lines,
            )

        # Still converging — keep the gripper<->cylinder CBF relaxed so the
        # standoff stays reachable.
        return TickOutput(
            target_twist=target_twist,
            new_state=self._state,
            ignore_cbf=f"+{cbf_name}",
            grasp_margin=self.GRASP_CBF_MARGIN,
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

        # Position-reached fallback: with the straight-line locked target, the
        # advance is finished once the EE is within 1 cm of it, even if the
        # gripper-box contact distance never crosses GRASP_CONTACT_DEPTH.
        pos_to_target = np.linalg.norm(
            inp.current_T_EE[:3, 3] - self.locked_grasp_pose[:3, 3])
        pos_reached = pos_to_target < 0.01

        if ang_ok and (contact_ok or pos_reached):
            self._transition("GRASP_CLOSE")
            self.grasp_timer = time.time()
            # Reset force-control state for the new closure attempt
            self.grip_position = 0.04  # Start from near-cylinder (fingers already around it)
            self.grip_contact_detected = False
            self.grip_force_stable_since = None
            self._lift_start_time = None  # Reset lift timer for the upcoming HOLDING phase
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
                ("warn", f"[GRASP FAILED] Approach timed out after {self.GRASP_APPROACH_TIMEOUT_S:.0f}s — "
                         f"contact depth {contact_d:.4f}m never reached the {self.GRASP_CONTACT_DEPTH}m "
                         f"threshold (cylinder likely not seated between the fingers). Backing out along "
                         f"the reverse approach axis and restoring CBF."))
            self._abort_lift_start = None
            self._abort_lift_color = color
            self._transition("ABORT_RETREAT")
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf=f"+{self.cylinders[color]['cbf_name']}",  # keep bypass active during retreat
                grasp_margin=None,
                gripper_cmd=f"OPEN_{inp.active_arm.upper()}",
                reset_trigger=True,
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
    # PHASE 3: GRASP_CLOSE (timed slow closure, then plugin attach)
    # ------------------------------------------------------------------
    # Force sensor data is IGNORED (corrupted). Grasp is confirmed purely by
    # a fixed closure time, after which the cylinder is welded to the gripper
    # via the Gazebo LinkAttacher plugin (handled by main_shared_autonomy on
    # the ATTACH gripper_cmd).
    CLOSURE_WAIT_S = 5.0  # seconds of slow closure before attaching

    def _grasp_close(self, inp: TickInput) -> TickOutput:
        self._transition("GRASP_CLOSE")
        log_lines = []
        color = inp.active_goal_key.split('_')[0]

        elapsed = time.time() - self.grasp_timer
        dt = 0.01  # 100 Hz tick

        target_twist = np.zeros(6)  # Arm frozen throughout closure

        # Close fingers slowly toward the cylinder surface
        self.grip_position -= self.GRIP_CLOSE_VELOCITY * dt
        self.grip_position = max(self.GRIP_FINAL_POSITION, self.grip_position)
        gripper_cmd = f"CLOSE_{inp.active_arm.upper()}_{self.grip_position:.4f}"

        # After CLOSURE_WAIT_S of closure → attach via plugin and LIFT the object
        if elapsed >= self.CLOSURE_WAIT_S:
            log_lines.append(
                ("info", f"[GRASP] Closure complete ({self.CLOSURE_WAIT_S:.0f}s). "
                         f"Attaching {color} cylinder. Lifting clear of the table."))
            self._transition("LIFT")
            self._lift_start_time = None  # _lift records the start on its first tick
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                # Clear the gripper↔cylinder CBF bypass: the ATTACH command
                # re-parents the cylinder as a real link of the arm chain (with
                # its own collision pairs vs the environment and a smooth 3s
                # barrier ramp), so from now on it must be treated as a robot
                # link — NOT bypassed. (Self-collision vs the gripper's own
                # fingers/wrist is already adjacency-excluded by the handler.)
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


    # ------------------------------------------------------------------
    # PHASE 4: LIFT (raise the grasped object a few cm clear of the table)
    # ------------------------------------------------------------------
    # Slow, short vertical lift just to break contact with the table before the
    # shared-autonomy placement phase takes over. Raised per request so the lift
    # is clearly felt on the handle and clears the object further:
    # 0.03 m/s * 3.0 s = 0.09 m (~9 cm).
    LIFT_VELOCITY = 0.03    # m/s upward (slow)
    LIFT_DURATION = 3.0     # s  -> 0.03 * 3.0 = 0.09 m = 9 cm lift
    LIFT_HEIGHT = LIFT_VELOCITY * LIFT_DURATION  # for logging only

    def _lift(self, inp: TickInput) -> TickOutput:
        """Vertical lift phase: command a slow Z-up twist for LIFT_DURATION, then HOLD.

        Runs blind (arm frozen in XY/orientation, no goal tracking). On completion
        it transitions to HOLDING, where the shared-autonomy loop resumes and the
        user may drive the (now loaded) gripper toward any remaining goal.
        """
        self._transition("LIFT")
        log_lines = []

        if self._lift_start_time is None:
            self._lift_start_time = time.time()
            log_lines.append(
                ("info", f"[LIFT] Raising grasped object ~{self.LIFT_HEIGHT * 100:.0f} cm "
                         f"clear of the table (slow)."))

        elapsed = time.time() - self._lift_start_time

        if elapsed < self.LIFT_DURATION:
            target_twist = np.array([0.0, 0.0, self.LIFT_VELOCITY, 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=None,
                grasp_margin=None,
                log_lines=log_lines,
            )

        # Lift complete -> hand control to HOLDING / shared autonomy.
        self._transition("HOLDING")
        log_lines.append(
            ("info", "[LIFT] Complete. Entering HOLDING — shared autonomy resumed."))
        return TickOutput(
            target_twist=np.zeros(6),
            new_state=self._state,
            ignore_cbf=None,
            grasp_margin=None,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 5: HOLDING (object in gripper; shared autonomy drives toward goals)
    # ------------------------------------------------------------------
    def _holding(self, inp: TickInput) -> TickOutput:
        """Loaded shared-autonomy phase.

        The object is grasped and lifted. This handler does NOT command motion of
        its own — it passes the outer loop's policy twist (inp.pi_max) straight
        through, so the user can steer the loaded gripper toward any remaining
        goal (e.g. the Platform placement goal, or the other cylinder for a
        robustness test) and the belief estimator keeps predicting over those
        goals. Committing a new grasp (PRE_GRASP) is intentionally impossible here.

        A trigger pull in HOLDING means "release / place": the node opens the
        gripper, detaches the payload, and the system falls back to
        SHARED_AUTONOMY exactly as if it had just started, now aware of the
        updated world (one cylinder already placed).
        """
        self._transition("HOLDING")
        log_lines = []

        if not self._holding_entered:
            self._holding_entered = True
            log_lines.append(
                ("info", "=== [HOLDING] Object grasped & lifted. Any REMAINING goal is "
                         "demandable (drive / belief only — no second grasp). "
                         "Type a goal (e.g. 'Platform_Place') to steer; trigger/'OPEN' to release. ==="))

        if inp.trigger_pulled:
            log_lines.append(
                ("info", "[HOLDING] Release requested — opening gripper and placing object."))
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,   # node's release routine performs the actual transition
                ignore_cbf=None,
                grasp_margin=None,
                release_object=True,
                log_lines=log_lines,
            )

        # Pass the outer-loop policy straight through (drive toward the active goal).
        return TickOutput(
            target_twist=inp.pi_max,
            new_state=self._state,
            ignore_cbf=None,
            grasp_margin=None,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 6: RELEASE_LIFT (post-OPEN: move clear of the just-placed object)
    # ------------------------------------------------------------------
    def _release_lift(self, inp: TickInput) -> TickOutput:
        """Dual of the post-CLOSE LIFT, executed after the object is released.

        Right after OPEN the gripper is sitting on top of the freshly-placed
        cylinder, which is being re-introduced into the collision world with a
        smooth barrier ramp. We drive a slow vertical lift to move clear of it
        BEFORE the barrier fully engages, then hand control back to the user
        (SHARED_AUTONOMY). Mirrors _lift but returns to SHARED_AUTONOMY instead
        of HOLDING.
        """
        self._transition("RELEASE_LIFT")
        log_lines = []

        if self._release_lift_start is None:
            self._release_lift_start = time.time()
            log_lines.append(
                ("info", f"[RELEASE-LIFT] Moving ~{self.LIFT_HEIGHT * 100:.0f} cm clear of the "
                         f"placed object before the barrier engages."))

        elapsed = time.time() - self._release_lift_start

        if elapsed < self.LIFT_DURATION:
            target_twist = np.array([0.0, 0.0, self.LIFT_VELOCITY, 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=None,
                grasp_margin=None,
                log_lines=log_lines,
            )

        # Clear of the object -> return control to the user.
        self._transition("SHARED_AUTONOMY")
        log_lines.append(("info", "[RELEASE-LIFT] Clear. Teleoperation resumed."))
        return TickOutput(
            target_twist=np.zeros(6),
            new_state=self._state,
            ignore_cbf=None,
            grasp_margin=None,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 7: ABORT_RETREAT (failed grasp — back out the way we came in)
    # ------------------------------------------------------------------
    def _abort_lift(self, inp: TickInput) -> TickOutput:
        """Retreat after a failed grasp (approach/align timeout).

        This is the EXACT OPPOSITE of the approach: the gripper backs out along
        the NEGATIVE approach axis (its local +X), retracing the insertion path
        away from the cylinder, with the fingers OPEN. The gripper<->cylinder CBF
        bypass stays active DURING the retreat (so no barrier spike while still
        overlapping); once clear, the CBF is restored (ignore_cbf="None") and
        control returns to SHARED_AUTONOMY.
        """
        self._transition("ABORT_RETREAT")
        log_lines = []

        if self._abort_lift_start is None:
            self._abort_lift_start = time.time()
            log_lines.append(
                ("info", f"[ABORT-RETREAT] Backing out ~{self.LIFT_HEIGHT * 100:.0f} cm along the "
                         f"reverse approach axis (gripper open) before restoring CBF."))

        elapsed = time.time() - self._abort_lift_start
        color = self._abort_lift_color or inp.active_goal_key.split('_')[0]
        cbf_name = self.cylinders[color]['cbf_name'] if color in self.cylinders else ""

        if elapsed < self.LIFT_DURATION:
            # Reverse approach: retreat along the gripper's local -X (approach axis
            # points +X into the object, so backing out is -X) in world frame.
            approach_axis = inp.current_T_EE[:3, :3][:, 0]
            v_lin = -self.LIFT_VELOCITY * approach_axis
            target_twist = np.array([v_lin[0], v_lin[1], v_lin[2], 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=f"+{cbf_name}",  # keep bypass active while retreating
                grasp_margin=None,
                log_lines=log_lines,
            )

        # Retreat complete → safe to restore CBF and hand control back.
        self._transition("SHARED_AUTONOMY")
        self._abort_lift_start = None
        self._abort_lift_color = None
        log_lines.append(("info", "[ABORT-RETREAT] Clear of the cylinder. CBF restored. Teleoperation resumed."))
        return TickOutput(
            target_twist=np.zeros(6),
            new_state=self._state,
            ignore_cbf="None",
            grasp_margin=CLEAR_MARGIN,
            log_lines=log_lines,
        )
