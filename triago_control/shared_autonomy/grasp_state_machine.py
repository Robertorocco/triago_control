#!/usr/bin/env python3
"""Dict-dispatch grasp state machine over TickInput/TickOutput dataclasses (no ROS/plot deps).
Belief entry/stay hysteresis (BELIEF_ENTER vs BELIEF_STAY) keeps a noisy belief dip from dropping
PRE_GRASP; every handler must return a concrete TickOutput, and new states are one handler + one
dict entry."""

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
    # None -> leave the margin topic untouched this tick (matches the caller contract,
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
    # ENTER shrinks the search GRASP_ALIGN must finish on its own: it drives to the
    # same standoff but demands ALIGN_POS_TOL (0.0209), so whatever ENTER allows
    # above that has to be closed autonomously inside ALIGN_TIMEOUT_S.
    POS_ERR_ENTER = 0.040
    ANG_ERR_ENTER = 0.16
    POS_ERR_STAY = 0.072
    ANG_ERR_STAY = 0.224

    BELIEF_ENTER = 0.90   # belief threshold required to *enter* PRE_GRASP
    BELIEF_STAY = 0.75    # relaxed belief threshold required to *stay* in PRE_GRASP
    #   ^ Fix for Problem A: without this, a single noisy EMA dip below 0.90
    #     would kick the robot out of PRE_GRASP even while perfectly aligned.

    GRASP_CBF_MARGIN = -0.08
    # Tightened ~10% (operator observed false-positive grasp confirmations: the
    # gate let GRASP_CLOSE start when the fingers weren't actually well-seated).
    # We have NO force/torque sensing on the real hardware (joint pos/vel + camera
    # only), so contact must be confirmed purely geometrically — tightening these
    # gates is the correct, sensor-realistic way to reduce false positives.
    # Gripper-box<->cylinder overlap required to trigger finger closure, PER GRASP TYPE.
    #   contact_ok = (contact_d <= threshold) and contact_d is NEGATIVE for overlap, so a
    #   LESS-negative threshold demands LESS overlap = EASIER to satisfy.
    #   - TOP  (-0.03): vertical approach; the arm can't seat the fingers deeply, so accept
    #     shallow overlap (a -0.0365 m reading now succeeds).
    #   - SIDE (-0.04): horizontal approach lets the fingers bracket the cylinder wall, so
    #     require deeper seating for a firmer, less slip-prone grasp.
    #   Both are reachable within the relaxed gripper<->cylinder CBF (GRASP_CBF_MARGIN=-0.08).
    #   Selected by grasp type via _contact_depth_threshold().
    GRASP_CONTACT_DEPTH_TOP = -0.03
    # Tracks GRASP_INSERTION_TRAVEL_SIDE: the two are coupled, since a shallower
    # insertion ends the advance further out and so reads a less-negative overlap.
    # Shortening the side travel to 0.068 tops the reading out near -0.045, which
    # the old -0.05 gate could not reach -- the approach then always timed out on
    # a grasp that was visually seated. Re-tune BOTH together, never one alone.
    GRASP_CONTACT_DEPTH_SIDE = -0.04
    APPROACH_ANG_TOL = 0.135         # rad -- approach-axis alignment at end of insertion
    APPROACH_POS_TOL = 0.009         # m -- position-reached fallback
    # Straight-line advance from the standoff along the approach axis (the DEPTH knob),
    # PER GRASP TYPE. The gripper drives this far from the standoff toward/into the
    # cylinder before closing; selected via _insertion_travel().
    #   - TOP: vertical insertion depth.
    #   - SIDE: shallower than top -- deep enough for the fingers to bracket the wall
    #     without shoving the cylinder sideways before they close.
    GRASP_INSERTION_TRAVEL_TOP = 0.09
    # Standoff is 0.05 from the cylinder surface, so travel > 0.05 ends PAST the
    # surface (bracketing the wall) and travel < 0.05 stops short of it.
    GRASP_INSERTION_TRAVEL_SIDE = 0.068
    GRASP_FORCE_THRESHOLD = 2.0
    GRASP_CLOSE_HOLD_S = 4.0
    GRASP_APPROACH_TIMEOUT_S = 15.0  # approach can be slow with the relaxed CBF

    # Force-controlled closure parameters
    GRIP_CLOSE_VELOCITY = 0.02   # rad/s — slow closure (0.05 rad over CLOSURE_WAIT_S)
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
                   (mirrors the GRASP_DEBUG flag).
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
        self._release_start_pos = None  # EE position at release retreat start (travel gate)
        self._abort_lift_start = None  # reset for ABORT_LIFT (failed grasp retreat) phase
        self._abort_lift_color = None  # color of the cylinder being retreated from
        self._abort_start_pos = None  # EE position at retreat start (travel-distance gate)
        self._recover_start = None  # reset for RECOVER (post-abort barrier-restore settle) phase
        self._recover_confirm_start = None  # debounce timer for a confirmed-clear contact reading
        self._recover_warn_count = 0  # edge-trigger for RECOVER's throttled "still stuck" warning
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
            "RECOVER": self._recover,
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

    _GRASPABLE_TYPES = ('Top', 'Side')

    def _is_graspable(self, inp: TickInput) -> bool:
        """True only for goal keys of a real grasp type ('Color_Top'/'Color_Side').

        Pure-reach goal types (e.g. 'Color_Front', see goal_set.py) have no
        physical cylinder behind them -- no CBF pair name, no grasp_contact
        signal, nothing for GRASP_APPROACH/GRASP_CLOSE to close on or ATTACH.
        Blocking PRE_GRASP entry here means the ENTIRE grasp-execution chain
        (GRASP_ALIGN/APPROACH/CLOSE/LIFT/HOLDING, and every ORANGE_/ATTACH_
        gripper_cmd + shared_autonomy_handler.attach_object_visually call they
        trigger) is categorically unreachable for such goals -- they stay in
        SHARED_AUTONOMY (belief-blended reach/hover) forever, which is exactly
        the intended behavior for a movement-only tutorial target.
        """
        parts = inp.active_goal_key.split('_')
        return len(parts) == 2 and parts[1] in self._GRASPABLE_TYPES

    def _contact_depth_threshold(self, goal_key: str) -> float:
        """Contact-overlap threshold for the active grasp type ('Color_Top'/'Color_Side').

        Top grasps accept shallow overlap (arm can't seat fingers deeply on a
        vertical approach); side grasps require deeper seating. Defaults to the
        (stricter) side value for anything unexpected.
        """
        parts = goal_key.split('_')
        gtype = parts[1] if len(parts) == 2 else ''
        return self.GRASP_CONTACT_DEPTH_TOP if gtype == 'Top' else self.GRASP_CONTACT_DEPTH_SIDE

    def _insertion_travel(self, goal_key: str) -> float:
        """Straight-line insertion depth for the active grasp type ('Color_Top'/'Color_Side').

        Top uses the full depth; side is 20% shallower so the fingers don't shove
        the cylinder sideways before closing. Defaults to the (shallower) side value
        for anything unexpected.
        """
        parts = goal_key.split('_')
        gtype = parts[1] if len(parts) == 2 else ''
        return self.GRASP_INSERTION_TRAVEL_TOP if gtype == 'Top' else self.GRASP_INSERTION_TRAVEL_SIDE

    def step(self, inp: TickInput) -> TickOutput:
        """Evaluates the active goal's transition guard, then dispatches to the handler.

        Priority ordering: GRASP_CLOSE
        and GRASP_APPROACH are "sticky" (handled purely by their own internal
        logic since they must run to completion / timeout), whereas the choice
        between PRE_GRASP and SHARED_AUTONOMY is re-evaluated every tick based on
        belief + alignment.
        """
        if self._state in ("GRASP_CLOSE", "GRASP_APPROACH", "LIFT", "HOLDING", "GRASP_ALIGN", "RELEASE_LIFT", "ABORT_RETREAT", "RECOVER"):
            # GRASP_* and LIFT are pure run-to-completion phases. HOLDING is
            # special: while an object is in the gripper the PRE_GRASP branch is
            # deliberately UNREACHABLE (you cannot commit a second grasp with the
            # same gripper). The user can still drive toward — and the belief
            # estimator can still predict — any remaining goal; that motion is
            # produced by the outer loop's policy (pi_max), which _holding passes
            # straight through. Only an explicit release (trigger) leaves HOLDING.
            out = self._handlers[self._state](inp)
        elif self._is_graspable(inp) and self._belief_ok(inp) and self._is_aligned(inp):
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
            grasp_margin=CLEAR_MARGIN,   # explicit margin clear on this transition
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
            log_lines.append(("info", "=== [PRE-GRASP READY] Alignment perfect! Type 'CLOSE' to execute. ==="))
            log_lines.append(
                ("info", f"[GRASP-DBG] PRE_GRASP goal={inp.active_goal_key} b_max={inp.b_max:.3f} "
                         f"pos_err={inp.pos_error:.4f}m ang_err={inp.ang_error:.4f} (waiting for CLOSE)"))

        # QP-constrained policy (pi_max), NOT the raw v_geo: at the standoff the raw velocity
        # fights the CBF (push-pushback oscillation); pi_max decelerates smoothly at the barrier.
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
    ALIGN_POS_TOL = 0.0209   # m -- within ~2.1 cm of the standoff
    ALIGN_ANG_TOL = 0.103    # rad -- approach-axis within ~5.9 deg of the goal
    # Lateral centering tolerance: the perpendicular distance from the cylinder
    # axis to the gripper's approach line must be within this. If not, the fingers
    # will hit one side of the cylinder and knock it over during the blind insertion.
    # Physically: gripper finger opening is ~7 cm, cylinder diameter is ~4 cm, so
    # up to ~1.5 cm lateral error still lets both fingers bracket the cylinder --
    # this tolerance still exceeds that physical bound, tightened only partway there.
    ALIGN_CENTERING_TOL = 0.0207  # m -- centered within ~20.7 mm of the cylinder axis
    ALIGN_TIMEOUT_S = 12.0  # s -- precise centring is slow; a premature abort nudges the cylinder

    def _grasp_align(self, inp: TickInput) -> TickOutput:
        """Drive precisely to the standoff pose before committing the blind insertion.

        The PRE_GRASP entry tolerances are generous (6 cm / 0.20 rad) so the user
        can trigger comfortably; but the straight-line insertion needs near-perfect
        centering on the cylinder axis, or the fingers shove the object sideways.

        The gripper<->cylinder CBF is RELAXED here (as in GRASP_APPROACH): with the nominal
        barrier the gripper is held short of the standoff and alignment can never converge.
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

        # Lateral centering error: the miss-distance between the gripper's
        # INSERTION LINE (through the EE, along the approach axis) and the
        # CYLINDER AXIS (a vertical line through the cylinder). This is what
        # decides whether the fingers will bracket the cylinder symmetrically.
        #
        # Grasp height is free and approach depth irrelevant: remove BOTH the vertical and the
        # along-approach components -- only the finger-opening offset can hit the cylinder.
        color = inp.active_goal_key.split('_')[0]
        cyl_pos = self.cylinders[color]['pos']
        approach_axis = self._align_target[:3, :3][:, 0]   # gripper +X (unit)
        cyl_axis = np.array([0.0, 0.0, 1.0])               # upright table cylinder
        ee_to_cyl = cyl_pos - inp.current_T_EE[:3, 3]
        n = np.cross(approach_axis, cyl_axis)
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-6:
            # Side grasp (approach ⊥ cylinder axis): miss-distance along the
            # common perpendicular = the finger-opening offset. Height & depth removed.
            n = n / n_norm
            centering_err = float(abs(np.dot(ee_to_cyl, n)))
        else:
            # Top grasp (approach ∥ cylinder axis): centering is the horizontal
            # radial distance from the EE to the axis (remove the vertical component).
            d_perp = ee_to_cyl - np.dot(ee_to_cyl, cyl_axis) * cyl_axis
            centering_err = float(np.linalg.norm(d_perp))

        pos_ok = pos_err < self.ALIGN_POS_TOL
        ang_ok = ang_err < self.ALIGN_ANG_TOL
        centered_ok = centering_err < self.ALIGN_CENTERING_TOL

        if pos_ok and ang_ok and centered_ok:
            log_lines.append(
                ("info", f"[GRASP] Alignment converged (pos={pos_err:.4f}m, ang={ang_err:.4f}, "
                         f"centering={centering_err*1000:.1f}mm). Starting straight-line approach."))
            T_base = self._align_target.copy()
            R_base = T_base[:3, :3]
            approach_axis = R_base[:, 0]
            locked = np.eye(4)
            locked[:3, :3] = R_base
            locked[:3, 3] = T_base[:3, 3] + approach_axis * self._insertion_travel(inp.active_goal_key)
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
            if not centered_ok:
                reasons.append(
                    f"LATERAL CENTERING off (centering_err={centering_err*1000:.1f}mm, "
                    f"need < {self.ALIGN_CENTERING_TOL*1000:.0f}mm — fingers would miss the cylinder)")
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
                grasp_margin=self.GRASP_CBF_MARGIN,  # relaxed -> contact keeps publishing, no barrier spike
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
        ang_ok = ang_fwd_err < self.APPROACH_ANG_TOL

        contact_threshold = self._contact_depth_threshold(inp.active_goal_key)
        contact_d = inp.grasp_contact.get(color.lower(), 1.0)
        contact_ok = contact_d <= contact_threshold

        # Position-reached fallback: with the straight-line locked target, the
        # advance is finished once the EE is within APPROACH_POS_TOL of it, even
        # if the gripper-box contact distance never crosses the per-type contact threshold.
        pos_to_target = np.linalg.norm(
            inp.current_T_EE[:3, 3] - self.locked_grasp_pose[:3, 3])
        pos_reached = pos_to_target < self.APPROACH_POS_TOL

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
                         f"contact depth {contact_d:.4f}m never reached the {contact_threshold}m "
                         f"threshold (cylinder likely not seated between the fingers). Backing out along "
                         f"the reverse approach axis and restoring CBF."))
            self._abort_lift_start = None
            self._abort_lift_color = color
            self._transition("ABORT_RETREAT")
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf=f"+{self.cylinders[color]['cbf_name']}",  # keep bypass active during retreat
                grasp_margin=self.GRASP_CBF_MARGIN,  # relaxed -> contact keeps publishing, no barrier spike
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
    CLOSURE_WAIT_S = 2.5  # seconds of slow closure before attaching

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
    # 0.06 m/s * 1.5 s = 0.09 m (~9 cm).
    LIFT_VELOCITY = 0.06    # m/s upward (slow)
    LIFT_DURATION = 1.5     # s  -> 0.06 * 1.5 = 0.09 m = 9 cm lift
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
    # Gated on ACTUAL EE travel, not elapsed time: a time-gated lift only
    # COMMANDS a velocity, and whatever the CBF/joint limits/tracking lag absorb
    # is silently lost -- the barrier then re-engages with the gripper still
    # overlapping the object it just placed, which is what wedges the arm. Same
    # failure and same remedy as ABORT_RETREAT below, whose numbers these mirror.
    RELEASE_RETREAT_DISTANCE = 0.14   # m — guaranteed EE clearance before handing back
    RELEASE_RETREAT_VELOCITY = 0.10   # m/s — decisive; twice the abort retreat's pace
    RELEASE_RETREAT_MAX_S = 3.5       # s — hard cap (covers 14 cm + tracking lag)

    def _release_lift(self, inp: TickInput) -> TickOutput:
        """Dual of the post-CLOSE LIFT, executed after the object is released.

        Right after OPEN the gripper still brackets the freshly-placed cylinder,
        which is being re-introduced into the collision world on a smooth barrier
        ramp. Back out along the reverse approach axis blended with world +Z (the
        same direction ABORT_RETREAT uses) until the EE has physically travelled
        RELEASE_RETREAT_DISTANCE, then hand control back to the user. A pure
        vertical lift does NOT de-approach a SIDE placement -- the fingers stay
        wrapped around the cylinder while rising, so the re-engaging barrier
        catches them still overlapping.
        """
        self._transition("RELEASE_LIFT")
        log_lines = []

        if self._release_lift_start is None:
            self._release_lift_start = time.time()
            self._release_start_pos = inp.current_T_EE[:3, 3].copy()
            log_lines.append(
                ("info", f"[RELEASE-LIFT] Backing ~{self.RELEASE_RETREAT_DISTANCE * 100:.0f} cm off "
                         f"the placed object along the reverse approach axis before the "
                         f"barrier engages."))

        elapsed = time.time() - self._release_lift_start
        traveled = float(np.linalg.norm(inp.current_T_EE[:3, 3] - self._release_start_pos))
        done = traveled >= self.RELEASE_RETREAT_DISTANCE

        if not done and elapsed < self.RELEASE_RETREAT_MAX_S:
            v_lin = self.RELEASE_RETREAT_VELOCITY * self._retreat_direction(inp.current_T_EE)
            target_twist = np.array([v_lin[0], v_lin[1], v_lin[2], 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=None,
                grasp_margin=None,
                log_lines=log_lines,
            )

        if not done:
            log_lines.append(
                ("warn", f"[RELEASE-LIFT] Retreat capped at {self.RELEASE_RETREAT_MAX_S:.0f}s "
                         f"(only {traveled * 100:.1f} cm of "
                         f"{self.RELEASE_RETREAT_DISTANCE * 100:.0f} cm — arm likely joint-limited "
                         f"or blocked). Handing back anyway; expect a tight barrier."))

        # Clear of the object -> return control to the user.
        self._transition("SHARED_AUTONOMY")
        log_lines.append(
            ("info", f"[RELEASE-LIFT] Clear ({traveled * 100:.1f} cm). Teleoperation resumed."))
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
    # Gate the retreat on ACTUAL EE travel, NOT the gripper<->cylinder gap: the
    # fingers open on abort, which widens the gripper contact box so its overlap
    # with the cylinder GROWS (gap reads more-negative) even as the arm backs
    # out -- a useless retreat criterion. Back out a guaranteed physical distance
    # instead, well clear before RECOVER restores the barrier.
    ABORT_RETREAT_DISTANCE = 0.10    # m — guaranteed EE back-out before restoring the barrier
    ABORT_RETREAT_VELOCITY = 0.05    # m/s — decisive (faster than a lift) so 10 cm clears quickly
    ABORT_RETREAT_MAX_S = 6.0        # s — hard cap (covers 10 cm + tracking lag; joint-limited case)
    # Blend a small UP component into the retreat direction (not pure reverse-
    # approach): a straight-line horizontal retreat for a SIDE grasp can drive
    # straight into a joint limit or the table's own (non-bypassed) CBF -- the
    # observed failure mode (retreat capped at 4 of 10 cm). Backing out while
    # also lifting is far less likely to be blocked by the SAME obstruction.
    ABORT_RETREAT_LIFT_BLEND = 0.4   # fraction of world +Z mixed into -approach_axis

    @classmethod
    def _retreat_direction(cls, current_T_EE):
        """Unit retreat direction: reverse approach axis blended with world +Z."""
        approach_axis = current_T_EE[:3, :3][:, 0]
        d = -approach_axis + cls.ABORT_RETREAT_LIFT_BLEND * np.array([0.0, 0.0, 1.0])
        return d / np.linalg.norm(d)

    def _abort_lift(self, inp: TickInput) -> TickOutput:
        """Retreat after a failed grasp (approach/align timeout).

        The EXACT OPPOSITE of the approach: the gripper backs out along the
        NEGATIVE approach axis (its local +X), fingers OPEN, until the EE has
        physically travelled ABORT_RETREAT_DISTANCE from where the retreat began
        (or the retreat is capped at ABORT_RETREAT_MAX_S). The CBF bypass +
        relaxed margin stay active DURING the retreat (no barrier spike while
        still overlapping); once clear, control passes to RECOVER, which restores
        the barrier smoothly -- never a one-tick snap to nominal.
        """
        self._transition("ABORT_RETREAT")
        log_lines = []
        color = self._abort_lift_color or inp.active_goal_key.split('_')[0]
        cbf_name = self.cylinders[color]['cbf_name'] if color in self.cylinders else ""

        if self._abort_lift_start is None:
            self._abort_lift_start = time.time()
            self._abort_start_pos = inp.current_T_EE[:3, 3].copy()
            log_lines.append(
                ("info", f"[ABORT-RETREAT] Backing out ~{self.ABORT_RETREAT_DISTANCE * 100:.0f} cm along "
                         f"the reverse approach axis (gripper open) before restoring the barrier."))

        elapsed = time.time() - self._abort_lift_start
        traveled = float(np.linalg.norm(inp.current_T_EE[:3, 3] - self._abort_start_pos))
        done = traveled >= self.ABORT_RETREAT_DISTANCE

        if not done and elapsed < self.ABORT_RETREAT_MAX_S:
            # Reverse approach blended with a lift component (see
            # ABORT_RETREAT_LIFT_BLEND) -- less likely to be blocked than a
            # pure horizontal retreat.
            v_lin = self.ABORT_RETREAT_VELOCITY * self._retreat_direction(inp.current_T_EE)
            target_twist = np.array([v_lin[0], v_lin[1], v_lin[2], 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf=f"+{cbf_name}",           # keep bypass active while retreating
                grasp_margin=self.GRASP_CBF_MARGIN,  # relaxed -> contact keeps publishing, no spike
                log_lines=log_lines,
            )

        # Cleared the distance (or capped) → hand to RECOVER for the smooth restore.
        if not done:
            log_lines.append(
                ("warn", f"[ABORT-RETREAT] Retreat capped at {self.ABORT_RETREAT_MAX_S:.0f}s "
                         f"(only {traveled * 100:.1f} cm — arm likely joint-limited). "
                         f"Restoring barrier and settling."))
        self._recover_start = None
        self._transition("RECOVER")
        return TickOutput(
            target_twist=np.zeros(6),
            new_state=self._state,
            ignore_cbf=f"+{cbf_name}",           # bypass held one more tick; RECOVER drops it
            grasp_margin=self.GRASP_CBF_MARGIN,
            log_lines=log_lines,
        )

    # ------------------------------------------------------------------
    # PHASE 8: RECOVER (post-abort: verify real clearance, THEN restore the barrier)
    # ------------------------------------------------------------------
    RECOVER_CLEARANCE = 0.02        # m — contact must read at least this before restoring starts
    RECOVER_CONFIRM_S = 0.5         # s — clearance must hold this long (debounce a noisy reading)
    RECOVER_RAMP_S = 1.5            # s — once confirmed clear, ramp margin -> nominal (arm held)
    RECOVER_NUDGE_VELOCITY = 0.03   # m/s — gentle extra retreat if ABORT_RETREAT was capped short
    RECOVER_WARN_S = 5.0            # s — start warning if still not clear past this long
    RECOVER_WARN_PERIOD_S = 5.0     # s — repeat interval for the "still stuck" warning
    RECOVER_SAFETY_BUFFER = 0.02    # m — margin stays this far behind the live-measured overlap
    RECOVER_MARGIN_FLOOR = -0.5     # m — sanity bound only, not a trust ceiling (see below)

    def _recover(self, inp: TickInput) -> TickOutput:
        """Post-abort settle: VERIFY real clearance with live contact, THEN restore the CBF.

        Fixes the residual crash risk in the first version of this phase: that
        version restored the barrier on a FIXED time schedule, trusting that
        ABORT_RETREAT's distance-based retreat always got fully clear. When the
        retreat was capped short (blocked by a joint limit / another CBF pair --
        the observed failure: 4 of the intended 10 cm), the schedule still ran
        to completion and restored a near-nominal barrier onto a gripper that
        was STILL overlapping the cylinder -> a real repulsion spike, not just
        the earlier open-finger sensor artifact.

        This version never restores past what `grasp_contact` currently
        supports: the margin is continuously clamped to (contact_d - buffer),
        so the barrier is a no-op at whatever the ACTUAL geometry is, however
        that came about. If still overlapping on entry, it keeps nudging clear
        (reverse-approach + lift blend, see _retreat_direction) instead of
        freezing and hoping. Only once clearance is confirmed (debounced) does
        it ramp to nominal and hand back to SHARED_AUTONOMY -- which unconditionally
        clears the margin on its own next tick, so handing back BEFORE genuine
        clearance would silently undo this whole safeguard.
        """
        self._transition("RECOVER")
        log_lines = []
        color = self._abort_lift_color or inp.active_goal_key.split('_')[0]
        contact_d = inp.grasp_contact.get(color.lower(), 1.0)
        cleared = contact_d >= self.RECOVER_CLEARANCE
        # Never trust a schedule past what's currently measured -- this bound
        # is applied to EVERY margin this method returns, cleared or not.
        contact_bound = contact_d - self.RECOVER_SAFETY_BUFFER

        if self._recover_start is None:
            self._recover_start = time.time()
            self._recover_confirm_start = None
            self._recover_warn_count = 0
            log_lines.append(
                ("info", f"[RECOVER] Verifying clearance (gap={contact_d * 100:.1f} cm) "
                         f"before restoring the barrier."))

        elapsed = time.time() - self._recover_start

        if not cleared:
            self._recover_confirm_start = None   # any confirm streak is broken
            margin = round(max(self.RECOVER_MARGIN_FLOOR, min(0.0, contact_bound)), 2)

            # Still overlapping: keep nudging clear (never freeze and hope -- the
            # arm is grasp_exec-locked here, so the operator's own teleop input is
            # ALSO zeroed at the node level; auto-nudging is the only thing that
            # can actually move it). Margin stays clamped to the live gap the
            # entire time, however long this takes -- no time-based restore.
            if elapsed > self.RECOVER_WARN_S and int(elapsed / self.RECOVER_WARN_PERIOD_S) > self._recover_warn_count:
                self._recover_warn_count = int(elapsed / self.RECOVER_WARN_PERIOD_S)
                log_lines.append(
                    ("warn", f"[RECOVER] Still overlapping the {color} cylinder "
                             f"(gap={contact_d * 100:.1f} cm) after {elapsed:.0f}s of nudging -- "
                             f"possibly wedged or joint-limited. Barrier held relaxed at "
                             f"{margin:+.3f} m (NOT restored), teleop still suspended. Continuing "
                             f"to nudge automatically; if this persists, a hardware-level "
                             f"intervention (E-stop / manual repositioning) may be needed."))
            v_lin = self.RECOVER_NUDGE_VELOCITY * self._retreat_direction(inp.current_T_EE)
            target_twist = np.array([v_lin[0], v_lin[1], v_lin[2], 0.0, 0.0, 0.0])
            return TickOutput(
                target_twist=target_twist,
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=margin,
                log_lines=log_lines,
            )

        # Clear -- debounce briefly (one good reading isn't proof) before ramping.
        if self._recover_confirm_start is None:
            self._recover_confirm_start = time.time()
        confirm_elapsed = time.time() - self._recover_confirm_start
        if confirm_elapsed < self.RECOVER_CONFIRM_S:
            margin = round(max(self.RECOVER_MARGIN_FLOOR, min(0.0, contact_bound)), 2)
            return TickOutput(
                target_twist=np.zeros(6),
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=margin,
                log_lines=log_lines,
            )

        ramp = max(0.0, min(1.0, (confirm_elapsed - self.RECOVER_CONFIRM_S) / self.RECOVER_RAMP_S))
        scheduled = (1.0 - ramp) * self.GRASP_CBF_MARGIN            # -0.08 -> 0
        # Quantized to 1 cm steps: the node re-logs the margin only when the
        # value changes, so a continuous ramp would spam ~150 lines.
        margin = round(max(self.RECOVER_MARGIN_FLOOR, min(scheduled, contact_bound, 0.0)), 2)

        if ramp < 1.0:
            return TickOutput(
                target_twist=np.zeros(6),   # arm frozen while the barrier ramps in
                new_state=self._state,
                ignore_cbf="None",
                grasp_margin=margin,
                log_lines=log_lines,
            )

        # Confirmed clear AND fully ramped → safe to hand control back.
        self._recover_start = None
        self._recover_confirm_start = None
        self._abort_lift_start = None
        self._abort_lift_color = None
        self._transition("SHARED_AUTONOMY")
        log_lines.append(("info", "[RECOVER] Clearance confirmed, barrier restored. Teleoperation resumed."))
        return TickOutput(
            target_twist=np.zeros(6),
            new_state=self._state,
            ignore_cbf="None",
            grasp_margin=CLEAR_MARGIN,
            log_lines=log_lines,
        )
