#!/usr/bin/env python3
"""GoalSet: cylinder geometry and dynamic SE(3) grasp-goal manifolds (pure geometry, no ROS deps)."""

import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin


def create_transform(pos, rot_mat):
    """Constructs a 4x4 homogeneous transformation matrix from translation and rotation.

    Lives here (and is re-exported) since GoalSet is the lowest-level geometry
    module and other modules need it without importing the full ROS node.
    """
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    return T


class GoalSet:
    """Owns the physical cylinder definitions and computes dynamic grasp-goal poses."""

    # pin.log3 is NaN at angle pi; within this epsilon of pi, fall back to a
    # Frobenius-norm distance instead (valid everywhere on SO(3)).
    _LOG3_SINGULARITY_EPS = 1e-3

    # Switch margin for the sticky Side/Front orientation choice (Top uses a
    # continuous roll); stops noise near the 180-deg bisector self-sustaining a flip.
    _ORIENTATION_HYSTERESIS = 0.05  # rad, in the _rotation_distance units

    # Confidence band for the position-based azimuth estimate (horizontal
    # anchor->axis vector): degenerates ON the axis and amplifies noise as d/r near it.
    _SIDE_AZIMUTH_POS_LO = 0.008   # m: at/below, position carries no azimuth
    _SIDE_AZIMUTH_POS_HI = 0.030   # m: at/above, azimuth is the raw radial

    # Heading-based estimate (gripper approach axis projected horizontally): fails
    # only where position doesn't (near-vertical gripper) -- blending covers both.
    _SIDE_HEADING_LO = 0.15
    _SIDE_HEADING_HI = 0.45

    # Minimum confidence-weighted azimuth-mix norm to trust; below it (both
    # estimates weak or cancelling) the last committed azimuth is held instead.
    _AZIMUTH_MIX_MIN_NORM = 0.15

    # Top-grasp reference frame: gripper +X (approach axis) along world -Z. Rolling
    # about world +Z leaves the approach axis fixed and sweeps the free DOF (_pick_top_roll).
    _R_TOP_DOWN = R.from_euler('y', 90, degrees=True).as_matrix()

    # Roll-singularity guard: objective amplitude (1 - x_anchor_z) vanishes only when
    # the anchor points straight up, where the Top goal is >160 deg away and inert.
    _TOP_ROLL_AMPLITUDE_EPS = 0.05

    # Normalized-EMA low-pass (wrap-free) on the two free grasp DOFs (Side azimuth,
    # Top roll), bounding how fast a goal can travel around its own manifold; 0 disables.
    GOAL_DIRECTION_FILTER_TAU = 4.0   # s
    _FILTER_NOMINAL_HZ = 100.0         # SharedControlNode.CONTROL_HZ; override per instance

    # Side-grasp standoff from the cylinder CENTRE (axis): the guidance manifold
    # pulls the reference to this radius, keeping it outside the cylinder (r ~0.02 m).
    SIDE_STANDOFF_FROM_CENTER = 0.03   # m (3 cm from the cylinder axis)

    # --- Platform placement goal ---------------------------------------------
    # Flat world disk where grasped cylinders are set down; hard constraint is the
    # cylinder axis vertical and footprint inside the disk -- rest is a free manifold.
    PLATFORM_KEY = 'Platform_Place'
    PLATFORM_POSE = np.array([1.000, 0.0, 0.701])   # world center of placement_area
    PLATFORM_RADIUS = 0.15                          # disk radius [m]
    PLATFORM_THICKNESS = 0.002                      # disk thickness [m]
    PLATFORM_PLACE_MARGIN = 0.03                    # keep the footprint this far inside the rim [m]

    # Grasp types every cylinder exposes when its own entry doesn't override
    # them via the `grasp_types` field (see __init__ docstring).
    _DEFAULT_GRASP_TYPES = ('Top', 'Side')

    def __init__(self, cylinders=None, target_keys=None, platform=None, platforms=None,
                 update_rate_hz=None):
        """Initializes the cylinder geometry table and the set of valid goal keys.

        Args:
            cylinders: dict of {color: {'pos', 'height', 'radius', 'cbf_name',
                'grasp_types' (optional)}}. `grasp_types` lists which suffixes
                ('Top', 'Side', 'Front' -- see get_dynamic_goal_pose) this
                entry offers, defaulting to _DEFAULT_GRASP_TYPES.
                grasp_types=['Front'] marks a pure reach/hover target with no
                physical object; grasp EXECUTION is still categorically
                blocked for non-Top/Side keys regardless (see
                grasp_state_machine.GraspStateMachine._is_graspable) -- this
                dict only controls which goal KEYS exist. Defaults to a
                two-cylinder Red/Blue table.
            target_keys: valid 'Color_GraspType' goal keys. Defaults to one
                key per (cylinder, grasp_type) pair plus one per platform_keys
                entry.
            platform: optional world_loader.PlatformSpec overriding the
                class-level PLATFORM_POSE/RADIUS/THICKNESS/PLACE_MARGIN
                defaults with the loaded world's placement disk (a reference
                pose/visual aid, not an obstacle). None keeps the defaults.
            platforms: optional list of PlatformSpec for multiple placement
                disks, producing one 'Platform_<name>' goal per entry.
            update_rate_hz: rate at which the owning node calls
                get_dynamic_goal_pose, used only to turn
                GOAL_DIRECTION_FILTER_TAU into a per-call EMA gain. None
                keeps _FILTER_NOMINAL_HZ.
        """
        # --- Placement platform(s) --------------------------------------------
        # `platform` -> one 'Platform_Place' goal; `platforms` -> one
        # 'Platform_<name>' goal per entry. Legacy PLATFORM_* attrs point at the first.
        specs = (list(platforms) if platforms
                 else ([platform] if platform is not None else []))
        self._platform_params = {}
        if specs:
            for spec in specs:
                name = getattr(spec, 'name', None) or 'Place'
                self._platform_params[f'Platform_{name}'] = {
                    'pose': np.asarray(spec.pose, dtype=float),
                    'radius': float(spec.radius),
                    'thickness': float(spec.thickness),
                    'place_margin': float(spec.place_margin),
                }
        else:
            # No world platform supplied: fall back to the class-level defaults.
            self._platform_params[self.PLATFORM_KEY] = {
                'pose': self.PLATFORM_POSE, 'radius': self.PLATFORM_RADIUS,
                'thickness': self.PLATFORM_THICKNESS,
                'place_margin': self.PLATFORM_PLACE_MARGIN,
            }
        self.platform_keys = list(self._platform_params.keys())
        # Primary platform = first key; legacy scalar attrs point at it so
        # get_platform_goal_pose's default and external readers resolve consistently.
        self.PLATFORM_KEY = self.platform_keys[0]
        _p0 = self._platform_params[self.PLATFORM_KEY]
        self.PLATFORM_POSE = _p0['pose']
        self.PLATFORM_RADIUS = _p0['radius']
        self.PLATFORM_THICKNESS = _p0['thickness']
        self.PLATFORM_PLACE_MARGIN = _p0['place_margin']

        if cylinders is None:
            cylinders = {
                'Red':  {'pos': np.array([0.800, -0.20, 0.775]), 'height': 0.15,
                         'radius': 0.02, 'cbf_name': 'red_cylinder'},
                'Blue': {'pos': np.array([0.800,  0.20, 0.775]), 'height': 0.15,
                         'radius': 0.02, 'cbf_name': 'blue_cylinder'},
            }
        self.cylinders = cylinders

        if target_keys is None:
            target_keys = []
            for color, spec in self.cylinders.items():
                for grasp_type in spec.get('grasp_types', self._DEFAULT_GRASP_TYPES):
                    target_keys.append(f'{color}_{grasp_type}')
            target_keys.extend(self.platform_keys)   # one key per placement disk
        self.target_keys = target_keys

        # Sticky orientation memory per goal key: None until first computed, then
        # 'primary' or 'flipped' (see _ORIENTATION_HYSTERESIS).
        self._last_orientation_choice = {k: None for k in self.target_keys}

        # Filtered Side-grasp approach azimuth (unit 2-vector, anchor->axis) per goal
        # key; None until first computed. See _pick_side_azimuth.
        self._last_side_radial = {k: None for k in self.target_keys}

        # Sticky Top-grasp roll about world +Z per goal key, as a unit 2-vector
        # (cos, sin); None until first computed. See _TOP_ROLL_AMPLITUDE_EPS.
        self._last_top_roll = {k: None for k in self.target_keys}

        # Per-call EMA gain realizing GOAL_DIRECTION_FILTER_TAU at the caller's rate;
        # the direction memories above double as this filter's state.
        rate = float(update_rate_hz) if update_rate_hz else self._FILTER_NOMINAL_HZ
        tau = float(self.GOAL_DIRECTION_FILTER_TAU)
        if tau > 0.0 and rate > 0.0:
            dt = 1.0 / rate
            self._dir_filter_alpha = dt / (tau + dt)
        else:
            self._dir_filter_alpha = 1.0

        # --- Grasped-object bookkeeping (set on grasp, cleared on release) ----
        # grasped_axis_local: cylinder axis in the gripper frame at grasp time, so
        # "R_gripper @ grasped_axis_local vertical" is the placement constraint.
        self.grasped_color = None
        self.grasped_axis_local = None
        self.grasped_z_offset = 0.0

    def cbf_name(self, color):
        """Returns the CBF pair name registered for this cylinder color."""
        return self.cylinders[color]['cbf_name']

    def radius(self, color):
        """Returns the cylinder radius (m) for this color."""
        return self.cylinders[color]['radius']

    # ------------------------------------------------------------------
    # Grasped-object bookkeeping (drives the Platform placement constraint)
    # ------------------------------------------------------------------
    def set_grasped(self, color, T_grasp):
        """Record what was grasped and how, at the instant of grasp.

        Args:
            color: 'red' / 'blue' (case-insensitive) — the grasped cylinder.
            T_grasp: 4x4 gripper (EE) pose at the moment the object was attached.

        We freeze the cylinder symmetry axis (world +Z, since the cylinders stand
        upright on the table) expressed in the *gripper* frame:
            grasped_axis_local = R_grasp^T @ [0, 0, 1]
        and the gripper's height offset relative to the cylinder center, so the
        placement goal can put the cylinder bottom flat on the platform.
        """
        self.grasped_color = color.capitalize()
        R_grasp = np.asarray(T_grasp, dtype=float)[:3, :3]
        world_axis = np.array([0.0, 0.0, 1.0])
        self.grasped_axis_local = R_grasp.T @ world_axis
        cyl_center_z = self.cylinders[self.grasped_color]['pos'][2]
        self.grasped_z_offset = float(T_grasp[2, 3] - cyl_center_z)

    def clear_grasped(self):
        """Forget the grasped object (called on release / placement)."""
        self.grasped_color = None
        self.grasped_axis_local = None
        self.grasped_z_offset = 0.0

    def relocate_cylinder(self, color, new_pos):
        """Updates a cylinder's believed world position after it was placed.

        Grasp goals (Color_Top/Side) are computed dynamically from
        cylinders[color]['pos'], so updating it here is sufficient. Resets the
        cylinder's sticky orientation/azimuth memory since its geometry changed.
        """
        color = color.capitalize()
        if color not in self.cylinders:
            return
        self.cylinders[color]['pos'] = np.asarray(new_pos, dtype=float).copy()
        for gt in ('Top', 'Side'):
            k = f"{color}_{gt}"
            if k in self._last_orientation_choice:
                self._last_orientation_choice[k] = None
            if k in self._last_side_radial:
                self._last_side_radial[k] = None
            if k in self._last_top_roll:
                self._last_top_roll[k] = None

    def platform_rest_z(self, color):
        """World Z of an upright cylinder of `color` resting on the table top (z=0.70)."""
        TABLE_TOP_Z = 0.70   # table center 0.35 + half-height 0.35
        half_h = self.cylinders[color.capitalize()]['height'] / 2.0
        return float(TABLE_TOP_Z + half_h)

    def get_platform_goal_pose(self, T_anchor, approach_offset=0.05, goal_key=None):
        """SE(3) placement goal on the platform disk, as a perpendicularity manifold.

        Position: anchor XY projected onto the disk (clamped to stay
        PLATFORM_PLACE_MARGIN inside the rim), at a Z that rests the grasped
        cylinder's bottom on the platform face plus approach_offset standoff.

        Orientation: minimal rotation of the anchor that brings the grasped
        cylinder axis to vertical ("axis ⊥ platform"), constraining 2 DOF and
        leaving yaw about vertical free (anchored to the current gripper yaw,
        so no wrist flip while hovering).
        """
        # Resolve which platform this goal key refers to; a None/unknown key
        # falls back to the primary (PLATFORM_KEY).
        params = self._platform_params.get(goal_key, self._platform_params[self.PLATFORM_KEY])
        center = params['pose']
        platform_radius = params['radius']
        platform_thickness = params['thickness']
        platform_place_margin = params['place_margin']

        p_anchor = np.asarray(T_anchor, dtype=float)[:3, 3]
        R_anchor = np.asarray(T_anchor, dtype=float)[:3, :3]

        # --- Position: project onto the disk, clamp inside the rim ---
        dxy = p_anchor[:2] - center[:2]
        r = float(np.linalg.norm(dxy))
        max_r = max(platform_radius - platform_place_margin, 0.0)
        if r > max_r and r > 1e-9:
            dxy = dxy / r * max_r
        p_xy = center[:2] + dxy

        # Place the cylinder bottom on the platform top face.
        if self.grasped_color is not None and self.grasped_color in self.cylinders:
            half_h = self.cylinders[self.grasped_color]['height'] / 2.0
        else:
            half_h = 0.075  # sane default (15 cm cylinder)
        platform_top = center[2] + platform_thickness / 2.0
        z_target = platform_top + half_h + self.grasped_z_offset + approach_offset
        p_target = np.array([p_xy[0], p_xy[1], z_target])

        # --- Orientation: minimal tilt so the cylinder axis becomes vertical ---
        axis_local = (self.grasped_axis_local
                      if self.grasped_axis_local is not None
                      else np.array([0.0, 0.0, 1.0]))
        cur_axis_world = R_anchor @ axis_local
        n = np.linalg.norm(cur_axis_world)
        if n > 1e-9:
            cur_axis_world = cur_axis_world / n

        # Snap to whichever vertical direction (+Z / -Z) is closer, so we never
        # demand a needless 180-degree flip of the held object.
        target_vert = np.array([0.0, 0.0, 1.0]) if cur_axis_world[2] >= 0.0 \
            else np.array([0.0, 0.0, -1.0])

        v = np.cross(cur_axis_world, target_vert)
        s = float(np.linalg.norm(v))
        c = float(np.dot(cur_axis_world, target_vert))
        if s < 1e-8:
            R_align = np.eye(3)  # already (anti)parallel to vertical
        else:
            angle = np.arctan2(s, c)
            R_align = R.from_rotvec((v / s) * angle).as_matrix()
        R_target = R_align @ R_anchor

        return create_transform(p_target, R_target)

    @staticmethod
    def _rotation_distance(R_candidate, R_anchor):
        """Distance between two rotations, safe at the pin.log3 singularity (angle = pi).

        pin.log3 returns NaN when the relative rotation angle is exactly pi
        (trace == -1); detected via the trace and falls back to a Frobenius-norm
        distance, valid (if not geodesic) everywhere on SO(3) and agreeing in
        ranking with the angle-axis distance away from the singularity.
        """
        R_rel = R_candidate @ R_anchor.T
        trace = np.trace(R_rel)
        # cos(theta) = (trace - 1) / 2  ->  trace == -1  <=>  theta == pi
        near_singularity = trace <= (-1.0 + GoalSet._LOG3_SINGULARITY_EPS)
        if near_singularity:
            return np.linalg.norm(R_candidate - R_anchor, ord='fro')
        return np.linalg.norm(pin.log3(R_rel))

    def _pick_orientation(self, goal_key, R_primary, R_flipped, R_anchor,
                          update_memory=True, hysteresis=None):
        """Chooses between two orientation candidates with hysteresis (sticky choice).

        Recomputing from scratch each tick would flip between R_primary and
        R_flipped whenever tracking noise crosses the comparison near their
        bisector, spiking ang_error into a tracking oscillation. This method only
        switches away from the previously selected candidate when the alternative
        is ahead by more than the hysteresis margin.

        Args:
            update_memory: if False, reads the sticky memory without writing to
                it -- used for speculative/lookahead evaluations so they don't
                perturb the real control loop's committed choice.
            hysteresis: switch margin (rad); None uses _ORIENTATION_HYSTERESIS.
                get_dynamic_goal_pose passes a distance-scaled value: large when
                far from the object (locking the choice), relaxing to nominal
                near it (where the user commits to an approach side).
        """
        if hysteresis is None:
            hysteresis = self._ORIENTATION_HYSTERESIS
        dist_primary = self._rotation_distance(R_primary, R_anchor)
        dist_flipped = self._rotation_distance(R_flipped, R_anchor)

        last_choice = self._last_orientation_choice.get(goal_key)

        if last_choice == 'flipped':
            # Currently committed to R_flipped: only switch back to R_primary
            # if it is clearly better, not just marginally.
            switch = dist_primary < (dist_flipped - hysteresis)
            choice = 'primary' if switch else 'flipped'
        elif last_choice == 'primary':
            switch = dist_flipped < (dist_primary - hysteresis)
            choice = 'flipped' if switch else 'primary'
        else:
            # First time this goal key is evaluated: no prior choice to be
            # sticky about, so just take the closer one.
            choice = 'primary' if dist_primary <= dist_flipped else 'flipped'

        if update_memory:
            self._last_orientation_choice[goal_key] = choice
        return R_primary if choice == 'primary' else R_flipped

    # Distance-scaled orientation-lock thresholds (see _orientation_hysteresis).
    _ORIENT_LOCK_NEAR = 0.12   # m: at/below this anchor->object distance, nominal hysteresis
    _ORIENT_LOCK_FAR = 0.30    # m: at/above this, the choice is effectively locked
    _ORIENT_LOCK_HYST = 10.0   # rad: "locked" margin (>> max rotation distance pi)

    def _orientation_hysteresis(self, d_obj):
        """Distance-scaled switch margin for _pick_orientation.

        Ramps (smoothstep) from the nominal margin near the object to a very
        large one far away, so the 180-deg-apart candidate cannot flip while the
        user is still far and uncommitted, yet stays choosable once close.
        """
        s = self._smoothstep(d_obj, self._ORIENT_LOCK_NEAR, self._ORIENT_LOCK_FAR)
        return self._ORIENTATION_HYSTERESIS + s * (self._ORIENT_LOCK_HYST
                                                   - self._ORIENTATION_HYSTERESIS)

    @staticmethod
    def _smoothstep(x, lo, hi):
        """Smooth 0->1 ramp over [lo, hi], clamped flat outside it."""
        t = float(np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
        return 3.0 * t ** 2 - 2.0 * t ** 3

    @staticmethod
    def _unit2(v):
        """Normalizes a 2-vector, returning None when it has no usable direction."""
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-9 else None

    def _filter_direction(self, memory, goal_key, v_new, update_memory=True):
        """Normalized EMA of a unit 2-vector, using `memory` as both state and output.

        Normalizing the blend (rather than averaging angles) keeps the filter free
        of wrap discontinuities at +-pi.

        update_memory=False (speculative/lookahead calls) returns the committed
        state untouched, so those calls neither advance the real filter nor let
        the real loop observe a goal one step ahead of what it's tracking.
        """
        prev = memory.get(goal_key)
        if prev is None:
            v_out = v_new
        elif not update_memory:
            return prev
        elif self._dir_filter_alpha >= 1.0:
            v_out = v_new
        else:
            a = self._dir_filter_alpha
            v_out = self._unit2((1.0 - a) * np.asarray(prev, dtype=float) + a * v_new)
            if v_out is None:
                # Blend cancelled: the estimate reversed within one tick. Step to it
                # directly rather than holding a direction the filter cannot leave.
                v_out = v_new
        if update_memory:
            memory[goal_key] = v_out
        return v_out

    def _pick_side_azimuth(self, goal_key, p_anchor, R_anchor, p_cyl, update_memory=True):
        """Which side of the cylinder a Side grasp approaches from, as a unit 2-vector.

        Blends two confidence-weighted estimates: the horizontal anchor->axis
        radial (degenerates ON the axis, see _SIDE_AZIMUTH_POS_LO/HI) and the
        gripper's horizontal heading (degenerates only near-vertical, see
        _SIDE_HEADING_LO/HI). Their failure modes are disjoint, so the mix stays
        defined through a gripper hovering over the cylinder top.

        Confidence ramps rather than switches on a hard radius threshold, since a
        hard cutoff re-commits to whatever direction the anchor is drifting in
        when it's crossed -- near the axis, that direction is noise.

        When both estimates are weak or cancel (see _AZIMUTH_MIX_MIN_NORM), the
        last committed azimuth is held instead.
        """
        d_xy = np.asarray(p_cyl, dtype=float)[:2] - np.asarray(p_anchor, dtype=float)[:2]
        r_pos = float(np.linalg.norm(d_xy))
        c_pos = self._smoothstep(r_pos, self._SIDE_AZIMUTH_POS_LO,
                                 self._SIDE_AZIMUTH_POS_HI)

        head_xy = np.asarray(R_anchor, dtype=float)[:2, 0]
        r_head = float(np.linalg.norm(head_xy))
        c_head = self._smoothstep(r_head, self._SIDE_HEADING_LO, self._SIDE_HEADING_HI)

        v_mix = np.zeros(2)
        if r_pos > 1e-9:
            v_mix = v_mix + c_pos * (d_xy / r_pos)
        if r_head > 1e-9:
            v_mix = v_mix + (1.0 - c_pos) * c_head * (head_xy / r_head)
        mix_norm = float(np.linalg.norm(v_mix))

        if mix_norm >= self._AZIMUTH_MIX_MIN_NORM:
            return self._filter_direction(self._last_side_radial, goal_key,
                                          v_mix / mix_norm, update_memory=update_memory)

        last_radial = self._last_side_radial.get(goal_key)
        if last_radial is not None:
            return last_radial
        v_rad = np.array([1.0, 0.0])
        if update_memory:
            self._last_side_radial[goal_key] = v_rad
        return v_rad

    def _pick_top_roll(self, goal_key, R_anchor, update_memory=True):
        """Closest Top-grasp orientation on the free-roll manifold about the cylinder axis.

        A cylinder is symmetric about its axis, so the Top goal is free to roll:
        R(theta) = Rz(theta) @ _R_TOP_DOWN. The goal is the point on this circle
        closest to R_anchor.

        Closed form: minimizing the geodesic distance means maximizing
        trace(Rz(theta) @ M) with M = _R_TOP_DOWN @ R_anchor^T, i.e.
            trace = (M00 + M11)*cos(theta) + (M01 - M10)*sin(theta) + M22
            theta* = atan2(M01 - M10, M00 + M11)
        The objective's amplitude is (1 - x_anchor_z), degenerating only when the
        anchor points straight up (_TOP_ROLL_AMPLITUDE_EPS).

        At theta* the residual angular error equals the tilt between the
        anchor's approach axis and world -Z; the roll term drops out entirely.

        Returned roll is low-passed (_filter_direction, GOAL_DIRECTION_FILTER_TAU).
        """
        R_anchor = np.asarray(R_anchor, dtype=float)
        M = self._R_TOP_DOWN @ R_anchor.T
        # (cos, sin) of theta*; its norm is the objective amplitude (1 - x_anchor_z).
        v_opt = np.array([M[0, 0] + M[1, 1], M[0, 1] - M[1, 0]])
        amplitude = float(np.linalg.norm(v_opt))
        last_roll = self._last_top_roll.get(goal_key)

        if amplitude >= self._TOP_ROLL_AMPLITUDE_EPS:
            v_roll = self._filter_direction(self._last_top_roll, goal_key,
                                            v_opt / amplitude,
                                            update_memory=update_memory)
        elif last_roll is not None:
            # Inside the singular zone: hold the last committed roll.
            v_roll = last_roll
        else:
            # No prior commit and the anchor points straight up: the gripper's own Z
            # axis is horizontal there, so its heading is a continuous tie-break.
            v_roll = self._unit2(R_anchor[:2, 2])
            if v_roll is None:
                v_roll = np.array([1.0, 0.0])
            if update_memory:
                self._last_top_roll[goal_key] = v_roll

        theta = float(np.arctan2(v_roll[1], v_roll[0]))
        return R.from_euler('z', theta).as_matrix() @ self._R_TOP_DOWN

    def get_dynamic_goal_pose(self, T_anchor, goal_key, approach_offset=0.05, update_memory=True):
        """Dynamically computes the target SE(3) pose for a given goal key.

        Args:
            approach_offset: distance from the cylinder surface; positive is
                standoff, negative is envelop/penetration depth during grasp.
            update_memory: False when T_anchor is a speculative/simulated pose
                (e.g. one-step-lookahead visualization) rather than the real
                EE/user pose, so it doesn't perturb the sticky orientation choice
                used by the real control loop (see _pick_orientation).

        Side grasp: orientation adaptively minimizes wrist flips relative to
        T_anchor (see _rotation_distance, _pick_orientation). Top grasp: position
        is fixed above the axis; roll is a free manifold resolved to the point
        closest to T_anchor (see _pick_top_roll).
        """
        if goal_key.startswith('Platform'):
            # Delegates to the matching platform (multi-platform); update_memory is
            # irrelevant here since orientation is derived from the anchor each tick.
            return self.get_platform_goal_pose(T_anchor, approach_offset=approach_offset,
                                               goal_key=goal_key)

        color, grasp_type = goal_key.split('_')
        cyl = self.cylinders[color]
        p_cyl = cyl['pos']
        h, r = cyl['height'], cyl['radius']

        p_anchor = T_anchor[:3, 3]
        R_anchor = T_anchor[:3, :3]  # Extract current user/EE orientation

        # Distance-scaled orientation-lock margin: locks far from the cylinder
        # (uncommitted), relaxes near it. See _orientation_hysteresis.
        hyst = self._orientation_hysteresis(float(np.linalg.norm(p_anchor - p_cyl)))

        if grasp_type == 'Top':
            p_target = p_cyl + np.array([0, 0, h / 2 + approach_offset])

            # Position pinned above the axis; roll tracks the anchor continuously, so
            # no discrete flip (hence no hysteresis) exists here, unlike Side/Front.
            R_target = self._pick_top_roll(goal_key, R_anchor, update_memory=update_memory)

        elif grasp_type == 'Side':
            # 1. Z-Height Tracking (user controls grasp height)
            z_target = np.clip(p_anchor[2], p_cyl[2] - h / 2 + 0.02, p_cyl[2] + h / 2 - 0.02)

            # 2. Approach azimuth. Drives BOTH the goal orientation and the standoff
            #    position below, so resolving it once keeps the two consistent.
            v_rad = self._pick_side_azimuth(goal_key, p_anchor, R_anchor, p_cyl,
                                            update_memory=update_memory)

            X_t = np.array([v_rad[0], v_rad[1], 0.0])

            # 3. FIXED GRIPPER PLANE: XY perpendicular to cylinder axis (world Z)
            Z_t = np.array([0., 0., 1.])
            Y_t = np.cross(Z_t, X_t)  # always well-conditioned (Z vertical, X horizontal)

            R_candidate = np.column_stack((X_t, Y_t, Z_t))
            R_flipped = np.column_stack((X_t, -Y_t, -Z_t))  # 180 deg around X_t (fingers down)

            # Closest-to-anchor orientation with hysteresis against chatter near the
            # bisector between the two candidates (see _pick_orientation).
            R_target = self._pick_orientation(goal_key, R_candidate, R_flipped, R_anchor,
                                               update_memory=update_memory, hysteresis=hyst)

            # Standoff from the cylinder CENTRE (not the surface) keeps the reference
            # outside the cylinder; blind insertion (GRASP_INSERTION_TRAVEL) handles the final reach.
            standoff = self.SIDE_STANDOFF_FROM_CENTER
            p_target = np.array([
                p_cyl[0] - X_t[0] * standoff,
                p_cyl[1] - X_t[1] * standoff,
                z_target
            ])

        elif grasp_type == 'Front':
            # Pure reach/hover target, no physical object or grasp execution (see
            # GraspStateMachine._is_graspable): a fixed pose with FIXED approach axis.
            p_target = np.array(p_cyl, dtype=float)

            # Orientation: gripper +X rigidly locked to world +X. Remaining roll DOF
            # uses the same sticky-hysteresis mechanism as Top/Side (up-vs-down fingers).
            X_t = np.array([1.0, 0.0, 0.0])
            Z_a = np.array([0.0, 0.0, 1.0])   # candidate A: fingers "up"
            Y_a = np.cross(Z_a, X_t)
            R_front_a = np.column_stack((X_t, Y_a, Z_a))
            R_front_b = np.column_stack((X_t, -Y_a, -Z_a))   # candidate B: 180 deg roll

            R_target = self._pick_orientation(goal_key, R_front_a, R_front_b, R_anchor,
                                               update_memory=update_memory, hysteresis=hyst)

        else:
            raise ValueError(f"Unknown grasp_type '{grasp_type}' in goal_key '{goal_key}'")

        return create_transform(p_target, R_target)
