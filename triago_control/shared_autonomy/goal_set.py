#!/usr/bin/env python3
"""GoalSet: cylinder geometry definitions and dynamic SE(3) grasp-goal computation.

Extracted from the monolithic SharedControlNode per the refactor plan in
shared_autonomy_analysis.md (Section 4 - Proposed class decomposition).

This module has NO ROS or matplotlib dependencies: it is pure geometry and is
trivially unit-testable.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin


def create_transform(pos, rot_mat):
    """Constructs a 4x4 homogeneous transformation matrix from translation and rotation.

    Shared utility -- kept here (and re-exported) since GoalSet is the lowest-level
    geometry module and several other modules need it without importing the full
    ROS node.
    """
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    return T


class GoalSet:
    """Owns the physical cylinder definitions and computes dynamic grasp-goal poses.

    Replaces the `self.cylinders` dict + `get_dynamic_goal_pose` method that used
    to live directly on SharedControlNode.
    """

    # Side-grasp orientation picker: pin.log3 is undefined (NaN) when the relative
    # rotation angle is exactly pi. We treat angles within this epsilon of pi as
    # "ambiguous" and fall back to a Frobenius-norm distance, which stays well
    # defined everywhere on SO(3).
    _LOG3_SINGULARITY_EPS = 1e-3

    # Hysteresis margin for the R_candidate-vs-R_flipped (and R_top_a-vs-R_top_b)
    # orientation choice. Without this, get_dynamic_goal_pose recomputes the goal
    # orientation from scratch every tick using the EE's *current* orientation as
    # the anchor. When the EE sits near the bisector between the two 180-degree-
    # apart candidates (which happens naturally while tracking toward either one),
    # ordinary tracking noise can tip the err_candidate <= err_flipped comparison
    # back and forth tick to tick. That flips the goal orientation by 180 degrees,
    # which spikes ang_error, the QP issues a large corrective twist toward the
    # new target, and the EE's resulting motion can tip the comparison back again
    # -- a self-sustaining limit cycle that shows up as tracking oscillation even
    # outside any grasp-specific logic. The fix: only switch the "sticky" choice
    # away from whichever orientation was picked last time if the alternative is
    # ahead by more than this margin, not merely whichever is marginally smaller.
    _ORIENTATION_HYSTERESIS = 0.05  # rad, in the _rotation_distance units

    # Azimuthal (polar) singularity guard for the Side grasp. The approach
    # direction is the horizontal vector from the anchor to the cylinder axis;
    # its DIRECTION is undefined when the anchor sits on the axis (gripper hovering
    # right over the cylinder top). Within this radius of the axis we FREEZE the
    # approach azimuth to its last committed value, so passing over the top no
    # longer swings the goal around the cylinder (the oscillation-from-indecision
    # the operator saw). Must stay below the grasp standoff (r + approach_offset)
    # so a real side approach still tracks the anchor normally.
    _SIDE_AZIMUTH_DEADZONE = 0.04   # m

    # --- Platform placement goal ---------------------------------------------
    # The placement_area model in the world: a flat yellow disk the grasped
    # cylinders must be set down on (sequentially). The ONLY hard constraint is
    # that the cylinder's symmetry axis ends up perpendicular to the platform
    # face (i.e. vertical, world +Z) and its footprint lands inside the disk.
    # Everything else (where on the disk, yaw about vertical) is a free manifold
    # the user picks by hovering — exactly mirroring how the Side-grasp goal is a
    # manifold around the cylinder rather than a single pose.
    PLATFORM_KEY = 'Platform_Place'
    PLATFORM_POSE = np.array([1.000, 0.0, 0.701])   # world center of placement_area
    PLATFORM_RADIUS = 0.15                          # disk radius [m]
    PLATFORM_THICKNESS = 0.002                      # disk thickness [m]
    PLATFORM_PLACE_MARGIN = 0.03                    # keep the footprint this far inside the rim [m]

    def __init__(self, cylinders=None, target_keys=None):
        """Initializes the cylinder geometry table and the set of valid goal keys.

        Args:
            cylinders: dict of {color: {'pos', 'height', 'radius', 'cbf_name'}}.
                       Defaults to the Red/Blue table used in the original script.
            target_keys: list of valid 'Color_GraspType' goal keys. Defaults to
                       the 4 flat goals (Red_Top, Red_Side, Blue_Top, Blue_Side).
        """
        if cylinders is None:
            cylinders = {
                'Red':  {'pos': np.array([0.800, -0.20, 0.775]), 'height': 0.15,
                         'radius': 0.02, 'cbf_name': 'red_cylinder'},
                'Blue': {'pos': np.array([0.800,  0.20, 0.775]), 'height': 0.15,
                         'radius': 0.02, 'cbf_name': 'blue_cylinder'},
            }
        self.cylinders = cylinders

        if target_keys is None:
            target_keys = ['Red_Top', 'Red_Side', 'Blue_Top', 'Blue_Side',
                           self.PLATFORM_KEY]
        self.target_keys = target_keys

        # Sticky orientation memory, per goal key: None until the first time
        # that goal's pose is computed, then holds whichever of the two
        # candidates ('primary' or 'flipped') was last selected. See
        # _ORIENTATION_HYSTERESIS above for why this exists.
        self._last_orientation_choice = {k: None for k in self.target_keys}

        # Sticky approach azimuth for the Side grasp (anti-singularity). Holds the
        # last committed unit radial direction (anchor->axis, horizontal) per goal
        # key; None until first computed. See _SIDE_AZIMUTH_DEADZONE.
        self._last_side_radial = {k: None for k in self.target_keys}

        # --- Grasped-object bookkeeping (set on grasp, cleared on release) ----
        # grasped_axis_local: the cylinder's symmetry axis expressed in the
        #   gripper frame at the instant of grasp. The placement constraint is
        #   then simply "R_gripper @ grasped_axis_local must be vertical", which
        #   stays valid no matter how the user later rotates the gripper about
        #   the vertical while hovering over the platform.
        # grasped_color / grasped_z_offset: used to compute the placement height
        #   so the cylinder bottom rests on the platform face.
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

    def get_platform_goal_pose(self, T_anchor, approach_offset=0.05):
        """SE(3) placement goal on the platform disk, as a perpendicularity manifold.

        Position: the anchor's XY projected onto the disk (clamped to stay
        PLATFORM_PLACE_MARGIN inside the rim), at a Z that rests the grasped
        cylinder's bottom on the platform face (plus the standoff approach_offset
        so the goal hovers above before descending). Because the XY tracks the
        anchor, the user freely chooses WHERE on the disk to place — the two
        cylinders can be set down at different spots without colliding.

        Orientation: the minimal rotation of the anchor orientation that brings
        the grasped cylinder axis to vertical. This is exactly the
        "cylinder axis ⊥ platform" rule: it constrains 2 DOF (the tilt) and
        leaves the yaw about vertical free, anchored to the current gripper yaw
        so there is no wrist flip while hovering.
        """
        center = self.PLATFORM_POSE
        p_anchor = np.asarray(T_anchor, dtype=float)[:3, 3]
        R_anchor = np.asarray(T_anchor, dtype=float)[:3, :3]

        # --- Position: project onto the disk, clamp inside the rim ---
        dxy = p_anchor[:2] - center[:2]
        r = float(np.linalg.norm(dxy))
        max_r = max(self.PLATFORM_RADIUS - self.PLATFORM_PLACE_MARGIN, 0.0)
        if r > max_r and r > 1e-9:
            dxy = dxy / r * max_r
        p_xy = center[:2] + dxy

        # Place the cylinder bottom on the platform top face.
        if self.grasped_color is not None and self.grasped_color in self.cylinders:
            half_h = self.cylinders[self.grasped_color]['height'] / 2.0
        else:
            half_h = 0.075  # sane default (15 cm cylinder)
        platform_top = center[2] + self.PLATFORM_THICKNESS / 2.0
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

        pin.log3(R_candidate @ R_anchor.T) returns NaN when the relative rotation
        angle is exactly pi (trace == -1). We detect that condition from the trace
        directly and fall back to a Frobenius-norm distance, ||R_candidate - R_anchor||_F,
        which is a valid (if not geodesic) distance everywhere on SO(3) and agrees
        in ranking with the angle-axis distance away from the singularity.
        """
        R_rel = R_candidate @ R_anchor.T
        trace = np.trace(R_rel)
        # cos(theta) = (trace - 1) / 2  ->  trace == -1  <=>  theta == pi
        near_singularity = trace <= (-1.0 + GoalSet._LOG3_SINGULARITY_EPS)
        if near_singularity:
            return np.linalg.norm(R_candidate - R_anchor, ord='fro')
        return np.linalg.norm(pin.log3(R_rel))

    def _pick_orientation(self, goal_key, R_primary, R_flipped, R_anchor, update_memory=True):
        """Chooses between two orientation candidates with hysteresis (sticky choice).

        Without memory, this choice is recomputed from scratch every tick by
        comparing distances to the EE's *current* orientation. If the EE sits
        near the bisector between R_primary and R_flipped (180 degrees apart),
        ordinary tracking noise flips the decision tick to tick, which spikes
        ang_error and drives a tracking oscillation (see _ORIENTATION_HYSTERESIS
        docstring above). This method only switches away from the previously
        selected candidate when the alternative is ahead by more than the
        hysteresis margin; otherwise it keeps repeating last tick's choice.

        Args:
            update_memory: if False, reads the sticky memory but does not write
                to it. Used for "scratch" evaluations -- the one-step-lookahead
                visualization in timer_callback evaluates this same goal_key
                anchored on a *simulated future* EE pose, not the real one.
                Without update_memory=False that speculative evaluation would
                overwrite the real sticky choice, and the next real control
                tick would inherit a decision based on a pose that was never
                actually reached.
        """
        dist_primary = self._rotation_distance(R_primary, R_anchor)
        dist_flipped = self._rotation_distance(R_flipped, R_anchor)

        last_choice = self._last_orientation_choice.get(goal_key)

        if last_choice == 'flipped':
            # Currently committed to R_flipped: only switch back to R_primary
            # if it is clearly better, not just marginally.
            switch = dist_primary < (dist_flipped - self._ORIENTATION_HYSTERESIS)
            choice = 'primary' if switch else 'flipped'
        elif last_choice == 'primary':
            switch = dist_flipped < (dist_primary - self._ORIENTATION_HYSTERESIS)
            choice = 'flipped' if switch else 'primary'
        else:
            # First time this goal key is evaluated: no prior choice to be
            # sticky about, so just take the closer one.
            choice = 'primary' if dist_primary <= dist_flipped else 'flipped'

        if update_memory:
            self._last_orientation_choice[goal_key] = choice
        return R_primary if choice == 'primary' else R_flipped

    def get_dynamic_goal_pose(self, T_anchor, goal_key, approach_offset=0.05, update_memory=True):
        """Dynamically computes the target SE(3) pose for a given goal key.

        approach_offset: Distance from cylinder surface (Positive = standoff,
                          Negative = envelop / penetration depth during grasp).

        update_memory: pass False when T_anchor is a speculative/simulated pose
            (e.g. the one-step-lookahead used for visualization) rather than the
            real current EE/user pose, so that speculative evaluation does not
            perturb the sticky orientation choice used by the real control loop.
            See _pick_orientation for why this matters.

        Side grasp: orientation is adaptively chosen to minimize wrist flips
        relative to T_anchor (current EE or user pose), using a singularity-safe
        rotation distance (see _rotation_distance) with hysteresis against chatter.

        Top grasp: orientation is also anchored, matching the Side grasp's
        adaptive behavior so switching goals does not cause unnecessary wrist
        flips (this fixes the inconsistency noted in the analysis: the original
        Top grasp ignored R_anchor entirely and always returned a fixed
        R.from_euler('y', 90 deg)).
        """
        if goal_key == self.PLATFORM_KEY or goal_key.startswith('Platform'):
            # Placement goal is a perpendicularity manifold over the disk, not a
            # cylinder grasp — delegate. (update_memory is irrelevant here since
            # the orientation is derived directly from the anchor each tick.)
            return self.get_platform_goal_pose(T_anchor, approach_offset=approach_offset)

        color, grasp_type = goal_key.split('_')
        cyl = self.cylinders[color]
        p_cyl = cyl['pos']
        h, r = cyl['height'], cyl['radius']

        p_anchor = T_anchor[:3, 3]
        R_anchor = T_anchor[:3, :3]  # Extract current user/EE orientation

        if grasp_type == 'Top':
            p_target = p_cyl + np.array([0, 0, h / 2 + approach_offset])

            # Top grasp: the gripper approach axis (gripper +X) must point DOWN
            # (world -Z) to descend onto the cylinder top. Both candidates keep
            # X pointing down; they differ only by a 180-deg roll about that
            # approach axis (mirrors the Side-grasp's two finger orientations).
            # Previously the second candidate used Ry(-90), which flips X to
            # point UP — physically wrong (gripper would approach from below).
            R_top_down = R.from_euler('y', 90, degrees=True).as_matrix()
            R_top_a = R_top_down
            R_top_b = (R.from_euler('y', 90, degrees=True) *
                       R.from_euler('x', 180, degrees=True)).as_matrix()

            R_target = self._pick_orientation(goal_key, R_top_a, R_top_b, R_anchor,
                                               update_memory=update_memory)

        elif grasp_type == 'Side':
            # 1. Z-Height Tracking (user controls grasp height)
            z_target = np.clip(p_anchor[2], p_cyl[2] - h / 2 + 0.02, p_cyl[2] + h / 2 - 0.02)

            # 2. Radial Projection (horizontal approach vector, anchor -> cylinder axis)
            #    Anti-singularity: when the anchor is within _SIDE_AZIMUTH_DEADZONE
            #    of the axis (hovering over the top), the radial DIRECTION is
            #    ill-defined and would swing wildly tick-to-tick. We commit to the
            #    last good azimuth instead, so the goal stays put until the anchor
            #    is unambiguously on one side again. This is what makes the
            #    behaviour safe to blend with user guidance — no indecision swing.
            v_rad_raw = p_cyl[:2] - p_anchor[:2]
            rad_norm = float(np.linalg.norm(v_rad_raw))
            last_radial = self._last_side_radial.get(goal_key)

            if rad_norm >= self._SIDE_AZIMUTH_DEADZONE:
                # Anchor clearly on a side: use (and commit) the true radial.
                v_rad = v_rad_raw / rad_norm
                if update_memory:
                    self._last_side_radial[goal_key] = v_rad
            elif last_radial is not None:
                # Inside the singular zone: FREEZE to the last committed azimuth.
                v_rad = last_radial
            else:
                # No prior commit and the anchor is right over the axis: fall back
                # to the gripper's own horizontal heading (continuous, reflects the
                # user's blended intent) rather than an undefined radial.
                approach_xy = R_anchor[:2, 0]
                a_norm = float(np.linalg.norm(approach_xy))
                v_rad = (approach_xy / a_norm) if a_norm > 1e-6 else np.array([1.0, 0.0])
                if update_memory:
                    self._last_side_radial[goal_key] = v_rad

            X_t = np.array([v_rad[0], v_rad[1], 0.0])

            # 3. FIXED GRIPPER PLANE: XY perpendicular to cylinder axis (world Z)
            Z_t = np.array([0., 0., 1.])
            Y_t = np.cross(Z_t, X_t)  # always well-conditioned (Z vertical, X horizontal)

            R_candidate = np.column_stack((X_t, Y_t, Z_t))
            R_flipped = np.column_stack((X_t, -Y_t, -Z_t))  # 180 deg around X_t (fingers down)

            # Pick the orientation closest to the anchor to avoid unnecessary wrist
            # flips, with hysteresis so the choice doesn't chatter near the
            # bisector between the two candidates (see _pick_orientation).
            R_target = self._pick_orientation(goal_key, R_candidate, R_flipped, R_anchor,
                                               update_memory=update_memory)

            # 4. Target Position
            standoff = r + approach_offset
            p_target = np.array([
                p_cyl[0] - X_t[0] * standoff,
                p_cyl[1] - X_t[1] * standoff,
                z_target
            ])

        else:
            raise ValueError(f"Unknown grasp_type '{grasp_type}' in goal_key '{goal_key}'")

        return create_transform(p_target, R_target)
