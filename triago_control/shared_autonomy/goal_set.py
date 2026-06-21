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
                         'radius': 0.035, 'cbf_name': 'red_cylinder'},
                'Blue': {'pos': np.array([0.800,  0.20, 0.775]), 'height': 0.15,
                         'radius': 0.035, 'cbf_name': 'blue_cylinder'},
            }
        self.cylinders = cylinders

        if target_keys is None:
            target_keys = ['Red_Top', 'Red_Side', 'Blue_Top', 'Blue_Side']
        self.target_keys = target_keys

        # Sticky orientation memory, per goal key: None until the first time
        # that goal's pose is computed, then holds whichever of the two
        # candidates ('primary' or 'flipped') was last selected. See
        # _ORIENTATION_HYSTERESIS above for why this exists.
        self._last_orientation_choice = {k: None for k in self.target_keys}

    def cbf_name(self, color):
        """Returns the CBF pair name registered for this cylinder color."""
        return self.cylinders[color]['cbf_name']

    def radius(self, color):
        """Returns the cylinder radius (m) for this color."""
        return self.cylinders[color]['radius']

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
        color, grasp_type = goal_key.split('_')
        cyl = self.cylinders[color]
        p_cyl = cyl['pos']
        h, r = cyl['height'], cyl['radius']

        p_anchor = T_anchor[:3, 3]
        R_anchor = T_anchor[:3, :3]  # Extract current user/EE orientation

        if grasp_type == 'Top':
            p_target = p_cyl + np.array([0, 0, h / 2 + approach_offset])

            # Top grasp candidates: gripper approach axis points down (-Z world),
            # with two opposite roll choices around the approach axis so that we
            # can pick whichever is closest to the anchor, exactly mirroring the
            # Side-grasp policy below instead of returning a hard fixed pose.
            R_top_a = R.from_euler('y', 90, degrees=True).as_matrix()
            R_top_b = R.from_euler('y', -90, degrees=True).as_matrix()

            R_target = self._pick_orientation(goal_key, R_top_a, R_top_b, R_anchor,
                                               update_memory=update_memory)

        elif grasp_type == 'Side':
            # 1. Z-Height Tracking (user controls grasp height)
            z_target = np.clip(p_anchor[2], p_cyl[2] - h / 2 + 0.02, p_cyl[2] + h / 2 - 0.02)

            # 2. Radial Projection (horizontal approach vector, pointing cylinder -> anchor)
            v_rad = p_cyl[:2] - p_anchor[:2]
            if np.linalg.norm(v_rad) < 1e-3:
                v_rad = np.array([1.0, 0.0])
            v_rad = v_rad / np.linalg.norm(v_rad)
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
