# collision_manager.py
"""
The Environmental Awareness module.

Owns the `hppfcl` geometry model and every proximity query the controller needs:
    * builds arm capsules from `calculate_offsets` (dominant-axis snapping),
    * adds simplified gripper bounding boxes, body colliders, ground, wall and
      the bimanual workspace obstacles (table + red/blue cylinders),
    * declares the collision-pair graph (with all the original exclusions),
    * runs the per-tick distance queries and aggregates them into TWO
      INDEPENDENT per-arm SoftMin Control Barrier Function gradients.

----------------------------------------------------------------------------
SoftMin CBF math (per arm X in {R, L}; PRESERVED per-pair math, now evaluated
over a PER-ARM subset of pairs -- see compute_softmin_jacobian / _arm_membership
for the coupling-fix rationale):

    h_soft_X(q) = -(1/alpha) * log( sum_{k in Pairs(X)} exp(-alpha * d_k(q)) )

    J_soft_X(q) = sum_{k in Pairs(X)} ( exp(-alpha*d_k(q)) / sum_{j in Pairs(X)} exp(-alpha*d_j(q)) ) * J_k(q)

  Pairs(X) = every active pair touching at least one geometry belonging to arm
  X (its own links/gripper, or a cylinder it currently holds). A genuine
  inter-arm pair belongs to BOTH Pairs(R) and Pairs(L). Each h_soft_X/J_soft_X
  is a differentiable approximation of the closest distance RESTRICTED to arm
  X's own pairs, whose gradient is the convex (softmax) blend of just those
  pairs' Jacobians -- so it has zero columns in the OTHER arm's joints unless a
  pair genuinely touches both.

Dynamic safety margin (PER-ARM, 2026-07-01 -- see the coupling-audit note in
compute_softmin_jacobian):

    d_safe_dynamic_X = d_safe_base + k_v_safe * ||v_X||      for X in {R, L}

  The barrier thickens with THAT ARM's OWN speed, so it brakes earlier when
  fast. Previously this was a SINGLE scalar computed from the COMBINED
  (both-arm) velocity norm and shared by both CBF rows -- a residual coupling
  channel that survived the per-arm Jacobian split: a fast active arm inflated
  the idle arm's margin too, tightening its (otherwise legitimately slack)
  barrier and forcing visible idle-arm motion even when nothing new approached
  it. Splitting this term removes that channel; see module-level notes below
  compute_softmin_jacobian for the full analysis.
----------------------------------------------------------------------------
"""

import threading

import pinocchio as pin
try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl
import numpy as np
import triago_control.qp_controller.config as cfg


def get_skew(v):
    """Skew-symmetric matrix of a 3-vector (used to shift Jacobians to a point)."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class CollisionManager:
    """Builds the collision world and computes the SoftMin CBF each control tick."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.cmodel = pin.GeometryModel()
        self.cdata = None

        # Serializes STRUCTURAL cmodel mutation (add/remove collision pairs on
        # grasp attach/detach) against a concurrent reader. Only used when the
        # real-hardware CBF runs on a worker thread (main_qp_controller_real.py):
        # the worker holds it around its compute, and the subclass's topology
        # seam holds it around attach/detach, so the two never touch cmodel at
        # once. Uncontended (and effectively free) in the synchronous/sim path,
        # where attach/detach is the only accessor and no worker exists.
        self.geom_lock = threading.Lock()

        # Geometry id bookkeeping
        self.right_geom_ids = []
        self.left_geom_ids = []
        self.head_geom_ids = []            # quasi-static obstacle capsules (see build_collision_model)
        self.body_geom_ids = []
        self.gripper_box_ids = {}          # {'right': id, 'left': id}
        self.workspace_obstacle_ids = []
        self.table_id = None               # set in build_collision_model (role=="table")
        self.top_active_pairs = []         # [(name1, name2, dist)] 3 closest enabled pairs (debug)
        # Cached once (right/left_geom_ids never change after build_collision_model)
        # by compute_softmin_jacobian -- see its "allowed_grasp_ids" comment.
        self._allowed_grasp_ids_cache = None
        # Global (unfiltered) closest-pair witness, refreshed every tick by
        # compute_softmin_jacobian -- consumed by qp_visualizer_tutorial.py's
        # publish_debug instead of it re-scanning cdata.distanceResults itself.
        self.witness_min_distance = float('nan')
        self.witness_min_points = None

        # Lazy sets (built on first use in _arm_membership) for O(1) "does this
        # geom_id belong to the right/left arm's own geometry" lookups.
        self._right_geom_id_set = set()
        self._left_geom_id_set = set()

    def calculate_offsets(self, chain, tool_link_name):
        # Build per-link capsule placements by snapping each link to its dominant axis.
        print(f"[Init] Calibrating offsets for {chain[0]}... (Dominant Axis Mode)")
        offsets = {}
        full_chain = chain + [tool_link_name]

        # Use the neutral pose to get clean, static joint locations
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        pin.updateFramePlacements(self.model, self.data)

        for i in range(len(chain)):
            link_name = chain[i]
            next_link = full_chain[i + 1]
            try:
                frame_i = self.model.frames[self.model.getFrameId(link_name)]
                frame_next = self.model.frames[self.model.getFrameId(next_link)]
                joint_i_id = frame_i.parentJoint
                joint_next_id = frame_next.parentJoint

                pose_i = self.data.oMi[joint_i_id]
                # For the final tool link use the frame placement (it may lack its own joint)
                if i + 1 == len(chain):
                    pose_next = self.data.oMf[self.model.getFrameId(next_link)]
                else:
                    pose_next = self.data.oMi[joint_next_id]

                # Vector from joint A to joint B, expressed in joint A's local frame
                vec_global = pose_next.translation - pose_i.translation
                vec_local = pose_i.rotation.T @ vec_global

                # --- DOMINANT AXIS SNAPPING ---
                # 1. Find which axis the CAD "tube" is actually built along
                dominant_idx = np.argmax(np.abs(vec_local))
                dominant_dir = np.zeros(3)
                dominant_dir[dominant_idx] = np.sign(vec_local[dominant_idx])
                # 2. Length strictly along that main tube, ignoring side-offsets
                length = abs(vec_local[dominant_idx])
                if length < 0.001:
                    length = 0.01
                # 3. Midpoint placed straight down the dominant axis
                midpoint = (length / 2.0) * dominant_dir
                # 4. Align the capsule's Z-axis with the dominant direction
                R_cyl = pin.Quaternion.FromTwoVectors(np.array([0., 0., 1.]), dominant_dir).matrix()

                # Store placement exactly relative to the JOINT
                offsets[link_name] = (pin.SE3(R_cyl, midpoint), length)
            except Exception as e:
                print(f"  Failed {link_name}: {e}")
        return offsets

    def _apply_capsule_override(self, link_name, placement_wrt_joint, length):
        """Applies this link's cfg.CAPSULE_OFFSET_OVERRIDES entry (if any) to
        its raw calculate_offsets() placement/length, and returns the
        possibly-corrected (placement, length, radius). A link with NO entry
        returns (placement_wrt_joint, length, cfg.CAPSULE_RADIUS) UNCHANGED --
        this function is a pure additive correction, never a rewrite of
        calculate_offsets' own straight-line-segment math.

        See config.py section 6b for the full field-by-field derivation.
        `lateral_offset` is applied in the capsule's OWN local frame (i.e.
        rotated by placement_wrt_joint.rotation), since it was computed
        relative to that same joint-relative axis by capsule_alignment_
        audit.py's `_compute_capsule_fix` -- NOT in the parent joint's raw
        x/y/z, which would apply the correction in the wrong direction
        whenever the dominant-axis snap picked something other than a pure
        Z-axis alignment relative to the joint frame.
        """
        override = cfg.CAPSULE_OFFSET_OVERRIDES.get(link_name)
        if override is None:
            return placement_wrt_joint, length, cfg.CAPSULE_RADIUS

        # lateral_offset/extensions are stored in mm (see config.py) -- convert once here.
        lateral_mm = np.array(override['lateral_offset'], dtype=float)
        lateral_local = lateral_mm / 1000.0
        prox_ext = override['proximal_extension'] / 1000.0
        dist_ext = override['distal_extension'] / 1000.0
        radius = override['radius'] / 1000.0

        # The capsule's own Z axis, expressed in the joint-local frame (same
        # convention calculate_offsets itself uses: "Align the capsule's
        # Z-axis with the dominant direction").
        z_axis_local = np.array([0., 0., 1.])

        # Shift the midpoint sideways (already _|_ to z_axis_local by
        # construction from _compute_capsule_fix -- see config.py), THEN
        # re-center to accommodate the axial extension: extending the
        # proximal end by prox_ext and the distal end by dist_ext grows the
        # segment by (prox_ext + dist_ext) and shifts its midpoint by
        # (dist_ext - prox_ext) / 2 along z_axis_local.
        new_length = length + prox_ext + dist_ext
        axial_recenter = (dist_ext - prox_ext) / 2.0

        # BUGFIX (2026-07-04): `lateral_offset` is NOT in the capsule's own
        # pre-rotation frame (where the axis is the literal [0,0,1]) -- it
        # was computed by capsule_alignment_audit.py's _compute_capsule_fix
        # from vertices/axis ALREADY expressed in the PARENT-JOINT-LOCAL
        # frame (z_axis there = placement.rotation @ [0,0,1], and
        # rel = verts - a used that same joint-local z_axis to split into
        # axial/perpendicular parts) -- i.e. lateral_offset is ALREADY in
        # this function's `placement_wrt_joint`-relative frame. Rotating it
        # AGAIN by placement_wrt_joint.rotation double-rotated it, which is a
        # no-op only when the offset happens to lie along a rotation-
        # invariant axis of that particular link's dominant-axis snap
        # (explaining why links 1/3/5 came out correct by coincidence while
        # 2/4/6 did not -- confirmed by re-running the audit after the
        # original, buggy version of this fix). Added DIRECTLY, no rotation.
        # The axial term is different: `z_axis_local = [0,0,1]` IS in the
        # capsule's own pre-rotation frame (hppfcl.Capsule's axis convention),
        # so it correctly still needs rotating into the joint frame.
        new_translation = (placement_wrt_joint.translation
                          + lateral_local
                          + placement_wrt_joint.rotation @ (axial_recenter * z_axis_local))
        new_placement = pin.SE3(placement_wrt_joint.rotation, new_translation)

        return new_placement, new_length, radius

    def build_collision_model(self, right_offsets, left_offsets, head_offsets=None,
                              world_scene=None):
        # Assemble the full collision geometry model: arms, grippers, body, ground, obstacles.
        #
        # `world_scene` (triago_control.qp_controller.world_loader.WorldScene, optional):
        # the loaded world's static obstacle layout (table + red/blue cylinders +
        # optional wall/extra obstacles -- see world_loader.py). When omitted, falls
        # back to the legacy cfg.TABLE_POS/RED_CYLINDER_POS/... constants (section 6
        # below), so any external caller that never passes a world_scene keeps the
        # exact old, unchanged behavior.

        # 1. ARM CAPSULES (from the calculated, joint-relative offsets)
        def add_arm_geoms(offsets_data, prefix, id_list):
            for link_name, (placement_wrt_joint, length) in offsets_data.items():
                if not self.model.existFrame(link_name) and not self.model.existBodyName(link_name):
                    continue
                try:
                    frame_id = self.model.getBodyId(link_name)
                except Exception:
                    frame_id = self.model.getFrameId(link_name)
                parent_joint_id = self.model.frames[frame_id].parentJoint
                # placement_wrt_joint is already relative to the joint origin (no extra multiply)
                #
                # PER-LINK ALIGNMENT OVERRIDE (2026-07-04, see config.py section
                # 6b for the full derivation/rationale): calculate_offsets' pure
                # joint-to-joint straight segment can leave the real visual mesh
                # poking outside the capsule (measured via
                # capsule_alignment_audit.py). A link WITHOUT an override entry
                # is completely unaffected -- byte-identical placement/length/
                # radius to before this feature existed.
                placement_fixed, length_fixed, radius = self._apply_capsule_override(
                    link_name, placement_wrt_joint, length)
                shape = hppfcl.Capsule(radius, length_fixed)
                obj = pin.GeometryObject(f"{prefix}_{link_name}", parent_joint_id, placement_fixed, shape)
                id_list.append(self.cmodel.addGeometryObject(obj))

        add_arm_geoms(right_offsets, "shadow_right", self.right_geom_ids)
        add_arm_geoms(left_offsets, "shadow_left", self.left_geom_ids)

        # 1b. HEAD CAPSULES (2026-07-01) -- same dominant-axis capsule geometry
        # as the arms (the head is the SAME hardware, hence calculate_offsets
        # is reused verbatim), but NEVER added to right_geom_ids/left_geom_ids:
        # this keeps the head OUT of _arm_membership (it belongs to neither
        # arm) and therefore OUT of idx_right/idx_left -- the arms' CBFs will
        # see it purely as geometry, with NO head joint columns ever appearing
        # in J_soft_r/J_soft_l (see compute_softmin_jacobian: a pair's
        # Jacobian is built from get_point_jacobian on the PARENT JOINT of
        # each geometry -- for a head capsule that is a head joint, but the
        # pair only "counts" toward an arm's SoftMin aggregate/gradient
        # column-wise through the OTHER geometry's (the arm's) parent joint
        # Jacobian branch of that same distance-rate row; the head-side
        # Jacobian columns are computed but never touch idx_right/idx_left,
        # so build_and_solve's dq_max_safe/dq_min_safe and the QP's actual
        # decision vector (n_joints = model.nv over idx_right+idx_left only,
        # sliced out downstream) never move the head).
        if head_offsets:
            add_arm_geoms(head_offsets, "shadow_head", self.head_geom_ids)

        # 2. SIMPLIFIED END-EFFECTOR BOXES (PAL PRO grippers)
        for side, id_list in (('right', self.right_geom_ids), ('left', self.left_geom_ids)):
            base_link = f'gripper_{side}_base_link'
            if self.model.existFrame(base_link):
                frame_id = self.model.getFrameId(base_link)
                parent_joint = self.model.frames[frame_id].parentJoint
                # Shift the box 5cm forward so it covers the fingers
                placement = self.model.frames[frame_id].placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.05]))
                geom = hppfcl.Box(0.05, 0.08, 0.25)
                obj = pin.GeometryObject(f"gripper_{side}_collision_box", parent_joint, placement, geom)
                geom_id = self.cmodel.addGeometryObject(obj)
                id_list.append(geom_id)
                self.gripper_box_ids[side] = geom_id

        # 3. BODY COLLIDERS (mobile base + torso pillar). Format: (frame, [x,y,z] size, [x,y,z] offset)
        body_parts = [
            ("base_link", [0.6, 0.5, 0.27], [0.0, 0.0, 0.09]),      # Mobile base box
            ("torso_lift_link", [0.2, 0.2, 0.6], [0.0, 0.0, 0.25]),  # Torso pillar (moves with lift)
        ]
        for parent_name, dims, offset in body_parts:
            if not self.model.existBodyName(parent_name) and not self.model.existFrame(parent_name):
                print(f"[Warning] Could not find frame {parent_name} for body collider.")
                continue
            if self.model.existBodyName(parent_name):
                frame_id = self.model.getBodyId(parent_name)
            else:
                frame_id = self.model.getFrameId(parent_name)
            parent_joint_id = self.model.frames[frame_id].parentJoint
            placement = self.model.frames[frame_id].placement * pin.SE3(np.eye(3), np.array(offset))
            obj = pin.GeometryObject(f"shadow_{parent_name}_box", parent_joint_id, placement, hppfcl.Box(*dims))
            self.body_geom_ids.append(self.cmodel.addGeometryObject(obj))

        # 4. GROUND PLANE
        ground_pose = pin.SE3.Identity()
        ground_pose.translation = np.array([0.0, 0.0, -0.5])
        self.ground_id = self.cmodel.addGeometryObject(
            pin.GeometryObject("ground_plane", 0, ground_pose, hppfcl.Box(20.0, 20.0, 1.0)))

        # 5+6. WORLD SCENE (virtual wall + bimanual workspace: table + graspable
        # cylinders + any future extra obstacles) -- built GENERICALLY from a
        # `world_scene.WorldScene` (see world_loader.py). Every obstacle in
        # `world_scene.static_obstacles` becomes exactly one hppfcl geometry, at
        # the SAME shape/pose/size the old hard-coded constants used to encode --
        # only the SOURCE of those numbers changed (YAML instead of config.py).
        #
        # Legacy fallback: if no world_scene is passed (e.g. an older caller),
        # rebuild the exact same objects from the deprecated cfg constants so
        # behavior is byte-for-byte unchanged for anyone not yet passing a scene.
        self._geom_id_by_obstacle_name = {}   # {obstacle name -> hppfcl geometry id}
        if world_scene is not None:
            for obs in world_scene.static_obstacles:
                # BUGFIX (2026-07-04): `collision: false` must mean "this
                # geometry does not exist in cmodel at all" -- exactly mirroring
                # the OLD `if cfg.WALL_COLLIDER:` gate, which only ever created
                # the wall's hppfcl geometry when enabled. The first version of
                # this loop created the geometry UNCONDITIONALLY and only used
                # `collision` to skip the pair/id bookkeeping below -- which left
                # a real (but un-colored, since color_collision_model only paints
                # workspace_obstacle_ids/wall_id) geometry sitting in cmodel, and
                # since cmodel IS what Meshcat renders (displayCollisions(True)),
                # a disabled wall still showed up there -- as a solid BLACK slab
                # (an hppfcl GeometryObject's default meshColor when nothing ever
                # sets one). Skipping geometry creation entirely for a disabled
                # obstacle restores the original "doesn't exist" semantics.
                if not obs.collision:
                    continue

                pose = pin.SE3(np.eye(3), np.array(obs.position, dtype=float))
                if obs.shape == 'box':
                    shape = hppfcl.Box(*obs.size)
                elif obs.shape == 'cylinder':
                    shape = hppfcl.Cylinder(*obs.size)
                else:
                    print(f"[Collision] WARNING: unknown obstacle shape "
                          f"'{obs.shape}' for '{obs.name}' -- skipped.")
                    continue

                geom_id = self.cmodel.addGeometryObject(
                    pin.GeometryObject(obs.name, 0, pose, shape))
                self._geom_id_by_obstacle_name[obs.name] = geom_id

                if obs.role == 'wall':
                    self.wall_id = geom_id
                else:
                    self.workspace_obstacle_ids.append(geom_id)
                    if obs.role == 'table':
                        # Explicit, name-based reference (NOT positional --
                        # see detach_object, which used to assume the table
                        # was always workspace_obstacle_ids[0]; a future world
                        # with obstacles listed in a different order would have
                        # silently broken that assumption).
                        self.table_id = geom_id

            # Resolve the grasp-role indirection (today's grasp state machine,
            # shared_autonomy_handler, and visualization_engine all key on the
            # literal attribute names `red_cyl_id`/`blue_cyl_id` -- this keeps
            # every one of those call sites working unchanged, just pointed at
            # whichever named obstacle plays that role in THIS world).
            red_name = world_scene.grasp_roles.get('red')
            blue_name = world_scene.grasp_roles.get('blue')
            if red_name and red_name in self._geom_id_by_obstacle_name:
                self.red_cyl_id = self._geom_id_by_obstacle_name[red_name]
            if blue_name and blue_name in self._geom_id_by_obstacle_name:
                self.blue_cyl_id = self._geom_id_by_obstacle_name[blue_name]
        else:
            # --- Legacy path (no world_scene given): unchanged from before. ---
            if cfg.WALL_COLLIDER:
                wall_pose = pin.SE3.Identity()
                wall_pose.translation = np.array(cfg.WALL_POS)
                self.wall_id = self.cmodel.addGeometryObject(
                    pin.GeometryObject("virtual_wall", 0, wall_pose, hppfcl.Box(*cfg.WALL_SIZE)))

            table_pose = pin.SE3(np.eye(3), np.array(cfg.TABLE_POS))
            self.table_id = self.cmodel.addGeometryObject(
                pin.GeometryObject("work_table", 0, table_pose, hppfcl.Box(*cfg.TABLE_SIZE)))
            self.workspace_obstacle_ids.append(self.table_id)

            red_pose = pin.SE3(np.eye(3), np.array(cfg.RED_CYLINDER_POS))
            self.red_cyl_id = self.cmodel.addGeometryObject(
                pin.GeometryObject("red_cylinder", 0, red_pose, hppfcl.Cylinder(*cfg.CYLINDER_SIZE)))
            self.workspace_obstacle_ids.append(self.red_cyl_id)

            blue_pose = pin.SE3(np.eye(3), np.array(cfg.BLUE_CYLINDER_POS))
            self.blue_cyl_id = self.cmodel.addGeometryObject(
                pin.GeometryObject("blue_cylinder", 0, blue_pose, hppfcl.Cylinder(*cfg.CYLINDER_SIZE)))
            self.workspace_obstacle_ids.append(self.blue_cyl_id)

        # Finalize the per-arm membership sets used by _arm_membership to split
        # the SoftMin CBF into independent per-arm barriers (right_geom_ids /
        # left_geom_ids already include each arm's gripper collision box).
        self._right_geom_id_set = set(self.right_geom_ids)
        self._left_geom_id_set = set(self.left_geom_ids)

    def define_collision_pairs(self):
        # Declare every checked collision pair, preserving the original exclusion rules.
        all_arm_ids = self.right_geom_ids + self.left_geom_ids
        base_joints_exclusions = ["arm_right_1", "arm_right_2", "arm_left_1", "arm_left_2"]

        # 1. Arm vs Arm (shoulders / link 1 excluded)
        for r_id in self.right_geom_ids:
            if "arm_right_1" in self.cmodel.geometryObjects[r_id].name:
                continue
            for l_id in self.left_geom_ids:
                if "arm_left_1" in self.cmodel.geometryObjects[l_id].name:
                    continue
                self.cmodel.addCollisionPair(pin.CollisionPair(r_id, l_id))

        # 2. Arms vs Body / Wall / Workspace (links 1 & 2 excluded to avoid base lockups)
        for arm_id in all_arm_ids:
            arm_name = self.cmodel.geometryObjects[arm_id].name
            if any(ex in arm_name for ex in base_joints_exclusions):
                continue
            for body_id in self.body_geom_ids:
                self.cmodel.addCollisionPair(pin.CollisionPair(arm_id, body_id))
            if hasattr(self, 'wall_id'):
                self.cmodel.addCollisionPair(pin.CollisionPair(self.wall_id, arm_id))
            for obs_id in self.workspace_obstacle_ids:
                self.cmodel.addCollisionPair(pin.CollisionPair(obs_id, arm_id))

        # 2d. Arms vs Head (2026-07-01): the head is a quasi-static CBF
        # obstacle for the arms only -- see config.py's HEAD_CHAIN docstring
        # and compute_softmin_jacobian's _arm_membership routing (a head
        # capsule belongs to NEITHER arm's own geometry set, so an
        # arm-vs-head pair contributes to exactly ONE arm's SoftMin
        # aggregate -- whichever arm's geometry is the OTHER side of the
        # pair -- never both, and never leaks a head joint-velocity "credit"
        # into the QP since the head's columns of the decision vector are
        # separately hard-locked to zero by the joint-limit rows in
        # qp_formulator, exactly like every other non-arm joint (torso,
        # base, gripper fingers) -- see the coupling note in
        # compute_softmin_jacobian's module docstring for the full math).
        # Per instruction: SKIP any pair touching arm_right_1 or arm_left_1
        # (that link cannot collide with the head chain).
        for arm_id in all_arm_ids:
            arm_name = self.cmodel.geometryObjects[arm_id].name
            if "arm_right_1" in arm_name or "arm_left_1" in arm_name:
                continue
            for head_id in self.head_geom_ids:
                self.cmodel.addCollisionPair(pin.CollisionPair(arm_id, head_id))

        # 2b. Cylinder vs Cylinder: so that when BOTH grippers hold a cylinder
        # (bimanual), the two held cylinders cannot inter-penetrate. Harmless while
        # both rest on the table (static, far apart); becomes a real CBF pair once
        # both are re-parented to the arm chains on grasp.
        if hasattr(self, 'red_cyl_id') and hasattr(self, 'blue_cyl_id'):
            self.cmodel.addCollisionPair(pin.CollisionPair(self.red_cyl_id, self.blue_cyl_id))

        # 3. Ground collision (HANDS ONLY: custom box + wrist), avoiding CAD finger noise
        hand_keywords = ["collision_box", "7_link"]
        for geom_id in all_arm_ids:
            name = self.cmodel.geometryObjects[geom_id].name.lower()
            if any(key in name for key in hand_keywords):
                self.cmodel.addCollisionPair(pin.CollisionPair(self.ground_id, geom_id))

        # 4. Intra-arm self-collision (distal hand vs the first 3 kinematic links)
        hand_kw_intra = ["palm", "knuck", "tip", "7_link", "gripper", "finger", "tool"]
        upper_arm_keywords = ["arm_right_1", "arm_right_2", "arm_right_3",
                              "arm_left_1", "arm_left_2", "arm_left_3"]

        def apply_intra_arm_collision(geom_ids):
            for i in range(len(geom_ids)):
                name_a = self.cmodel.geometryObjects[geom_ids[i]].name.lower()
                is_hand_a = any(k in name_a for k in hand_kw_intra)
                is_upper_a = any(k in name_a for k in upper_arm_keywords)
                for j in range(i + 1, len(geom_ids)):
                    name_b = self.cmodel.geometryObjects[geom_ids[j]].name.lower()
                    is_hand_b = any(k in name_b for k in hand_kw_intra)
                    is_upper_b = any(k in name_b for k in upper_arm_keywords)
                    # Pair only a clear hand against a strictly-base arm link
                    if (is_hand_a and is_upper_b) or (is_upper_a and is_hand_b):
                        self.cmodel.addCollisionPair(pin.CollisionPair(geom_ids[i], geom_ids[j]))

        apply_intra_arm_collision(self.right_geom_ids)
        apply_intra_arm_collision(self.left_geom_ids)

        # Finalize: create cdata and request nearest points on every distance query
        self.cdata = self.cmodel.createData()
        for req in self.cdata.distanceRequests:
            req.enable_nearest_points = True

        print("--------------------------------------------------")
        print("[Collision] OPTIMIZED MODEL BUILT.")
        print(f"            - Distance Pairs: {len(self.cmodel.collisionPairs)}")
        if hasattr(self, 'wall_id'):
            print("            - Virtual Wall Protection: ACTIVE (Excl. J1/J2)")
        print("--------------------------------------------------")

    def update_geometry(self, current_q, data=None, cdata=None):
        # Refresh geometry placements and run all pairwise distance queries.
        # data/cdata default to self.* (the shared main-thread pair). The real-
        # hardware CBF worker (main_qp_controller_real.py) passes its OWN private
        # pin.Data / GeometryData so it never races the main thread's kinematics
        # refresh -- see that file's isolation invariant. cmodel/model are shared
        # read-only structure and stay self.*.
        data = self.data if data is None else data
        cdata = self.cdata if cdata is None else cdata
        pin.updateGeometryPlacements(self.model, data, self.cmodel, cdata, current_q)
        pin.computeDistances(self.cmodel, cdata)

    def add_attached_object_pairs(self, cyl_id, arm_side, current_q):
        """Treat a grasped cylinder as a moving link of `arm_side`.

        Builds the adjacency-exclusion set (own-arm links 6, 7, gripper, fingers
        — too close to the grasp to check) and dynamically CREATES the collision
        pairs that were missing while the cylinder was a static obstacle:
        cylinder vs base, torso, ground, wall, table and the other cylinder.
        Pairs vs both arms already exist from startup; after re-parenting they
        become self-collision (own arm links 3/4/5 stay active) and other-arm
        avoidance.

        Returns: (added_names, skipped_names, adjacency_set)
        Raises on any failure so the caller can report a red error.
        """
        own_ids = self.right_geom_ids if arm_side == 'right' else self.left_geom_ids

        # 1. Adjacency exclusion: own-arm geometry fused to / overlapping the grasp.
        adjacency = set()
        for gid in own_ids:
            nm = self.cmodel.geometryObjects[gid].name.lower()
            if "6_link" in nm or "7_link" in nm or "gripper" in nm or "finger" in nm:
                adjacency.add(gid)

        # 2. Snapshot the pairs that already exist (unordered) to avoid duplicates.
        existing = set()
        for k in range(len(self.cmodel.collisionPairs)):
            cp = self.cmodel.collisionPairs[k]
            existing.add((cp.first, cp.second))
            existing.add((cp.second, cp.first))

        # 3. Targets the cylinder must now be checked against (it's a moving link).
        targets = list(self.body_geom_ids)          # base box + torso pillar
        targets.append(self.ground_id)              # ground plane
        if hasattr(self, 'wall_id'):
            targets.append(self.wall_id)            # virtual wall
        for obs in self.workspace_obstacle_ids:     # table + the OTHER cylinder
            if obs != cyl_id:
                targets.append(obs)

        added_names, skipped_names = [], []
        for t in targets:
            if (cyl_id, t) in existing:
                skipped_names.append(self.cmodel.geometryObjects[t].name)
                continue
            self.cmodel.addCollisionPair(pin.CollisionPair(cyl_id, t))
            added_names.append(self.cmodel.geometryObjects[t].name)

        # 4. Rebuild cdata for the enlarged pair set and revalidate immediately so
        #    the same control tick can query distances without a stale/empty array.
        self.cdata = self.cmodel.createData()
        for req in self.cdata.distanceRequests:
            req.enable_nearest_points = True
        pin.updateGeometryPlacements(self.model, self.data, self.cmodel, self.cdata, current_q)
        pin.computeDistances(self.cmodel, self.cdata)

        return added_names, skipped_names, adjacency

    def detach_object(self, cyl_id, current_q, world_pos=None):
        """Re-parent a previously-attached cylinder back to the world (inverse of attach).

        Re-parents the cylinder geometry onto the universe joint (id 0). If
        `world_pos` is given (the perfect-fall placement position computed by the
        shared-autonomy node), the cylinder is placed UPRIGHT there; otherwise it
        is frozen at its current live pose. The collision pairs created at attach
        time are kept (all valid for a static obstacle); only the parentJoint and
        placement change, mirroring attach_object_visually.

        The caller is responsible for dropping the cylinder from the attached_*
        bookkeeping and (optionally) restarting the barrier ramp for a smooth
        re-engagement of the gripper<->cylinder pair.
        """
        # Make sure oMg reflects the current configuration before snapshotting.
        pin.updateGeometryPlacements(self.model, self.data, self.cmodel, self.cdata, current_q)

        if world_pos is not None:
            # Upright cylinder resting at the placement location (perfect fall).
            world_pose = pin.SE3(np.eye(3), np.asarray(world_pos, dtype=float))
        else:
            world_pose = self.cdata.oMg[cyl_id].copy()

        geom = self.cmodel.geometryObjects[cyl_id]
        # parentJoint 0 == universe, so the geometry placement IS the world pose.
        geom.placement = world_pose
        geom.parentJoint = 0

        # Remove the cylinder↔table pair that was added during attach (the cylinder
        # was a moving arm link then, so table collision made sense). Now that the
        # cylinder is RESTING on the table, that pair fires at distance ~0 and
        # explodes the SoftMin. The original pre-grasp world never had this pair
        # (cylinders on the table are in workspace_obstacle_ids and only check vs
        # the arms, not vs the table). Also remove cylinder↔ground for the same reason.
        # Explicit, name/role-based reference (see build_collision_model's
        # world_scene branch, which sets self.table_id from the obstacle whose
        # role=="table" -- NOT assumed to be workspace_obstacle_ids[0]).
        table_id = getattr(self, 'table_id', None)
        pairs_to_remove = []
        for k in range(len(self.cmodel.collisionPairs)):
            cp = self.cmodel.collisionPairs[k]
            ids = {cp.first, cp.second}
            if cyl_id not in ids:
                continue
            other = (ids - {cyl_id}).pop()
            # Remove pairs vs table and ground (the cylinder rests on/touches these)
            if other == table_id or other == self.ground_id:
                pairs_to_remove.append(k)
            # Also remove pairs vs body (base, torso) that were added at attach —
            # a static resting cylinder should only be checked vs the arms.
            if other in set(self.body_geom_ids):
                pairs_to_remove.append(k)

        # Remove in reverse order to keep indices valid.
        for k in sorted(set(pairs_to_remove), reverse=True):
            self.cmodel.removeCollisionPair(self.cmodel.collisionPairs[k])

        # Rebuild cdata for the (unchanged-size) pair set and revalidate so the
        # same control tick can query distances without a stale array.
        self.cdata = self.cmodel.createData()
        for req in self.cdata.distanceRequests:
            req.enable_nearest_points = True
        pin.updateGeometryPlacements(self.model, self.data, self.cmodel, self.cdata, current_q)
        pin.computeDistances(self.cmodel, self.cdata)

    def _arm_membership(self, geom_id, attached_object_arm=None):
        """Classify a geometry id as belonging to 'right', 'left', 'both', or None.

        Used to split the single SoftMin CBF into two INDEPENDENT per-arm
        barriers (see compute_softmin_jacobian). A geometry belongs to an arm if:
          * it is one of that arm's own capsules/gripper box (self.right_geom_ids
            / self.left_geom_ids), OR
          * it is a cylinder currently ATTACHED (grasped) by that arm
            (attached_object_arm: {cyl_id: 'right'/'left'}) — a held object moves
            with its owning arm, so it must count as part of that arm's "body"
            for the purpose of collision-avoidance authority.

        Static world geometry (table, ground, wall, body boxes, un-grasped
        cylinders) belongs to NEITHER arm on its own — a pair between two such
        geometries never occurs (they don't move relative to each other), but a
        pair between an arm's geometry and a static object correctly attributes
        to just that one arm.

        Returns a set subset of {'right', 'left'} (empty set if neither).
        """
        membership = set()
        if geom_id in self._right_geom_id_set:
            membership.add('right')
        if geom_id in self._left_geom_id_set:
            membership.add('left')
        if attached_object_arm and geom_id in attached_object_arm:
            membership.add(attached_object_arm[geom_id])
        return membership

    def compute_softmin_jacobian(self, current_v, idx_right, idx_left,
                                 margin_targets, attached_objs, attached_adjacency,
                                 ignored_targets, publish_counter=0,
                                 attach_ramp_shifts=None, attached_object_arm=None,
                                 data=None, cdata=None):
        """
        Aggregate active collision pairs into TWO INDEPENDENT per-arm SoftMin CBFs
        (plus the legacy combined pair for backward-compatible telemetry).

        RATIONALE (per-arm coupling fix): a single SoftMin row mixes ALL active
        pairs' Jacobians via one softmax weighting, so its gradient can have
        nonzero columns in an arm's joints even when NONE of that arm's own pairs
        are actually close to binding -- merely because some OTHER pair (possibly
        involving only the other arm) was among the K closest. The QP then
        legitimately (but uselessly) recruits the "innocent" arm's joints to
        satisfy a barrier that has nothing to do with it, causing the inactive
        arm to twitch/oscillate whenever the active arm nears ANY obstacle.

        FIX: build one SoftMin aggregate per arm, where a pair contributes to
        arm A's aggregate iff at least one of its two geometries belongs to arm A
        (via _arm_membership -- includes a cylinder A is currently holding). A
        genuine inter-arm pair (or two held cylinders getting close) touches BOTH
        arms and correctly contributes to BOTH aggregates -- preserving the
        desired "arm A may yield to let arm B reduce its tracking error" behavior.
        A pair that touches only arm A's geometry (vs. a static obstacle, e.g.
        the table) NEVER appears in arm B's aggregate, so arm B's barrier row has
        an EXACTLY ZERO gradient there -- eliminating the spurious coupling.

        Returns: (J_soft_R, h_soft_R, J_soft_L, h_soft_L, d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance)
            J_soft_X : (nv,) gradient of arm X's SoftMin barrier (0 when no interaction)
            h_soft_X : scalar SoftMin distance value for arm X (1.0 when no interaction)
            d_safe_dynamic_X : velocity-inflated safety margin for arm X's OWN barrier row
                (PER-ARM, 2026-07-01 -- see the coupling audit below)
            abs_min_distance : true closest distance over ALL pairs (for telemetry)

        ---------------------------------------------------------------------
        COUPLING AUDIT (2026-07-01): why d_safe_dynamic had to be split too.

        The per-arm Jacobian split (previous fix) makes J_soft_R/J_soft_L's
        NONZERO COLUMNS correctly disjoint (modulo genuine inter-arm pairs).
        But d_safe_dynamic was still ONE SHARED SCALAR, computed from the
        COMBINED (both-arm) velocity norm, and used identically in BOTH rows:

            b_col_X = -GAMMA_CBF * (h_soft_X - d_safe_dynamic)

        If the ACTIVE arm moves fast, the combined v_norm grows, inflating
        d_safe_dynamic for BOTH rows -- including the IDLE arm's row, even
        though nothing changed about the idle arm's own geometry or motion.
        h_soft_L is essentially never exactly 1.0 in a cluttered scene (the
        idle arm always has SOME finite closest distance to the table/body/
        other arm, just usually a large, harmless one) -- shrinking its
        margin against that fixed h_soft_L tightens b_col_L and can flip an
        otherwise-slack barrier active, forcing the idle arm's (few, but
        nonzero -- e.g. via genuinely shared inter-arm pairs, or simply to
        satisfy a now-infeasible-if-ignored row) joints to move. This was a
        coupling channel that survived the Jacobian split entirely, and it
        tracks exactly the reported symptom: idle-arm oscillation correlated
        with FAST ACTIVE-ARM MOTION near ANY obstacle (its own speed, not the
        idle arm's, is what was inflating the idle arm's threshold).

        FIX: compute d_safe_dynamic per arm, from ONLY that arm's own joint
        velocities. An idle arm (near-zero q_dot) now keeps a near-BASE
        (tight) margin regardless of how fast the other arm is moving; only
        an arm that is ITSELF moving fast gets an inflated margin.
        ---------------------------------------------------------------------
        """
        # data/cdata default to self.* (shared main-thread pair). The real-
        # hardware CBF worker passes its OWN private pin.Data / GeometryData
        # (already refreshed via update_geometry on that same data) so this whole
        # aggregation runs race-free off the main thread. model/cmodel are shared
        # read-only structure and stay self.*. The self.witness_*/top_active_pairs
        # writes below are read back SAME-THREAD by the caller right after this
        # returns (never cross-thread), so they carry no race either.
        data = self.data if data is None else data
        cdata = self.cdata if cdata is None else cdata

        # --- Dynamic margin: thicken the barrier with EACH ARM'S OWN speed
        # (computed FIRST, before the SoftMin shifts, so high velocity still
        # correctly pushes that arm away from a grasp target -- but no longer
        # leaks into the OTHER arm's margin). ---
        v_norm_r = np.linalg.norm(current_v[idx_right]) if idx_right else 0.0
        v_norm_l = np.linalg.norm(current_v[idx_left]) if idx_left else 0.0
        d_safe_dynamic_r = cfg.D_SAFE_BASE + (cfg.K_V_SAFE * v_norm_r)
        d_safe_dynamic_l = cfg.D_SAFE_BASE + (cfg.K_V_SAFE * v_norm_l)
        # The per-pair grasp-margin shift math below picks the correct scalar
        # per pair (a gripper<->cylinder pair belongs to exactly one arm in
        # practice, resolved via _arm_membership) -- see the shift computation.

        # STEP 1: Collect candidate pairs within range, then keep the K closest.
        # Also track the TRUE (unfiltered) global-closest pair in this SAME pass
        # -- used only for the RViz collision witness line (qp_visualizer_
        # tutorial.py's publish_debug), which used to re-scan cdata.distanceResults
        # from scratch on every telemetry tick: a second full pass over the exact
        # same data this loop already visits. Folding it in here makes it ONE scan
        # instead of two; the visualizer's own draw threshold (0.20m) is unchanged,
        # it just reads self.witness_min_distance/points instead of recomputing them.
        pair_distances = []
        witness_dist = float('inf')
        witness_res = None
        for k, res in enumerate(cdata.distanceResults):
            d = res.min_distance
            if d < witness_dist:
                witness_dist = d
                witness_res = res
            if d <= cfg.DISTANCE_FILTER_THRESHOLD:
                pair_distances.append((d, k, res))
        pair_distances.sort()   # tuples sort by [0] (distance) first -- no lambda needed
        active_pairs = pair_distances[:cfg.K_MAX_PAIRS]
        # NaN (not a fake 1.0) when NO pair is within DISTANCE_FILTER_THRESHOLD,
        # so /qp_debug/min_distance stays truthful -- "no pair in range" instead
        # of a spurious 1 m reading that plots as a discontinuous jump. Telemetry
        # only (never consumed by the control law; h_soft carries the CBF value).
        abs_min_distance = float(pair_distances[0][0]) if pair_distances else float("nan")

        self.witness_min_distance = witness_dist if witness_res is not None else float('nan')
        if witness_res is not None:
            if hasattr(witness_res, 'nearest_points'):
                self.witness_min_points = (witness_res.nearest_points[0], witness_res.nearest_points[1])
            elif hasattr(witness_res, 'getNearestPoint1'):
                self.witness_min_points = (witness_res.getNearestPoint1(), witness_res.getNearestPoint2())
            else:
                self.witness_min_points = (witness_res.o1, witness_res.o2)
        else:
            self.witness_min_points = None

        # Per-pair "core math" (contact normal, distance-rate Jacobian, softmax
        # weight) is collected here and processed in ONE vectorized batch AFTER
        # the loop, instead of a separate small numpy call (exp, weighted-sum) on
        # every single pair -- numpy's per-call overhead dominates at this array
        # size (up to K_MAX_PAIRS), so batching M calls into 1-2 is a real saving.
        # The per-pair HOOKS below (skip/margin/ramp logic) are UNCHANGED and
        # still run per-pair exactly as before -- only the final weighting/
        # summation step is batched.
        d_eff_list = []
        Jdist_list = []
        route_r_list = []
        route_l_list = []
        jacobian_cache = {}
        enabled_pairs = []   # (raw_distance, geom_id_first, geom_id_second) actually contributing

        # Geometry set allowed to "touch" a cylinder during a grasp (boxes + wrist
        # + fingers). STATIC for the node's lifetime (right_geom_ids/left_geom_ids
        # never change after build_collision_model), so compute it ONCE and cache
        # it instead of rebuilding it via string ops on every single tick.
        if self._allowed_grasp_ids_cache is None:
            allowed_grasp_ids = set(self.gripper_box_ids.values())
            for gid in self.right_geom_ids + self.left_geom_ids:
                name = self.cmodel.geometryObjects[gid].name.lower()
                if "7_link" in name or "gripper" in name or "finger" in name:
                    allowed_grasp_ids.add(gid)
            self._allowed_grasp_ids_cache = allowed_grasp_ids
        allowed_grasp_ids = self._allowed_grasp_ids_cache

        for d, k, res in active_pairs:
            pair = self.cmodel.collisionPairs[k]
            first, second = pair.first, pair.second

            # --- SHARED-AUTONOMY HOOK 1: attached-payload adjacency exclusion ---
            # A carried cylinder is a fixed link; skip only the links it is fused to.
            skip_pair = False
            for cyl_id in attached_objs:
                if first == cyl_id or second == cyl_id:
                    other_id = second if first == cyl_id else first
                    if other_id in attached_adjacency.get(cyl_id, set()):
                        skip_pair = True
                    break

            # --- SHARED-AUTONOMY HOOK 2: explicitly bypass ignored targets ---
            # Drops a target cylinder entirely from the CBF during blind insertion.
            name1 = self.cmodel.geometryObjects[first].name
            name2 = self.cmodel.geometryObjects[second].name
            if name1 in ignored_targets or name2 in ignored_targets:
                skip_pair = True
            if skip_pair:
                continue

            # Record this pair as an ACTUALLY-ENABLED contributor (for debug plot)
            enabled_pairs.append((d, first, second))

            # --- SHARED-AUTONOMY HOOK 3: per-pair negative grasp margin ---
            # The barrier stays ACTIVE; only this gripper<->cylinder pair's safe
            # distance is relaxed (via a SoftMin shift) so controlled contact is allowed.
            shift = 0.0
            if margin_targets:
                cyl_gid = None
                if first in margin_targets and second in allowed_grasp_ids:
                    cyl_gid = first
                elif second in margin_targets and first in allowed_grasp_ids:
                    cyl_gid = second
                if cyl_gid is not None:
                    # Shift uses the DYNAMIC distance of the GRASPING arm (not a
                    # combined value) -- the other geometry in this pair is the
                    # gripper/wrist/finger, whose _arm_membership tells us which
                    # arm's own margin applies. Falls back to the max of both
                    # (safe/conservative) if membership is somehow ambiguous.
                    other_gid = second if cyl_gid == first else first
                    grasp_arm = self._arm_membership(other_gid, attached_object_arm)
                    if 'right' in grasp_arm and 'left' not in grasp_arm:
                        d_safe_for_shift = d_safe_dynamic_r
                    elif 'left' in grasp_arm and 'right' not in grasp_arm:
                        d_safe_for_shift = d_safe_dynamic_l
                    else:
                        d_safe_for_shift = max(d_safe_dynamic_r, d_safe_dynamic_l)
                    shift = d_safe_for_shift - margin_targets[cyl_gid]
                    if cfg.GRASP_DEBUG and publish_counter % 200 == 0:
                        if ("red_cylinder" in (name1, name2)) and \
                           ("gripper_right" in name1 or "gripper_right" in name2):
                            print("\n--- SHIFT TRACKER ---")
                            print(f"margin_ids keys: {list(margin_targets.keys())}")
                            print(f"Pair IDs -> first: {first}, second: {second}")
                            print(f"Was shift applied? shift = {shift:.4f}")
                            print(f"d_raw = {d:.4f} | d_eff = {d + shift:.4f}")
                            print("---------------------\n")

            # --- SHARED-AUTONOMY HOOK 4: attached-payload smooth barrier ramp ---
            # A freshly-attached cylinder's pairs start relaxed (large +shift ->
            # invisible to the SoftMin) and the true distance is restored as the
            # ramp shift decays to 0 over ~3 s, avoiding an instantaneous jump.
            if attach_ramp_shifts:
                for cyl_id, sh in attach_ramp_shifts.items():
                    if first == cyl_id or second == cyl_id:
                        shift += sh
                        break

            # Extract the nearest points (API differs across hppfcl versions)
            if hasattr(res, 'nearest_points'):
                p1, p2 = res.nearest_points[0], res.nearest_points[1]
            elif hasattr(res, 'getNearestPoint1'):
                p1, p2 = res.getNearestPoint1(), res.getNearestPoint2()
            else:
                p1, p2 = res.o1, res.o2

            # Contact normal between the two nearest points
            diff = p1 - p2
            norm = np.linalg.norm(diff)
            n = np.array([1, 0, 0]) if norm < 1e-6 else diff / norm

            # Per-point translational Jacobians (cached per joint), shifted to the contact point
            def get_point_jacobian(geom_id, p_target):
                j_id = self.cmodel.geometryObjects[geom_id].parentJoint
                if j_id not in jacobian_cache:
                    jacobian_cache[j_id] = pin.getJointJacobian(
                        self.model, data, j_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_6D = jacobian_cache[j_id]
                return J_6D[:3, :] - np.dot(get_skew(p_target - data.oMi[j_id].translation), J_6D[3:, :])

            J1_p1 = get_point_jacobian(first, p1)
            J2_p2 = get_point_jacobian(second, p2)
            J_dist_k = np.dot(n, (J1_p1 - J2_p2))  # Scalar distance-rate Jacobian for this pair

            # --- PER-ARM ROUTING (the coupling fix) ---
            # A pair contributes to arm X's SoftMin aggregate iff at least one of
            # its two geometries belongs to arm X (own links/gripper, or a
            # cylinder that arm is currently holding). A pair touching BOTH arms
            # (inter-arm contact, or two held cylinders) contributes to BOTH --
            # correctly preserving "arm A may yield to help arm B" -- while a pair
            # touching only ONE arm's geometry (e.g. that arm vs. the static
            # table) NEVER pollutes the other arm's barrier.
            touched = (self._arm_membership(first, attached_object_arm)
                      | self._arm_membership(second, attached_object_arm))

            # --- EXPERIMENTAL: INTER-ARM CLOSING-MARGIN HOOK (see config.py's
            # ENABLE_INTER_ARM_CLOSING_MARGIN) ---
            # d_safe_dynamic_r/l (computed once above) already inflate each row's
            # margin from THAT arm's own speed alone -- correct for a static
            # obstacle, but for a pair touching BOTH arms this under-estimates the
            # true closing rate (row R never "sees" the left arm's contribution,
            # and vice versa). Shrink this pair's effective distance by the
            # missing term, using the conservative max() since one shared d_eff
            # feeds both rows' softmin sums below. Static-obstacle pairs (touched
            # has only one arm) are completely unaffected.
            if cfg.ENABLE_INTER_ARM_CLOSING_MARGIN and 'right' in touched and 'left' in touched:
                shift -= cfg.K_V_SAFE * max(v_norm_r, v_norm_l)

            # Effective (shifted) distance -- SoftMax weighting itself is now
            # batched below, after the loop.
            d_eff = d + shift

            d_eff_list.append(d_eff)
            Jdist_list.append(J_dist_k)
            route_r_list.append('right' in touched)
            route_l_list.append('left' in touched)

        # --- Record the 3 closest ACTUALLY-ENABLED pairs for the debug plot ---
        enabled_pairs.sort()
        self.top_active_pairs = []
        for d_p, g1, g2 in enabled_pairs[:3]:
            n1 = self.cmodel.geometryObjects[g1].name
            n2 = self.cmodel.geometryObjects[g2].name
            self.top_active_pairs.append((n1, n2, float(d_p)))

        # --- Batched SoftMax weighting + per-arm accumulation --------------
        # Mathematically identical to the old per-pair "weight = exp(...);
        # sum_exp_X += weight; J_soft_sum_X += weight * J_dist_k" loop, just
        # computed as 1-2 vectorized numpy calls over all collected pairs
        # instead of up to K_MAX_PAIRS individual exp()/accumulate calls.
        if d_eff_list:
            d_eff_arr = np.asarray(d_eff_list)
            weights = np.exp(-cfg.ALPHA_SOFTMIN * d_eff_arr)
            Jdist_arr = np.asarray(Jdist_list)          # (M, nv)
            mask_r = np.asarray(route_r_list, dtype=bool)
            mask_l = np.asarray(route_l_list, dtype=bool)
            active_interaction_r = bool(mask_r.any())
            active_interaction_l = bool(mask_l.any())
            sum_exp_r = float(weights[mask_r].sum()) if active_interaction_r else 0.0
            sum_exp_l = float(weights[mask_l].sum()) if active_interaction_l else 0.0
            J_soft_sum_r = ((weights[mask_r, None] * Jdist_arr[mask_r]).sum(axis=0)
                           if active_interaction_r else np.zeros(self.model.nv))
            J_soft_sum_l = ((weights[mask_l, None] * Jdist_arr[mask_l]).sum(axis=0)
                           if active_interaction_l else np.zeros(self.model.nv))
        else:
            active_interaction_r, active_interaction_l = False, False
            sum_exp_r, sum_exp_l = 0.0, 0.0
            J_soft_sum_r = np.zeros(self.model.nv)
            J_soft_sum_l = np.zeros(self.model.nv)

        # No active interaction -> that arm's barrier is silent (open space).
        # Each arm's SoftMin is computed and returned INDEPENDENTLY.
        if not active_interaction_r or sum_exp_r < 1e-6:
            J_soft_r, h_soft_r = np.zeros(self.model.nv), 1.0
        else:
            J_soft_r = J_soft_sum_r / sum_exp_r
            h_soft_r = -(1.0 / cfg.ALPHA_SOFTMIN) * np.log(sum_exp_r)

        if not active_interaction_l or sum_exp_l < 1e-6:
            J_soft_l, h_soft_l = np.zeros(self.model.nv), 1.0
        else:
            J_soft_l = J_soft_sum_l / sum_exp_l
            h_soft_l = -(1.0 / cfg.ALPHA_SOFTMIN) * np.log(sum_exp_l)

        return J_soft_r, h_soft_r, J_soft_l, h_soft_l, d_safe_dynamic_r, d_safe_dynamic_l, abs_min_distance
