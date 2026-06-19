# collision_manager.py
"""
The Environmental Awareness module.

Owns the `hppfcl` geometry model and every proximity query the controller needs:
    * builds arm capsules from `calculate_offsets` (dominant-axis snapping),
    * adds simplified gripper bounding boxes, body colliders, ground, wall and
      the bimanual workspace obstacles (table + red/blue cylinders),
    * declares the collision-pair graph (with all the original exclusions),
    * runs the per-tick distance queries and aggregates them into a single
      SoftMin Control Barrier Function gradient.

----------------------------------------------------------------------------
SoftMin CBF math (PRESERVED EXACTLY):

    h_soft(q) = -(1/alpha) * log( sum_k exp(-alpha * d_k(q)) )

    J_soft(q) = sum_k ( exp(-alpha * d_k(q)) / sum_j exp(-alpha * d_j(q)) ) * J_k(q)

  i.e. a differentiable approximation of the single closest distance, whose
  gradient is the convex (softmax) blend of every active pair Jacobian J_k.

Dynamic safety margin (PRESERVED EXACTLY):

    d_safe_dynamic = d_safe_base + k_v_safe * ||v_norm||

  The barrier thickens with arm speed so the robot brakes earlier when fast.
----------------------------------------------------------------------------
"""

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

        # Geometry id bookkeeping
        self.right_geom_ids = []
        self.left_geom_ids = []
        self.body_geom_ids = []
        self.gripper_box_ids = {}          # {'right': id, 'left': id}
        self.workspace_obstacle_ids = []

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

    def build_collision_model(self, right_offsets, left_offsets):
        # Assemble the full collision geometry model: arms, grippers, body, ground, obstacles.

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
                shape = hppfcl.Capsule(cfg.CAPSULE_RADIUS, length)
                obj = pin.GeometryObject(f"{prefix}_{link_name}", parent_joint_id, placement_wrt_joint, shape)
                id_list.append(self.cmodel.addGeometryObject(obj))

        add_arm_geoms(right_offsets, "shadow_right", self.right_geom_ids)
        add_arm_geoms(left_offsets, "shadow_left", self.left_geom_ids)

        # 2. SIMPLIFIED END-EFFECTOR BOXES (PAL PRO grippers)
        for side, id_list in (('right', self.right_geom_ids), ('left', self.left_geom_ids)):
            base_link = f'gripper_{side}_base_link'
            if self.model.existFrame(base_link):
                frame_id = self.model.getFrameId(base_link)
                parent_joint = self.model.frames[frame_id].parent
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

        # 5. VIRTUAL WALL (XZ plane) -- optional
        if cfg.WALL_COLLIDER:
            wall_pose = pin.SE3.Identity()
            wall_pose.translation = np.array(cfg.WALL_POS)
            self.wall_id = self.cmodel.addGeometryObject(
                pin.GeometryObject("virtual_wall", 0, wall_pose, hppfcl.Box(*cfg.WALL_SIZE)))

        # 6. BIMANUAL WORKSPACE (table + red/blue cylinders)
        table_pose = pin.SE3(np.eye(3), np.array(cfg.TABLE_POS))
        self.workspace_obstacle_ids.append(self.cmodel.addGeometryObject(
            pin.GeometryObject("work_table", 0, table_pose, hppfcl.Box(*cfg.TABLE_SIZE))))

        red_pose = pin.SE3(np.eye(3), np.array(cfg.RED_CYLINDER_POS))
        self.red_cyl_id = self.cmodel.addGeometryObject(
            pin.GeometryObject("red_cylinder", 0, red_pose, hppfcl.Cylinder(*cfg.CYLINDER_SIZE)))
        self.workspace_obstacle_ids.append(self.red_cyl_id)

        blue_pose = pin.SE3(np.eye(3), np.array(cfg.BLUE_CYLINDER_POS))
        self.blue_cyl_id = self.cmodel.addGeometryObject(
            pin.GeometryObject("blue_cylinder", 0, blue_pose, hppfcl.Cylinder(*cfg.CYLINDER_SIZE)))
        self.workspace_obstacle_ids.append(self.blue_cyl_id)

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

    def update_geometry(self, current_q):
        # Refresh geometry placements and run all pairwise distance queries.
        pin.updateGeometryPlacements(self.model, self.data, self.cmodel, self.cdata, current_q)
        pin.computeDistances(self.cmodel, self.cdata)

    def compute_softmin_jacobian(self, current_v, idx_right, idx_left,
                                 margin_targets, attached_objs, attached_adjacency,
                                 ignored_targets, publish_counter=0):
        """
        Aggregate all active collision pairs into one SoftMin CBF.

        Returns: (J_soft, h_soft, d_safe_dynamic, abs_min_distance)
            J_soft : (nv,) gradient of the SoftMin barrier (0 when no interaction)
            h_soft : scalar SoftMin distance value (1.0 when no interaction)
            d_safe_dynamic : velocity-inflated safety margin used by the barrier
            abs_min_distance : true closest distance (for telemetry)
        """
        # --- Dynamic margin: thicken the barrier with arm speed (computed FIRST) ---
        # It must be known before the SoftMin shifts, otherwise high velocity would
        # push the robot away from a grasp target.
        active_v = np.zeros(0)
        if idx_right:
            active_v = np.concatenate((active_v, current_v[idx_right]))
        if idx_left:
            active_v = np.concatenate((active_v, current_v[idx_left]))
        v_norm = np.linalg.norm(active_v) if len(active_v) > 0 else 0.0
        d_safe_dynamic = cfg.D_SAFE_BASE + (cfg.K_V_SAFE * v_norm)

        # STEP 1: Collect candidate pairs within range, then keep the K closest
        pair_distances = [(res.min_distance, k, res)
                          for k, res in enumerate(self.cdata.distanceResults)
                          if res.min_distance <= cfg.DISTANCE_FILTER_THRESHOLD]
        pair_distances.sort(key=lambda x: x[0])
        active_pairs = pair_distances[:cfg.K_MAX_PAIRS]
        abs_min_distance = float(pair_distances[0][0]) if pair_distances else 1.0

        # SoftMin accumulators
        sum_exp = 0.0
        J_soft_sum = np.zeros(self.model.nv)
        active_interaction = False
        jacobian_cache = {}

        # Geometry set allowed to "touch" a cylinder during a grasp (boxes + wrist + fingers)
        allowed_grasp_ids = set(self.gripper_box_ids.values())
        for gid in self.right_geom_ids + self.left_geom_ids:
            name = self.cmodel.geometryObjects[gid].name.lower()
            if "7_link" in name or "gripper" in name or "finger" in name:
                allowed_grasp_ids.add(gid)

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
                    # Shift uses the DYNAMIC distance, not the static base distance
                    shift = d_safe_dynamic - margin_targets[cyl_gid]
                    if cfg.GRASP_DEBUG and publish_counter % 200 == 0:
                        if ("red_cylinder" in (name1, name2)) and \
                           ("gripper_right" in name1 or "gripper_right" in name2):
                            print("\n--- SHIFT TRACKER ---")
                            print(f"margin_ids keys: {list(margin_targets.keys())}")
                            print(f"Pair IDs -> first: {first}, second: {second}")
                            print(f"Was shift applied? shift = {shift:.4f}")
                            print(f"d_raw = {d:.4f} | d_eff = {d + shift:.4f}")
                            print("---------------------\n")

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
                        self.model, self.data, j_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_6D = jacobian_cache[j_id]
                return J_6D[:3, :] - np.dot(get_skew(p_target - self.data.oMi[j_id].translation), J_6D[3:, :])

            J1_p1 = get_point_jacobian(first, p1)
            J2_p2 = get_point_jacobian(second, p2)
            J_dist_k = np.dot(n, (J1_p1 - J2_p2))  # Scalar distance-rate Jacobian for this pair

            # SoftMax weighting on the (shifted) effective distance
            d_eff = d + shift
            weight = np.exp(-cfg.ALPHA_SOFTMIN * d_eff)
            sum_exp += weight
            J_soft_sum += weight * J_dist_k
            active_interaction = True

        # No active interaction -> barrier is silent (open space)
        if not active_interaction or sum_exp < 1e-6:
            return np.zeros(self.model.nv), 1.0, d_safe_dynamic, abs_min_distance

        # Normalize the blended gradient and recover the scalar SoftMin distance
        J_soft = J_soft_sum / sum_exp
        h_soft = -(1.0 / cfg.ALPHA_SOFTMIN) * np.log(sum_exp)
        return J_soft, h_soft, d_safe_dynamic, abs_min_distance
