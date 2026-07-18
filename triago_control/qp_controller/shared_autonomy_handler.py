# shared_autonomy_handler.py
"""HRI interface: turns shared-autonomy commands (gripper_cmd, target_ignore, grasp_margin) into safe
collision-world topology changes; attach re-parents a grasped cylinder with zero geometric teleport."""

from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import time


class SharedAutonomyHandler:
    """Parses shared-autonomy commands and mutates grasp / CBF-exclusion state."""

    # Fresh-payload barrier ramp: pairs start relaxed and the true margin returns linearly,
    # avoiding the instantaneous constraint jump when a resting cylinder becomes an arm link.
    ATTACH_RAMP_S = 3.0            # seconds to ramp the barrier in
    ATTACH_RAMP_SHIFT_MAX = 0.30   # initial distance shift [m] (silences the pair)

    # Measured firm-grasp closure for the REAL gripper (rad, real convention 0=open ~0.7=closed);
    # commanding past it would drive through the physical cylinder's wall.
    REAL_GRASP_FIRM_CLOSE = 0.33

    def __init__(self, node, col_manager, kinematics, viz_engine):
        self.node = node
        self.col = col_manager
        self.kin = kinematics
        self.viz = viz_engine

        # --- DYNAMIC CBF / GRASP STATE ---
        self.ignored_targets = set()           # Names entirely bypassed by the CBF
        self.attached_objects = set()           # Cylinders permanently fused post-grasp
        self.attached_object_arm = {}           # {cyl_id: 'right'/'left'} owning arm
        self.attached_relative_transforms = {}  # {cyl_id: pin.SE3} relative pose at pick
        self.attached_adjacency = {}            # {cyl_id: set(geom_id)} rigidly-fused links
        self.attached_time = {}                 # {cyl_id: attach timestamp} for the barrier ramp
        self.grasp_margin_targets = {}          # {cyl_geom_id: negative margin}
        self.pending_attach = None              # (arm_side, color) processed in the QP loop
        self.pending_detach = None              # (arm_side, color) processed in the QP loop

        # --- SUBSCRIBERS ---
        self.node.create_subscription(String, '/shared_autonomy/gripper_cmd', self.gripper_cmd_callback, 10)
        self.node.create_subscription(String, '/shared_autonomy/target_ignore', self.ignore_col_callback, 10)
        self.node.create_subscription(String, '/shared_autonomy/grasp_margin', self.grasp_margin_callback, 10)

        # Signed gripper<->cylinder distance so the shared layer can confirm contact
        self.pub_grasp_contact = self.node.create_publisher(Float64MultiArray, '/shared_autonomy/grasp_contact', 10)

        # --- GRIPPER ACTION CLIENTS ---
        self.gripper_right_client = ActionClient(self.node, FollowJointTrajectory, '/gripper_right_controller/follow_joint_trajectory')
        self.gripper_left_client = ActionClient(self.node, FollowJointTrajectory, '/gripper_left_controller/follow_joint_trajectory')

    def close_gripper(self, side="right", position=0.0):
        """Sends a FollowJointTrajectory goal driving one gripper to `position` (rad)."""
        if self.node.REAL_HARDWARE:
            # The real gripper's joint direction is inverted vs sim; flip so the same
            # command means the same physical grip, then hard-cap at the firm-grasp reference.
            position = 0.7 - position
            position = min(position, self.REAL_GRASP_FIRM_CLOSE)
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f'gripper_{side}_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [position]   # 0.0 = fully closed; ~radius wraps without crushing
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        client = self.gripper_right_client if side == "right" else self.gripper_left_client
        # Never block the single-threaded executor: a no-timeout wait_for_server() would silently
        # freeze the whole safety loop; drop the command instead (the grasp FSM re-issues it).
        if not client.server_is_ready():
            self.node.get_logger().warn(
                f"[GRIPPER] {side} action server not ready -- dropping command "
                f"(control loop kept alive).", throttle_duration_sec=2.0)
            return
        client.send_goal_async(goal_msg)

    def gripper_cmd_callback(self, msg):
        """Parses CLOSE_/ORANGE_/ATTACH_/DETACH_ commands; attach/detach defer to the QP loop for fresh FK."""
        cmd = msg.data
        if cmd.startswith("CLOSE_"):
            parts = cmd.split('_')
            side = parts[1].lower()
            position = float(parts[2]) if len(parts) > 2 else 0.0  # optional CLOSE_RIGHT_0.0200
            self.close_gripper(side, position=position)
        elif cmd.startswith("ORANGE_"):
            parts = cmd.split('_')
            self.viz.paint_grasp_intent(parts[1].lower(), parts[2].lower(), self.col)
        elif cmd.startswith("ATTACH_"):
            parts = cmd.split('_')
            self.pending_attach = (parts[1].lower(), parts[2].lower())
        elif cmd.startswith("DETACH_"):
            parts = cmd.split('_')
            # Optional trailing floats carry the placement pose: DETACH_<ARM>_<COLOR>_<x>_<y>_<z>.
            arm = parts[1].lower()
            color = parts[2].lower()
            world_pos = None
            if len(parts) >= 6:
                try:
                    world_pos = [float(parts[3]), float(parts[4]), float(parts[5])]
                except ValueError:
                    world_pos = None
            self.pending_detach = (arm, color, world_pos)

    def ignore_col_callback(self, msg):
        """Adds/removes targets from the CBF bypass set (+name / -name / CLEAR protocol)."""
        command = msg.data
        if command in ("None", "CLEAR"):
            if self.ignored_targets:
                self.ignored_targets.clear()
                self.node.get_logger().info("[CBF RESTORED] All collision protections fully active.")
        elif command.startswith("+"):
            target = command[1:]
            if target not in self.ignored_targets:
                self.ignored_targets.add(target)
                self.node.get_logger().info(f"[CBF BYPASS] Added {target} to permitted contacts.")
        elif command.startswith("-"):
            target = command[1:]
            if target in self.ignored_targets:
                self.ignored_targets.discard(target)
                self.node.get_logger().info(f"[CBF RESTORED] Removed {target} from permitted contacts.")

    def grasp_margin_callback(self, msg):
        """Sets/clears the per-pair negative CBF margin ('name:margin' or CLEAR)."""
        command = msg.data.strip()
        if command in ("None", "clear", "CLEAR", ""):
            if self.grasp_margin_targets:
                self.grasp_margin_targets.clear()
                self.node.get_logger().info("[CBF MARGIN] All grasp margins restored to full safety.")
            return

        name, _, m_str = command.partition(":")
        try:
            margin = float(m_str)
        except ValueError:
            self.node.get_logger().warn(f"[CBF MARGIN] Malformed grasp_margin '{command}'. Expected 'name:margin'.")
            return

        # Resolve the id directly in cmodel so the math-engine id always matches.
        gid = None
        for i, obj in enumerate(self.col.cmodel.geometryObjects):
            if obj.name == name:
                gid = i
                break
        if gid is not None:
            if self.grasp_margin_targets.get(gid) != margin:
                self.grasp_margin_targets[gid] = margin
                self.node.get_logger().info(
                    f"[CBF MARGIN] {name} (cmodel ID {gid}): gripper-pair safe distance relaxed "
                    f"to {margin:+.4f} m (barrier still active).")
        else:
            self.node.get_logger().error(f"[CBF MARGIN] CRITICAL: '{name}' not found in cmodel geometry objects!")

    def attach_object_visually(self, arm_side, color):
        """Rigidly re-parents a grasped cylinder onto the gripper wrist joint (no geometric teleport)."""
        self.node.get_logger().info(f"\033[93m[TOPOLOGY] Attaching {color} cylinder to {arm_side} gripper.\033[0m")
        cyl_id = self.col.red_cyl_id if color == "red" else self.col.blue_cyl_id

        # 1. Promote to a permanent payload and drop the temporary grasp margin
        self.attached_objects.add(cyl_id)
        self.attached_object_arm[cyl_id] = arm_side
        self.grasp_margin_targets.pop(cyl_id, None)

        # 2. RE-PARENT GEOMETRY TO THE GRIPPER WRIST JOINT (J_soft continuity)
        tcp_frame = f'gripper_{arm_side}_grasping_link'
        if self.kin.model.existFrame(tcp_frame) and cyl_id < len(self.col.cmodel.geometryObjects):
            wrist_joint_id = self.kin.model.frames[self.kin.model.getFrameId(tcp_frame)].parentJoint

            # jMc = oMj^-1 * oMc from LIVE kinematics keeps the world pose bit-for-bit preserved:
            # distances, nearest points and the pair set stay continuous, only the Jacobian changes.
            oMj = self.kin.data.oMi[wrist_joint_id]
            oMc = self.col.cdata.oMg[cyl_id]
            jMc = oMj.actInv(oMc)

            geom = self.col.cmodel.geometryObjects[cyl_id]
            geom.placement = jMc            # relative pose first ...
            geom.parentJoint = wrist_joint_id  # ... then re-parent
            self.attached_relative_transforms[cyl_id] = jMc.copy()

            # Build the adjacency exclusion and create the cylinder's new world collision pairs.
            try:
                added, skipped, adjacency = self.col.add_attached_object_pairs(
                    cyl_id, arm_side, self.kin.current_q)
                self.attached_adjacency[cyl_id] = adjacency
                adj_names = sorted(self.col.cmodel.geometryObjects[g].name for g in adjacency)
                self.attached_time[cyl_id] = time.time()
                self.node.get_logger().info(
                    f"\033[92m[TOPOLOGY OK] {color} cylinder is now a link of the "
                    f"{arm_side} arm.\n"
                    f"  NEW collision pairs created: {added}\n"
                    f"  Already-existing pairs kept: {skipped} + both-arm pairs\n"
                    f"  Adjacency-EXCLUDED (own links 6/7/gripper/fingers): {adj_names}\n"
                    f"  Self-collision vs own arm links 3/4/5: ACTIVE.\033[0m")
                self.node.get_logger().info(
                    f"\033[96m[CBF RAMP] Smoothly engaging the collision barrier for the "
                    f"{color} cylinder over {self.ATTACH_RAMP_S:.1f}s "
                    f"(d_safe ramps from -{self.ATTACH_RAMP_SHIFT_MAX:.2f} m to nominal).\033[0m")
            except Exception as e:
                self.node.get_logger().error(
                    f"\033[91m[TOPOLOGY FAIL] Could not create collision pairs for "
                    f"the attached {color} cylinder: {e}\033[0m")
                # Fall back to the minimal adjacency so we don't crash the loop
                self.attached_adjacency[cyl_id] = set()

        # 3. Update Meshcat visuals (opaque orange) via the thread-safe viz engine
        self.viz.paint_grasp_intent(arm_side, color, self.col, opaque=True)

    def detach_object_visually(self, arm_side, color, world_pos=None):
        """Inverse of attach: releases a carried cylinder back into the world (upright at world_pos if given)."""
        self.node.get_logger().info(
            f"\033[93m[TOPOLOGY] Detaching {color} cylinder from {arm_side} gripper.\033[0m")
        cyl_id = self.col.red_cyl_id if color == "red" else self.col.blue_cyl_id

        if cyl_id not in self.attached_objects:
            self.node.get_logger().warn(
                f"[TOPOLOGY] {color} cylinder is not attached — nothing to detach.")
            return

        # 1. Re-parent geometry back to the world (only parentJoint/placement change).
        self.col.detach_object(cyl_id, self.kin.current_q, world_pos=world_pos)

        # 2. Drop the payload bookkeeping so the SoftMin stops treating it as a fused arm link.
        self.attached_objects.discard(cyl_id)
        self.attached_object_arm.pop(cyl_id, None)
        self.attached_relative_transforms.pop(cyl_id, None)
        self.attached_adjacency.pop(cyl_id, None)

        # 3. Re-arm the barrier ramp: the gripper still overlaps the released cylinder,
        # so the pair re-engages smoothly instead of spiking.
        self.attached_time[cyl_id] = time.time()

        # 4. Restore the original Meshcat colors (un-orange cylinder + gripper).
        self.viz.restore_object_color(arm_side, color, self.col)
        self.node.get_logger().info(
            f"\033[92m[TOPOLOGY OK] {color} cylinder released to the world; "
            f"barrier re-engaging smoothly over {self.ATTACH_RAMP_S:.1f}s.\033[0m")

    def get_attach_ramp_shifts(self):
        """Returns {cyl_id: shift_m} decaying from ATTACH_RAMP_SHIFT_MAX to 0 over ATTACH_RAMP_S."""
        shifts = {}
        now = time.time()
        for cyl_id, t0 in self.attached_time.items():
            elapsed = now - t0
            if elapsed < self.ATTACH_RAMP_S:
                ramp = max(0.0, min(1.0, elapsed / self.ATTACH_RAMP_S))  # 0 -> 1
                shifts[cyl_id] = (1.0 - ramp) * self.ATTACH_RAMP_SHIFT_MAX
        return shifts

    def publish_contact_distances(self):
        """Publishes the signed gripper<->cylinder distance [red, blue] for geometric grasp confirmation."""
        if not (self.col.gripper_box_ids and (self.grasp_margin_targets or self.attached_objects)):
            return
        box_ids = set(self.col.gripper_box_ids.values())
        contact = {'red': 1.0, 'blue': 1.0}
        for k, res in enumerate(self.col.cdata.distanceResults):
            pair = self.col.cmodel.collisionPairs[k]
            ids = {pair.first, pair.second}
            if not (ids & box_ids):
                continue
            if hasattr(self.col, 'red_cyl_id') and self.col.red_cyl_id in ids:
                contact['red'] = min(contact['red'], float(res.min_distance))
            elif hasattr(self.col, 'blue_cyl_id') and self.col.blue_cyl_id in ids:
                contact['blue'] = min(contact['blue'], float(res.min_distance))
        self.pub_grasp_contact.publish(Float64MultiArray(data=[contact['red'], contact['blue']]))
