# shared_autonomy_handler.py
"""
The Human-Robot-Interaction (HRI) interface.

Processes commands coming from the high-level shared-control node and turns
them into safe, topological changes to the collision world:

    * /shared_autonomy/gripper_cmd   -> CLOSE_/ORANGE_/ATTACH_ commands
    * /shared_autonomy/target_ignore -> dynamic CBF bypass set (+/- protocol)
    * /shared_autonomy/grasp_margin  -> per-pair negative CBF margins
    * /shared_autonomy/grasp_contact -> published signed gripper<->cylinder distance

The crown jewel here is `attach_object_visually`: it re-parents a grasped
cylinder from the world to the gripper wrist joint WITHOUT any geometric
teleport, so every collision distance, nearest point and the collision-pair
SET stay continuous across the topology change.
"""

from std_msgs.msg import String, Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import numpy as np
import triago_control.qp_controller.config as cfg


class SharedAutonomyHandler:
    """Parses shared-autonomy commands and mutates grasp / CBF-exclusion state."""

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
        self.grasp_margin_targets = {}          # {cyl_geom_id: negative margin}
        self.pending_attach = None              # (arm_side, color) processed in the QP loop

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
        # Send a FollowJointTrajectory goal to close one gripper to `position` (rad).
        self.node.get_logger().info(f"[GRIPPER] Closing {side} gripper to position={position:.4f} rad")
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [f'gripper_{side}_finger_joint']
        point = JointTrajectoryPoint()
        point.positions = [position]   # 0.0 = fully closed; ~radius wraps without crushing
        point.time_from_start.sec = 2
        goal_msg.trajectory.points = [point]
        client = self.gripper_right_client if side == "right" else self.gripper_left_client
        client.wait_for_server()
        client.send_goal_async(goal_msg)

    def gripper_cmd_callback(self, msg):
        # Parse high-level shared-autonomy gripper commands (CLOSE / ORANGE / ATTACH).
        cmd = msg.data
        if cmd.startswith("CLOSE_"):
            parts = cmd.split('_')
            side = parts[1].lower()
            position = float(parts[2]) if len(parts) > 2 else 0.0  # Optional CLOSE_RIGHT_0.0200
            self.close_gripper(side, position=position)
        elif cmd.startswith("ORANGE_"):
            parts = cmd.split('_')
            self.viz.paint_grasp_intent(parts[1].lower(), parts[2].lower(), self.col)
        elif cmd.startswith("ATTACH_"):
            parts = cmd.split('_')
            # Defer to the QP loop: re-parenting needs the freshly-updated kinematics
            self.pending_attach = (parts[1].lower(), parts[2].lower())

    def ignore_col_callback(self, msg):
        # Dynamically add/remove targets from the CBF bypass set (+/-/CLEAR protocol).
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
        # Set/clear the per-pair negative CBF margin for a gripper<->cylinder pair.
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

        # Look the id up directly in cmodel so the math-engine id always matches
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
        # Rigidly re-parent a grasped cylinder onto the gripper wrist (no geometric teleport).
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

            # Relative transform captured from the LIVE kinematics at the pick instant:
            #   jMc = oMj^-1 * oMc
            # Reading the current oMj/oMc keeps the cylinder's WORLD pose bit-for-bit
            # preserved (oMj * jMc == oMc), so distances, nearest points and the pair
            # set are continuous; only the parentJoint (hence the Jacobian) changes.
            oMj = self.kin.data.oMi[wrist_joint_id]
            oMc = self.col.cdata.oMg[cyl_id]
            jMc = oMj.actInv(oMc)

            geom = self.col.cmodel.geometryObjects[cyl_id]
            geom.placement = jMc            # set relative pose first ...
            geom.parentJoint = wrist_joint_id  # ... then re-parent
            self.attached_relative_transforms[cyl_id] = jMc.copy()

            # Adjacency-exclusion set: the cylinder is fused only to the wrist (link 7),
            # gripper box and fingers. Arm links 1-6 and the OTHER arm keep checking it.
            arm_ids = self.col.right_geom_ids if arm_side == 'right' else self.col.left_geom_ids
            adjacency = set()
            for gid in arm_ids:
                nm = self.col.cmodel.geometryObjects[gid].name.lower()
                if "_7_link" in nm or "gripper" in nm or "finger" in nm:
                    adjacency.add(gid)
            self.attached_adjacency[cyl_id] = adjacency

            self.node.get_logger().info(
                f"[TOPOLOGY] {color} cylinder bound to joint {wrist_joint_id} "
                f"(rel pos {np.round(jMc.translation, 3)}). Adjacency-excluded geoms: "
                f"{sorted(adjacency)}. Collision vs arm 1-6 + other arm STAYS ACTIVE.")

        # 3. Update Meshcat visuals (opaque orange) via the thread-safe viz engine
        self.viz.paint_grasp_intent(arm_side, color, self.col, opaque=True)

    def publish_contact_distances(self):
        # Publish signed gripper<->cylinder distance [red, blue] for grasp confirmation.
        # Only meaningful during an active grasp (margin set or cylinder attached).
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

        if cfg.GRASP_DEBUG and self.node.publish_counter % 200 == 0:
            margin_view = {self.col.cmodel.geometryObjects[g].name: m
                           for g, m in self.grasp_margin_targets.items()}
            self.node.get_logger().info(
                f"[GRASP-DBG/teleop] grasp_margin={margin_view} "
                f"attached={sorted(self.attached_objects)} | "
                f"gripper-cyl dist red={contact['red']:.4f} blue={contact['blue']:.4f} m "
                f"(margin keeps barrier active; closes near the margin value)")
