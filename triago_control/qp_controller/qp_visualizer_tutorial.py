import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray, String
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R # NEW IMPORT

class QPVisualizer:
    def __init__(self, node, ref_frame='base_link'):
        self.node = node
        self.ref_frame = ref_frame
        self.pub = node.create_publisher(MarkerArray, '/qp_debug_visualization', 10)

        #  Dedicated channel for Teleoperation graphics
        self.teleop_pub = node.create_publisher(MarkerArray, '/teleop_debug_visualization', 10)
        
        # --- State variables to hold the live telemetry ---
        self.ee_pos_right = None
        self.ee_vel_right = None
        self.ee_pos_left = None
        self.ee_vel_left = None

        # --- State variables to hold the COMMANDED telemetry ---
        self.cmd_pos = None
        self.cmd_rot_matrix = None
        self.active_arm = 'right'

        # Frozen (inactive) arm's held pose — shown as a grey gripper in RViz.
        # Updated when the active arm switches: the old active arm's last known
        # reference becomes the frozen pose for the grey gripper.
        self.frozen_pos = None
        self.frozen_rot_matrix = None
        self._inactive_last_pos = None
        self._inactive_last_rot = None

        # --- Subscribe directly to the controller's published data ---
        self.ee_sub = node.create_subscription(
            Float64MultiArray,
            '/qp_debug/ee_real',
            self.ee_callback,
            10
        )

        # --- Subscribe to BOTH arm references; only the active one updates cmd_pos ---
        node.create_subscription(
            Float64MultiArray,
            '/arm_right/cartesian_reference',
            self.cmd_callback_right,
            10
        )
        node.create_subscription(
            Float64MultiArray,
            '/arm_left/cartesian_reference',
            self.cmd_callback_left,
            10
        )
        # --- Subscribe to the active-arm switch topic ---
        node.create_subscription(
            String,
            '/shared_autonomy/active_arm',
            self.active_arm_cb,
            10
        )

        # Periodic full sweep of both marker topics this class owns, so any
        # marker that gets orphaned by a code path we haven't foreseen (id
        # collisions, dropped messages, etc.) self-heals instead of lingering on
        # screen forever. Defense-in-depth alongside the fixed-id fix above.
        self.MARKER_CLEANUP_PERIOD_S = 3.0
        node.create_timer(self.MARKER_CLEANUP_PERIOD_S, self._sweep_all_markers)

    def _sweep_all_markers(self):
        """Publish Marker.DELETEALL on every topic this class owns."""
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        ma = MarkerArray()
        ma.markers.append(delete_all)
        self.pub.publish(ma)
        self.teleop_pub.publish(ma)

    def ee_callback(self, msg):
        # Protocol: [Px_R, Py_R, Pz_R, Vx_R, Vy_R, Vz_R, Px_L, Py_L, Pz_L, Vx_L, Vy_L, Vz_L]
        if len(msg.data) >= 12:
            self.ee_pos_right = np.array(msg.data[0:3])
            self.ee_vel_right = np.array(msg.data[3:6])
            self.ee_pos_left = np.array(msg.data[6:9])
            self.ee_vel_left = np.array(msg.data[9:12])

    def active_arm_cb(self, msg):
        """Switches which arm's commanded reference is visualized.
        The old active arm's last known cmd_pos becomes the grey frozen gripper."""
        if msg.data in ('right', 'left') and msg.data != self.active_arm:
            # Snapshot: the arm we're leaving becomes the frozen grey gripper.
            self.frozen_pos = self.cmd_pos.copy() if self.cmd_pos is not None else None
            self.frozen_rot_matrix = self.cmd_rot_matrix.copy() if self.cmd_rot_matrix is not None else None
            # Also start tracking the inactive arm's reference for the frozen gripper
            # (if the inactive arm receives reference updates from _freeze_arm, we
            # pick those up in the non-active callback and keep frozen_pos fresh).
            self.active_arm = msg.data
            # Reset active-arm cmd so it re-anchors from the new arm's next message.
            self.cmd_pos = None
            self.cmd_rot_matrix = None
            # BUG FIX: without an explicit DELETE, the blue "qp_debug_gripper"
            # markers drawn for the arm we just left stay on screen FOREVER —
            # publish_debug only re-ADDs at the SAME ns/id when cmd_pos is set, and
            # here we just set it to None, so nothing ever overwrites/removes the
            # old blue gripper. Clear it explicitly on every switch.
            self._delete_marker_group("qp_debug_gripper", start_id=0, count=3)

    def _delete_marker_group(self, ns, start_id, count):
        """Publish DELETE for `count` markers in `ns` starting at `start_id`."""
        ma = MarkerArray()
        now = self.node.get_clock().now().to_msg()
        for i in range(count):
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = now
            m.ns = ns
            m.id = start_id + i
            m.action = Marker.DELETE
            ma.markers.append(m)
        self.pub.publish(ma)

    def cmd_callback_right(self, msg):
        """Updates commanded pose from the right-arm reference."""
        if len(msg.data) < 6:
            return
        pos = np.array(msg.data[0:3])
        rot = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False).as_matrix()
        if self.active_arm == 'right':
            self.cmd_pos = pos
            self.cmd_rot_matrix = rot
        else:
            # Inactive arm: keep the frozen grey gripper at its held pose.
            self.frozen_pos = pos
            self.frozen_rot_matrix = rot

    def cmd_callback_left(self, msg):
        """Updates commanded pose from the left-arm reference."""
        if len(msg.data) < 6:
            return
        pos = np.array(msg.data[0:3])
        rot = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False).as_matrix()
        if self.active_arm == 'left':
            self.cmd_pos = pos
            self.cmd_rot_matrix = rot
        else:
            # Inactive arm: keep the frozen grey gripper at its held pose.
            self.frozen_pos = pos
            self.frozen_rot_matrix = rot

    def _build_gripper(self, p_center, R_mat, opacity, start_id, timestamp,
                       color=None, ns="qp_debug_gripper"):
        """Builds a 3-part gripper for a commanded/frozen pose.
           Shifted backward so the reference point aligns with the fingertips (TCP).
        """
        markers = []
        quat = R.from_matrix(R_mat).as_quat()
        if color is None:
            color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=opacity)  # default blue
        else:
            color.a = opacity

        # --- The TCP Shift ---
        # Shift the entire gripper backward along the local X-axis (approach axis)
        # by exactly the distance from the palm to the fingertips (0.06m).
        tcp_offset = np.array([-0.06, 0.0, 0.0])
        p_base = p_center + (R_mat @ tcp_offset)

        # 1. The Base (Palm)
        base = Marker()
        base.header.frame_id = self.ref_frame
        base.header.stamp = timestamp
        base.ns = ns
        base.id = start_id
        base.type = Marker.CUBE
        base.action = Marker.ADD
        
        # Position is now p_base, NOT p_center
        base.pose.position.x, base.pose.position.y, base.pose.position.z = p_base[0], p_base[1], p_base[2]
        
        base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w = quat[0], quat[1], quat[2], quat[3]
        base.scale.x, base.scale.y, base.scale.z = 0.02, 0.08, 0.03  
        base.color = color
        markers.append(base)

        # 2. Left Finger
        offset_l = np.array([0.03, 0.035, 0.0]) 
        # Calculate finger position relative to the NEW shifted base
        p_left = p_base + (R_mat @ offset_l) 
        
        left = Marker()
        left.header = base.header
        left.ns = base.ns
        left.id = start_id + 1
        left.type = Marker.CUBE
        left.action = Marker.ADD
        left.pose.position.x, left.pose.position.y, left.pose.position.z = p_left[0], p_left[1], p_left[2]
        left.pose.orientation = base.pose.orientation  
        left.scale.x, left.scale.y, left.scale.z = 0.06, 0.01, 0.02  
        left.color = base.color
        markers.append(left)

        # 3. Right Finger
        offset_r = np.array([0.03, -0.035, 0.0]) 
        # Calculate finger position relative to the NEW shifted base
        p_right = p_base + (R_mat @ offset_r)
        
        right = Marker()
        right.header = base.header
        right.ns = base.ns
        right.id = start_id + 2
        right.type = Marker.CUBE
        right.action = Marker.ADD
        right.pose.position.x, right.pose.position.y, right.pose.position.z = p_right[0], p_right[1], p_right[2]
        right.pose.orientation = base.pose.orientation
        right.scale.x, right.scale.y, right.scale.z = 0.06, 0.01, 0.02
        right.color = base.color
        markers.append(right)

        return markers

    # TELEOPERATION VISUALIZATION
    def publish_teleop_tether(self):
        """
        Publishes a dynamic 3D tether connecting the real End-Effector to the commanded pose.
        The tether changes color (Green -> Red) and thickness based on the tracking error.
        A 0.4m error corresponds to the max 10N force (Kp = 25 N/m).
        """
        # Ensure we have both real and commanded positions before computing
        if self.cmd_pos is None:
            return
        # Use the active arm's EE position
        ee_pos = self.ee_pos_right if self.active_arm == 'right' else self.ee_pos_left
        if ee_pos is None:
            return

        markers = MarkerArray()
        timestamp = self.node.get_clock().now().to_msg()
        
        # 1. Compute the Cartesian Position Error
        error_vec = self.cmd_pos - ee_pos
        error_mag = np.linalg.norm(error_vec)

        # 2. Compute the Error Ratio (0.0 to 1.0)
        # Max error is 0.4m. We use min() to clamp the ratio at 1.0 so colors don't overflow.
        max_error = 0.4
        ratio = min(error_mag / max_error, 1.0)

        # 3. Interpolate Color (Green -> Yellow -> Red)
        # At ratio=0.0: r=0.0, g=1.0 (Green)
        # At ratio=1.0: r=1.0, g=0.0 (Red)
        r_color = float(ratio)
        g_color = float(1.0 - ratio)
        b_color = 0.0

        # 4. Interpolate Thickness
        # Starts as a thin 5mm string, grows up to a thick 3cm cylinder at max force
        min_thick = 0.005
        max_thick = 0.030
        thickness = min_thick + (ratio * (max_thick - min_thick))

        # 5. Build the Tether Marker
        tether = Marker()
        tether.header.frame_id = self.ref_frame
        tether.header.stamp = timestamp
        tether.ns = "teleop_tether"
        tether.id = 0
        
        # We use the ARROW type but hide the head to create a perfect cylinder between 2 points
        tether.type = Marker.ARROW
        tether.action = Marker.ADD
        
        # scale.x is shaft diameter, scale.y is head diameter, scale.z is head length
        tether.scale.x = thickness
        tether.scale.y = 0.0 
        tether.scale.z = 0.0 
        
        tether.color = ColorRGBA(r=r_color, g=g_color, b=b_color, a=0.8)

        # Define the start (Real TCP) and end (Commanded TCP) points
        p_start = Point(x=float(ee_pos[0]), 
                        y=float(ee_pos[1]), 
                        z=float(ee_pos[2]))
        
        p_end = Point(x=float(self.cmd_pos[0]), 
                      y=float(self.cmd_pos[1]), 
                      z=float(self.cmd_pos[2]))

        tether.points = [p_start, p_end]
        markers.markers.append(tether)

        # Publish to the dedicated teleop channel
        self.teleop_pub.publish(markers)

    def publish_debug(self, model, data, cdata, q, q_dot, target_right, target_left, id_right, id_left, limit_buffer):
        """
        Visualizes:
        1. Collision Witness (Red Line + Distance Text)
        2. WORST Joint Limit (Arms Only, Smaller Text)
        3. End Effector Desires vs Reality (Arrows)
        4. Commanded Target Gripper (Blue)
        """
        markers = MarkerArray()
        idx = 0
        timestamp = rclpy.time.Time().to_msg() 

        # --- Helpers ---
        def make_point(v):
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            return p

        def create_marker(id, type, scale, color, position=None, points=None, text=""):
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = timestamp
            m.ns = "qp_debug"
            m.id = id
            m.type = type
            m.action = Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = color
            m.text = text
            if position is not None:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = position
            if points:
                m.points = points
            return m

        # Colors
        c_warn   = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0) # Orange
        c_crit   = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # Red
        c_text   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # White text
        
        # --- 1. COLLISION SAFETY (Red Line + Text) ---
        min_d = 100.0
        p1_w, p2_w = None, None

        for res in cdata.distanceResults:
            if res.min_distance < min_d:
                min_d = res.min_distance
                if hasattr(res, 'nearest_points'):
                    p1_w, p2_w = res.nearest_points[0], res.nearest_points[1]
                elif hasattr(res, 'getNearestPoint1'):
                    p1_w, p2_w = res.getNearestPoint1(), res.getNearestPoint2()
                else:
                    p1_w, p2_w = res.o1, res.o2
        
        if min_d < 0.20 and p1_w is not None:
            col = c_warn if min_d > 0.05 else c_crit
            markers.markers.append(create_marker(idx, Marker.LINE_LIST, (0.01, 0.0, 0.0), col, points=[make_point(p1_w), make_point(p2_w)]))
            idx += 1
            midpoint = (p1_w + p2_w) / 2.0
            midpoint[2] += 0.05 
            markers.markers.append(create_marker(idx, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.03), c_text, position=midpoint, text=f"{min_d:.3f}m"))
            idx += 1

        # --- 2. WORST JOINT LIMIT --- (REMOVED: the text/sphere at the near-limit
        # joint cluttered the RViz view; joint-limit proximity is monitored via the
        # plotter's lambda_joints trace instead.)

        # --- 3. ARM VECTORS (Arrows) ---
        def process_arm(pos_curr, vel_curr, target_pose):
            nonlocal idx
            
            if pos_curr is None or vel_curr is None: 
                return

            if target_pose is not None:
                pos_targ = target_pose.translation
                
                # Distance/Reference Arrow (White)
                Kp = 1.0
                v_ref = (pos_targ - pos_curr) * Kp
                start = make_point(pos_curr)
                end_des = make_point(pos_curr + v_ref)
                markers.markers.append(create_marker(idx, Marker.ARROW, (0.01, 0.02, 0.0), ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5), points=[start, end_des]))
                idx += 1
                
                # Actual Velocity Arrow (Yellow)
                end_act = make_point(pos_curr + vel_curr)
                markers.markers.append(create_marker(idx, Marker.ARROW, (0.015, 0.03, 0.0), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), points=[start, end_act]))
                idx += 1

        process_arm(self.ee_pos_right, self.ee_vel_right, target_right)
        process_arm(self.ee_pos_left, self.ee_vel_left, target_left)

        # --- 4. COMMANDED GRIPPER VISUALIZATION (Blue, follows active arm) ---
        # BUG FIX: these two grippers use a FIXED start_id (0), NOT the shared
        # `idx` counter. `idx` accumulates from the conditional markers above
        # (collision line/text, joint-limit sphere/text — each appears only
        # SOMETIMES), so under the old code the gripper's id shifted tick to
        # tick even though its ns ("qp_debug_gripper" / "frozen_gripper") never
        # changed. RViz treats (ns, id) as the marker's identity, so a shifting
        # id meant a NEW marker was created every time instead of overwriting
        # the previous one — the old id was never touched again (no lifetime,
        # no DELETE) and stayed on screen forever. Both namespaces are unique to
        # these two grippers, so a fixed id=0..2 here cannot collide with the
        # "qp_debug" markers above (different namespace).
        if self.cmd_pos is not None and self.cmd_rot_matrix is not None:
            gripper_markers = self._build_gripper(self.cmd_pos, self.cmd_rot_matrix, 0.8, 0, timestamp)
            markers.markers.extend(gripper_markers)
        else:
            self._delete_marker_group("qp_debug_gripper", start_id=0, count=3)

        # --- 5. FROZEN (INACTIVE) ARM GRIPPER (Grey, shows where the CLF holds it) ---
        # Disappears when the arm becomes active (cmd_pos takes over → blue gripper).
        if self.frozen_pos is not None and self.frozen_rot_matrix is not None:
            grey_markers = self._build_gripper(
                self.frozen_pos, self.frozen_rot_matrix, 0.5, 0, timestamp,
                color=ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.5), ns="frozen_gripper")
            markers.markers.extend(grey_markers)
        else:
            self._delete_marker_group("frozen_gripper", start_id=0, count=3)
            
        # =========================================================
        # --- 5. PINHOLE TASK ENVIRONMENT (STATIC OBSTACLES) ---
        # =========================================================
        # Broadcasts exact CBF geometric primitives to RViz for operator transparency
        
        def create_box_marker(m_id, pos, scale, color):
            """Helper function to instantiate primitive Box markers."""
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = timestamp
            m.ns = "task_environment"
            m.id = m_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos[0], pos[1], pos[2]
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale[0], scale[1], scale[2]
            m.color = color
            return m

        def create_cylinder_marker(m_id, pos, radius, length, color):
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = self.node.get_clock().now().to_msg()
            m.ns = "environment"
            m.id = m_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            
            # Default upright orientation
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 1.0
            
            # Scale for a cylinder: x and y are the diameter, z is the length
            m.scale.x = float(radius * 2)
            m.scale.y = float(radius * 2)
            m.scale.z = float(length)
            m.color = color
            return m

        # --- 1. Work Table (Wood Color) ---
        markers.markers.append(create_box_marker(
            idx, 
            [1.0, 0.0, 0.35], 
            [0.6, 0.5, 0.7], 
            ColorRGBA(r=0.6, g=0.4, b=0.2, a=0.8)
        ))
        idx += 1

        # --- 2. Red Cylinder (Right Hand Manipulandum) ---
        # REMOVED: drawn by visualization_engine.py at its live collision pose
        # (grey when grasped, red otherwise). Keeping a static copy here
        # creates a phantom duplicate on the table after grasp.

        # --- 3. Blue Cylinder (Left Hand Manipulandum) ---
        # REMOVED: same reason as above.

        # --- 4. Target Placement Area (Yellow Manifold on Table) ---
        # Represented as a very flat cylinder slightly elevated above the table surface
        markers.markers.append(create_cylinder_marker(
            idx, 
            [1.000, 0.0, 0.701], 
            0.15, 0.002, 
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6)
        ))
        idx += 1

        self.pub.publish(markers)