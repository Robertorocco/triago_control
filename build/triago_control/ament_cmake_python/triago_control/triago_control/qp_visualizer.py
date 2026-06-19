import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray
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

        # --- NEW: State variables to hold the COMMANDED telemetry ---
        self.cmd_pos_right = None
        self.cmd_rot_matrix_right = None

        # --- Subscribe directly to the controller's published data ---
        self.ee_sub = node.create_subscription(
            Float64MultiArray,
            '/qp_debug/ee_real',
            self.ee_callback,
            10
        )

        # --- NEW: Subscribe directly to the commanded reference ---
        self.cmd_sub = node.create_subscription(
            Float64MultiArray,
            '/arm_right/cartesian_reference',
            self.cmd_callback,
            10
        )

    def ee_callback(self, msg):
        # Protocol: [Px_R, Py_R, Pz_R, Vx_R, Vy_R, Vz_R, Px_L, Py_L, Pz_L, Vx_L, Vy_L, Vz_L]
        if len(msg.data) >= 12:
            self.ee_pos_right = np.array(msg.data[0:3])
            self.ee_vel_right = np.array(msg.data[3:6])
            self.ee_pos_left = np.array(msg.data[6:9])
            self.ee_vel_left = np.array(msg.data[9:12])

    def cmd_callback(self, msg):
        """NEW: Listens to the user's teleoperation commands"""
        if len(msg.data) >= 6:
            # Extract position
            self.cmd_pos_right = np.array(msg.data[0:3])
            # Extract RPY and convert to Rotation Matrix for the gripper builder
            rpy = np.array(msg.data[3:6])
            self.cmd_rot_matrix_right = R.from_euler('xyz', rpy, degrees=False).as_matrix()

    def _build_gripper(self, p_center, R_mat, opacity, start_id, timestamp):
        """NEW: Builds a 3-part BLUE gripper for the commanded pose.
           Shifted backward so the reference point aligns with the fingertips (TCP).
        """
        markers = []
        quat = R.from_matrix(R_mat).as_quat()

        # --- The TCP Shift ---
        # Shift the entire gripper backward along the local X-axis (approach axis)
        # by exactly the distance from the palm to the fingertips (0.06m).
        tcp_offset = np.array([-0.06, 0.0, 0.0])
        p_base = p_center + (R_mat @ tcp_offset)

        # 1. The Base (Palm)
        base = Marker()
        base.header.frame_id = self.ref_frame
        base.header.stamp = timestamp
        base.ns = "qp_debug_gripper"
        base.id = start_id
        base.type = Marker.CUBE
        base.action = Marker.ADD
        
        # Position is now p_base, NOT p_center
        base.pose.position.x, base.pose.position.y, base.pose.position.z = p_base[0], p_base[1], p_base[2]
        
        base.pose.orientation.x, base.pose.orientation.y, base.pose.orientation.z, base.pose.orientation.w = quat[0], quat[1], quat[2], quat[3]
        base.scale.x, base.scale.y, base.scale.z = 0.02, 0.08, 0.03  
        base.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=opacity) # BLUE
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
        if self.ee_pos_right is None or self.cmd_pos_right is None:
            return

        markers = MarkerArray()
        timestamp = self.node.get_clock().now().to_msg()
        
        # 1. Compute the Cartesian Position Error
        error_vec = self.cmd_pos_right - self.ee_pos_right
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
        p_start = Point(x=float(self.ee_pos_right[0]), 
                        y=float(self.ee_pos_right[1]), 
                        z=float(self.ee_pos_right[2]))
        
        p_end = Point(x=float(self.cmd_pos_right[0]), 
                      y=float(self.cmd_pos_right[1]), 
                      z=float(self.cmd_pos_right[2]))

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

        # --- 2. WORST JOINT LIMIT (Arms Only, Exclude 7) ---
        worst_margin = 100.0
        worst_joint_idx = -1
        is_upper = False
        
        for i, joint in enumerate(model.joints):
            if i == 0 or joint.nq != 1: continue 
            
            # --- FILTER ---
            name = model.names[i]
            if "arm" not in name: continue  # Must be an arm joint
            if "7" in name: continue        # EXCLUDE WRIST ROTATION
            # --------------

            q_val = q[joint.idx_q]
            q_u = model.upperPositionLimit[joint.idx_q]
            q_l = model.lowerPositionLimit[joint.idx_q]
            
            dist_u = q_u - q_val
            dist_l = q_val - q_l
            current_min = min(dist_u, dist_l)
            
            if current_min < worst_margin:
                worst_margin = current_min
                worst_joint_idx = i
                is_upper = (dist_u < dist_l)

        if worst_margin < limit_buffer and worst_joint_idx != -1:
            pos_joint = data.oMi[worst_joint_idx].translation
            joint_name = model.names[worst_joint_idx]
            
            label_type = "MAX" if is_upper else "MIN"
            col = c_crit if worst_margin < (limit_buffer/2) else c_warn
            
            text_str = f"{joint_name}\n{label_type} ({worst_margin:.2f})"
            
            # Draw Text
            pos_text = pos_joint.copy()
            pos_text[2] += 0.1 
            markers.markers.append(create_marker(idx, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.025), c_text, position=pos_text, text=text_str))
            idx += 1
            
            markers.markers.append(create_marker(idx, Marker.SPHERE, (0.04, 0.04, 0.04), col, position=pos_joint))
            idx += 1

        # --- 3. ARM VECTORS (Arrows) ---
        def process_arm(pos_curr, vel_curr, target_pose):
            nonlocal idx
            
            if pos_curr is None or vel_curr is None: 
                return
            
            # 1. The Blue Dot (Reality)
            markers.markers.append(create_marker(idx, Marker.SPHERE, (0.04, 0.04, 0.04), ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8), position=pos_curr))
            idx += 1

            if target_pose is not None:
                # Target Dot (Green)
                pos_targ = target_pose.translation
                markers.markers.append(create_marker(idx, Marker.SPHERE, (0.04, 0.04, 0.04), ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), position=pos_targ))
                idx += 1
                
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

        # --- 4. NEW: COMMANDED GRIPPER VISUALIZATION ---
        if self.cmd_pos_right is not None and self.cmd_rot_matrix_right is not None:
            # Opacity set to 0.8 as requested
            gripper_markers = self._build_gripper(self.cmd_pos_right, self.cmd_rot_matrix_right, 0.8, idx, timestamp)
            markers.markers.extend(gripper_markers)
            # We used 3 markers (Base, Left, Right), so bump the id by 3
            idx += 3
            
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

        def create_cylinder_marker(m_id, pos, scale, color, quat):
            """Helper function to instantiate primitive Cylinder markers with arbitrary rotation."""
            m = Marker()
            m.header.frame_id = self.ref_frame
            m.header.stamp = timestamp
            m.ns = "task_environment"
            m.id = m_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos[0], pos[1], pos[2]
            m.pose.orientation.x = quat[0]
            m.pose.orientation.y = quat[1]
            m.pose.orientation.z = quat[2]
            m.pose.orientation.w = quat[3]
            m.scale.x, m.scale.y, m.scale.z = scale[0], scale[1], scale[2] 
            m.color = color
            return m

        # 1. Work Table (Brown, 50% opacity)
        markers.markers.append(create_box_marker(idx, [1.0, 0.0, 0.35], [0.6, 0.5, 0.7], ColorRGBA(r=0.6, g=0.4, b=0.2, a=0.5)))
        idx += 1

        # 2. Battery Holder (Blue, 100% opacity) - Sized X=0.12, Y=0.12, Z=0.10
        markers.markers.append(create_box_marker(idx, [0.86, -0.06, 0.75], [0.12, 0.12, 0.10], ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)))
        idx += 1

        # 3. Safety Shield (White, 50% opacity)
        markers.markers.append(create_box_marker(idx, [0.70, 0.00, 0.75], [0.02, 0.35, 0.30], ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.5)))
        idx += 1

        # 4. HV Testing Assembly Block (Black, 100% opacity)
        markers.markers.append(create_box_marker(idx, [1.195, -0.06, 0.775], [0.15, 0.15, 0.15], ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)))
        idx += 1

        # --- High Voltage Cables ---
        # RViz scaling for Cylinders requires X and Y to be the Diameter (0.008 * 2 = 0.016m)
        cable_scale = [0.016, 0.016, 0.20]
        cable_quat = R.from_euler('y', 90, degrees=True).as_quat() # Transforms pitch to [x, y, z, w]

        # 5. Positive Cable (Yellow, 100% opacity)
        markers.markers.append(create_cylinder_marker(idx, [1.02, -0.02, 0.76], cable_scale, ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0), cable_quat))
        idx += 1

        # 6. Negative Cable (Green, 100% opacity)
        markers.markers.append(create_cylinder_marker(idx, [1.02, -0.10, 0.76], cable_scale, ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), cable_quat))
        idx += 1

        self.pub.publish(markers)