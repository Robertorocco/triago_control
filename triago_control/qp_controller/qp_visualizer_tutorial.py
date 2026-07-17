import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Float64MultiArray, String
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R # NEW IMPORT

class QPVisualizer:
    def __init__(self, node, ref_frame='base_link', world_scene=None):
        self.node = node
        self.ref_frame = ref_frame
        # Loaded world scene (world_loader.WorldScene, optional): drives the
        # placement-disk marker(s) below so RViz shows the CURRENT world's
        # platform(s) at their real poses/radii instead of a hard-coded default
        # center disk. None -> legacy single centered yellow disk.
        self.world_scene = world_scene
        self.pub = node.create_publisher(MarkerArray, '/qp_debug_visualization', 10)

        #  Dedicated channel for Teleoperation graphics
        self.teleop_pub = node.create_publisher(MarkerArray, '/teleop_debug_visualization', 10)
        
        # --- State variables to hold the live telemetry ---
        self.ee_pos_right = None
        self.ee_vel_right = None
        self.ee_pos_left = None
        self.ee_vel_left = None

        # --- State variables to hold the COMMANDED/REFERENCE telemetry ---
        # PER-ARM, INDEPENDENT (2026-07-01 redesign): each arm has its OWN
        # reference pose and its OWN frozen flag, decoupled from any notion of
        # a single global "active_arm". This is what lets BOTH grippers render
        # BLUE simultaneously when both arms are actively driven (e.g. by
        # trajectory_generator.py), while STILL rendering exactly one blue +
        # one grey in the single-arm teleop paradigm (one arm frozen by the
        # CLF hold, the other tracking live input) — the old design only ever
        # tracked one arm's pose at a time (cmd_pos/frozen_pos + active_arm),
        # so a second simultaneously-active arm's reference was silently
        # routed to the "frozen" grey slot instead of getting its own blue
        # gripper.
        self.ref_pos_right = None
        self.ref_rot_right = None
        self.ref_pos_left = None
        self.ref_rot_left = None
        # True == that arm's CLF is currently holding a FROZEN pose (draw
        # grey); False == that arm is actively tracking a live reference
        # (draw blue). Sourced from main_qp_controller's own right_frozen /
        # left_frozen state (the ground truth), via /qp_debug/arm_frozen —
        # NOT re-derived here from topic activity, which would be guesswork.
        self.frozen_right = False
        self.frozen_left = False

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
        # --- Subscribe to the per-arm frozen/active state (ground truth from
        # main_qp_controller.py's right_frozen / left_frozen) ---
        node.create_subscription(
            Float64MultiArray,
            '/qp_debug/arm_frozen',
            self.frozen_state_cb,
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

    def frozen_state_cb(self, msg):
        """[right_frozen, left_frozen] as 0.0/1.0 floats -- ground truth from
        main_qp_controller.py's self.right_frozen / self.left_frozen. Drives
        which color (blue=active, grey=frozen) each arm's OWN gripper is drawn
        in, independently of the other arm."""
        if len(msg.data) >= 2:
            self.frozen_right = msg.data[0] > 0.5
            self.frozen_left = msg.data[1] > 0.5

    def _delete_marker_group(self, ns, start_id, count, pub=None):
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
        (pub if pub is not None else self.pub).publish(ma)

    def cmd_callback_right(self, msg):
        """Updates the RIGHT arm's own reference pose. Independent of the left
        arm's state -- each arm always gets its own gripper marker, colored by
        its own frozen flag (see publish_debug)."""
        if len(msg.data) < 6:
            return
        self.ref_pos_right = np.array(msg.data[0:3])
        self.ref_rot_right = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False).as_matrix()

    def cmd_callback_left(self, msg):
        """Updates the LEFT arm's own reference pose. Independent of the right
        arm's state -- see cmd_callback_right."""
        if len(msg.data) < 6:
            return
        self.ref_pos_left = np.array(msg.data[0:3])
        self.ref_rot_left = R.from_euler('xyz', np.array(msg.data[3:6]), degrees=False).as_matrix()

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
        Publishes a dynamic 3D tether connecting each arm's real End-Effector
        to its own commanded/reference pose (PER-ARM, independent -- draws up
        to TWO tethers, one per arm, whichever has a reference). The tether
        changes color (Green -> Red) and thickness based on the tracking error.
        A 0.4m error corresponds to the max 10N force (Kp = 25 N/m).
        """
        markers = MarkerArray()
        timestamp = self.node.get_clock().now().to_msg()

        def build_tether(ref_pos, ee_pos, marker_id):
            if ref_pos is None or ee_pos is None:
                return None
            error_vec = ref_pos - ee_pos
            error_mag = np.linalg.norm(error_vec)
            max_error = 0.4
            ratio = min(error_mag / max_error, 1.0)
            r_color = float(ratio)
            g_color = float(1.0 - ratio)
            b_color = 0.0
            min_thick = 0.005
            max_thick = 0.030
            thickness = min_thick + (ratio * (max_thick - min_thick))

            tether = Marker()
            tether.header.frame_id = self.ref_frame
            tether.header.stamp = timestamp
            tether.ns = "teleop_tether"
            tether.id = marker_id
            tether.type = Marker.ARROW
            tether.action = Marker.ADD
            tether.scale.x = thickness
            tether.scale.y = 0.0
            tether.scale.z = 0.0
            tether.color = ColorRGBA(r=r_color, g=g_color, b=b_color, a=0.8)
            tether.points = [
                Point(x=float(ee_pos[0]), y=float(ee_pos[1]), z=float(ee_pos[2])),
                Point(x=float(ref_pos[0]), y=float(ref_pos[1]), z=float(ref_pos[2])),
            ]
            return tether

        t_r = build_tether(self.ref_pos_right, self.ee_pos_right, 0)
        t_l = build_tether(self.ref_pos_left, self.ee_pos_left, 1)
        if t_r is not None:
            markers.markers.append(t_r)
        else:
            self._delete_marker_group("teleop_tether", start_id=0, count=1, pub=self.teleop_pub)
        if t_l is not None:
            markers.markers.append(t_l)
        else:
            self._delete_marker_group("teleop_tether", start_id=1, count=1, pub=self.teleop_pub)

        # Publish to the dedicated teleop channel
        self.teleop_pub.publish(markers)

    def publish_debug(self, model, data, witness_info, q, q_dot, target_right, target_left, id_right, id_left, limit_buffer):
        """
        Visualizes:
        1. Collision Witness (Red Line + Distance Text)
        2. WORST Joint Limit (Arms Only, Smaller Text)
        3. End Effector Desires vs Reality (Arrows)
        4. Commanded Target Gripper (Blue)

        `witness_info` = (min_distance, (p1, p2) or None): the TRUE global
        closest collision-pair distance/points, computed ONCE per tick by
        CollisionManager.compute_softmin_jacobian (self.witness_min_distance/
        points) -- this used to be recomputed here via a second full scan of
        cdata.distanceResults, duplicating that same scan every telemetry tick.
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
        # Reused from CollisionManager.compute_softmin_jacobian's own scan of
        # cdata.distanceResults (same tick) instead of re-scanning it here.
        min_d, witness_points = witness_info
        if min_d is None or (isinstance(min_d, float) and min_d != min_d):  # NaN-safe (no pairs at all)
            min_d = 100.0
        p1_w, p2_w = witness_points if witness_points is not None else (None, None)

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

        # --- 4/5. PER-ARM GRIPPER VISUALIZATION (2026-07-01 redesign) ---
        # Each arm gets its OWN gripper marker, colored independently by its
        # OWN frozen flag (self.frozen_right / self.frozen_left, sourced from
        # main_qp_controller's ground-truth right_frozen/left_frozen state):
        #   frozen=False (actively tracking a live reference) -> BLUE, ns
        #     "qp_debug_gripper_right"/"_left"
        #   frozen=True  (CLF holding a fixed pose)            -> GREY, ns
        #     "frozen_gripper_right"/"_left"
        # This is what lets BOTH grippers render BLUE simultaneously when both
        # arms are actively driven (e.g. trajectory_generator.py publishing to
        # both /arm_right and /arm_left cartesian_reference at once), while
        # STILL rendering exactly one blue + one grey in the classic
        # single-arm teleop paradigm (the inactive arm is frozen by the CLF
        # hold). Each arm ALWAYS occupies exactly one of the two mutually
        # exclusive namespaces at a time; the other is explicitly DELETEd so
        # switching states never leaves a stale marker on screen.
        #
        # Fixed ids (0..2) per namespace -- see the historical bug-fix note
        # this replaced: a shifting id (from the shared conditional `idx`
        # counter above) meant RViz never overwrote the previous marker,
        # leaving permanent ghosts. Four DISTINCT namespaces here (two per
        # arm) can never collide with each other or with "qp_debug".
        def render_arm_gripper(ref_pos, ref_rot, frozen, side):
            """Add the correct gripper markers (blue or grey) for one arm
            directly into the `markers` MarkerArray, plus DELETE markers for
            the alternate namespace — all in the SAME array, so everything
            goes out in a SINGLE publish (no separate _delete_marker_group
            calls that would flood the topic with extra publishes and cause
            RViz subscriber-queue starvation / visual lag)."""
            blue_ns = f"qp_debug_gripper_{side}"
            grey_ns = f"frozen_gripper_{side}"

            def _inline_delete(ns, start_id, count):
                for i in range(count):
                    m = Marker()
                    m.header.frame_id = self.ref_frame
                    m.header.stamp = timestamp
                    m.ns = ns
                    m.id = start_id + i
                    m.action = Marker.DELETE
                    markers.markers.append(m)

            if ref_pos is None or ref_rot is None:
                _inline_delete(blue_ns, 0, 3)
                _inline_delete(grey_ns, 0, 3)
                return
            if frozen:
                grey_markers = self._build_gripper(
                    ref_pos, ref_rot, 0.5, 0, timestamp,
                    color=ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.5), ns=grey_ns)
                markers.markers.extend(grey_markers)
                _inline_delete(blue_ns, 0, 3)
            else:
                blue_markers = self._build_gripper(ref_pos, ref_rot, 0.8, 0, timestamp, ns=blue_ns)
                markers.markers.extend(blue_markers)
                _inline_delete(grey_ns, 0, 3)

        render_arm_gripper(self.ref_pos_right, self.ref_rot_right, self.frozen_right, "right")
        render_arm_gripper(self.ref_pos_left, self.ref_rot_left, self.frozen_left, "left")
            
        # =========================================================
        # --- 5. PINHOLE TASK ENVIRONMENT (STATIC OBSTACLES) ---
        # =========================================================
        # Broadcasts exact CBF geometric primitives to RViz for operator transparency
        
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

        # --- 2. Red Cylinder (Right Hand Manipulandum) ---
        # REMOVED: drawn by visualization_engine.py at its live collision pose
        # (grey when grasped, red otherwise). Keeping a static copy here
        # creates a phantom duplicate on the table after grasp.

        # --- 3. Blue Cylinder (Left Hand Manipulandum) ---
        # REMOVED: same reason as above.

        # --- 4. Target Placement Area(s) (flat disks on the table) ---
        # Draw EVERY platform of the loaded world (multi-platform), each a very
        # flat cylinder slightly above the table. Colors cycle so distinct disks
        # are distinguishable and match the Gazebo scheme (1st yellow, 2nd green,
        # ...). Falls back to the legacy single centered yellow disk when no world
        # scene was provided. (The periodic DELETEALL sweep clears any stale disk
        # from a previous world, so switching worlds never leaves an orphan.)
        _platform_rgba = [
            ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.6),   # yellow
            ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.6),   # green
            ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.6),   # cyan
            ColorRGBA(r=0.8, g=0.0, b=0.8, a=0.6),   # magenta
        ]
        _platforms = (self.world_scene.platforms
                      if (self.world_scene is not None
                          and getattr(self.world_scene, 'platforms', None))
                      else None)
        if _platforms:
            for pi, p in enumerate(_platforms):
                markers.markers.append(create_cylinder_marker(
                    idx, p.pose, float(p.radius), max(float(p.thickness), 0.004),
                    _platform_rgba[pi % len(_platform_rgba)]))
                idx += 1
        else:
            markers.markers.append(create_cylinder_marker(
                idx, [1.000, 0.0, 0.701], 0.15, 0.002, _platform_rgba[0]))
            idx += 1

        self.pub.publish(markers)