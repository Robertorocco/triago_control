import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from visualization_msgs.msg import Marker
import pinocchio as pin
import numpy as np
import time
import quadprog 
from rcl_interfaces.srv import GetParameters
import tempfile
import os
from controller_manager_msgs.srv import SwitchController, ListControllers
from geometry_msgs.msg import TwistStamped, Point
try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl

from urdf_parser_py.urdf import URDF

# --- CONFIGURATION ---
HEAD_CONTROLLER = "arm_head_joint_space_controller_vel"

# Kinematic Chain (Must match URDF names)
HEAD_CHAIN = ['arm_head_1_link', 'arm_head_2_link', 'arm_head_3_link',
              'arm_head_4_link', 'arm_head_5_link', 'arm_head_6_link', 'arm_head_7_link']
HEAD_JOINTS = ['arm_head_1_joint', 'arm_head_2_joint', 'arm_head_3_joint',
               'arm_head_4_joint', 'arm_head_5_joint', 'arm_head_6_joint', 'arm_head_7_joint']

# Frames to Track
CAMERA_FRAME = 'gripper_head_camera_rgbd_color_optical_frame'
RIGHT_HAND_FRAME = 'arm_right_tool_link'
LEFT_HAND_FRAME  = 'arm_left_tool_link'

# Camera Intrinsics (e.g., 720p RealSense D405 approximations)
CAM_W, CAM_H = 1280, 720
CAM_FX, CAM_FY = 640.0, 640.0
CAM_CX, CAM_CY = 640.0, 360.0

# Target Pixel Center and Depth
TARGET_U = CAM_CX
TARGET_V = CAM_CY
TARGET_Z = 1.0  # Keep centroid 1 meter away

# QP Weights and Gains
LAMBDA_VISUAL = 1     # CLF Tracking Gain
GAMMA_FOV = 5.0          # CBF FOV Safety Gain
GAMMA_JOINT = 10.0       # CBF Joint Limit Gain

# Pixels are big numbers (e.g. 100^2 = 10000). Meters are tiny (e.g. 0.1^2 = 0.01).
# If we want the controller to treat 1cm (10^-2) error in depth as 10 pixel error in the image
#(Same modification could be applied to LAMDA_VISUAL)
W_SLACK_PIXELS = 1
W_SLACK_DEPTH = W_SLACK_PIXELS* 1e4 #SHould be 1e6 but  scared of numerical instability

FOV_MARGIN = 50.0        # Pixels from the edge to trigger CBF

class VisualServoingHead(Node):
    def __init__(self):
        super().__init__('qp_head_visual_servo')

        # --- Environment & Collision (Stored state) ---
        self.wall_size = [1.0, 0.02, 2.0]
        self.wall_pos = [0.5, 0.0, 1.0]

        # --- State Tracking ---
        self.joint_name_to_idx = {}
        self.is_ready = False

        # --- ROS 2 Interfaces ---
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
       # Publish Joint Commands
        self.pub_head_cmd = self.create_publisher(Float64MultiArray, f'/{HEAD_CONTROLLER}/joint_velocity_cmd', 10)

        
        # Plotter & Debug Telemetry
        self.pub_qdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_err', 10)
        self.pub_visual_err = self.create_publisher(Float64MultiArray, '/qp_debug/xdot_err', 10)
        self.pub_wall_marker = self.create_publisher(Marker, '/qp_debug/virtual_wall_marker', 10)
        
        # NEW: Cartesian Command Debugger
        self.pub_cartesian_cmd = self.create_publisher(TwistStamped, '/qp_debug/head_cartesian_cmd', 10)

        # NEW: Automatically switch controllers on boot
        self.check_and_switch_controllers()

        # --- RViz Visualizers ---
        self.pub_ray = self.create_publisher(Marker, '/qp_debug/camera_ray', 10)
        self.pub_centroid = self.create_publisher(Marker, '/qp_debug/target_centroid', 10)

    def check_and_switch_controllers(self):
        """Automatically activates the velocity controller and deactivates the trajectory controller."""
        self.get_logger().info("Checking ROS 2 Controller Manager states...")
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        
        if not list_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller manager service not available!")
            return

        future = list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        active_controllers = [c.name for c in response.controller if c.state == 'active']        
        to_start = []
        to_stop = []

        if HEAD_CONTROLLER not in active_controllers:
            to_start.append(HEAD_CONTROLLER)
        
        # The conflicting default trajectory controller
        conflict = "arm_head_controller"
        if conflict in active_controllers:
            to_stop.append(conflict)

        if not to_start and not to_stop:
            self.get_logger().info("Controllers are already in the correct state.")
            return

        self.get_logger().info(f"Switching Controllers -> START: {to_start}, STOP: {to_stop}")
        
        switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        switch_client.wait_for_service()
        
        req = SwitchController.Request()
        req.activate_controllers = to_start
        req.deactivate_controllers = to_stop
        req.strictness = SwitchController.Request.STRICT
        
        future = switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().ok:
            self.get_logger().info("Successfully switched controllers!")
        else:
            self.get_logger().error("Failed to switch controllers.")

    def get_urdf(self):
        """Fetches the URDF string dynamically from the robot_state_publisher."""
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=2.0): 
            self.get_logger().error("Robot state publisher not available!")
            return None
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value

    def setup_pinocchio(self, urdf_path):
        # 1. Build Pinocchio model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 2. Setup neutral config and velocities
        self.q_real = pin.neutral(self.model)
        self.v_real = np.zeros(self.model.nv)
        
        # ==========================================================
        # 3. OFFICIAL URDF PARSER: Extract Soft Limits
        # ==========================================================
        self.soft_limits = {}
        try:
            from urdf_parser_py.urdf import URDF
            robot_urdf = URDF.from_xml_file(urdf_path)
            
            for joint in robot_urdf.joints:
                if joint.safety_controller is not None:
                    soft_min = joint.safety_controller.soft_lower_limit
                    soft_max = joint.safety_controller.soft_upper_limit
                    
                    if soft_min is not None and soft_max is not None:
                        self.soft_limits[joint.name] = (float(soft_min), float(soft_max))
                        
            self.get_logger().info(f"Loaded {len(self.soft_limits)} soft limits via urdf_parser_py!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse soft limits from URDF: {e}")
            
        # ==========================================================
        # 4. RESTORED: Map Head Joints to Pinocchio Indices
        # ==========================================================
        self.nq_head = len(HEAD_JOINTS)
        self.head_q_idx = []
        self.head_v_idx = []
        
        for name in HEAD_JOINTS:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.head_q_idx.append(self.model.joints[joint_id].idx_q)
                self.head_v_idx.append(self.model.joints[joint_id].idx_v)
            else:
                self.get_logger().error(f"[FATAL] Joint {name} not found in URDF!")

    def build_collision_model(self):
        """ Stripped down collision model: Only Head Links vs Wall/Base """
        self.cmodel = pin.GeometryModel()
        self.head_geom_ids = []

        # 1. Base / Torso
        body_parts = [("base_link", [0.6, 0.5, 0.27], [0.0, 0.0, 0.09]),
                      ("torso_lift_link", [0.2, 0.2, 0.6], [0.0, 0.0, 0.25])]
        self.body_geom_ids = []
        for name, dims, offset in body_parts:
            if self.model.existBodyName(name) or self.model.existFrame(name):
                fid = self.model.getFrameId(name) if self.model.existFrame(name) else self.model.getBodyId(name)
                pid = self.model.frames[fid].parentJoint
                placement = self.model.frames[fid].placement * pin.SE3(np.eye(3), np.array(offset))
                obj = pin.GeometryObject(f"col_{name}", pid, placement, hppfcl.Box(*dims))
                self.body_geom_ids.append(self.cmodel.addGeometryObject(obj))

        # 2. Head Links
        for link in HEAD_CHAIN:
            if self.model.existFrame(link):
                fid = self.model.getFrameId(link)
                pid = self.model.frames[fid].parentJoint
                placement = self.model.frames[fid].placement
                # Simple capsule for head links
                obj = pin.GeometryObject(f"col_{link}", pid, placement, hppfcl.Capsule(0.08, 0.2))
                self.head_geom_ids.append(self.cmodel.addGeometryObject(obj))

        # 3. Virtual Wall
        wall_shape = hppfcl.Box(self.wall_size[0], self.wall_size[1], self.wall_size[2])
        wall_pose = pin.SE3(np.eye(3), np.array(self.wall_pos))
        wall_geom = pin.GeometryObject("virtual_wall", 0, wall_pose, wall_shape)
        self.wall_id = self.cmodel.addGeometryObject(wall_geom)

        # Generate Pairs: Head vs Body & Head vs Wall
        for hid in self.head_geom_ids:
            for bid in self.body_geom_ids:
                self.cmodel.addCollisionPair(pin.CollisionPair(hid, bid))
            self.cmodel.addCollisionPair(pin.CollisionPair(hid, self.wall_id))

        self.cdata = self.cmodel.createData()
        for req in self.cdata.distanceRequests:
            req.enable_nearest_points = True

    def joint_cb(self, msg):
        """Updates Pinocchio state with any incoming joints, handling split ROS 2 messages."""
        
        # --- NEW: Race Condition Shield ---
        if not hasattr(self, 'model'):
            return  # Ignore messages until the URDF is downloaded and model is built!
        # ----------------------------------
        
        # 1. Unconditionally save every joint in this specific message
        for i, name in enumerate(msg.name):
            self.joint_name_to_idx[name] = True  
            if self.model.existJointName(name):
                q_idx = self.model.joints[self.model.getJointId(name)].idx_q
                self.q_real[q_idx] = msg.position[i]

        # 2. Check if we have received the head joints at least once to start the solver
        if not self.is_ready:
            missing_head = [j for j in HEAD_JOINTS if j not in self.joint_name_to_idx]
            if len(missing_head) == 0:
                self.is_ready = True

    def get_interaction_matrix(self, u, v, Z):
        """ Computes the 3x6 Interaction Matrix Ls mapping spatial camera velocity to (u_dot, v_dot, Z_dot) """
        x = (u - CAM_CX) / CAM_FX
        y = (v - CAM_CY) / CAM_FY
        Ls = np.zeros((3, 6))
        
        # du/dt
        Ls[0, 0] = -CAM_FX / Z
        Ls[0, 2] = CAM_FX * x / Z
        Ls[0, 3] = CAM_FX * x * y
        Ls[0, 4] = -CAM_FX * (1 + x**2)
        Ls[0, 5] = CAM_FX * y
        # dv/dt
        Ls[1, 1] = -CAM_FY / Z
        Ls[1, 2] = CAM_FY * y / Z
        Ls[1, 3] = CAM_FY * (1 + y**2)
        Ls[1, 4] = -CAM_FY * x * y
        Ls[1, 5] = -CAM_FY * x
        # dZ/dt
        Ls[2, 2] = -1.0
        Ls[2, 3] = -y * Z
        Ls[2, 4] = x * Z
        
        return Ls

    def solve_and_publish(self):
        if not self.is_ready: 
            return

        # 1. Update Kinematics
        pin.forwardKinematics(self.model, self.data, self.q_real)
        pin.updateFramePlacements(self.model, self.data)

        fid_cam = self.model.getFrameId(CAMERA_FRAME)
        fid_r   = self.model.getFrameId(RIGHT_HAND_FRAME)
        fid_l   = self.model.getFrameId(LEFT_HAND_FRAME)

        T_cam = self.data.oMf[fid_cam]
        T_r   = self.data.oMf[fid_r]
        T_l   = self.data.oMf[fid_l]

        # 2. Transform Hands to Camera Frame
        P_r_cam = T_cam.inverse().act(T_r.translation)
        P_l_cam = T_cam.inverse().act(T_l.translation)
        # --- NEW DEBUG BLOCK ---
        self.get_logger().info(f"--- PINOCCHIO TF DEBUG ---", throttle_duration_sec=2.0)
        self.get_logger().info(f"Global Camera Pos: {np.round(T_cam.translation, 2)}", throttle_duration_sec=2.0)
        self.get_logger().info(f"Global Right Hand: {np.round(T_r.translation, 2)}", throttle_duration_sec=2.0)
        self.get_logger().info(f"Camera Z-Axis (Forward): {np.round(T_cam.rotation[:, 2], 2)}", throttle_duration_sec=2.0)
        self.get_logger().info(f"Local Hand in Cam: {np.round(P_r_cam, 2)}", throttle_duration_sec=2.0)
        # -----
        
        # Centroid in 3D Camera Frame
        P_c_cam = (P_r_cam + P_l_cam) / 2.0


        # 3. Determine FOV State
        in_fov = False
        
        # Only calculate pixels if hands are physically in front of the lens (Z > 0.05)
        if P_r_cam[2] > 0.05 and P_l_cam[2] > 0.05 and P_c_cam[2] > 0.05:
            
            # Calculate pixel projections for BOTH hands
            u_r = CAM_FX * (P_r_cam[0] / P_r_cam[2]) + CAM_CX
            v_r = CAM_FY * (P_r_cam[1] / P_r_cam[2]) + CAM_CY
            
            u_l = CAM_FX * (P_l_cam[0] / P_l_cam[2]) + CAM_CX
            v_l = CAM_FY * (P_l_cam[1] / P_l_cam[2]) + CAM_CY
            
            # Calculate centroid for Phase 2 tracking
            u_c = CAM_FX * (P_c_cam[0] / P_c_cam[2]) + CAM_CX
            v_c = CAM_FY * (P_c_cam[1] / P_c_cam[2]) + CAM_CY
            
            # Ensure BOTH hands are strictly inside the CBF safe margin!
            # (If the screen is 640x480 and margin is 50, they must be between 50 and 590)
            in_fov = (
                FOV_MARGIN < u_r < (CAM_W - FOV_MARGIN) and
                FOV_MARGIN < u_l < (CAM_W - FOV_MARGIN) and
                FOV_MARGIN < v_r < (CAM_H - FOV_MARGIN) and
                FOV_MARGIN < v_l < (CAM_H - FOV_MARGIN)
            )

        # Common QP Setup
        J_cam_full = pin.computeFrameJacobian(self.model, self.data, self.q_real, fid_cam, pin.ReferenceFrame.LOCAL)
        J_cam = J_cam_full[:, self.head_v_idx]
        
        # QP Variables: [dq_head (7), slack (3)]
        n_vars = self.nq_head + 3
        H = np.eye(n_vars)
        
        # 1. DAMPING FACTOR: Penalize joint velocities to make the arm move heavier/slower
        W_QDOT = 2.0  # <--- INCREASE THIS NUMBER TO DAMPEN SPEED (e.g. 5.0, 10.0, 50.0)
        # Instead of H[:7, :7] = np.eye(7) * W_QDOT
        # We assign custom weights to each joint!
        weights = np.array([50.0, 40.0, 30.0, 10.0, 5.0, 1.0, 1.0]) 
        H[:7, :7] = np.diag(weights)
        
        # ==========================================================
        # 2.5D VISUAL SERVOING: SLACK WEIGHTING MATRIX (W)
        # ==========================================================
        
        
        H[7, 7] = W_SLACK_PIXELS  # Slack for u-pixel error
        H[8, 8] = W_SLACK_PIXELS  # Slack for v-pixel error
        H[9, 9] = W_SLACK_DEPTH   # Slack for Cartesian Z-depth error
        # ==========================================================
        g = np.zeros(n_vars)

        # ==========================================================
        # SECONDARY POSTURAL TASK (Null-Space Projection)
        # ==========================================================
        K_POSTURE = 0.05  # Increased slightly now that it scales correctly
        
        dq_posture = np.zeros(self.nq_head)
        
        for i, joint_name in enumerate(HEAD_JOINTS):
            # CRITICAL FIX: Use idx_q to read positions and limits!
            joint_id = self.model.getJointId(joint_name)
            idx_q = self.model.joints[joint_id].idx_q
            
            q_i = self.q_real[idx_q]
            # Fetch hard limits, but override with soft limits if they exist
            if joint_name in getattr(self, 'soft_limits', {}):
                q_min = self.soft_limits[joint_name][0]
                q_max = self.soft_limits[joint_name][1]
                #print("Parsed soft limits for {}: [{:.2f}, {:.2f}]".format(joint_name, q_min, q_max))

            else:
                q_max = self.model.upperPositionLimit[idx_q]
                q_min = self.model.lowerPositionLimit[idx_q]
        
            
            if (q_max - q_min) > 0.01 and q_max < 100.0 and q_min > -100.0:
                q_center = (q_max + q_min) / 2.0
                dq_posture[i] = -K_POSTURE * (q_i - q_center)
            
            
        # CRITICAL FIX: Scale the g vector by H so the posture task isn't overpowered by damping!
        g[:7] = H[:7, :7] @ dq_posture
        
        C, b = [], []

        # ==========================================================
        # CONTROL STATE MACHINE (Unchanged)
        # ==========================================================
        if in_fov:
            # --- STAGE 2: IBVS (Pixel-Based Visual Servoing) ---
            self.get_logger().info("Target in FOV. Active: Pixel IBVS Controller.", throttle_duration_sec=1.0)
            
            e_visual = np.array([u_c - TARGET_U, v_c - TARGET_V, P_c_cam[2] - TARGET_Z])
            Ls_centroid = self.get_interaction_matrix(u_c, v_c, P_c_cam[2])
            J_task = Ls_centroid @ J_cam

            A_eq = np.zeros((3, n_vars))
            A_eq[:, :7] = J_task
            A_eq[:, 7:] = -np.eye(3)
            b_eq = -LAMBDA_VISUAL * e_visual

            def add_fov_cbf(u, v, Z):
                Ls_hand = self.get_interaction_matrix(u, v, Z)
                grad_u = Ls_hand[0, :] @ J_cam
                grad_v = Ls_hand[1, :] @ J_cam
                
                C.append(np.concatenate([grad_u, np.zeros(3)])); b.append(-GAMMA_FOV * (u - FOV_MARGIN))
                C.append(np.concatenate([-grad_u, np.zeros(3)])); b.append(-GAMMA_FOV * ((CAM_W - FOV_MARGIN) - u))
                C.append(np.concatenate([grad_v, np.zeros(3)])); b.append(-GAMMA_FOV * (v - FOV_MARGIN))
                C.append(np.concatenate([-grad_v, np.zeros(3)])); b.append(-GAMMA_FOV * ((CAM_H - FOV_MARGIN) - v))

            u_r = CAM_FX * (P_r_cam[0] / P_r_cam[2]) + CAM_CX
            v_r = CAM_FY * (P_r_cam[1] / P_r_cam[2]) + CAM_CY
            u_l = CAM_FX * (P_l_cam[0] / P_l_cam[2]) + CAM_CX
            v_l = CAM_FY * (P_l_cam[1] / P_l_cam[2]) + CAM_CY
            add_fov_cbf(u_r, v_r, P_r_cam[2])
            add_fov_cbf(u_l, v_l, P_l_cam[2])
            
            pub_error = e_visual

        else:
            # --- STAGE 1: PBVS (3D Rotational Look-At) ---
            self.get_logger().warn(f"Target LOST. Active: 3D Rotational Look-At. (Z={P_c_cam[2]:.2f}m)", throttle_duration_sec=1.0)
            
            P_norm = np.linalg.norm(P_c_cam)
            if P_norm > 0.01:
                dir_vec = P_c_cam / P_norm
            else:
                dir_vec = np.array([0.0, 0.0, 1.0])

            z_axis = np.array([0.0, 0.0, 1.0])
            omega_des = LAMBDA_VISUAL * np.cross(z_axis, dir_vec)

            if dir_vec[2] < -0.95: 
                omega_des[1] = LAMBDA_VISUAL  

            J_rot = J_cam[3:, :]

            A_eq = np.zeros((3, n_vars))
            A_eq[:, :7] = J_rot
            A_eq[:, 7:] = -np.eye(3)
            b_eq = omega_des
            
            pub_error = omega_des
            
        # ==========================================================
        # JOINT LIMITS, VELOCITY LIMITS & SOLVE (Always Active)
        # ==========================================================
        MAX_VELOCITY = 0.15  # rad/s (Super slow and safe)

        for i, joint_name in enumerate(HEAD_JOINTS):
            # CRITICAL FIX: Use idx_q to read positions and limits!
            joint_id = self.model.getJointId(joint_name)
            idx_q = self.model.joints[joint_id].idx_q
            
            q_i = self.q_real[idx_q]
            # Fetch hard limits, but override with soft limits if they exist
            if joint_name in getattr(self, 'soft_limits', {}):
                q_min = self.soft_limits[joint_name][0]
                q_max = self.soft_limits[joint_name][1]
            else:
                q_max = self.model.upperPositionLimit[idx_q]
                q_min = self.model.lowerPositionLimit[idx_q]
        
            
            if (q_max - q_min) < 0.01 or q_max > 100.0 or q_min < -100.0:
                upper_bound = MAX_VELOCITY
                lower_bound = -MAX_VELOCITY
            else:
                range_total = q_max - q_min
                SAFE_BUF = min(0.15, range_total * 0.1) 
                LOCAL_GAMMA = 2.0 
                
                v_req_upper = LOCAL_GAMMA * (q_max - q_i - SAFE_BUF)
                v_req_lower = -LOCAL_GAMMA * (q_i - q_min - SAFE_BUF)
                
                upper_bound = min(MAX_VELOCITY, v_req_upper)
                lower_bound = max(-MAX_VELOCITY, v_req_lower)
                
                if lower_bound >= upper_bound:
                    midpoint = (upper_bound + lower_bound) / 2.0
                    upper_bound = midpoint + 0.01
                    lower_bound = midpoint - 0.01

            row_upper = np.zeros(n_vars); row_upper[i] = -1.0
            C.append(row_upper); b.append(-upper_bound)
            
            row_lower = np.zeros(n_vars); row_lower[i] = 1.0
            C.append(row_lower); b.append(lower_bound)

        C_mat = np.array(C).T if len(C) > 0 else np.zeros((n_vars, 0))
        b_vec = np.array(b) if len(b) > 0 else np.zeros(0)
        
        try:
            res = quadprog.solve_qp(H, g, np.hstack((A_eq.T, C_mat)), np.hstack((b_eq, b_vec)), meq=3)
            dq_opt = res[0][:7]
        except ValueError:
            self.get_logger().warn("[QP] Infeasible! Handing zero velocity.", throttle_duration_sec=0.5)
            dq_opt = np.zeros(7)

        
        # Publish Telemetry for Plotter
        self.pub_visual_err.publish(Float64MultiArray(data=pub_error.tolist()))
        self.pub_qdot_err.publish(Float64MultiArray(data=dq_opt.tolist()))

        # --- NEW: DEBUG CARTESIAN COMMAND ---
        # Map the chosen joint velocities back to Cartesian space to see what the head is doing
        v_cam_cmd = J_cam_full[:, self.head_v_idx] @ dq_opt 
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = CAMERA_FRAME
        twist_msg.twist.linear.x = float(v_cam_cmd[0])
        twist_msg.twist.linear.y = float(v_cam_cmd[1])
        twist_msg.twist.linear.z = float(v_cam_cmd[2])
        twist_msg.twist.angular.x = float(v_cam_cmd[3])
        twist_msg.twist.angular.y = float(v_cam_cmd[4])
        twist_msg.twist.angular.z = float(v_cam_cmd[5])
        self.pub_cartesian_cmd.publish(twist_msg)

       # Publish Joint Commands
        msg_cmd = Float64MultiArray()
        msg_cmd.data = [float(x) for x in dq_opt]
        self.pub_head_cmd.publish(msg_cmd)

        # ==========================================================
        # RVIZ DEBUG VISUALIZERS
        # ==========================================================
        # 1. Publish the Target Centroid (Bright Green Dot)
        msg_dot = Marker()
        msg_dot.header.frame_id = CAMERA_FRAME
        msg_dot.header.stamp.sec = 0
        msg_dot.header.stamp.nanosec = 0
        msg_dot.ns = "centroid"
        msg_dot.id = 0
        msg_dot.type = Marker.SPHERE
        msg_dot.action = Marker.ADD
        msg_dot.pose.position.x = float(P_c_cam[0])
        msg_dot.pose.position.y = float(P_c_cam[1])
        msg_dot.pose.position.z = float(P_c_cam[2])
        msg_dot.scale.x = 0.08  # 8cm sphere
        msg_dot.scale.y = 0.08
        msg_dot.scale.z = 0.08
        msg_dot.color.r = 0.0
        msg_dot.color.g = 1.0
        msg_dot.color.b = 0.0
        msg_dot.color.a = 1.0   # Solid color
        self.pub_centroid.publish(msg_dot)

        # 2. Publish the Optical Axis Ray (Semi-transparent Green Arrow)
        msg_ray = Marker()
        msg_ray.header.frame_id = CAMERA_FRAME
        msg_ray.header.stamp.sec = 0
        msg_ray.header.stamp.nanosec = 0
        msg_ray.ns = "optical_ray"
        msg_ray.id = 1
        msg_ray.type = Marker.ARROW
        msg_ray.action = Marker.ADD
        
        # Arrow starts at the lens (0,0,0) and shoots 2 meters forward along Z
        p_start = Point(); p_start.x = 0.0; p_start.y = 0.0; p_start.z = 0.0
        p_end = Point(); p_end.x = 0.0; p_end.y = 0.0; p_end.z = 2.0
        msg_ray.points = [p_start, p_end]
        
        msg_ray.scale.x = 0.01  # Shaft diameter
        msg_ray.scale.y = 0.03  # Arrow-head diameter
        msg_ray.scale.z = 0.05  # Arrow-head length
        msg_ray.color.r = 0.0
        msg_ray.color.g = 1.0
        msg_ray.color.b = 0.0
        msg_ray.color.a = 0.5   # 50% transparent so it doesn't block the view
        self.pub_ray.publish(msg_ray)
        # ==========================================================

def main():
    rclpy.init()
    node = VisualServoingHead()
    
    # --- PHASE 1: SETUP ---
    print("[Main] Fetching URDF from robot_state_publisher...")
    urdf_str = node.get_urdf()
    
    if urdf_str is None:
        print("[Error] Could not fetch URDF. Exiting.")
        return
        
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
        f.write(urdf_str)
        urdf_path = f.name
    
    print("[Main] Building Pinocchio Kinematic & Collision Models...")
    node.setup_pinocchio(urdf_path)
    
    # Optional: Clean up the temp file
    os.remove(urdf_path)
    print("[Main] Setup Complete! Starting visual servoing loop...")

    # --- PHASE 2: CONTROL LOOP ---
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            node.solve_and_publish()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()