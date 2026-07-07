#!/usr/bin/env python3
"""
head_active_arm_tracking.py — Head controller that tracks the ACTIVE teleop arm.

CONTEXT (see .kiro/context.md §7)
--------------------------------
During teleoperation exactly one arm is "active" at a time (the operator drives
it; the other is frozen). The `main_shared_autonomy.py` node publishes the
currently active side on `/shared_autonomy/active_arm` (std_msgs/String
"right"/"left"). This node consumes that signal and steers the head camera so
that:

  1. CENTERING  — the active arm's end-effector is kept in the centre of the
                  camera field of view (optical axis pointed at the hand).
  2. STAND-OFF  — the camera keeps a fixed working distance of ~0.5 m from the
                  active hand (close enough for a useful close-up, far enough to
                  see the whole gripper + a grasp target).

No perception / image recognition is used: the hand pose is obtained purely from
the robot's forward kinematics (Pinocchio FK of the arm chain). The controller
just points the optical axis using that kinematic knowledge.

This node is a sibling of `qp_head_visual_servo.py` and shares its structure
(independent QP, own velocity controller, no shared state with the arm QP,
head chain is NOT part of the arm decision vector). The differences:

  * It tracks a SINGLE point (the active arm tool link) instead of the centroid
    of both hands.
  * The desired stand-off distance is TARGET_DISTANCE (0.5 m) instead of 1.0 m.
  * The active side is selected live from /shared_autonomy/active_arm.
  * A live Matplotlib plotting THREAD (in this same file) reports tracking
    performance: pixel-centering error, angular-centering error, and the
    stand-off distance held vs. the 0.5 m target.

CONTROL (2.5D visual servoing, kinematic target)
------------------------------------------------
Decision vector: x = [dq_head (7), slack (4)].

Two-stage state machine (identical philosophy to qp_head_visual_servo.py):
  * PBVS (Look-At): hand behind camera / outside FOV -> 3D rotational servoing,
    rotate the optical axis onto the direction of the hand.
  * IBVS (2.5D):    hand inside FOV -> interaction matrix maps pixel + depth
    error to a camera twist. This channel simultaneously centres the hand
    (u,v -> image centre) AND regulates the stand-off (Z -> TARGET_DISTANCE).

HORIZON LEVELING (roll about the optical axis)
----------------------------------------------
Point tracking alone leaves the camera roll unconstrained (rotating a centred
point about the optical axis is in the null space of the u,v,Z task), so the
head could drift to any roll — including a fully inverted, upside-down view.
A dedicated roll task removes that freedom: world-up expressed in the camera
frame must point along -Y_cam (image "up"). The signed roll error is
`phi = atan2(up_cam_x, -up_cam_y)` (0 = upright, +/-pi = inverted); it is
regulated by commanding `wz = K_ROLL * phi` about the optical axis — a 4th
equality DOF in IBVS and the roll component of the look-at twist in PBVS. A
gimbal guard disables it when the optical axis is near-vertical (roll undefined).

RUN:
    ros2 run triago_control head_active_arm_tracking.py
    ros2 run triago_control head_active_arm_tracking.py --ros-args -p plot:=false
"""

import os
import tempfile
import threading
import time
from collections import deque

import numpy as np
import quadprog

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import TwistStamped, Point
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import SwitchController, ListControllers

import pinocchio as pin

try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl

# =============================================================================
# CONFIGURATION
# =============================================================================
HEAD_CONTROLLER = "arm_head_joint_space_controller_vel"

# Kinematic Chain (must match URDF names)
HEAD_JOINTS = ['arm_head_1_joint', 'arm_head_2_joint', 'arm_head_3_joint',
               'arm_head_4_joint', 'arm_head_5_joint', 'arm_head_6_joint',
               'arm_head_7_joint']

# Frames
CAMERA_FRAME = 'gripper_head_camera_rgbd_color_optical_frame'
ARM_FRAMES = {
    'right': 'arm_right_tool_link',
    'left':  'arm_left_tool_link',
}

# Camera intrinsics (720p RealSense-style approximation, matches visual servo)
CAM_W, CAM_H = 1280, 720
CAM_FX, CAM_FY = 640.0, 640.0
CAM_CX, CAM_CY = 640.0, 360.0

# Targets: keep the active hand at the image centre, TARGET_DISTANCE away.
TARGET_U = CAM_CX
TARGET_V = CAM_CY
TARGET_DISTANCE = 0.5   # [m] desired stand-off from the active hand (50 cm)

# QP gains
LAMBDA_VISUAL = 1.0     # CLF tracking gain (pixel + depth convergence rate)
GAMMA_FOV = 5.0         # FOV barrier (CBF) gain
K_ROLL = 1.0            # horizon-leveling servo gain (roll about the optical axis)
MAX_VELOCITY = 0.15     # [rad/s] per-head-joint velocity cap (slow & safe)

# Below this in-plane magnitude of world-up, the optical axis is ~vertical and
# roll is geometrically ill-defined -> the leveling task is disabled (gimbal).
ROLL_GIMBAL_EPS = 0.1

# Slack weights: pixels are O(100), metres are O(0.1). Scale the depth slack up
# so a 1 cm depth error is treated comparably to a ~100 px image error.
W_SLACK_PIXELS = 1.0
W_SLACK_DEPTH = W_SLACK_PIXELS * 1e4
# Roll (rad/s) is a firm secondary task: heavy enough that the QP levels the
# horizon using the null-space DOF, but the pixel/depth centering stays primary.
W_SLACK_ROLL = 100.0

FOV_MARGIN = 50.0       # [px] distance from the image edge that triggers the CBF

# Per-joint velocity regularization (heavier on proximal/base joints so the
# head moves "heavier" from the shoulder and light at the wrist).
JOINT_WEIGHTS = np.array([50.0, 40.0, 30.0, 10.0, 5.0, 1.0, 1.0])

# Secondary postural centering spring (null-space, keeps joints mid-range).
K_POSTURE = 0.05

# Control loop rate
CONTROL_HZ = 50.0

# Plotting
PLOT_WINDOW_S = 40.0    # seconds of history shown in the live dashboard


class HeadActiveArmTracker(Node):
    def __init__(self):
        super().__init__('head_active_arm_tracking')

        # --- Parameters ---
        self.declare_parameter('plot', True)
        self.declare_parameter('target_distance', TARGET_DISTANCE)
        self.enable_plot = self.get_parameter('plot').value
        self.target_distance = float(self.get_parameter('target_distance').value)

        # --- State ---
        self.joint_name_seen = {}
        self.is_ready = False
        self.active_arm = 'right'          # default; overwritten by topic

        # --- Telemetry buffers (shared with the plotting thread via lock) ---
        self.plot_lock = threading.Lock()
        self.t0 = time.time()
        maxlen = int(PLOT_WINDOW_S * CONTROL_HZ)
        self.buf_t = deque(maxlen=maxlen)
        self.buf_u_err = deque(maxlen=maxlen)      # [px] u - TARGET_U
        self.buf_v_err = deque(maxlen=maxlen)      # [px] v - TARGET_V
        self.buf_ang_err = deque(maxlen=maxlen)    # [deg] optical-axis vs hand
        self.buf_roll_err = deque(maxlen=maxlen)   # [deg] camera roll (horizon)
        self.buf_dist = deque(maxlen=maxlen)       # [m] camera->hand distance
        self.buf_in_fov = deque(maxlen=maxlen)     # 1.0 IBVS / 0.0 PBVS
        self.buf_active = deque(maxlen=maxlen)     # 1.0 right / 0.0 left

        # --- ROS interfaces ---
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10)
        self.sub_active = self.create_subscription(
            String, '/shared_autonomy/active_arm', self.active_arm_cb, 10)

        self.pub_head_cmd = self.create_publisher(
            Float64MultiArray, f'/{HEAD_CONTROLLER}/joint_velocity_cmd', 10)

        # Telemetry topics (also usable by an external plotter if desired)
        self.pub_track_err = self.create_publisher(
            Float64MultiArray, '/head_active_tracking/error', 10)
        self.pub_qdot = self.create_publisher(
            Float64MultiArray, '/head_active_tracking/qdot', 10)
        self.pub_cartesian_cmd = self.create_publisher(
            TwistStamped, '/head_active_tracking/cartesian_cmd', 10)

        # RViz debug
        self.pub_ray = self.create_publisher(Marker, '/head_active_tracking/camera_ray', 10)
        self.pub_target = self.create_publisher(Marker, '/head_active_tracking/target', 10)

        # Auto-switch to the velocity controller
        self.check_and_switch_controllers()

    # ------------------------------------------------------------------ #
    # Controller manager
    # ------------------------------------------------------------------ #
    def check_and_switch_controllers(self):
        """Activate the head velocity controller, deactivate the trajectory one."""
        self.get_logger().info("Checking ROS 2 Controller Manager states...")
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        if not list_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Controller manager service not available!")
            return

        future = list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        active = [c.name for c in response.controller if c.state == 'active']
        to_start, to_stop = [], []
        if HEAD_CONTROLLER not in active:
            to_start.append(HEAD_CONTROLLER)
        conflict = "arm_head_controller"
        if conflict in active:
            to_stop.append(conflict)

        if not to_start and not to_stop:
            self.get_logger().info("Controllers already in the correct state.")
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

    # ------------------------------------------------------------------ #
    # URDF / Pinocchio setup
    # ------------------------------------------------------------------ #
    def get_urdf(self):
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
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_real = pin.neutral(self.model)

        # Soft limits via official URDF parser
        self.soft_limits = {}
        try:
            from urdf_parser_py.urdf import URDF
            robot_urdf = URDF.from_xml_file(urdf_path)
            for joint in robot_urdf.joints:
                if joint.safety_controller is not None:
                    lo = joint.safety_controller.soft_lower_limit
                    hi = joint.safety_controller.soft_upper_limit
                    if lo is not None and hi is not None:
                        self.soft_limits[joint.name] = (float(lo), float(hi))
            self.get_logger().info(f"Loaded {len(self.soft_limits)} soft limits via urdf_parser_py!")
        except Exception as e:
            self.get_logger().error(f"Failed to parse soft limits from URDF: {e}")

        # Map head joints to Pinocchio indices
        self.nq_head = len(HEAD_JOINTS)
        self.head_v_idx = []
        for name in HEAD_JOINTS:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                self.head_v_idx.append(self.model.joints[jid].idx_v)
            else:
                self.get_logger().error(f"[FATAL] Joint {name} not found in URDF!")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def active_arm_cb(self, msg):
        side = msg.data.strip().lower()
        if side in ARM_FRAMES and side != self.active_arm:
            self.active_arm = side
            self.get_logger().info(f"Active arm switched -> {side.upper()} "
                                   f"(tracking {ARM_FRAMES[side]})")

    def joint_cb(self, msg):
        if not hasattr(self, 'model'):
            return  # ignore until the model is built
        for i, name in enumerate(msg.name):
            self.joint_name_seen[name] = True
            if self.model.existJointName(name):
                q_idx = self.model.joints[self.model.getJointId(name)].idx_q
                self.q_real[q_idx] = msg.position[i]

        if not self.is_ready:
            missing = [j for j in HEAD_JOINTS if j not in self.joint_name_seen]
            if not missing:
                self.is_ready = True

    # ------------------------------------------------------------------ #
    # Interaction matrix
    # ------------------------------------------------------------------ #
    def get_interaction_matrix(self, u, v, Z):
        """3x6 interaction matrix Ls mapping camera spatial velocity to
        (u_dot, v_dot, Z_dot)."""
        x = (u - CAM_CX) / CAM_FX
        y = (v - CAM_CY) / CAM_FY
        Ls = np.zeros((3, 6))
        # du/dt
        Ls[0, 0] = -CAM_FX / Z
        Ls[0, 2] = CAM_FX * x / Z
        Ls[0, 3] = CAM_FX * x * y
        Ls[0, 4] = -CAM_FX * (1 + x ** 2)
        Ls[0, 5] = CAM_FX * y
        # dv/dt
        Ls[1, 1] = -CAM_FY / Z
        Ls[1, 2] = CAM_FY * y / Z
        Ls[1, 3] = CAM_FY * (1 + y ** 2)
        Ls[1, 4] = -CAM_FY * x * y
        Ls[1, 5] = -CAM_FY * x
        # dZ/dt
        Ls[2, 2] = -1.0
        Ls[2, 3] = -y * Z
        Ls[2, 4] = x * Z
        return Ls

    # ------------------------------------------------------------------ #
    # Main solve
    # ------------------------------------------------------------------ #
    def solve_and_publish(self):
        if not self.is_ready:
            return

        # 1. FK
        pin.forwardKinematics(self.model, self.data, self.q_real)
        pin.updateFramePlacements(self.model, self.data)

        fid_cam = self.model.getFrameId(CAMERA_FRAME)
        arm_frame = ARM_FRAMES[self.active_arm]
        fid_hand = self.model.getFrameId(arm_frame)

        T_cam = self.data.oMf[fid_cam]
        T_hand = self.data.oMf[fid_hand]

        # 2. Active hand in camera frame
        P_cam = T_cam.inverse().act(T_hand.translation)   # [x, y, z] optical frame
        dist = float(np.linalg.norm(P_cam))               # true camera->hand distance

        # Angular centering error: angle between optical axis (+Z) and hand dir.
        if dist > 1e-6:
            cos_ang = np.clip(P_cam[2] / dist, -1.0, 1.0)
            ang_err_deg = float(np.degrees(np.arccos(cos_ang)))
        else:
            ang_err_deg = 0.0

        # Horizon-leveling (roll about the optical axis). World-up expressed in
        # the camera frame is the 3rd row of R_cam. For an upright image it must
        # point along -Y_cam, i.e. phi = atan2(up_x, -up_y) -> 0. wz = K_ROLL*phi
        # rolls the camera back to level. Disabled when the optical axis is
        # near-vertical (in-plane component of world-up vanishes -> roll undefined).
        up_cam = np.asarray(T_cam.rotation[2, :]).flatten()
        in_plane = float(np.hypot(up_cam[0], up_cam[1]))
        if in_plane > ROLL_GIMBAL_EPS:
            phi_roll = float(np.arctan2(up_cam[0], -up_cam[1]))
            wz_desired = K_ROLL * phi_roll
        else:
            phi_roll = 0.0
            wz_desired = 0.0
        roll_err_deg = float(np.degrees(phi_roll))

        self.get_logger().info(
            f"[{self.active_arm.upper()}] hand_in_cam={np.round(P_cam, 2)} "
            f"dist={dist:.2f}m ang_err={ang_err_deg:.1f}deg roll_err={roll_err_deg:.1f}deg",
            throttle_duration_sec=2.0)

        # 3. FOV state (hand physically in front of the lens)
        in_fov = False
        u_h = v_h = None
        if P_cam[2] > 0.05:
            u_h = CAM_FX * (P_cam[0] / P_cam[2]) + CAM_CX
            v_h = CAM_FY * (P_cam[1] / P_cam[2]) + CAM_CY
            in_fov = (FOV_MARGIN < u_h < (CAM_W - FOV_MARGIN) and
                      FOV_MARGIN < v_h < (CAM_H - FOV_MARGIN))

        # 4. Common QP setup
        J_cam_full = pin.computeFrameJacobian(
            self.model, self.data, self.q_real, fid_cam, pin.ReferenceFrame.LOCAL)
        J_cam = J_cam_full[:, self.head_v_idx]

        n_vars = self.nq_head + 4   # [dq_head(7), slack(4): u, v, Z, roll]
        H = np.eye(n_vars)
        H[:7, :7] = np.diag(JOINT_WEIGHTS)
        H[7, 7] = W_SLACK_PIXELS     # u slack
        H[8, 8] = W_SLACK_PIXELS     # v slack
        H[9, 9] = W_SLACK_DEPTH      # depth slack
        H[10, 10] = W_SLACK_ROLL     # roll (horizon) slack
        g = np.zeros(n_vars)

        # Secondary postural centering spring (null-space)
        dq_posture = np.zeros(self.nq_head)
        for i, jn in enumerate(HEAD_JOINTS):
            jid = self.model.getJointId(jn)
            idx_q = self.model.joints[jid].idx_q
            q_i = self.q_real[idx_q]
            if jn in self.soft_limits:
                q_min, q_max = self.soft_limits[jn]
            else:
                q_max = self.model.upperPositionLimit[idx_q]
                q_min = self.model.lowerPositionLimit[idx_q]
            if (q_max - q_min) > 0.01 and abs(q_max) < 100.0 and abs(q_min) < 100.0:
                q_center = 0.5 * (q_max + q_min)
                dq_posture[i] = -K_POSTURE * (q_i - q_center)
        g[:7] = H[:7, :7] @ dq_posture

        C, b = [], []

        # 5. State machine
        if in_fov:
            # --- IBVS: 2.5D centering + stand-off regulation --------------
            self.get_logger().info("Active hand in FOV -> IBVS (centering + 50cm stand-off).",
                                    throttle_duration_sec=1.0)
            e_visual = np.array([u_h - TARGET_U,
                                 v_h - TARGET_V,
                                 P_cam[2] - self.target_distance])
            Ls = self.get_interaction_matrix(u_h, v_h, P_cam[2])
            J_task = Ls @ J_cam

            # 4 equality rows: [u, v, Z] centering/stand-off + roll leveling.
            # The roll row directly commands wz (angular-Z of the camera twist,
            # row 5 of the LOCAL frame Jacobian) to the leveling velocity.
            A_eq = np.zeros((4, n_vars))
            A_eq[:3, :7] = J_task
            A_eq[3, :7] = J_cam[5, :]
            A_eq[:3, 7:10] = -np.eye(3)   # u, v, Z slacks
            A_eq[3, 10] = -1.0            # roll slack
            b_eq = np.concatenate([-LAMBDA_VISUAL * e_visual, [wz_desired]])

            # FOV barriers on the tracked hand (keep it inside the frame)
            grad_u = Ls[0, :] @ J_cam
            grad_v = Ls[1, :] @ J_cam
            C.append(np.concatenate([grad_u, np.zeros(4)])); b.append(-GAMMA_FOV * (u_h - FOV_MARGIN))
            C.append(np.concatenate([-grad_u, np.zeros(4)])); b.append(-GAMMA_FOV * ((CAM_W - FOV_MARGIN) - u_h))
            C.append(np.concatenate([grad_v, np.zeros(4)])); b.append(-GAMMA_FOV * (v_h - FOV_MARGIN))
            C.append(np.concatenate([-grad_v, np.zeros(4)])); b.append(-GAMMA_FOV * ((CAM_H - FOV_MARGIN) - v_h))

            pub_error = e_visual
        else:
            # --- PBVS: rotate the optical axis onto the hand -------------
            self.get_logger().warn(
                f"Active hand outside FOV -> PBVS look-at (Z={P_cam[2]:.2f}m).",
                throttle_duration_sec=1.0)
            P_norm = np.linalg.norm(P_cam)
            dir_vec = P_cam / P_norm if P_norm > 0.01 else np.array([0.0, 0.0, 1.0])
            z_axis = np.array([0.0, 0.0, 1.0])
            omega_des = LAMBDA_VISUAL * np.cross(z_axis, dir_vec)
            if dir_vec[2] < -0.95:   # hand directly behind: break the singularity
                omega_des[1] = LAMBDA_VISUAL
            # cross(z_axis, dir) has no roll component -> inject the horizon
            # leveling command directly on the optical-axis rotation, so the
            # image is already upright when the hand re-enters the FOV.
            omega_des[2] = wz_desired

            J_rot = J_cam[3:, :]
            A_eq = np.zeros((3, n_vars))
            A_eq[:, :7] = J_rot
            A_eq[:, 7:10] = -np.eye(3)   # wx, wy, wz slacks (roll slack unused here)
            b_eq = omega_des
            pub_error = omega_des

        # 6. Joint-limit / velocity CBF (always active)
        for i, jn in enumerate(HEAD_JOINTS):
            jid = self.model.getJointId(jn)
            idx_q = self.model.joints[jid].idx_q
            q_i = self.q_real[idx_q]
            if jn in self.soft_limits:
                q_min, q_max = self.soft_limits[jn]
            else:
                q_max = self.model.upperPositionLimit[idx_q]
                q_min = self.model.lowerPositionLimit[idx_q]

            if (q_max - q_min) < 0.01 or abs(q_max) > 100.0 or abs(q_min) > 100.0:
                upper_bound = MAX_VELOCITY
                lower_bound = -MAX_VELOCITY
            else:
                range_total = q_max - q_min
                safe_buf = min(0.15, range_total * 0.1)
                local_gamma = 2.0
                v_req_upper = local_gamma * (q_max - q_i - safe_buf)
                v_req_lower = -local_gamma * (q_i - q_min - safe_buf)
                upper_bound = min(MAX_VELOCITY, v_req_upper)
                lower_bound = max(-MAX_VELOCITY, v_req_lower)
                if lower_bound >= upper_bound:
                    mid = 0.5 * (upper_bound + lower_bound)
                    upper_bound, lower_bound = mid + 0.01, mid - 0.01

            row_u = np.zeros(n_vars); row_u[i] = -1.0
            C.append(row_u); b.append(-upper_bound)
            row_l = np.zeros(n_vars); row_l[i] = 1.0
            C.append(row_l); b.append(lower_bound)

        C_mat = np.array(C).T if len(C) > 0 else np.zeros((n_vars, 0))
        b_vec = np.array(b) if len(b) > 0 else np.zeros(0)

        # 7. Solve
        try:
            res = quadprog.solve_qp(H, g, np.hstack((A_eq.T, C_mat)),
                                    np.hstack((b_eq, b_vec)), meq=A_eq.shape[0])
            dq_opt = res[0][:7]
        except ValueError:
            self.get_logger().warn("[QP] Infeasible! Commanding zero velocity.",
                                   throttle_duration_sec=0.5)
            dq_opt = np.zeros(7)

        # 8. Publish command + telemetry
        self.pub_head_cmd.publish(Float64MultiArray(data=[float(x) for x in dq_opt]))
        self.pub_track_err.publish(Float64MultiArray(data=[float(x) for x in pub_error]))
        self.pub_qdot.publish(Float64MultiArray(data=[float(x) for x in dq_opt]))

        v_cam_cmd = J_cam @ dq_opt
        tw = TwistStamped()
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.header.frame_id = CAMERA_FRAME
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = map(float, v_cam_cmd[:3])
        tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z = map(float, v_cam_cmd[3:])
        self.pub_cartesian_cmd.publish(tw)

        # 9. RViz debug markers
        self._publish_markers(P_cam)

        # 10. Push telemetry to the plotting buffers
        with self.plot_lock:
            self.buf_t.append(time.time() - self.t0)
            self.buf_u_err.append((u_h - TARGET_U) if u_h is not None else np.nan)
            self.buf_v_err.append((v_h - TARGET_V) if v_h is not None else np.nan)
            self.buf_ang_err.append(ang_err_deg)
            self.buf_roll_err.append(roll_err_deg)
            self.buf_dist.append(dist)
            self.buf_in_fov.append(1.0 if in_fov else 0.0)
            self.buf_active.append(1.0 if self.active_arm == 'right' else 0.0)

    def _publish_markers(self, P_cam):
        # Target sphere at the tracked hand (camera frame)
        m = Marker()
        m.header.frame_id = CAMERA_FRAME
        m.ns = "active_hand"; m.id = 0
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, P_cam)
        m.scale.x = m.scale.y = m.scale.z = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
        self.pub_target.publish(m)

        # Optical-axis ray
        ray = Marker()
        ray.header.frame_id = CAMERA_FRAME
        ray.ns = "optical_ray"; ray.id = 1
        ray.type = Marker.ARROW; ray.action = Marker.ADD
        p0 = Point(); p0.x = p0.y = p0.z = 0.0
        p1 = Point(); p1.x = 0.0; p1.y = 0.0; p1.z = 2.0
        ray.points = [p0, p1]
        ray.scale.x, ray.scale.y, ray.scale.z = 0.01, 0.03, 0.05
        ray.color.r, ray.color.g, ray.color.b, ray.color.a = 0.0, 1.0, 0.0, 0.5
        self.pub_ray.publish(ray)


# =============================================================================
# LIVE PLOTTING THREAD
# =============================================================================
def plotting_thread(node: HeadActiveArmTracker, stop_event: threading.Event):
    """Runs a live Matplotlib dashboard of tracking performance in its own thread.

    Shows CENTERING (pixel + angular error, target = 0) and the STAND-OFF
    DISTANCE held vs. the 0.5 m target. Fully decoupled from the control loop:
    it only reads the node's telemetry buffers under a lock, so a slow/absent
    display never stalls the controller. Any failure (e.g. no X display) is
    logged and the controller keeps running headless.
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        node.get_logger().warn(f"[Plotter] Matplotlib unavailable ({e}); running headless.")
        return

    try:
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        fig.canvas.manager.set_window_title("Head Active-Arm Tracking — Live Performance")

        ax_px = axs[0, 0]      # pixel centering error
        ax_ang = axs[0, 1]     # angular centering error
        ax_dist = axs[1, 0]    # stand-off distance
        ax_disterr = axs[1, 1] # distance error

        ax_px.set_title("Centering — Pixel Error (target = 0)")
        ax_px.set_ylabel("Error [px]"); ax_px.set_xlabel("Time [s]")
        ax_px.grid(True, alpha=0.3)
        ax_px.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax_px.axhline(FOV_MARGIN, color="lightgray", linestyle=":", linewidth=1)
        ax_px.axhline(-FOV_MARGIN, color="lightgray", linestyle=":", linewidth=1)
        line_u, = ax_px.plot([], [], "b-", linewidth=1.5, label="u error")
        line_v, = ax_px.plot([], [], "r-", linewidth=1.5, label="v error")
        ax_px.legend(loc="upper right", fontsize=8)

        ax_ang.set_title("Angular Errors — pointing (centering) + roll (horizon)")
        ax_ang.set_ylabel("Error [deg]"); ax_ang.set_xlabel("Time [s]")
        ax_ang.grid(True, alpha=0.3)
        ax_ang.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        line_ang, = ax_ang.plot([], [], "m-", linewidth=1.5, label="pointing error")
        line_roll, = ax_ang.plot([], [], "orange", linewidth=1.5, label="roll error (0=upright)")
        ax_ang.legend(loc="upper right", fontsize=8)

        ax_dist.set_title(f"Stand-off Distance vs Target ({node.target_distance:.2f} m)")
        ax_dist.set_ylabel("Distance [m]"); ax_dist.set_xlabel("Time [s]")
        ax_dist.grid(True, alpha=0.3)
        ax_dist.axhline(node.target_distance, color="gray", linestyle="--",
                        linewidth=1.5, label="target")
        line_dist, = ax_dist.plot([], [], "g-", linewidth=1.5, label="measured")
        ax_dist.legend(loc="upper right", fontsize=8)

        ax_disterr.set_title("Stand-off Error (measured - target)")
        ax_disterr.set_ylabel("Error [cm]"); ax_disterr.set_xlabel("Time [s]")
        ax_disterr.grid(True, alpha=0.3)
        ax_disterr.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        line_disterr, = ax_disterr.plot([], [], "c-", linewidth=1.5)

        fig.tight_layout()
        plt.show(block=False)

        while rclpy.ok() and not stop_event.is_set():
            with node.plot_lock:
                t = list(node.buf_t)
                ue = list(node.buf_u_err)
                ve = list(node.buf_v_err)
                ang = list(node.buf_ang_err)
                roll = list(node.buf_roll_err)
                dist = list(node.buf_dist)
                in_fov = list(node.buf_in_fov)

            if t:
                line_u.set_data(t, ue)
                line_v.set_data(t, ve)
                line_ang.set_data(t, ang)
                line_roll.set_data(t, roll)
                line_dist.set_data(t, dist)
                disterr = [(d - node.target_distance) * 100.0 for d in dist]
                line_disterr.set_data(t, disterr)

                current_t = t[-1]
                lo = max(0.0, current_t - PLOT_WINDOW_S)
                for ax in (ax_px, ax_ang, ax_dist, ax_disterr):
                    ax.set_xlim(lo, current_t + 1)
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)

                # Title annotation with live FOV/IBVS state
                mode = "IBVS" if (in_fov and in_fov[-1] > 0.5) else "PBVS (look-at)"
                arm = node.active_arm.upper()
                fig.suptitle(f"Tracking {arm} arm  |  mode: {mode}", fontsize=11)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.1)

    except Exception as e:  # pragma: no cover
        node.get_logger().warn(f"[Plotter] Plotting thread stopped: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    rclpy.init()
    node = HeadActiveArmTracker()

    print("[Main] Fetching URDF from robot_state_publisher...")
    urdf_str = node.get_urdf()
    if urdf_str is None:
        print("[Error] Could not fetch URDF. Exiting.")
        return

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
        f.write(urdf_str)
        urdf_path = f.name

    print("[Main] Building Pinocchio kinematic model...")
    node.setup_pinocchio(urdf_path)
    os.remove(urdf_path)
    print("[Main] Setup complete! Tracking the active arm with the head camera.")

    # Launch the live plotting thread (optional)
    stop_event = threading.Event()
    plot_thread = None
    if node.enable_plot:
        plot_thread = threading.Thread(target=plotting_thread, args=(node, stop_event),
                                       daemon=True)
        plot_thread.start()

    # Control loop
    period = 1.0 / CONTROL_HZ
    try:
        while rclpy.ok():
            t_start = time.time()
            rclpy.spin_once(node, timeout_sec=0.001)
            node.solve_and_publish()
            elapsed = time.time() - t_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if plot_thread is not None:
            plot_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
