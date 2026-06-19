#!/usr/bin/env python3
# main_qp_controller.py
"""
The Orchestrator (ROS 2 Node).

Wires together the five specialized modules and drives the safety-critical
control loop:

    RobotKinematics  -> FK / Jacobians / filtered velocity / digital twin
    CollisionManager -> SoftMin CBF gradient + dynamic safety margin
    SharedAutonomyHandler -> grasp / CBF-bypass / attachment commands
    QPFormulator     -> the CLF-CBF-QP that produces safe joint velocities
    VisualizationEngine -> Meshcat + RViz telemetry (off the critical path)

Responsibilities:
    * fetch the URDF and initialize every sub-module,
    * own the cartesian-reference and joint-state callbacks + motion watchdogs,
    * run `solve_and_publish` on a CONFIGURABLE-frequency timer (see
      `set_control_frequency`, which replaces the original hard-coded 1/300 dt),
    * keep the TSID JS velocity controllers active,
    * publish joint velocity commands and all dashboard telemetry.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64, String
from controller_manager_msgs.srv import SwitchController, ListControllers
from rcl_interfaces.srv import GetParameters
from tf2_ros import Buffer, TransformListener
import pinocchio as pin
import numpy as np
import time
import tempfile
import os

import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.qp_controller.collision_manager import CollisionManager
from triago_control.qp_controller.shared_autonomy_handler import SharedAutonomyHandler
from triago_control.qp_controller.qp_formulator import QPFormulator
from triago_control.qp_controller.visualization_engine import VisualizationEngine


class SafetyQPController(Node):
    """ROS 2 node orchestrating the bimanual QP-CLF-CBF safety controller."""

    def __init__(self):
        super().__init__('safety_qp_controller')

        # Configurable control frequency (replaces the hard-coded 1/300 target dt)
        self._control_freq = cfg.CONTROL_FREQ_DEFAULT
        self.loop_timer = None

        # --- TF (kept for the start-up transform wait) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- BUILD MODEL + SUB-MODULES ---
        urdf_str = self.get_urdf()
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
            f.write(urdf_str)
            self.urdf_path = f.name

        self.kin = RobotKinematics(self.urdf_path)
        self.col = CollisionManager(self.kin.model, self.kin.data)

        right_offsets = self.col.calculate_offsets(cfg.RIGHT_CHAIN, 'gripper_right_base_link')
        left_offsets = self.col.calculate_offsets(cfg.LEFT_CHAIN, 'gripper_left_base_link')
        self.col.build_collision_model(right_offsets, left_offsets)
        self.col.define_collision_pairs()

        self.viz = VisualizationEngine(self, self.kin.model, self.col.cmodel, self.urdf_path)
        self.viz.add_gripper_visual_boxes(self.col)
        self.hri = SharedAutonomyHandler(self, self.col, self.kin, self.viz)
        self.qp = QPFormulator(self.kin.model)

        # --- CONTROL MODE / REFERENCES ---
        self.orientation_ctrl = cfg.ORIENTATION_CTRL
        self.x_ref_right = None; self.rpy_ref_right = None; self.xdot_ref_right = None; self.w_ref_right = None
        self.x_ref_left = None;  self.rpy_ref_left = None;  self.xdot_ref_left = None;  self.w_ref_left = None
        self.task_dim_right = 6.0
        self.task_dim_left = 6.0

        # --- WATCHDOGS ---
        self.right_imposed_motion = False
        self.left_imposed_motion = False
        self.last_right_msg_time = time.time()
        self.last_left_msg_time = time.time()

        # --- LOOP / SIM STATE ---
        self.active_controller_mode = False
        self.publish_counter = 0
        self.publish_every_n = cfg.PUBLISH_EVERY_N
        self.last_freq_pub_time = time.perf_counter()
        self.last_sim_time = None
        self.last_qdot_cmd_14 = np.zeros(14)

        # --- COMMAND PUBLISHERS ---
        self.pub_right = self.create_publisher(Float64MultiArray, f'/{cfg.RIGHT_CONTROLLER}/joint_velocity_cmd', 1)
        self.pub_left = self.create_publisher(Float64MultiArray, f'/{cfg.LEFT_CONTROLLER}/joint_velocity_cmd', 1)

        # --- TELEMETRY PUBLISHERS ---
        self.pub_qdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_err', 10)
        self.pub_xdot_err = self.create_publisher(Float64MultiArray, '/qp_debug/xdot_err', 10)
        self.pub_slacks = self.create_publisher(Float64MultiArray, '/qp_debug/slacks', 10)
        self.pub_ee_state = self.create_publisher(Float64MultiArray, '/qp_debug/ee_real', 10)
        self.pub_debug_h = self.create_publisher(Float64, '/qp_debug/safety_margin', 10)
        self.pub_loop_freq = self.create_publisher(Float64, '/qp_debug/loop_freq', 10)
        self.pub_min_dist = self.create_publisher(Float64, '/qp_debug/min_distance', 10)
        self.pub_lambda_cbf = self.create_publisher(Float64, '/qp_debug/lambda_cbf', 10)
        self.pub_lambda_joints = self.create_publisher(Float64MultiArray, '/qp_debug/lambda_joints', 10)
        self.pub_dynamic_weights = self.create_publisher(Float64MultiArray, '/qp_debug/dynamic_weights', 10)
        self.pub_d_safe_dynamic = self.create_publisher(Float64, '/qp_debug/d_safe_dynamic', 10)
        self.pub_qdot_cmd = self.create_publisher(Float64MultiArray, '/qp_debug/qdot_cmd', 10)
        self.pub_shared_col = self.create_publisher(Float64MultiArray, '/collision_constraints', 10)

        # --- SUBSCRIBERS ---
        self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.ref_cb_right, 10)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.ref_cb_left, 10)

        # Services for controller switching
        self.switch_srv = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_srv = self.create_client(ListControllers, '/controller_manager/list_controllers')

        # Low-rate RViz obstacle marker timer (matches original 0.5s cadence)
        self.timer_obs = self.create_timer(0.5, lambda: self.viz.publish_obstacle_marker(self.hri))

    # =====================================================================
    # CONFIGURABLE FREQUENCY GOVERNOR
    # =====================================================================
    @property
    def control_frequency(self):
        # Current control loop frequency [Hz].
        return self._control_freq

    def set_control_frequency(self, freq_hz):
        # Dynamically govern the solve_and_publish loop, recreating the timer.
        if freq_hz <= 0:
            self.get_logger().warn(f"[FREQ] Ignoring non-positive frequency {freq_hz}.")
            return
        self._control_freq = float(freq_hz)
        if self.loop_timer is not None:
            self.destroy_timer(self.loop_timer)
        self.loop_timer = self.create_timer(1.0 / self._control_freq, self.solve_and_publish)
        self.get_logger().info(f"[FREQ] Control loop set to {self._control_freq:.1f} Hz.")

    def start_control_loop(self):
        # Engage the real-time loop at the configured frequency.
        self.set_control_frequency(self._control_freq)

    # =====================================================================
    # SETUP HELPERS
    # =====================================================================
    def get_urdf(self):
        # Fetch the robot_description string from robot_state_publisher.
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value

    def wait_for_tf(self):
        # Block until the base->wrist transform is available (mirrors original startup).
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tf_buffer.can_transform(cfg.REF_FRAME, cfg.RIGHT_CHAIN[-1], rclpy.time.Time()):
                break

    def check_and_switch_controllers(self):
        # Smart switch: activate TSID JS velocity controllers, deactivate conflicts.
        if not self.list_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("List Controllers Service unavailable.")
            return False
        future = self.list_srv.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        if not future.result():
            return False

        current_state = {c.name: c.state for c in future.result().controller}
        to_activate = [t for t in [cfg.RIGHT_CONTROLLER, cfg.LEFT_CONTROLLER] if current_state.get(t) != "active"]
        to_deactivate = [c for c in cfg.CONFLICTING_CONTROLLERS if current_state.get(c) == "active"]

        if not to_activate and not to_deactivate:
            self.get_logger().info("Controllers already correct.")
            self.active_controller_mode = True
            return True
        if not self.switch_srv.wait_for_service(timeout_sec=1.0):
            return False

        req = SwitchController.Request()
        req.activate_controllers = to_activate
        req.deactivate_controllers = to_deactivate
        req.strictness = SwitchController.Request.STRICT
        self.get_logger().info(f"Switching: +{to_activate} | -{to_deactivate}")
        future = self.switch_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result().ok:
            self.active_controller_mode = True
            return True
        self.get_logger().error("Switch Service Failed.")
        return False

    # =====================================================================
    # CALLBACKS
    # =====================================================================
    def joint_callback(self, msg):
        # Parse physical joint positions and hand them to the kinematics filter.
        if self.kin.model is None:
            return
        q_physical = pin.neutral(self.kin.model)
        for i, name in enumerate(msg.name):
            if self.kin.model.existJointName(name):
                idx_q = self.kin.model.joints[self.kin.model.getJointId(name)].idx_q
                if idx_q >= 0:
                    q_physical[idx_q] = msg.position[i]
        time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.kin.update_from_joint_state(q_physical, time_stamp)

    def ref_cb_right(self, msg):
        # Right-arm cartesian reference (12+ float protocol, 6-float fallback).
        if len(msg.data) >= 12:
            self.x_ref_right = np.array(msg.data[0:3])
            self.rpy_ref_right = np.array(msg.data[3:6])
            self.xdot_ref_right = np.array(msg.data[6:9])
            self.w_ref_right = np.array(msg.data[9:12])
            self.task_dim_right = msg.data[12] if len(msg.data) >= 13 else 6.0
            self.right_imposed_motion = True
            self.last_right_msg_time = time.time()
        elif len(msg.data) >= 6:
            self.x_ref_right = np.array(msg.data[0:3])
            self.xdot_ref_right = np.array(msg.data[3:6])
            self.right_imposed_motion = True
            self.last_right_msg_time = time.time()

    def ref_cb_left(self, msg):
        # Left-arm cartesian reference (12+ float protocol, 6-float fallback).
        if len(msg.data) >= 12:
            self.x_ref_left = np.array(msg.data[0:3])
            self.rpy_ref_left = np.array(msg.data[3:6])
            self.xdot_ref_left = np.array(msg.data[6:9])
            self.w_ref_left = np.array(msg.data[9:12])
            self.task_dim_left = msg.data[12] if len(msg.data) >= 13 else 6.0
            self.left_imposed_motion = True
            self.last_left_msg_time = time.time()
        elif len(msg.data) >= 6:
            self.x_ref_left = np.array(msg.data[0:3])
            self.xdot_ref_left = np.array(msg.data[3:6])
            self.left_imposed_motion = True
            self.last_left_msg_time = time.time()

    # =====================================================================
    # TASK ERROR EXTRACTION (5D / 6D / 3D)
    # =====================================================================
    def _arm_task_error(self, ee_id, x_ref, rpy_ref, xdot_ref, w_ref, task_dim):
        # Compute the (weighted-ready) task error + feedforward velocity for one arm.
        if x_ref is None or ee_id is None:
            return np.zeros(3), np.zeros(3)
        x_real = self.kin.data.oMf[ee_id].translation
        e_pos = x_ref - x_real
        if self.orientation_ctrl and rpy_ref is not None:
            R_real = self.kin.data.oMf[ee_id].rotation
            R_des = pin.rpy.rpyToMatrix(rpy_ref[0], rpy_ref[1], rpy_ref[2])
            if task_dim == 5.0:
                e_R = np.cross(R_real[:, 0], R_des[:, 0])     # 5D: align X-axis only
            else:
                e_R = pin.log3(R_des @ R_real.T)              # 6D: full SO(3) error
            e_task = np.concatenate([e_pos, e_R])
            v_task = np.concatenate([xdot_ref, w_ref])
        else:
            e_task = e_pos                                    # 3D: translation only
            v_task = xdot_ref
        return e_task, v_task

    def extract_task_errors(self):
        # Build both arms' task errors for the CLF rows.
        e_r, v_r = self._arm_task_error(self.kin.ee_id_right, self.x_ref_right, self.rpy_ref_right,
                                        self.xdot_ref_right, self.w_ref_right, self.task_dim_right)
        e_l, v_l = self._arm_task_error(self.kin.ee_id_left, self.x_ref_left, self.rpy_ref_left,
                                        self.xdot_ref_left, self.w_ref_left, self.task_dim_left)
        return e_r, v_r, e_l, v_l

    # =====================================================================
    # MAIN CONTROL LOOP
    # =====================================================================
    def solve_and_publish(self):
        # One control tick: kinematics -> SoftMin -> QP -> publish -> evolve twin.
        if self.kin.current_q is None:
            return

        # --- Watchdog timeout (freeze an arm if its reference went stale) ---
        if self.left_imposed_motion and (time.time() - self.last_left_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self.left_imposed_motion = False
            print("[Safety] Watchdog Timeout: Left motion stopped.")
        if self.right_imposed_motion and (time.time() - self.last_right_msg_time) > cfg.WATCHDOG_TIMEOUT:
            self.right_imposed_motion = False
            print("[Safety] Watchdog Timeout: Right motion stopped.")

        # --- 0. Kinematics + geometry refresh ---
        self.kin.update_kinematics()
        self.kin.debug_interrogate()
        self.col.update_geometry(self.kin.current_q)

        # --- Deferred attachment (needs fresh oMi / oMg) ---
        if self.hri.pending_attach is not None:
            arm_side, color = self.hri.pending_attach
            self.hri.pending_attach = None
            try:
                self.hri.attach_object_visually(arm_side, color)
            except Exception as e:
                self.get_logger().warn(f"[TOPOLOGY] Attach failed: {e}")

        # --- Grasp contact distance telemetry ---
        self.hri.publish_contact_distances()

        # --- Low-level tracking error (commanded vs measured) ---
        qdot_err_14, xdot_err_6 = self.kin.compute_tracking_errors(self.last_qdot_cmd_14)

        # --- 1. SoftMin CBF aggregation ---
        J_soft, h_soft, d_safe_dynamic, abs_min_distance = self.col.compute_softmin_jacobian(
            self.kin.current_v, self.kin.idx_right, self.kin.idx_left,
            self.hri.grasp_margin_targets, self.hri.attached_objects,
            self.hri.attached_adjacency, self.hri.ignored_targets, self.publish_counter)

        # --- 2. Task errors ---
        e_r, v_r, e_l, v_l = self.extract_task_errors()

        # --- 3. Build + solve the CLF-CBF-QP ---
        dt = 1.0 / self._control_freq
        q_dot_safe, slack_r, slack_l, b_col, lambda_joints_total = self.qp.build_and_solve(
            self.kin, J_soft, h_soft, d_safe_dynamic,
            self.right_imposed_motion, self.left_imposed_motion,
            self.xdot_ref_right, self.xdot_ref_left, e_r, v_r, e_l, v_l, dt)

        self.publish_counter += 1

        # --- 4. Downsampled telemetry publishing ---
        if self.publish_counter % self.publish_every_n == 0:
            self._publish_telemetry(q_dot_safe, slack_r, slack_l, b_col, lambda_joints_total,
                                    J_soft, h_soft, d_safe_dynamic, abs_min_distance,
                                    qdot_err_14, xdot_err_6)

        # --- 5. Command publishing + hardware override ---
        cmd_data_r = [0.0] * 7
        cmd_data_l = [0.0] * 7
        if self.active_controller_mode:
            if self.kin.idx_right:
                cmd_data_r = q_dot_safe[self.kin.idx_right].tolist() if self.right_imposed_motion else [0.0] * len(self.kin.idx_right)
                self.pub_right.publish(Float64MultiArray(data=cmd_data_r))
            if self.kin.idx_left:
                cmd_data_l = q_dot_safe[self.kin.idx_left].tolist() if self.left_imposed_motion else [0.0] * len(self.kin.idx_left)
                self.pub_left.publish(Float64MultiArray(data=cmd_data_l))

        # Save the exact command sent to hardware for next tick's tracking-error math
        self.last_qdot_cmd_14 = np.concatenate((cmd_data_r, cmd_data_l))
        if self.publish_counter % self.publish_every_n == 0:
            self.pub_qdot_cmd.publish(Float64MultiArray(data=self.last_qdot_cmd_14.tolist()))

        # --- 6. Evolve the digital twin (ideal kinematics) ---
        if cfg.SIMULATE_IDEAL_KINEMATICS:
            current_time = time.perf_counter()
            if self.last_sim_time is None:
                dt_sim = 0.001
            else:
                dt_sim = current_time - self.last_sim_time
            if dt_sim > 0.1:
                dt_sim = 0.001
            self.kin.integrate_simulated_state(q_dot_safe, dt_sim)
            self.last_sim_time = current_time

        # --- 7. External debug visualizer (optional tethers / overlays) ---
        if self.publish_counter % self.publish_every_n == 0:
            if not cfg.DISABLE_CBF:
                self.pub_debug_h.publish(Float64(data=float(h_soft - d_safe_dynamic)))
            self.viz.publish_debug(
                self.kin.model, self.kin.data, self.col.cdata, self.kin.current_q,
                q_dot_safe, None, None, self.kin.ee_id_right, self.kin.ee_id_left,
                cfg.JOINT_LIMIT_BUFFER_BASE)
            self.viz.publish_teleop_tether()

        # --- Diagnostic brake tracker ---
        if self.publish_counter % 200 == 0:
            print("\n=== DECOUPLED QP BRAKES ===")
            print(f"Collision Brakes:  {self.qp.last_lambda_col:.4f}")
            print(f"Joint Brakes (R):  {self.qp.last_lambda_joints_right:.4f}")
            print(f"Joint Brakes (L):  {self.qp.last_lambda_joints_left:.4f}")
            print("===========================\n")

    def _publish_telemetry(self, q_dot_safe, slack_r, slack_l, b_col, lambda_joints_total,
                           J_soft, h_soft, d_safe_dynamic, abs_min_distance,
                           qdot_err_14, xdot_err_6):
        # Publish the full dashboard telemetry set (downsampled, off the hot path).
        # Slacks + shadow prices
        self.pub_slacks.publish(Float64MultiArray(data=[float(abs(slack_r)), float(abs(slack_l))]))
        self.pub_lambda_cbf.publish(Float64(data=self.qp.last_lambda_col))
        if self.kin.idx_right and self.kin.idx_left:
            max_lambda_r = float(np.max(lambda_joints_total[self.kin.idx_right]))
            max_lambda_l = float(np.max(lambda_joints_total[self.kin.idx_left]))
        else:
            max_lambda_r = max_lambda_l = 0.0
        self.pub_lambda_joints.publish(Float64MultiArray(data=[max_lambda_r, max_lambda_l]))

        # End-effector state (pos + lin vel + RPY, 18 floats)
        if self.kin.ee_id_right is not None and self.kin.ee_id_left is not None:
            p_real_r = self.kin.data.oMf[self.kin.ee_id_right].translation
            p_real_l = self.kin.data.oMf[self.kin.ee_id_left].translation
            v_real_r = pin.getFrameVelocity(self.kin.model, self.kin.data, self.kin.ee_id_right, pin.LOCAL_WORLD_ALIGNED).linear
            v_real_l = pin.getFrameVelocity(self.kin.model, self.kin.data, self.kin.ee_id_left, pin.LOCAL_WORLD_ALIGNED).linear
            rpy_real_r = pin.rpy.matrixToRpy(self.kin.data.oMf[self.kin.ee_id_right].rotation)
            rpy_real_l = pin.rpy.matrixToRpy(self.kin.data.oMf[self.kin.ee_id_left].rotation)
            ee_data = []
            ee_data.extend(p_real_r.tolist()); ee_data.extend(v_real_r.tolist())
            ee_data.extend(p_real_l.tolist()); ee_data.extend(v_real_l.tolist())
            ee_data.extend(rpy_real_r.tolist()); ee_data.extend(rpy_real_l.tolist())
            self.pub_ee_state.publish(Float64MultiArray(data=ee_data))

        # Loop frequency (measured)
        current_time = time.perf_counter()
        elapsed = current_time - self.last_freq_pub_time
        if elapsed > 0:
            self.pub_loop_freq.publish(Float64(data=self.publish_every_n / elapsed))
        self.last_freq_pub_time = current_time

        # Min distance + dynamic weights
        self.pub_min_dist.publish(Float64(data=abs_min_distance))
        self.pub_dynamic_weights.publish(Float64MultiArray(data=[float(self.qp.weight_slack), float(self.qp.gamma_clf)]))
        self.pub_d_safe_dynamic.publish(Float64(data=float(d_safe_dynamic)))

        # Tracking errors
        if qdot_err_14 is not None:
            self.pub_qdot_err.publish(Float64MultiArray(data=qdot_err_14.tolist()))
        if xdot_err_6 is not None:
            self.pub_xdot_err.publish(Float64MultiArray(data=xdot_err_6.tolist()))

        # Virtual wall marker
        self.viz.publish_wall_marker()

        # Cartesian projection of the collision gradient for shared autonomy (13 floats)
        if self.kin.ee_id_right is not None and self.kin.ee_id_left is not None:
            J_EE_R_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_right, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_EE_L_6D = pin.getFrameJacobian(self.kin.model, self.kin.data, self.kin.ee_id_left, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_c_cart_R = np.dot(J_soft, np.linalg.pinv(J_EE_R_6D))
            J_c_cart_L = np.dot(J_soft, np.linalg.pinv(J_EE_L_6D))
            self.pub_shared_col.publish(Float64MultiArray(
                data=[float(b_col)] + J_c_cart_R.tolist() + J_c_cart_L.tolist()))


def main():
    rclpy.init()
    node = SafetyQPController()

    # --- PHASE 1: wait for TF, then verify controller state ---
    node.get_logger().info("[Main] Waiting for TF...")
    node.wait_for_tf()

    node.get_logger().info("[Main] Verifying Controller State...")
    if node.check_and_switch_controllers():
        print("------------------------------------------------")
        print("SAFETY CONTROLLER RUNNING (Velocity Mode)")
        print("------------------------------------------------")
    else:
        print("[Error] Could not switch controllers. Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- PHASE 2: visualization + diagnostics ---
    node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    node.start_control_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
