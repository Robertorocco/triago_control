#!/usr/bin/env python3
from std_msgs.msg import Float64MultiArray, Float64, String
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D  
import threading
from collections import deque
import numpy as np


class TriagoDashboard(Node):
    # Constructor: Sets up 4-quadrant monitoring for 14 DoF total
    def __init__(self):
        super().__init__('triago_dashboard')
        
# 1. Configuration: Expanded memory for full-episode recording
        # 5000 samples @ 100Hz = 50 seconds of continuous data retention
        self.history_len = 5000  
        self.left_joints = [f'arm_left_{i}_joint' for i in range(1, 8)]
        self.right_joints = [f'arm_right_{i}_joint' for i in range(1, 8)]
        self.all_joints = self.left_joints + self.right_joints
        
        # --- Shared Autonomy Time Tracking ---
        self.start_time = None  # Holds the absolute start time for continuous tracking

        # 2. Data Buffers: (Keep your existing deque initialization here)
        self.time_buffer = deque(maxlen=self.history_len)
        self.q_buffers = {name: deque(maxlen=self.history_len) for name in self.all_joints}
        self.dq_buffers = {name: deque(maxlen=self.history_len) for name in self.all_joints}
        

        self.err_time = deque(maxlen=self.history_len)
        self.err_pos_r = deque(maxlen=self.history_len)
        self.err_pos_l = deque(maxlen=self.history_len)
        self.err_vel_r = deque(maxlen=self.history_len)
        self.err_vel_l = deque(maxlen=self.history_len)

        self.h_buffer = deque(maxlen=self.history_len)
        self.h_time = deque(maxlen=self.history_len)
        
        self.freq_buffer = deque(maxlen=self.history_len)
        self.freq_time = deque(maxlen=self.history_len)

        # [ADD] Minimum Distance Buffers
        self.min_dist_buffer = deque(maxlen=self.history_len)
        self.min_dist_time = deque(maxlen=self.history_len)


        # [ADD] Cartesian Position Buffers for 3D Plotting
        self.ref_pos_r = deque(maxlen=self.history_len)
        self.ref_pos_l = deque(maxlen=self.history_len)
        self.real_pos_r = deque(maxlen=self.history_len)
        self.real_pos_l = deque(maxlen=self.history_len)
        
        # [ADD] Slack Buffers
        self.slack_buffer = deque(maxlen=self.history_len)
        self.slack_time = deque(maxlen=self.history_len)

        # [ADD] Slack dimension mode tracker
        self.slack_mode = None  # Will be 'scalar' or 'vector' once detected
        self.slack_legend_updated = False

        # [ADD] Adaptive Controller Buffers
        self.dyn_weights_buffer = deque(maxlen=self.history_len) # Stores [w_slack, gamma_clf]
        self.dyn_weights_time = deque(maxlen=self.history_len)
        
        self.time_scale_buffer = deque(maxlen=self.history_len)  # Stores sigma
        self.time_scale_time = deque(maxlen=self.history_len)

        # --- NEW: Lagrangian (Shadow Price) Buffers ---
        self.lambda_cbf_buffer = deque(maxlen=self.history_len)
        self.lambda_cbf_time = deque(maxlen=self.history_len)
        
        self.lambda_joints_buffer = deque(maxlen=self.history_len)
        self.lambda_joints_time = deque(maxlen=self.history_len)

        # Commanded joint velocity buffers from QP solver
        self.qdot_cmd_time = deque(maxlen=self.history_len)
        self.qdot_cmd_r_buffer = deque(maxlen=self.history_len)
        self.qdot_cmd_l_buffer = deque(maxlen=self.history_len)

        # --- NEW: DUAL TRACKING ERROR BUFFERS ---
        self.qdot_err_time = deque(maxlen=self.history_len)
        self.qdot_err_r_buffer = deque(maxlen=self.history_len)
        self.qdot_err_l_buffer = deque(maxlen=self.history_len)

        self.xdot_err_time = deque(maxlen=self.history_len)
        self.xdot_err_r_buffer = deque(maxlen=self.history_len)
        self.xdot_err_l_buffer = deque(maxlen=self.history_len)

        # 3. State Management
        self.first_msg_time = None
        self.warned_no_velocity = False


        # 4. QoS: Best Effort is critical for high-freq plotting without lag
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # [ADD] Slack Subscriber
        self.create_subscription(Float64MultiArray, '/qp_debug/slacks', self.slack_callback, qos_profile)
        
        # 1. Cartesian References from Shared Autonomy Node
        self.ref_right = np.zeros(13)
        self.ref_left = np.zeros(13)
        self.has_ref_right = False
        self.has_ref_left = False
        self.create_subscription(Float64MultiArray, '/arm_right/cartesian_reference', self.cb_ref_right, qos_profile)
        self.create_subscription(Float64MultiArray, '/arm_left/cartesian_reference', self.cb_ref_left, qos_profile)
        
        # 2. Real State from QP Controller Subscriber
        self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.cb_real, qos_profile)

        # [ADD] Minimum Distance Subscriber
        self.create_subscription(Float64, '/qp_debug/min_distance', self.min_dist_callback, qos_profile)
        
        # [ADD] Performance and Safety Subscribers
        self.create_subscription(Float64, '/qp_debug/loop_freq', self.freq_callback, qos_profile)
        self.create_subscription(Float64, '/qp_debug/safety_margin', self.h_callback, qos_profile)

        # [ADD] Adaptive Controller Subscribers
        self.create_subscription(Float64MultiArray, '/qp_debug/dynamic_weights', self.dyn_weights_callback, qos_profile)
        self.create_subscription(Float64, '/trajectory/time_scale', self.time_scale_callback, qos_profile)

        # --- Dynamic CBF margin subscriber ---
        self.d_safe_buffer = deque(maxlen=self.history_len)
        self.d_safe_time = deque(maxlen=self.history_len)
        self.create_subscription(Float64, '/qp_debug/d_safe_dynamic', self.d_safe_callback, qos_profile)

        # --- NEW: TRACKING ERROR SUBSCRIBERS ---
        self.sub_qdot_err = self.create_subscription(
            Float64MultiArray, '/qp_debug/qdot_err', self.qdot_err_callback, 10)
        self.sub_xdot_err = self.create_subscription(
            Float64MultiArray, '/qp_debug/xdot_err', self.xdot_err_callback, 10)


        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            qos_profile 
        )
        self.get_logger().info('Dashboard Initialized. Waiting for 14-DoF stream...')
    
        # --- NEW: Lagrangian Subscribers ---
        self.create_subscription(Float64, '/qp_debug/lambda_cbf', self.lambda_cbf_callback, qos_profile)
        self.create_subscription(Float64MultiArray, '/qp_debug/lambda_joints', self.lambda_joints_callback, qos_profile)

        self.sub_qdot_cmd = self.create_subscription(
            Float64MultiArray,
            '/qp_debug/qdot_cmd',
            self.cmd_callback,
            10
        )

    def get_time(self):
        """Returns the current time in seconds, normalized to start exactly at t=0.0"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = current_time
        return current_time - self.start_time
    
    def qdot_err_callback(self, msg):
        """ Stores the measured hardware tracking error at the joint level (14 DoF). """
        if len(msg.data) >= 14:
            t = self.get_time()
            self.qdot_err_time.append(t)
            self.qdot_err_r_buffer.append(list(msg.data[:7]))
            self.qdot_err_l_buffer.append(list(msg.data[7:14]))

    def xdot_err_callback(self, msg):
        """ Stores the resulting Cartesian drift velocity (6 DoF: 3 Right + 3 Left). """
        if len(msg.data) >= 6:
            t = self.get_time()
            self.xdot_err_time.append(t)
            self.xdot_err_r_buffer.append(list(msg.data[:3]))
            self.xdot_err_l_buffer.append(list(msg.data[3:6]))
    
    def cmd_callback(self, msg):
        """ Stores the raw velocity commands evaluated by the QP solver. """
        if len(msg.data) >= 14:
            t = self.get_time()
            self.qdot_cmd_time.append(t)
            self.qdot_cmd_r_buffer.append(list(msg.data[:7]))
            self.qdot_cmd_l_buffer.append(list(msg.data[7:14]))


    # --- NEW: Adaptive Controller Callbacks ---
    def dyn_weights_callback(self, msg):
        t = self.get_time()
        self.dyn_weights_time.append(t)
        self.dyn_weights_buffer.append(list(msg.data))

    def time_scale_callback(self, msg):
        t = self.get_time()
        self.time_scale_time.append(t)
        self.time_scale_buffer.append(msg.data)

    def d_safe_callback(self, msg):
        t = self.get_time()
        self.d_safe_time.append(t)
        self.d_safe_buffer.append(msg.data)
    
    def freq_callback(self, msg):
        t = self.get_time()
        self.freq_time.append(t)
        self.freq_buffer.append(msg.data)

    def h_callback(self, msg):
        t = self.get_time()
        self.h_time.append(t)
        self.h_buffer.append(msg.data)

    def min_dist_callback(self, msg):
        t = self.get_time()
        self.min_dist_time.append(t)
        self.min_dist_buffer.append(msg.data)

    def cb_ref_right(self, msg):
        if len(msg.data) >= 12:
            self.ref_right = np.array(msg.data)
            self.has_ref_right = True

    def cb_ref_left(self, msg):
        if len(msg.data) >= 12:
            self.ref_left = np.array(msg.data)
            self.has_ref_left = True


    def cb_real(self, msg):
        if not self.has_ref_right: 
            return
            
        real = np.array(msg.data)
        p_real_r = real[0:3]
        v_real_r = real[3:6]
        p_real_l = real[6:9]
        v_real_l = real[9:12]
        
        # Right Arm Error
        e_p_r = np.linalg.norm(self.ref_right[0:3] - p_real_r)
        e_v_r = np.linalg.norm(self.ref_right[6:9] - v_real_r)
        
        # Left Arm Error
        e_p_l = np.linalg.norm(self.ref_left[0:3] - p_real_l) if self.has_ref_left else 0.0
        e_v_l = np.linalg.norm(self.ref_left[6:9] - v_real_l) if self.has_ref_left else 0.0
        
        t = self.get_time()
        self.err_time.append(t)
        self.err_pos_r.append(e_p_r)
        self.err_vel_r.append(e_v_r)
        self.err_pos_l.append(e_p_l)
        self.err_vel_l.append(e_v_l)
        
        self.ref_pos_r.append(self.ref_right[0:3].copy())
        self.real_pos_r.append(p_real_r.copy())
        
        if self.has_ref_left:
            self.ref_pos_l.append(self.ref_left[0:3].copy())
        else:
            self.ref_pos_l.append(np.zeros(3))
            
        self.real_pos_l.append(p_real_l.copy())

    def slack_callback(self, msg):
        t = self.get_time()
        data = list(msg.data)
        
        # Auto-detect mode on first message
        if self.slack_mode is None:
            if len(data) == 2:
                self.slack_mode = 'scalar'
                self.get_logger().info("[SLACK] Detected SCALAR mode (2D)")
            elif len(data) == 6:
                self.slack_mode = 'vector'
                self.get_logger().info("[SLACK] Detected VECTOR mode (6D: x,y,z per arm)")
            else:
                self.get_logger().error(f"[SLACK] Invalid data length: {len(data)}")
                return
        self.slack_time.append(t)
        self.slack_buffer.append(data)


    # --- NEW: Lagrangian Callbacks ---
    def lambda_cbf_callback(self, msg):
        t = self.get_time()
        self.lambda_cbf_time.append(t)
        self.lambda_cbf_buffer.append(msg.data)

    def lambda_joints_callback(self, msg):
        t = self.get_time()
        self.lambda_joints_time.append(t)
        self.lambda_joints_buffer.append(list(msg.data))

    def listener_callback(self, msg):
        t = self.get_time()

        if len(msg.name) != len(msg.position): return
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        if not all(j in name_to_idx for j in self.all_joints):
            return

        self.time_buffer.append(t)

        has_velocity = len(msg.velocity) == len(msg.position)
        if not has_velocity and not self.warned_no_velocity:
            self.get_logger().warning("Driver is NOT publishing velocities! Plotting zeros.")
            self.warned_no_velocity = True

        for j in self.all_joints:
            idx = name_to_idx[j]
            self.q_buffers[j].append(msg.position[idx])
            self.dq_buffers[j].append(msg.velocity[idx] if has_velocity else 0.0)

# Threading wrapper for ROS spin
def ros_thread_entry(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def update_plot(frame, node, lines_map, axs1, axs2, axs3, dyn_plots, figs):
    artists = []
    window = 10.0

    # --- PART 1: Joint positions + velocities (rows 0-1 of axs1) ---
    if node.time_buffer:
        t = list(node.time_buffer)

        def update_subset(joint_list, buffer_dict, suffix):
            for j in joint_list:
                y = list(buffer_dict[j])
                min_len = min(len(t), len(y))
                if min_len > 0:
                    lines_map[j + suffix].set_data(t[:min_len], y[:min_len])
                    artists.append(lines_map[j + suffix])

        # Row 0: positions
        update_subset(node.left_joints, node.q_buffers, '_pos')
        update_subset(node.right_joints, node.q_buffers, '_pos')
        # Row 1: velocity from driver
        update_subset(node.left_joints, node.dq_buffers, '_vel')
        update_subset(node.right_joints, node.dq_buffers, '_vel')

    # --- PART 2: Raw Encoder vel + Servo Error (rows 2-3 of axs1) ---
    if node.time_buffer:
        t_meas = list(node.time_buffer)

        # Row 2: raw encoder velocity (from /joint_states)
        for i, j in enumerate(node.left_joints):
            dq_data = list(node.dq_buffers[j])
            min_len = min(len(t_meas), len(dq_data))
            if min_len > 0:
                lines_map[j + '_raw_vel_l'].set_data(t_meas[:min_len], dq_data[:min_len])

        for i, j in enumerate(node.right_joints):
            dq_data = list(node.dq_buffers[j])
            min_len = min(len(t_meas), len(dq_data))
            if min_len > 0:
                lines_map[j + '_raw_vel_r'].set_data(t_meas[:min_len], dq_data[:min_len])


        # Row 3: QP command minus raw measured (servo error)
        if node.qdot_cmd_time and len(node.qdot_cmd_time) > 1 and len(t_meas) > 2:
            t_cmd = list(node.qdot_cmd_time)
            cmd_r = np.array(list(node.qdot_cmd_r_buffer)) if node.qdot_cmd_r_buffer else None
            cmd_l = np.array(list(node.qdot_cmd_l_buffer)) if node.qdot_cmd_l_buffer else None

            if cmd_r is not None:
                min_cmd = min(len(t_cmd), len(cmd_r))
                for i, j in enumerate(node.right_joints):
                    dq_data = list(node.dq_buffers[j])
                    min_meas = min(len(t_meas), len(dq_data))
                    if min_meas > 2 and min_cmd > 0:
                        meas_interp = np.interp(t_cmd[:min_cmd], t_meas[:min_meas], dq_data[:min_meas])
                        err = cmd_r[:min_cmd, i] - meas_interp
                        lines_map[j + '_verr_r'].set_data(t_cmd[:min_cmd], err)

            if cmd_l is not None:
                min_cmd_l = min(len(t_cmd), len(cmd_l))
                for i, j in enumerate(node.left_joints):
                    dq_data = list(node.dq_buffers[j])
                    min_meas = min(len(t_meas), len(dq_data))
                    if min_meas > 2 and min_cmd_l > 0:
                        meas_interp = np.interp(t_cmd[:min_cmd_l], t_meas[:min_meas], dq_data[:min_meas])
                        err = cmd_l[:min_cmd_l, i] - meas_interp
                        lines_map[j + '_verr_l'].set_data(t_cmd[:min_cmd_l], err)

        # Window scaling for all axs1 subplots
        if t_meas:
            max_t = t_meas[-1]
            for ax in axs1.flatten():
                ax.set_xlim(max(0, max_t - window), max_t + 0.1)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)


    # --- PART 3: Slacks + lambdas + freq + margin + min_dist (axs2) ---
    # 3A. Update Slacks
    if node.slack_time and node.slack_mode is not None:
        if not node.slack_legend_updated:
            if node.slack_mode == 'scalar':
                axs2[0].legend(handles=[lines_map['slack_right_scalar']], loc='upper right', fontsize='x-small')
                axs2[1].legend(handles=[lines_map['slack_left_scalar']], loc='upper right', fontsize='x-small')
            elif node.slack_mode == 'vector':
                axs2[0].legend(handles=[lines_map['slack_right_x'], lines_map['slack_right_y'], lines_map['slack_right_z']], loc='upper right', fontsize='x-small')
                axs2[1].legend(handles=[lines_map['slack_left_x'], lines_map['slack_left_y'], lines_map['slack_left_z']], loc='upper right', fontsize='x-small')
            node.slack_legend_updated = True

        t_s = list(node.slack_time)
        data_s_list = list(node.slack_buffer)
        min_len = min(len(t_s), len(data_s_list))

        if min_len > 0:
            data_s = np.array(data_s_list[:min_len])

            if node.slack_mode == 'scalar':
                lines_map['slack_right_scalar'].set_data(t_s[:min_len], data_s[:, 0])
                lines_map['slack_left_scalar'].set_data(t_s[:min_len], data_s[:, 1])
                artists.extend([lines_map['slack_right_scalar'], lines_map['slack_left_scalar']])
                for key in ['slack_right_x', 'slack_right_y', 'slack_right_z',
                            'slack_left_x', 'slack_left_y', 'slack_left_z']:
                    lines_map[key].set_data([], [])

            elif node.slack_mode == 'vector':
                lines_map['slack_right_x'].set_data(t_s[:min_len], data_s[:, 0])
                lines_map['slack_right_y'].set_data(t_s[:min_len], data_s[:, 1])
                lines_map['slack_right_z'].set_data(t_s[:min_len], data_s[:, 2])
                lines_map['slack_left_x'].set_data(t_s[:min_len], data_s[:, 3])
                lines_map['slack_left_y'].set_data(t_s[:min_len], data_s[:, 4])
                lines_map['slack_left_z'].set_data(t_s[:min_len], data_s[:, 5])
                artists.extend([
                    lines_map['slack_right_x'], lines_map['slack_right_y'], lines_map['slack_right_z'],
                    lines_map['slack_left_x'], lines_map['slack_left_y'], lines_map['slack_left_z']
                ])
                lines_map['slack_right_scalar'].set_data([], [])
                lines_map['slack_left_scalar'].set_data([], [])


    # 3B. Update CBF Lambda
    if node.lambda_cbf_time:
        t_lc = list(node.lambda_cbf_time)
        data_lc = list(node.lambda_cbf_buffer)
        min_len = min(len(t_lc), len(data_lc))
        if min_len > 0:
            lines_map['lambda_cbf'].set_data(t_lc[:min_len], data_lc[:min_len])
            artists.append(lines_map['lambda_cbf'])

    # 3C. Update Joint Lambdas
    if node.lambda_joints_time:
        t_lj = list(node.lambda_joints_time)
        data_lj_list = list(node.lambda_joints_buffer)
        min_len = min(len(t_lj), len(data_lj_list))
        if min_len > 0:
            data_lj = np.array(data_lj_list[:min_len])
            lines_map['lambda_joints_r'].set_data(t_lj[:min_len], data_lj[:, 0])
            lines_map['lambda_joints_l'].set_data(t_lj[:min_len], data_lj[:, 1])
            artists.extend([lines_map['lambda_joints_r'], lines_map['lambda_joints_l']])

    # 3D. Frequency
    if node.freq_time:
        t_f = list(node.freq_time)
        y_f = list(node.freq_buffer)
        m_f = min(len(t_f), len(y_f))
        if m_f > 0:
            lines_map['loop_freq'].set_data(t_f[:m_f], y_f[:m_f])
            artists.append(lines_map['loop_freq'])

    # 3E. Safety Margin h
    if node.h_time:
        t_h = list(node.h_time)
        y_h = list(node.h_buffer)
        m_h = min(len(t_h), len(y_h))
        if m_h > 0:
            lines_map['margin_h'].set_data(t_h[:m_h], y_h[:m_h])
            artists.append(lines_map['margin_h'])

    # 3F. Min Distance
    if node.min_dist_time:
        t_m = list(node.min_dist_time)
        y_m = list(node.min_dist_buffer)
        m_m = min(len(t_m), len(y_m))
        if m_m > 0:
            lines_map['min_dist'].set_data(t_m[:m_m], y_m[:m_m])
            artists.append(lines_map['min_dist'])


    # 3G. Window scaling for all axs2 subplots
    time_lists_2 = [node.slack_time, node.lambda_cbf_time, node.lambda_joints_time,
                    node.freq_time, node.h_time, node.min_dist_time]
    valid_times_2 = [tl for tl in time_lists_2 if tl]
    if valid_times_2:
        max_t = max(list(tl)[-1] for tl in valid_times_2)
        for ax in axs2:
            ax.set_xlim(max(0, max_t - window), max_t + 0.1)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            ymin, ymax = ax.get_ylim()
            if abs(ymax - ymin) < 1e-6:
                ax.set_ylim(-0.1, max(1.0, ymax + 0.5))

    # --- PART 4: Cartesian error (axs3 rows 0-1) ---
    if node.err_time:
        t_e = list(node.err_time)
        min_len = min(len(t_e), len(node.err_pos_r))
        if min_len > 0:
            lines_map['err_pos_r'].set_data(t_e[:min_len], list(node.err_pos_r)[:min_len])
            lines_map['err_pos_l'].set_data(t_e[:min_len], list(node.err_pos_l)[:min_len])
            lines_map['err_vel_r'].set_data(t_e[:min_len], list(node.err_vel_r)[:min_len])
            lines_map['err_vel_l'].set_data(t_e[:min_len], list(node.err_vel_l)[:min_len])
            artists.extend([
                lines_map['err_pos_r'], lines_map['err_pos_l'],
                lines_map['err_vel_r'], lines_map['err_vel_l']
            ])

        current_max_time = t_e[-1]
        view_min = max(0, current_max_time - window)
        view_max = current_max_time + 0.1
        for ax in axs3.flatten():
            ax.set_xlim(view_min, view_max)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)


    # --- PART 5: Dynamic weights (axs3 rows 2+) ---
    if dyn_plots and node.dyn_weights_time:
        t_dw = list(node.dyn_weights_time)
        data_dw = list(node.dyn_weights_buffer)
        t_ds = list(node.d_safe_time)
        data_ds = list(node.d_safe_buffer)

        for idx, (key, ylabel, title) in enumerate(dyn_plots):
            if key == 'weight_slack':
                min_len = min(len(t_dw), len(data_dw))
                if min_len > 0:
                    y_data = [d[0] for d in data_dw[:min_len]]
                    lines_map[f'dyn_{key}'].set_data(t_dw[:min_len], y_data)
            elif key == 'gamma_clf':
                min_len = min(len(t_dw), len(data_dw))
                if min_len > 0:
                    y_data = [d[1] for d in data_dw[:min_len]]
                    lines_map[f'dyn_{key}'].set_data(t_dw[:min_len], y_data)
            elif key == 'd_safe_dynamic':
                min_len = min(len(t_ds), len(data_ds))
                if min_len > 0:
                    lines_map[f'dyn_{key}'].set_data(t_ds[:min_len], data_ds[:min_len])

        # Window scaling for dynamic weight rows in axs3
        all_t = list(t_dw) + list(t_ds)
        if all_t:
            max_t = max(all_t)
            n_fixed = 2  # rows 0 and 1 are error plots
            for row_idx in range(len(dyn_plots)):
                ax = axs3[n_fixed + row_idx]
                ax.set_xlim(max(0, max_t - window), max_t + 0.1)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

    # Manual redraws for figs[1] and figs[2]
    figs[1].canvas.draw_idle()
    figs[2].canvas.draw_idle()

    return artists


def main(args=None):
    import triago_control.qp_controller.config as cfg_plot

    rclpy.init(args=args)
    node = TriagoDashboard()
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

    spinner = threading.Thread(target=ros_thread_entry, args=(node,), daemon=True)
    spinner.start()

    # --- LAYOUT CONSTANTS ---
    # Positions from wmctrl (slightly aligned borders)
    def place_window(fig, x, y, w, h):
        """Position a TkAgg figure window at (x,y) with size (w,h) in pixels."""
        fig.set_size_inches(w / fig.dpi, h / fig.dpi)
        mng = fig.canvas.manager
        mng.window.wm_geometry(f"+{x}+{y}")

    colors = plt.cm.jet(np.linspace(0, 1, 7))
    lines_map = {}


    # ===================================================================
    # WINDOW 1 (left): "Joint Data" — 4x2 subplot matrix
    # ===================================================================
    fig1, axs1 = plt.subplots(4, 2, sharex=True)
    fig1.suptitle('Joint Data')

    row_titles = ['L- Position', 'L- Velocity from driver',
                  'L- Raw Encoder vel', 'L- Servo Error cmd-meas']
    row_titles_r = ['R- Position', 'R- Velocity from driver',
                    'R- Raw Encoder vel', 'R- Servo Error cmd-meas']

    # Row 0: Position
    axs1[0, 0].set_title('L- Position', fontsize='small')
    axs1[0, 1].set_title('R- Position', fontsize='small')
    for i, j in enumerate(node.left_joints):
        l, = axs1[0, 0].plot([], [], color=colors[i])
        lines_map[j + '_pos'] = l
    for i, j in enumerate(node.right_joints):
        l, = axs1[0, 1].plot([], [], color=colors[i])
        lines_map[j + '_pos'] = l

    # Row 1: Velocity from driver
    axs1[1, 0].set_title('L- Velocity from driver', fontsize='small')
    axs1[1, 1].set_title('R- Velocity from driver', fontsize='small')
    for i, j in enumerate(node.left_joints):
        l, = axs1[1, 0].plot([], [], color=colors[i])
        lines_map[j + '_vel'] = l
    for i, j in enumerate(node.right_joints):
        l, = axs1[1, 1].plot([], [], color=colors[i])
        lines_map[j + '_vel'] = l

    # Row 2: Raw Encoder vel
    axs1[2, 0].set_title('L- Raw Encoder vel', fontsize='small')
    axs1[2, 1].set_title('R- Raw Encoder vel', fontsize='small')
    for i, j in enumerate(node.left_joints):
        l, = axs1[2, 0].plot([], [], color=colors[i], linewidth=0.9)
        lines_map[j + '_raw_vel_l'] = l
    for i, j in enumerate(node.right_joints):
        l, = axs1[2, 1].plot([], [], color=colors[i], linewidth=0.9)
        lines_map[j + '_raw_vel_r'] = l


    # Row 3: Servo Error cmd-meas
    axs1[3, 0].set_title('L- Servo Error cmd-meas', fontsize='small')
    axs1[3, 1].set_title('R- Servo Error cmd-meas', fontsize='small')
    axs1[3, 0].axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    axs1[3, 1].axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    for i, j in enumerate(node.left_joints):
        l, = axs1[3, 0].plot([], [], color=colors[i], linewidth=0.8)
        lines_map[j + '_verr_l'] = l
    for i, j in enumerate(node.right_joints):
        l, = axs1[3, 1].plot([], [], color=colors[i], linewidth=0.8)
        lines_map[j + '_verr_r'] = l

    # Legend below the title (centered, using figure-level legend)
    legend_handles = [Line2D([0], [0], color=colors[i], label=f'J{i+1}') for i in range(7)]
    fig1.legend(handles=legend_handles, loc='upper center', fontsize='x-small',
                ncol=7, bbox_to_anchor=(0.5, 0.95))

    # Y-axis labels on every row left column only
    axs1[0, 0].set_ylabel('[rad]')
    axs1[1, 0].set_ylabel('[rad/s]')
    axs1[2, 0].set_ylabel('[rad/s]')
    axs1[3, 0].set_ylabel('[rad/s]')

    # X-axis label only on bottom row
    axs1[3, 0].set_xlabel('Time [s]')
    axs1[3, 1].set_xlabel('Time [s]')

    for ax in axs1.flatten():
        ax.grid(True, alpha=0.3)


    # ===================================================================
    # WINDOW 2 (center): "QP Data" — 7x1 stacked
    # ===================================================================
    fig2, axs2 = plt.subplots(7, 1, sharex=True)
    fig2.suptitle('QP Data')

    # Row 0: Right Slack
    l_sr_scalar, = axs2[0].plot([], [], 'r-', label=r'$\delta_{R}$ (scalar)', linewidth=2)
    l_sr_x, = axs2[0].plot([], [], 'r-', label=r'$\delta_{R,x}$', alpha=0.8)
    l_sr_y, = axs2[0].plot([], [], 'g-', label=r'$\delta_{R,y}$', alpha=0.8)
    l_sr_z, = axs2[0].plot([], [], 'b-', label=r'$\delta_{R,z}$', alpha=0.8)
    axs2[0].set_ylabel('R Slack')
    axs2[0].yaxis.set_label_position('right')
    lines_map['slack_right_scalar'] = l_sr_scalar
    lines_map['slack_right_x'] = l_sr_x
    lines_map['slack_right_y'] = l_sr_y
    lines_map['slack_right_z'] = l_sr_z

    # Row 1: Left Slack
    l_sl_scalar, = axs2[1].plot([], [], 'b-', label=r'$\delta_{L}$ (scalar)', linewidth=2)
    l_sl_x, = axs2[1].plot([], [], 'r-', label=r'$\delta_{L,x}$', alpha=0.8)
    l_sl_y, = axs2[1].plot([], [], 'g-', label=r'$\delta_{L,y}$', alpha=0.8)
    l_sl_z, = axs2[1].plot([], [], 'b-', label=r'$\delta_{L,z}$', alpha=0.8)
    axs2[1].set_ylabel('L Slack')
    axs2[1].yaxis.set_label_position('right')
    lines_map['slack_left_scalar'] = l_sl_scalar
    lines_map['slack_left_x'] = l_sl_x
    lines_map['slack_left_y'] = l_sl_y
    lines_map['slack_left_z'] = l_sl_z

    # Row 2: CBF Lambda
    l_lc, = axs2[2].plot([], [], 'm-', label=r'$\lambda_{CBF}$')
    axs2[2].set_ylabel('CBF Price')
    axs2[2].yaxis.set_label_position('right')
    axs2[2].legend(loc='upper left', fontsize='x-small')
    lines_map['lambda_cbf'] = l_lc

    # Row 3: Joint Lambdas (R and L on same plot)
    l_lj_r, = axs2[3].plot([], [], 'r-', label=r'$\lambda_{Joints}$ R')
    l_lj_l, = axs2[3].plot([], [], 'b-', label=r'$\lambda_{Joints}$ L')
    axs2[3].set_ylabel('Joint Prices')
    axs2[3].yaxis.set_label_position('right')
    axs2[3].legend(loc='upper left', fontsize='x-small')
    lines_map['lambda_joints_r'] = l_lj_r
    lines_map['lambda_joints_l'] = l_lj_l


    # Row 4: Loop Frequency
    l_freq, = axs2[4].plot([], [], 'g-', label='Loop Hz')
    axs2[4].set_ylabel('Freq [Hz]')
    axs2[4].yaxis.set_label_position('right')
    axs2[4].legend(loc='upper left', fontsize='x-small')
    lines_map['loop_freq'] = l_freq

    # Row 5: Safety Margin (with y=0 red dashed line)
    l_h, = axs2[5].plot([], [], 'm-', label='Softmin h')
    axs2[5].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axs2[5].set_ylabel('Margin [m]')
    axs2[5].yaxis.set_label_position('right')
    axs2[5].legend(loc='upper left', fontsize='x-small')
    lines_map['margin_h'] = l_h

    # Row 6: Min Distance (with y=0 red dashed line)
    l_min_dist, = axs2[6].plot([], [], 'c-', label='Abs Min Dist')
    axs2[6].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axs2[6].set_ylabel('Dist [m]')
    axs2[6].yaxis.set_label_position('right')
    axs2[6].legend(loc='upper left', fontsize='x-small')
    lines_map['min_dist'] = l_min_dist

    # X-axis label only on bottom
    axs2[6].set_xlabel('Time [s]')

    for ax in axs2:
        ax.grid(True, alpha=0.3)


    # ===================================================================
    # WINDOW 3 (right): "Task Error & Adaptation" — variable rows
    # ===================================================================
    # Determine dynamic weight subplots
    dyn_plots = []
    if cfg_plot.DYNAMIC_CBF:
        dyn_plots.append(('d_safe_dynamic', r'$d_{safe}^{dyn}$ [m]', 'Dynamic Safety Margin'))
    if cfg_plot.DYNAMIC_GAMMA_CLF:
        dyn_plots.append(('gamma_clf', r'$\gamma_{CLF}$', 'CLF Convergence Rate'))
    if cfg_plot.DYNAMIC_SLACK_WEIGHT:
        dyn_plots.append(('weight_slack', r'$w_{\delta}$', 'Slack Weight'))

    n_rows_3 = 2 + len(dyn_plots)
    fig3, axs3 = plt.subplots(n_rows_3, 1, sharex=True, squeeze=False)
    axs3 = axs3.flatten()
    fig3.suptitle('Task Error & Adaptation')

    # Row 0: Position error (R and L)
    l_epr, = axs3[0].plot([], [], 'r-', label='Pos Err R')
    l_epl, = axs3[0].plot([], [], 'b-', label='Pos Err L')
    axs3[0].set_ylabel('Error [m]')
    axs3[0].grid(True, alpha=0.3)
    axs3[0].legend(loc='upper right', fontsize='x-small')
    lines_map['err_pos_r'] = l_epr
    lines_map['err_pos_l'] = l_epl

    # Row 1: Velocity error (R and L)
    l_evr, = axs3[1].plot([], [], 'r-', label='Vel Err R')
    l_evl, = axs3[1].plot([], [], 'b-', label='Vel Err L')
    axs3[1].set_ylabel('Error [m/s]')
    axs3[1].grid(True, alpha=0.3)
    axs3[1].legend(loc='upper right', fontsize='x-small')
    lines_map['err_vel_r'] = l_evr
    lines_map['err_vel_l'] = l_evl

    # Rows 2+: Dynamic weight subplots
    for idx, (key, ylabel, title) in enumerate(dyn_plots):
        ax = axs3[2 + idx]
        ax.set_title(title, fontsize='small')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        l, = ax.plot([], [], 'm-', linewidth=1.2)
        lines_map[f'dyn_{key}'] = l

    # X-axis label only on bottom
    axs3[-1].set_xlabel('Time [s]')


    # ===================================================================
    # WINDOW PLACEMENT (from wmctrl, slightly aligned)
    # ===================================================================
    figs = [fig1, fig2, fig3]

    place_window(fig1, 86, 126, 832, 1131)    # Left: Joint Data
    place_window(fig2, 924, 118, 541, 1126)   # Center: QP Data
    place_window(fig3, 1471, 126, 499, 1131)  # Right: Task Error

    # ===================================================================
    # ANIMATION
    # ===================================================================
    ani = FuncAnimation(fig1, update_plot,
                        fargs=(node, lines_map, axs1, axs2, axs3, dyn_plots, figs),
                        interval=100)
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
