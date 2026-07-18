# robot_kinematics.py
"""Pinocchio wrapper: URDF model/data, arm joint indices, FK/Jacobians, and the joint-velocity pipeline.
Velocity is ALWAYS derived from position (Lie-manifold diff + EMA) on sim and real alike -- raw sensor
velocity is noisy and gets amplified by the QP's rate-damping term."""

import pinocchio as pin
import numpy as np
import triago_control.qp_controller.config as cfg


class RobotKinematics:
    """Owns the Pinocchio model/data and all kinematic + filtering operations."""

    def __init__(self, urdf_path, real_hardware=False):
        # Build the full TRIAGo model (WARNING: this includes EVERY joint).
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.urdf_path = urdf_path
        self.real_hardware = real_hardware

        # --- KINEMATIC BYPASS (DIGITAL TWIN) ---
        self.q_sim = None              # Mathematically perfect joint state
        self.is_sim_anchored = False   # Latch: capture the physical start posture once
        self.current_q = None          # Active joint configuration used by the QP
        self.current_v = None          # Active joint velocity (filtered)

        # Memory buffers for numerical differentiation + EMA filtering
        self.last_q_meas = None
        self.last_v_filt = None
        self.last_msg_time = None

        # Indices of the actuated arm joints in the velocity vector (q_dot)
        self.idx_right = []
        self.idx_left = []
        self._cache_joint_indices()

        # End-effector frame IDs — inject grasping frames if the URDF lacks them
        self._ensure_grasping_frames()
        self.ee_id_right = self.model.getFrameId(cfg.RIGHT_TCP_FRAME) if self.model.existFrame(cfg.RIGHT_TCP_FRAME) else None
        self.ee_id_left = self.model.getFrameId(cfg.LEFT_TCP_FRAME) if self.model.existFrame(cfg.LEFT_TCP_FRAME) else None

    def _cache_joint_indices(self):
        # Pre-compute the velocity-vector indices of right/left arm joints (QP variables).
        for name in cfg.RIGHT_JOINTS:
            if self.model.existJointName(name):
                self.idx_right.append(self.model.joints[self.model.getJointId(name)].idx_v)
        for name in cfg.LEFT_JOINTS:
            if self.model.existJointName(name):
                self.idx_left.append(self.model.joints[self.model.getJointId(name)].idx_v)
        print(f"[Init] Mapped {len(self.idx_right)} Right Joints and {len(self.idx_left)} Left Joints.")

    def _ensure_grasping_frames(self):
        """Injects gripper_*_grasping_link frames when the real TRIAGo URDF lacks them
        ([0,0,0.157] + Ry(-90 deg) from gripper_*_base_link, matching the hardware static TF)."""
        R_offset = pin.rpy.rpyToMatrix(0.0, -1.5708, 0.0)
        t_offset = np.array([0.0, 0.0, 0.157])
        placement = pin.SE3(R_offset, t_offset)

        frames_to_add = [
            (cfg.RIGHT_TCP_FRAME, 'gripper_right_base_link'),
            (cfg.LEFT_TCP_FRAME,  'gripper_left_base_link'),
        ]

        for tcp_name, parent_body_name in frames_to_add:
            if self.model.existFrame(tcp_name):
                continue  # Already in URDF, nothing to do
            if not self.model.existFrame(parent_body_name):
                print(f"[WARN] Cannot inject {tcp_name}: parent frame '{parent_body_name}' not found in model.")
                continue
            parent_frame_id = self.model.getFrameId(parent_body_name)
            parent_joint_id = self.model.frames[parent_frame_id].parentJoint
            # Compose: new frame placement = parent_frame_placement * offset
            parent_placement = self.model.frames[parent_frame_id].placement
            frame_placement = parent_placement * placement
            new_frame = pin.Frame(
                tcp_name,
                parent_joint_id,
                parent_frame_id,
                frame_placement,
                pin.FrameType.OP_FRAME,
            )
            self.model.addFrame(new_frame)
            print(f"[Init] Injected frame '{tcp_name}' into Pinocchio model (parent: {parent_body_name}).")

        # Rebuild data to account for new frames
        self.data = self.model.createData()

    def update_from_joint_state(self, q_physical, time_stamp, v_direct=None):
        """Updates the joint state; velocity is derived + EMA-filtered from positions (v_direct bypass unused)."""
        if v_direct is not None and self.real_hardware:
            v_physical = v_direct
        elif self.last_q_meas is not None and self.last_msg_time is not None:
            dt = time_stamp - self.last_msg_time
            if dt > 1e-5:  # Guard against duplicate / zero-dt messages
                # Raw velocity on the Lie manifold (safe for quaternion floating base)
                v_raw = pin.difference(self.model, self.last_q_meas, q_physical) / dt
                # First-Order Low-Pass Filter (Exponential Moving Average)
                v_physical = (cfg.ALPHA_FILTER * v_raw) + ((1.0 - cfg.ALPHA_FILTER) * self.last_v_filt)
            else:
                v_physical = self.last_v_filt.copy()  # Hold previous velocity
        else:
            self.last_v_filt = np.zeros(self.model.nv)  # Initialization tick
            v_physical = self.last_v_filt.copy()

        # Update historical buffers for the next k+1 tick
        self.last_q_meas = q_physical.copy()
        self.last_v_filt = v_physical.copy()
        self.last_msg_time = time_stamp

        # Branch on the simulation flag
        if cfg.SIMULATE_IDEAL_KINEMATICS:
            if not self.is_sim_anchored:
                # First tick: anchor the digital twin to the real starting posture
                self.q_sim = q_physical.copy()
                self.current_q = self.q_sim.copy()
                self.is_sim_anchored = True
                print("[Bypass] Anchored digital twin. Physical sensors disconnected.")
            # Ignore q_physical thereafter, but always refresh measured velocity for error math
            self.current_v = v_physical.copy()
        else:
            # Normal hardware operation: trust the physical sensors continuously
            self.current_q = q_physical.copy()
            self.current_v = v_physical.copy()

    def integrate_simulated_state(self, q_dot_cmd, dt):
        # Evolve the digital twin via Pinocchio's Lie-group exponential map integration.
        v_full = np.zeros(self.model.nv)
        if self.idx_right:
            v_full[self.idx_right] = q_dot_cmd[self.idx_right]
        if self.idx_left:
            v_full[self.idx_left] = q_dot_cmd[self.idx_left]
        # Convert velocity to a tangent-space displacement and integrate
        self.q_sim = pin.integrate(self.model, self.q_sim, v_full * dt)
        self.current_q = self.q_sim.copy()  # Next QP iteration uses the perfect state

    def update_kinematics(self):
        """Refreshes FK, frame placements and Jacobians; velocity is passed so frame twists are true."""
        v = self.current_v if self.current_v is not None else np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, self.current_q, v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.current_q)

    def debug_interrogate(self):
        # Optional console dump of the right wrist / TCP world positions (DEBUG only).
        if not cfg.DEBUG:
            return
        wrist_id = self.model.getFrameId('arm_right_tool_link')
        tcp_id = self.model.getFrameId('gripper_right_grasping_link')
        p_wrist = self.data.oMf[wrist_id].translation
        p_tcp = self.data.oMf[tcp_id].translation
        print("--- PINOCCHIO MATH ---")
        print(f"WRIST (tool_link):  [{p_wrist[0]:.3f}, {p_wrist[1]:.3f}, {p_wrist[2]:.3f}]")
        print(f"TCP (grasp_link):   [{p_tcp[0]:.3f}, {p_tcp[1]:.3f}, {p_tcp[2]:.3f}]")
        print("----------------------")

    def compute_tracking_errors(self, last_qdot_cmd_14):
        # Low-level joint + Cartesian velocity tracking error (commanded vs measured).
        if not (self.idx_right and self.idx_left):
            return None, None

        # 1. Joint-space error (14-DoF): commanded minus measured
        meas_v_14 = np.concatenate((self.current_v[self.idx_right], self.current_v[self.idx_left]))
        qdot_err_14 = last_qdot_cmd_14 - meas_v_14

        # 2. Cartesian-space error (6-DoF: 3 right + 3 left) via translation Jacobians
        xdot_err_6 = None
        if self.ee_id_right is not None and self.ee_id_left is not None:
            v_err_full = np.zeros(self.model.nv)
            v_err_full[self.idx_right] = qdot_err_14[:7]
            v_err_full[self.idx_left] = qdot_err_14[7:]
            J_r = pin.getFrameJacobian(self.model, self.data, self.ee_id_right, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            J_l = pin.getFrameJacobian(self.model, self.data, self.ee_id_left, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            xdot_err_6 = np.concatenate((np.dot(J_r, v_err_full), np.dot(J_l, v_err_full)))
        return qdot_err_14, xdot_err_6

    def get_joint_limits(self, joint_names):
        """Returns (lower, upper) URDF limits for the named joints; unbounded/missing fall back to +/-3.15 rad."""
        lower, upper = [], []
        for name in joint_names:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                idx_q = self.model.joints[jid].idx_q
                lo = float(self.model.lowerPositionLimit[idx_q])
                hi = float(self.model.upperPositionLimit[idx_q])
                if lo <= -1e10 or hi >= 1e10:
                    lo, hi = -3.15, 3.15
                lower.append(lo)
                upper.append(hi)
            else:
                print(f"[WARN] get_joint_limits: joint '{name}' not found in model.")
                lower.append(-3.15)
                upper.append(3.15)
        return lower, upper

    def print_joint_limits_table(self, logger=None):
        # Print a formatted table of the URDF lower/upper position limits for both arms.
        log = logger.info if logger is not None else print
        log("\n" + "=" * 65)
        log(f"{'Joint Name':<25} | {'Lower Limit (rad)':<15} | {'Upper Limit (rad)':<15}")
        log("-" * 65)
        for joint in self.model.joints:
            if joint.id == 0 or joint.nq != 1:  # Skip universe / multi-DOF joints
                continue
            if joint.idx_v in self.idx_right or joint.idx_v in self.idx_left:
                q_l = self.model.lowerPositionLimit[joint.idx_q]
                q_u = self.model.upperPositionLimit[joint.idx_q]
                log(f"{self.model.names[joint.id]:<25} | {q_l:>15.4f} | {q_u:>15.4f}")
        log("=" * 65 + "\n")
