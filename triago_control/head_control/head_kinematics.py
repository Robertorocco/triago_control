"""
Head kinematics: Pinocchio model wrapper for the 7-DOF head chain.

Responsibilities (extracted & cleaned from qp_head_visual_servo.py):
    * Fetch the URDF at runtime from /robot_state_publisher and build a
      Pinocchio model (a static copy also exists at triago_extracted.urdf).
    * Parse *soft* joint limits from the URDF <safety_controller> tags, which
      are tighter (and safer) than the hard limits.
    * Track the live joint configuration from /joint_states (split messages
      handled — TRIAGo publishes arms/head/base in separate messages).
    * Provide, each control tick:
        - T_cam_base : SE(3) pose of the camera optical frame in base_footprint
        - J_cam      : 6x7 LOCAL Jacobian of the camera frame w.r.t. head joints
    * Activate the head velocity controller and deactivate the conflicting
      trajectory controller via the controller_manager services.

WHY express the camera relative to base_footprint explicitly (instead of
trusting Pinocchio's model root): the URDF root link is not guaranteed to be
base_footprint. We look up both frames and compute
    T_cam_base = oMf[base]^-1 * oMf[cam]
so the result is correct no matter what the root is.
"""

import numpy as np
import pinocchio as pin

from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import SwitchController, ListControllers
import rclpy

import triago_control.head_control.config as cfg


class HeadKinematics:
    def __init__(self, node):
        self._node = node
        self._log = node.get_logger()

        self.model = None
        self.data = None
        self.q_real = None

        self.head_v_idx = []        # Pinocchio velocity-space indices of head joints
        self.head_q_idx = []        # Pinocchio config-space indices of head joints
        self.soft_limits = {}       # joint_name -> (min, max)
        self._seen_joints = set()
        self._ready = False

        # EMA velocity reconstruction (same approach as arm QP — encoder vels
        # are unreliable, so we derive from position differences + filter).
        self._last_q = None
        self._last_time = None
        self._v_filtered = None     # (nv,) EMA-filtered velocity

    # ================================================================== #
    # Model construction                                                  #
    # ================================================================== #
    def fetch_urdf(self) -> str:
        """Pull the robot_description string from robot_state_publisher."""
        client = self._node.create_client(GetParameters, "/robot_state_publisher/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            self._log.error("robot_state_publisher not available — cannot fetch URDF.")
            return None
        req = GetParameters.Request()
        req.names = ["robot_description"]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        if future.result() is None:
            self._log.error("Timed out fetching robot_description.")
            return None
        return future.result().values[0].string_value

    def build(self, urdf_path: str):
        """Build the Pinocchio model and map the head joints."""
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_real = pin.neutral(self.model)

        # Map head joints to config/velocity indices.
        for name in cfg.HEAD_JOINTS:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                self.head_q_idx.append(self.model.joints[jid].idx_q)
                self.head_v_idx.append(self.model.joints[jid].idx_v)
            else:
                self._log.error(f"[FATAL] head joint '{name}' not in URDF!")

        # Parse soft limits from the URDF safety_controller tags.
        try:
            from urdf_parser_py.urdf import URDF
            robot = URDF.from_xml_file(urdf_path)
            for j in robot.joints:
                if j.safety_controller is not None:
                    lo = j.safety_controller.soft_lower_limit
                    hi = j.safety_controller.soft_upper_limit
                    if lo is not None and hi is not None:
                        self.soft_limits[j.name] = (float(lo), float(hi))
            self._log.info(f"Loaded {len(self.soft_limits)} soft joint limits from URDF.")
        except Exception as e:                                   # noqa: BLE001
            self._log.warn(f"Could not parse soft limits ({e}); using hard limits.")

        # Cache frame ids we need every tick.
        self._fid_cam = self.model.getFrameId(cfg.CAMERA_OPTICAL_FRAME)
        self._fid_base = (
            self.model.getFrameId(cfg.BASE_FRAME)
            if self.model.existFrame(cfg.BASE_FRAME)
            else None
        )
        if self._fid_base is None:
            self._log.warn(
                f"Frame '{cfg.BASE_FRAME}' not found in model; using model root as base."
            )

    # ================================================================== #
    # Live state                                                          #
    # ================================================================== #
    def update_joint_states(self, names, positions, stamp_sec=None):
        """Absorb a (possibly partial) /joint_states message.

        Also derives joint velocity via finite-difference + EMA filter
        (encoder velocities are unreliable — see Critical Hardware Quirks §7).
        """
        if self.model is None:
            return
        for name, pos in zip(names, positions):
            self._seen_joints.add(name)
            if self.model.existJointName(name):
                idx_q = self.model.joints[self.model.getJointId(name)].idx_q
                self.q_real[idx_q] = pos
        if not self._ready:
            if all(j in self._seen_joints for j in cfg.HEAD_JOINTS):
                self._ready = True
                self._v_filtered = np.zeros(self.model.nv)

        # EMA velocity reconstruction from position differences.
        if self._ready and stamp_sec is not None:
            if self._last_q is not None and self._last_time is not None:
                dt = stamp_sec - self._last_time
                if dt > 1e-5:
                    v_raw = pin.difference(self.model, self._last_q, self.q_real) / dt
                    alpha = cfg.ALPHA_VELOCITY_FILTER
                    self._v_filtered = alpha * v_raw + (1.0 - alpha) * self._v_filtered
            self._last_q = self.q_real.copy()
            self._last_time = stamp_sec

    def is_ready(self) -> bool:
        return self._ready

    # ================================================================== #
    # Kinematics queries                                                  #
    # ================================================================== #
    def forward(self):
        """Run FK once; return (T_cam_base : pin.SE3, J_cam : 6x7 LOCAL)."""
        pin.forwardKinematics(self.model, self.data, self.q_real)
        pin.updateFramePlacements(self.model, self.data)

        oMf_cam = self.data.oMf[self._fid_cam]
        if self._fid_base is not None:
            oMf_base = self.data.oMf[self._fid_base]
            T_cam_base = oMf_base.inverse() * oMf_cam
        else:
            T_cam_base = oMf_cam

        J_full = pin.computeFrameJacobian(
            self.model, self.data, self.q_real, self._fid_cam, pin.ReferenceFrame.LOCAL
        )
        J_cam = J_full[:, self.head_v_idx]          # 6 x 7
        return T_cam_base, J_cam

    def get_head_joint_positions(self):
        return np.array([self.q_real[i] for i in self.head_q_idx])

    def get_head_joint_velocities(self):
        """Return EMA-filtered velocities for the 7 head joints (rad/s)."""
        if self._v_filtered is None:
            return np.zeros(len(cfg.HEAD_JOINTS))
        return np.array([self._v_filtered[i] for i in self.head_v_idx])

    def get_head_joint_limits(self):
        """Return (q_min, q_max) arrays for the 7 head joints (soft if known)."""
        q_min = np.zeros(len(cfg.HEAD_JOINTS))
        q_max = np.zeros(len(cfg.HEAD_JOINTS))
        for i, name in enumerate(cfg.HEAD_JOINTS):
            if name in self.soft_limits:
                q_min[i], q_max[i] = self.soft_limits[name]
            else:
                idx_q = self.head_q_idx[i]
                q_min[i] = self.model.lowerPositionLimit[idx_q]
                q_max[i] = self.model.upperPositionLimit[idx_q]
        return q_min, q_max

    # ================================================================== #
    # Controller switching                                                #
    # ================================================================== #
    def switch_controllers(self):
        """Activate the head velocity controller; stop the trajectory one."""
        list_client = self._node.create_client(
            ListControllers, "/controller_manager/list_controllers"
        )
        if not list_client.wait_for_service(timeout_sec=3.0):
            self._log.error("controller_manager/list_controllers unavailable.")
            return False

        future = list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=3.0)
        if future.result() is None:
            self._log.error("Timed out listing controllers.")
            return False

        active = [c.name for c in future.result().controller if c.state == "active"]
        to_start, to_stop = [], []
        if cfg.HEAD_CONTROLLER not in active:
            to_start.append(cfg.HEAD_CONTROLLER)
        if cfg.HEAD_CONFLICTING_CONTROLLER in active:
            to_stop.append(cfg.HEAD_CONFLICTING_CONTROLLER)

        if not to_start and not to_stop:
            self._log.info("Head controllers already in the correct state.")
            return True

        self._log.info(f"Switching controllers -> START {to_start}, STOP {to_stop}")
        switch_client = self._node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        if not switch_client.wait_for_service(timeout_sec=3.0):
            self._log.error("controller_manager/switch_controller unavailable.")
            return False

        req = SwitchController.Request()
        req.activate_controllers = to_start
        req.deactivate_controllers = to_stop
        req.strictness = SwitchController.Request.STRICT
        future = switch_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=3.0)

        ok = future.result() is not None and future.result().ok
        if ok:
            self._log.info("Head controller switch succeeded.")
        else:
            self._log.error("Head controller switch FAILED.")
        return ok
