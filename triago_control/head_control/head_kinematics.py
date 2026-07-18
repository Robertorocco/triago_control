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

import contextlib
import os

import numpy as np
import pinocchio as pin

from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import SwitchController, ListControllers
import rclpy

import triago_control.head_control.config as cfg


@contextlib.contextmanager
def _suppress_native_output():
    """Pinocchio's URDF parser warns on raw stdout/stderr (C++-level, not
    Python logging) for tags it doesn't recognise -- TIAGo's ros2_control/
    suspension/laser tags, harmless and not ours to fix upstream."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


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

        self._ctrl_started = []     # controllers WE activated (restore_controllers reverses)
        self._ctrl_stopped = []     # controllers WE deactivated

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

        self._ensure_camera_frame()
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

    def _ensure_camera_frame(self):
        """Inject cfg.CAMERA_OPTICAL_FRAME into the model if the URDF lacks it.

        Real hardware: the camera is a separately-mounted module, not baked
        into the base URDF the way Gazebo's sim camera plugin is, so
        getFrameId would silently return an out-of-range sentinel (crashing
        forward()'s oMf indexing). Same pattern as robot_kinematics.py's
        _ensure_grasping_frames() -- inject at the MEASURED arm_head_tool_link
        -> head_arm_rgbd_link mount offset (see launch/head_real.launch.py's
        static_transform_publisher). This is the camera body's own reference
        frame, not the exact colour/depth sensor optical centre within it --
        adequate for the look-at controller's coarse aiming, not precision work.
        """
        if self.model.existFrame(cfg.CAMERA_OPTICAL_FRAME):
            return  # sim: already native to the URDF
        parent_body_name = 'arm_head_tool_link'
        if not self.model.existFrame(parent_body_name):
            self._log.error(
                f"Cannot inject '{cfg.CAMERA_OPTICAL_FRAME}': parent frame "
                f"'{parent_body_name}' not found in model.")
            return
        t_mount = np.array([-0.056783, 0.034171, 0.011676])
        R_mount = pin.Quaternion(0.69556217, 0.00914972, -0.71833972, 0.00987883).matrix()
        # realsense2_camera (like every ROS camera driver) publishes the mount
        # link in BODY convention (X-fwd, Y-left, Z-up) then a fixed rotation to
        # the OPTICAL convention (Z-fwd, X-right, Y-down) that images -- and
        # look_at_controller.py's "z-axis is the current look direction" law --
        # actually use. Standard ROS body->optical quaternion (x,y,z,w) =
        # (-0.5, 0.5, -0.5, 0.5); apply it on top of the measured mount offset,
        # or the injected frame's Z axis doesn't point where the lens looks.
        R_optical = pin.Quaternion(0.5, -0.5, 0.5, -0.5).matrix()
        placement = pin.SE3(R_mount, t_mount) * pin.SE3(R_optical, np.zeros(3))
        parent_frame_id = self.model.getFrameId(parent_body_name)
        parent_joint_id = self.model.frames[parent_frame_id].parentJoint
        parent_placement = self.model.frames[parent_frame_id].placement
        frame_placement = parent_placement * placement
        new_frame = pin.Frame(
            cfg.CAMERA_OPTICAL_FRAME, parent_joint_id, parent_frame_id,
            frame_placement, pin.FrameType.OP_FRAME,
        )
        self.model.addFrame(new_frame)
        self.data = self.model.createData()   # rebuild data for the new frame
        self._log.info(
            f"[Init] Injected frame '{cfg.CAMERA_OPTICAL_FRAME}' into Pinocchio "
            f"model (parent: {parent_body_name}).")

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

    def solve_posture_for_position(self, target_pos, iters=150, tol=0.03):
        """Numeric position-only IK for the camera ORIGIN to reach target_pos
        (base_footprint, (3,)): damped least-squares task + null-space
        regularisation toward the CURRENT posture (so the solve stays close to
        a sensible configuration rather than wandering). AIM is deliberately
        NOT solved here -- the live look-at QP (LookAtController) already
        closes that loop every control tick; this only needs to produce a
        reachable null-space POSTURE target (config's HEAD_POSTURE_TARGET
        role), used by main_head.py's close-inspection sequence to derive a
        closer/differently-angled view from the LIVE measured table geometry
        instead of an offline-computed constant.

        Runs on a LOCAL copy of q_real -- never mutates live state (self.data
        gets overwritten with the real q on the next forward() tick regardless).

        Returns (q_head, info): the BEST-EFFORT head-joint vector (7,) ALWAYS
        (clipped to the soft joint limits -- the closest the head can get), plus
        a diagnostic dict so the caller can drive there anyway and explain WHY a
        view is only partially reached:
            reachable   : bool  -- final residual within `tol`
            residual_m  : float -- ‖target - achieved‖ (m)
            err_base    : (3,)  -- target - achieved in base frame (which way it's short)
            achieved    : (3,)  -- camera origin actually reached, base frame
            target      : (3,)  -- the requested camera position
            saturated   : list of (joint_name, side, value, limit) for head joints
                          pinned at a soft limit -- the mechanical reason it's short.
        """
        q_full = self.q_real.copy()
        q_min, q_max = self.get_head_joint_limits()
        buf = 0.15   # same safety buffer used elsewhere (JOINT_LIMIT_BUFFER-ish)
        lo, hi = q_min + buf, q_max - buf
        q_seed = np.array([q_full[i] for i in self.head_q_idx])
        q_head = q_seed.copy()
        err = np.zeros(3)
        p = np.zeros(3)

        for _ in range(iters):
            for i, idx in enumerate(self.head_q_idx):
                q_full[idx] = q_head[i]
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            oMf_cam = self.data.oMf[self._fid_cam]
            p = (self.data.oMf[self._fid_base].inverse() * oMf_cam).translation \
                if self._fid_base is not None else oMf_cam.translation
            err = np.asarray(target_pos, dtype=float) - p
            if np.linalg.norm(err) < 1e-3:
                break
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q_full, self._fid_cam,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J_full[:3, self.head_v_idx]              # 3x7 position Jacobian
            JJt = J @ J.T + 1e-4 * np.eye(3)
            dq_task = J.T @ np.linalg.solve(JJt, err)
            Jpinv = J.T @ np.linalg.inv(JJt)
            N = np.eye(len(self.head_v_idx)) - Jpinv @ J
            dq_null = 0.2 * (q_seed - q_head)
            dq = dq_task + N @ dq_null
            q_head = np.clip(q_head + np.clip(dq, -0.15, 0.15), lo, hi)

        # Which head joints ended pinned at a soft limit -- the joint(s) that
        # physically block reaching the target (buf away from the raw limit).
        saturated = []
        for i, name in enumerate(cfg.HEAD_JOINTS):
            if q_head[i] <= lo[i] + 1e-3:
                saturated.append((name, "lower", float(q_head[i]), float(q_min[i])))
            elif q_head[i] >= hi[i] - 1e-3:
                saturated.append((name, "upper", float(q_head[i]), float(q_max[i])))

        residual = float(np.linalg.norm(err))
        info = {
            "reachable": residual <= tol,
            "residual_m": residual,
            "err_base": np.asarray(err, dtype=float).copy(),
            "achieved": np.asarray(p, dtype=float).copy(),
            "target": np.asarray(target_pos, dtype=float).copy(),
            "saturated": saturated,
        }
        return q_head, info

    def get_head_joint_positions(self):
        return np.array([self.q_real[i] for i in self.head_q_idx])

    def get_frame_in_base(self, frame_name):
        """Pinocchio FK pose (R, t) of an arbitrary frame in base_footprint.

        Used to cross-check against TF. Returns (None, None) if the frame is
        not in the model. Assumes forward() / FK has run with current q.
        """
        if not self.model.existFrame(frame_name):
            return None, None
        pin.forwardKinematics(self.model, self.data, self.q_real)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.model.getFrameId(frame_name)]
        if self._fid_base is not None:
            T = self.data.oMf[self._fid_base].inverse() * oMf
        else:
            T = oMf
        return T.rotation.copy(), T.translation.copy()

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
            self._ctrl_started = to_start
            self._ctrl_stopped = to_stop
        else:
            self._log.error("Head controller switch FAILED.")
        return ok

    def restore_controllers(self):
        """Reverse switch_controllers() on shutdown: stop what we started,
        start what we stopped -- returns the robot to its pre-launch state."""
        if not self._ctrl_started and not self._ctrl_stopped:
            return
        print(f"[Shutdown] Restoring original head controllers: "
              f"+{self._ctrl_stopped} | -{self._ctrl_started} ...", flush=True)
        switch_client = self._node.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )
        if not switch_client.wait_for_service(timeout_sec=2.0):
            msg = ("[Shutdown] Switch Controller service unavailable -- cannot restore "
                   f"original head controller state (was: +{self._ctrl_started} "
                   f"-{self._ctrl_stopped}).")
            print(msg, flush=True)
            self._log.error(msg)
            return
        req = SwitchController.Request()
        req.activate_controllers = self._ctrl_stopped
        req.deactivate_controllers = self._ctrl_started
        req.strictness = SwitchController.Request.STRICT
        future = switch_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=3.0)
        if future.result() is not None and future.result().ok:
            print("[Shutdown] Original head controller state restored.", flush=True)
            self._log.info("Original head controller state restored.")
        else:
            print("[Shutdown] Restore Switch Service Failed.", flush=True)
            self._log.error("Restore Switch Service Failed.")
