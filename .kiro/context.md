# AI Agent Context — triago_control

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-06-26 (added §13.5 simulation flow; fixed F_guide → velocity-field guidance, green-gripper marker viz robustness, and SA frequency-plot layout)

---

## 1. Project Identity

- **Package name**: `triago_control`
- **ROS 2 distribution**: Humble (Ubuntu 22.04)
- **Robot**: PAL Robotics TRIAGo++ (bimanual variant, mobile base, lift torso, head)
- **Maintainer**: Roberto Rocco (roberto.rocco@irisa.fr)
- **Repository**: https://github.com/Robertorocco/triago_control
- **Build system**: `ament_cmake` + `ament_cmake_python` (hybrid C++/Python package)
- **Runtime environment**: Dockerized ROS 2 workspace, shared via `~/exchange/` with host

---

## 2. Workspace Layout

```
~/exchange/ros2-ws/
├── build/          (colcon output — not tracked)
├── install/        (colcon output — not tracked)
├── log/            (colcon output — not tracked)
└── src/
    ├── triago_control/              ← THIS REPO (git-tracked, contains both packages)
    │   ├── (triago_control package files)
    │   └── haption_teleoperation/   ← haptic device interface package (inside same repo)
    ├── haption_interface/           ← hardware driver (not maintained by user)
    ├── pal-packages/                ← PAL vendor packages (not maintained)
    ├── demo-square-cpp/             ← legacy demo (unused)
    └── tsid_ros2/                   ← legacy TSID controller (superseded by this package)
```

---

## 3. Package Structure

```
triago_control/
├── CMakeLists.txt
├── package.xml
├── LICENSE                          (BSD-3-Clause)
├── README.md
├── triago_extracted.urdf            (full TRIAGo URDF, extracted from robot_state_publisher)
├── .kiro/
│   └── context.md                   ← THIS FILE
├── config/
│   ├── qp_debug.rviz               (RViz layout for live telemetry)
│   ├── Recording_Rviz.rviz
│   └── trajectory_endpoints.yaml   (endpoint presets + flags for trajectory_generator.py)
├── launch/
│   └── visualize.launch.py
├── scripts/                         ← EXECUTABLE ENTRY POINTS (ros2 run targets)
│   ├── qp_arm_teleop/
│   │   ├── main_qp_controller.py       ★ primary: QP-CLF-CBF safety loop
│   │   ├── main_shared_autonomy.py     ★ primary: intent prediction + blending
│   │   ├── trajectory_generator.py     ★ open-loop quintic reference source (robustness tests)
│   │   ├── base_controller.py          mobile base velocity teleop
│   │   ├── keyboard_teleop.py          keyboard cartesian jog
│   │   ├── plotter.py                  live matplotlib dashboard
│   │   └── drift_evaluator_node.py     tracking error analysis
│   ├── head_controller/
│   │   └── qp_head_visual_servo.py     ★ primary: QP-based visual servoing for head camera
│   ├── visualize_live_shadow.py
│   └── workspace_mapper.py
└── triago_control/                  ← IMPORTABLE PYTHON LIBRARY
    ├── __init__.py
    ├── qp_visualizer.py             (shared utility: debug overlays for RViz)
    ├── qp_controller/               ← QP safety math (used by main_qp_controller)
    │   ├── __init__.py
    │   ├── config.py                    ALL tunable parameters (single source of truth)
    │   ├── robot_kinematics.py          Pinocchio model, FK, EMA filter, digital twin
    │   ├── collision_manager.py         hppfcl geometry, SoftMin CBF, dynamic margin
    │   ├── qp_formulator.py            CLF-CBF-QP: H/g/C/b assembly, quadprog solver
    │   ├── shared_autonomy_handler.py   gripper cmds, CBF-bypass, cylinder re-parenting
    │   ├── visualization_engine.py      thread-safe Meshcat + RViz markers
    │   └── qp_visualizer_tutorial.py    debug tether/overlay helper (legacy name)
    └── shared_autonomy/             ← intent prediction (used by main_shared_autonomy)
        ├── __init__.py
        ├── belief_estimator.py          Bayesian intent inference
        ├── goal_set.py                  dynamic goal pose computation
        ├── grasp_state_machine.py       pick FSM (approach→contact→close→attach)
        └── plot_manager.py              live plot helper for shared autonomy telemetry
```

---

## 4. haption_teleoperation Package (Haptic Device Interface)

A **separate ROS 2 package** living inside the same repository, responsible for the bidirectional interface between the Haption Virtuose haptic device and the TRIAGo teleoperation pipeline.

### 4.1 Package Structure

```
haption_teleoperation/
├── CMakeLists.txt               (ament_cmake, links VirtuoseAPI + libtirpc)
├── package.xml                  (depends: rclcpp, geometry_msgs, sensor_msgs, rclpy)
├── include/
│   └── VirtuoseAPI.h            (proprietary C header, v4.04, Haption S.A.)
├── lib/
│   └── libVirtuoseAPI.so        (proprietary shared library — device driver)
├── src/                         ← C++ NODES (only code that touches the hardware API)
│   ├── virtuose_server_node.cpp     ★ primary: 150Hz impedance-mode device server
│   └── calibration_main.cpp         utility: manual joint-limit discovery tool
└── scripts/                     ← PYTHON NODES (teleoperation logic)
    ├── teleop_triago_clutch.py      ★ active: clutch-indexing teleop (mouse-mode)
    ├── haptic_force_manager.py      ★ active: force-feedback superposition & passivity
    ├── teleop_triago.py             forward teleop (no clutch, continuous integration)
    ├── teleop_demo_integrator.py    RViz-only demo (no robot, visualizes in "map" frame)
    ├── haption_plotter.py           live matplotlib: pose/vel/force from virtuose topics
    └── workspace_debug_visualizer.py  6-window 3D workspace alignment debugger
```

### 4.2 Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  HAPTIC DEVICE (Haption Virtuose, 6-DOF)                     │
│                                                                             │
│   virtGetPosition / virtGetPhysicalSpeed / virtGetButton (read)             │
│   virtSetForce (write, impedance mode)                                      │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ VirtuoseAPI calls @ 150 Hz
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  virtuose_server_node (C++)                                                  │
│  ─────────────────────────────                                               │
│  Publishes:  virtuose/pose  (Pose, quat [x,y,z,w])                          │
│              virtuose/velocity (Twist, 6-DOF)                                │
│              virtuose/button (Bool, right button = clutch)                    │
│              virtuose/articular_position (Float64MultiArray, 6 joints)        │
│  Subscribes: virtuose/force_cmd (Wrench) → virtSetForce every tick           │
└──────────┬──────────────────────────────────┬────────────────────────────────┘
           │                                  │
    (reads pose/vel/button)            (writes force_cmd)
           │                                  │
           ▼                                  │
┌──────────────────────────┐    ┌─────────────┴─────────────────────────────────┐
│ teleop_triago_clutch.py  │    │ haptic_force_manager.py                        │
│ ─────────────────────────│    │ ──────────────────────                         │
│ Clutch-indexing teleop:  │    │ Force feedback computation:                    │
│ • Maps Haption twist to  │    │ • F_sync (spring-damper tether)                │
│   TRIAGo frame (180° Z)  │    │ • F_cbf (repulsive obstacle force, LPF'd)     │
│ • Integrates pose when   │    │ • F_guide (belief-weighted policy blend)        │
│   clutch released         │    │ • F_limit (75Hz vibration near joint limits)   │
│ • Freezes when clutch     │    │ • Clutch alignment torque (orientation guide)  │
│   pressed                 │    │ • Passivity Observer + Controller              │
│                           │    │ • Global damping, safety clipping              │
│ Publishes:               │    │                                                │
│ /arm_right/cartesian_    │    │ Subscribes to:                                 │
│ reference (13-float msg)  │    │  /arm_right/cartesian_reference, /qp_debug/*,  │
│                           │    │  /collision_constraints, /shared_autonomy/*     │
│ Subscribes to:           │    │  virtuose/velocity, virtuose/button, etc.       │
│  virtuose/velocity       │    │                                                │
│  virtuose/button         │    │ Publishes:                                     │
│  /qp_debug/ee_real       │    │  virtuose/force_cmd (Wrench)                   │
└──────────────────────────┘    └────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ main_qp_controller.py (triago_control) │
│ Consumes /arm_right/cartesian_reference│
│ and tracks with CLF-CBF safety         │
└────────────────────────────────────────┘
```

### 4.3 C++ Node: virtuose_server_node

- **Frequency**: 150 Hz (microsecond-precise wall timer)
- **Command mode**: `COMMAND_TYPE_IMPEDANCE` (force in, position out)
- **Indexing**: `INDEXING_NONE` (button must be held for the device to track)
- **IP**: `127.0.0.1#53210` (communicates via `libtirpc` with device controller)
- **Startup sequence**: open → configure → power on → 3s relay wait → loop
- **Force subscribe pattern**: asynchronous `ForceCallback` writes to `current_force[6]`; the 150 Hz timer reads and applies it with `virtSetForce` every tick

### 4.4 Key Script: teleop_triago_clutch.py

Implements **clutch-indexing** (mouse-mode) teleoperation:
- **Initialization**: waits for `/qp_debug/ee_real` to anchor integration at current robot EE pose
- **Frame mapping**: Haption→TRIAGo = 180° rotation around Z (negate X, negate Y, keep Z)
- **Clutch logic**: when button pressed → pose frozen, zero velocity published; when released → integration resumes from frozen pose
- **Output protocol**: 13-element `Float64MultiArray` = `[pos(3), rpy(3), vel_lin(3), vel_ang(3), task_dim(1)]`
- **task_dim** flag: 6.0 = full 6D control, 5.0 = free rotation around approach axis

### 4.5 Key Script: haptic_force_manager.py

> **Active script name**: the running force-feedback node is
> **`haptic_force_manager_tutorial.py`** (see §13.5). The description below
> documents the multi-layer force architecture; the tutorial variant currently
> runs with `DEBUG_ONLY_GUIDE = True`, emitting only `F_guide`.

Multi-layer force-feedback superposition node. Computes and sums:

| Layer | Symbol | Description |
|-------|--------|-------------|
| Sync | F_sync | Spring-damper (Kp=10, Kd=0) tethering user to robot tracking error |
| CBF | F_cbf | Repulsive force from collision barrier gradient × λ_cbf, tanh-saturated, LPF'd (α=0.15) |
| Guide | F_guide | Belief-weighted policy rendered as a **velocity field** the handle should follow: `F = D·(map(pi_blend) − v_handle)·confidence`, entropy-gated, tanh-saturated (`D_lin=28`, `D_ang=0.45`, `MAX 3.5N/0.25Nm`), LPF'd. Intrinsically damped → no runaway |
| Limit | F_limit | 75 Hz square-wave vibration when Haption joints approach mechanical limits |
| Clutch align | — | Rotational spring (K=10 Nm/rad) pulling handle toward target orientation during clutch |
| Global damping | — | Viscous Kd_lin=0.7, Kd_ang=0.1 for stability |

**Passivity architecture**:
- **Observer (PO)**: integrates power = −(wrench · twist) to track energy balance
- **Controller (PC)**: when energy < 0 (active), injects dissipative damping β·v, saturated at MAX_PC_FORCE=5N / MAX_PC_TORQUE=0.5Nm
- **PC enable toggle**: `ENABLE_PASSIVITY_CONTROL` flag (currently `False` for tuning)

**Safety clipping**: global MAX_FORCE=10N, MAX_TORQUE=1Nm after all layers summed.

**Live plotting**: 3 matplotlib windows (force superposition 5×2 grid, passivity observer, twist analyzer) running on main thread with ROS spinning on daemon thread.

### 4.6 Frame Convention (Haption ↔ TRIAGo Mapping)

The Haption device base frame has **X pointing toward the user** and **Y to the right** (operator's perspective). The TRIAGo `base_footprint` has X forward and Y left. The relationship is a **pure 180° rotation around Z**:

```
TRIAGo_vel.x = -Haption_vel.x
TRIAGo_vel.y = -Haption_vel.y
TRIAGo_vel.z = +Haption_vel.z
(same for angular velocities)
```

For force feedback (Haption←TRIAGo), the **same** negation applies (transpose of rotation = same rotation for 180°).

### 4.7 Build & Run (haption_teleoperation)

```bash
# Build (separate package)
cd ~/exchange/ros2-ws
colcon build --packages-select haption_teleoperation
source install/setup.bash

# Run device server (requires hardware or simulator on 127.0.0.1#53210)
ros2 run haption_teleoperation virtuose_server_node

# Run clutch teleop
ros2 run haption_teleoperation teleop_triago_clutch.py

# Run force feedback
ros2 run haption_teleoperation haptic_force_manager.py

# Calibration utility (discover joint limits by manually moving device)
ros2 run haption_teleoperation virtuose_calibration

# Debug/visualization
ros2 run haption_teleoperation haption_plotter.py
ros2 run haption_teleoperation workspace_debug_visualizer.py
```

### 4.8 Gazebo Link Attacher (IFRA_LinkAttacher)

External dependency for kinematic object attachment during grasping in Gazebo.
Creates a fixed joint between the gripper and a grasped object via a ROS 2 service.

```bash
# Install (clone into workspace src/ — NOT part of triago_control repo)
cd ~/exchange/ros2-ws/src
git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git
cd ~/exchange/ros2-ws
colcon build --packages-up-to ros2_linkattacher
source install/setup.bash

# Required in world file:
#   <plugin name="ros2_linkattacher" filename="libgazebo_link_attacher.so"/>

# Required environment (before launching Gazebo):
#   export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/exchange/ros2-ws/install/ros2_linkattacher/lib

# Services exposed:
#   /ATTACHLINK (linkattacher_msgs/srv/AttachLink)
#   /DETACHLINK (linkattacher_msgs/srv/DetachLink — if available)

# Manual test (attach cylinder to gripper):
ros2 service call /ATTACHLINK linkattacher_msgs/srv/AttachLink \
  "{model1_name: 'tiago', link1_name: 'gripper_right_grasping_link', model2_name: 'red_cylinder', link2_name: 'link'}"
```

---

## 5. Head Controller (Vision-Based Independent Head Servoing)

An **independent subsystem** controlling TRIAGo's 7-DOF head arm to keep both hands in the camera field-of-view. Runs at its own frequency, decoupled from the arm QP safety loop — the head does NOT share the CLF-CBF formulation used for the arms.

### 5.1 Design Philosophy

The head is mechanically identical to the left/right arms (7-DOF, same hardware) but serves a fundamentally different purpose: it carries a camera (RealSense D405 RGBD) and must keep the operator's working hands visible. Future evolution will add image-processing-based algorithms (e.g., object detection, gaze prediction), but the current starting point uses **kinematic hand tracking** (Pinocchio FK projects hand positions into the camera frame — no actual image data required yet).

**Key architectural decision**: the head controller is **fully independent** from the arm controller. It:
- Has its own QP solver instance (not shared with `main_qp_controller.py`)
- Commands its own velocity controller (`arm_head_joint_space_controller_vel`)
- Runs at its own loop rate (currently event-driven via `spin_once`)
- Does NOT subscribe to or publish `/arm_*/cartesian_reference`
- Does NOT participate in the arm CBF collision pairs

### 5.2 Kinematic Chain

```
Head joints (7-DOF):  arm_head_1_joint → arm_head_7_joint
Head links:           arm_head_1_link  → arm_head_7_link
End-effector frame:   gripper_head_camera_rgbd_color_optical_frame
Tracked targets:      arm_right_tool_link, arm_left_tool_link (both hands centroid)
```

### 5.3 Control Architecture (2.5D Visual Servoing QP)

The controller uses a **two-stage state machine** based on whether the hands are currently visible in the camera FOV:

| Stage | Condition | Strategy |
|-------|-----------|----------|
| **PBVS (Look-At)** | Hands outside FOV or behind camera | 3D rotational servoing: cross(z_cam, dir_to_centroid) → angular velocity via J_rot |
| **IBVS (Pixel Tracking)** | Both hands inside FOV margin | 2.5D image-based visual servoing: interaction matrix Ls maps pixel + depth error to camera twist |

**QP formulation** (both stages):

Decision vector: `x = [dq_head (7), slack (3)]`

Cost:
- Joint velocity regularization with per-joint weights `[50, 40, 30, 10, 5, 1, 1]` (heavier on base joints → smoother motion, wrist joints freer)
- Slack penalty: `W_SLACK_PIXELS=1` for u,v errors; `W_SLACK_DEPTH=1e4` for depth (normalizes pixel vs. meter scales)
- Secondary postural task: centering spring toward mid-range (K_POSTURE=0.05)

Equality constraint (CLF-like):
- `J_task · dq - slack = -λ · e` (λ_visual = 1)
- In IBVS: J_task = Ls @ J_cam (3×7), e = [u-u_target, v-v_target, Z-Z_target]
- In PBVS: J_task = J_rot (3×7), e = ω_desired (cross-product look-at)

Inequality constraints (CBF-style):
- **FOV barriers** (IBVS only): each hand must stay ≥ FOV_MARGIN=50px from image edges. Per-hand, 4 barriers (left, right, top, bottom) using the interaction matrix gradient.
- **Joint limits**: velocity-aware position buffer (SAFE_BUF=min(0.15, 10% of range), γ=2.0), capped at MAX_VELOCITY=0.15 rad/s. Uses **soft limits** from URDF safety_controller tags when available.

Solver: `quadprog.solve_qp` (same as arm QP).

### 5.4 Camera Parameters

```python
# RealSense D405 (720p approximation)
CAM_W, CAM_H = 1280, 720
CAM_FX, CAM_FY = 640.0, 640.0
CAM_CX, CAM_CY = 640.0, 360.0

# Servoing targets
TARGET_U = CAM_CX      # Keep centroid at image center (u)
TARGET_V = CAM_CY      # Keep centroid at image center (v)
TARGET_Z = 1.0         # Keep centroid 1 meter from camera
```

### 5.5 Controller Switching

The node automatically handles controller activation on startup:
- **Activates**: `arm_head_joint_space_controller_vel`
- **Deactivates** (conflicting): `arm_head_controller` (default trajectory controller)
- Uses `/controller_manager/list_controllers` + `/controller_manager/switch_controller` services

### 5.6 Collision Avoidance (Simplified)

Unlike the arm QP (which uses SoftMin CBF over 60 pairs), the head has a **lightweight collision model**:
- Head links: capsules (radius=0.08, length=0.2) for each of 7 links
- Body parts: boxes for `base_link` (0.6×0.5×0.27) and `torso_lift_link` (0.2×0.2×0.6)
- Virtual wall: box at (0.5, 0.0, 1.0) of size (1.0×0.02×2.0)
- Collision pairs: head-vs-body + head-vs-wall only (no inter-arm pairs)

**Note**: collision avoidance constraints from this model are NOT currently wired into the QP as CBF inequalities — the model is built but the distance-based barriers are not yet formulated. This is a planned extension.

### 5.7 ROS 2 Interface

**Subscriptions:**
| Topic | Type | Purpose |
|-------|------|---------|
| `/joint_states` | JointState | Full robot state (head + arms, split messages handled) |

**Publications:**
| Topic | Type | Purpose |
|-------|------|---------|
| `/arm_head_joint_space_controller_vel/joint_velocity_cmd` | Float64MultiArray | 7-DOF head velocity command |
| `/qp_debug/qdot_err` | Float64MultiArray | Solved joint velocities (telemetry) |
| `/qp_debug/xdot_err` | Float64MultiArray | Visual/rotational error (telemetry) |
| `/qp_debug/head_cartesian_cmd` | TwistStamped | Cartesian camera velocity (debug) |
| `/qp_debug/camera_ray` | Marker | Optical axis arrow in RViz |
| `/qp_debug/target_centroid` | Marker | Green sphere at hands centroid |
| `/qp_debug/virtual_wall_marker` | Marker | Wall visualization |

### 5.8 Build & Run

```bash
# Build (part of triago_control package)
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

# Run head visual servoing
ros2 run triago_control qp_head_visual_servo.py
```

### 5.9 Current Limitations & Future Work

- **No actual image processing yet**: hand positions are computed via Pinocchio FK, not from camera images. This is the "starting point" — future work will add detection/tracking from the RGB stream.
- **Collision CBF not wired**: the hppfcl collision model is built but distance constraints are not yet formulated as QP inequalities.
- **No shared config file**: gains are hard-coded in-script (unlike the arm QP which uses `config.py`). Will be refactored as the module matures.
- **Loop rate**: currently event-driven (`spin_once` + `solve_and_publish` per iteration). Future: dedicated timer at a fixed frequency.

---

## 6. Entry Point → Library Dependency Map

```
main_qp_controller.py
  imports: triago_control.qp_controller.config
           triago_control.qp_controller.robot_kinematics.RobotKinematics
           triago_control.qp_controller.collision_manager.CollisionManager
           triago_control.qp_controller.qp_formulator.QPFormulator
           triago_control.qp_controller.shared_autonomy_handler.SharedAutonomyHandler
           triago_control.qp_controller.visualization_engine.VisualizationEngine

main_shared_autonomy.py
  imports: triago_control.shared_autonomy.belief_estimator.BeliefEstimator
           triago_control.shared_autonomy.goal_set.GoalSet
           triago_control.shared_autonomy.grasp_state_machine.GraspStateMachine
           triago_control.shared_autonomy.plot_manager.PlotManager
  publishes to: /arm_right/cartesian_reference, /arm_left/cartesian_reference
  subscribes to: /collision_constraints (from main_qp_controller)

trajectory_generator.py
  reads: config/trajectory_endpoints.yaml (endpoint presets + behaviour flags;
         overridable at runtime via the `config_file` ROS parameter)
  subscribes to: /qp_debug/ee_real (sample start pose), /qp_debug/lambda_cbf (time scaling)
  publishes to: /arm_right/cartesian_reference, /arm_left/cartesian_reference
                (13-float 6-DOF refs: [xyz, rpy, xdot, w, task_dim]),
                /trajectory/phase, /trajectory/phase_marker,
                /trajectory/reference_state, /trajectory/time_scale
  NOTE: does NOT import or modify main_qp_controller — it is just another source
        on the existing cartesian-reference contract (like keyboard_teleop).

[haption_teleoperation package]

virtuose_server_node (C++, 150 Hz)
  hardware API: VirtuoseAPI (impedance mode)
  publishes: virtuose/pose, virtuose/velocity, virtuose/button,
             virtuose/articular_position
  subscribes: virtuose/force_cmd

teleop_triago_clutch.py
  subscribes: virtuose/velocity, virtuose/button, /qp_debug/ee_real
  publishes: /arm_right/cartesian_reference (13-float protocol)
  NOTE: another source on the cartesian-reference contract (replaces keyboard_teleop
        or trajectory_generator as the active teleop input)

haptic_force_manager.py
  subscribes: /arm_right/cartesian_reference, /qp_debug/ee_real,
              virtuose/velocity, virtuose/button, virtuose/pose,
              virtuose/articular_position, /collision_constraints,
              /qp_debug/lambda_cbf, /shared_autonomy/goal_names,
              /shared_autonomy/goal_probabilities, /shared_autonomy/user_policy
  publishes: virtuose/force_cmd (Wrench, consumed by virtuose_server_node)

[head_controller — independent subsystem]

qp_head_visual_servo.py
  subscribes: /joint_states (full robot, for FK of head + hands)
  publishes: /arm_head_joint_space_controller_vel/joint_velocity_cmd (7-DOF velocities)
             /qp_debug/qdot_err, /qp_debug/xdot_err, /qp_debug/head_cartesian_cmd
             /qp_debug/camera_ray, /qp_debug/target_centroid (RViz markers)
  NOTE: fully independent — does NOT share the arm QP solver, does NOT
        subscribe to /arm_*/cartesian_reference. Uses its own Pinocchio model
        instance and quadprog call. Future: will add image-based input.
```

---

## 7. Import Convention

All library imports use the **fully-qualified package path**:

```python
import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.shared_autonomy.belief_estimator import BeliefEstimator
```

**Never** use bare `import config` — it collides with system modules. Always anchor to `triago_control.*`.

---

## 8. Critical Hardware Quirks

1. **Corrupted encoder velocities (SIMULATION ONLY)**: In Gazebo, TRIAGo's joint_states `velocity` field is unreliable. The controller derives velocity from position differences and filters with a first-order EMA (`ALPHA_FILTER = 0.5`). On **real hardware**, the velocity sensors work correctly and are used directly — no differentiation or filtering.

2. **REAL_HARDWARE auto-detection**: The system automatically detects whether it is running on real hardware or in simulation by inspecting the URDF fetched from `robot_state_publisher`:
   - **Gazebo URDF** contains `gripper_right_grasping_link` and `gripper_left_grasping_link` natively → `REAL_HARDWARE = False`
   - **Real TIAGo Pro URDF** does NOT contain these frames → `REAL_HARDWARE = True`
   
   This detection happens at startup in `main_qp_controller.py` before building the Pinocchio model. When `REAL_HARDWARE = True`:
   - The missing grasping frames are **injected** into the Pinocchio model (via `robot_kinematics._ensure_grasping_frames()`) and **broadcast as static TFs** (so RViz and other nodes see them too).
   - Joint velocities are read **directly** from `/joint_states` `msg.velocity` (no EMA differentiation).
   
   A colored console banner announces the detected environment at startup:
   - Cyan: `[ENV] REAL HARDWARE detected`
   - Green: `[ENV] SIMULATION detected`

3. **Meshcat thread safety**: Meshcat's WebSocket is NOT thread-safe. ROS callbacks must NEVER call the viewer. Only the dedicated `_run_viz` thread (in `visualization_engine.py`) owns Meshcat WebSocket calls. Callbacks mutate `meshColor` under a `threading.Lock` and set `meshcat_reload_pending = True`.

4. **Controller switching**: TRIAGo requires explicit activation of velocity controllers (`arm_right_joint_space_controller_vel`, `arm_left_joint_space_controller_vel`) and deactivation of conflicting trajectory controllers before the QP can command the arms.

---

## 9. Mathematical Core (QP-CLF-CBF)

Decision vector: `x = [q_dot (nv), delta_right, delta_left]`

**Cost** (minimize):
- Joint velocity regularization (damping λ = 10.0)
- Posture centering spring toward neutral (Kp = 0.1)
- Slack penalty (adaptive per-arm weighting)

**Constraints** (C'x >= b):
- **CLF (task tracking)**: Perfect Scalar Inequality CLF with diagonal task weights [pos=10, ori=1]. Two formulations available (`COMPARISON_CLF` flag): normalized (unit-error) or raw.
- **CBF (collision avoidance)**: SoftMin aggregation over K_MAX_PAIRS=60 closest collision pairs. Dynamic margin = d_safe_base + k_v_safe * ||v||.
- **Joint limits**: velocity-aware position buffer (CBF-style).

Solver: `quadprog.solve_qp` (active-set method).

---

## 10. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting**: each arm's slack weight drops (toward `BASE_WEIGHT_SLACK=5`) when its shadow price grows, letting the slack absorb more tracking error near obstacles. In free space it rises (toward `MAX_WEIGHT_SLACK=50`) for tighter tracking.
- **Dynamic gamma (CLF)**: the CLF convergence rate γ drops exponentially with the collision Lagrangian λ_col, low-pass filtered (τ=0.125s). This gives tracking priority in free space but yields to safety near obstacles.

---

## 11. Shared Autonomy Architecture

The `main_shared_autonomy.py` node implements:
- **Bayesian belief estimation** over a discrete goal set (with goal **exclusion** support)
- **Local QP policy** (separate from the safety QP) for constrained intent following
- **Grasp state machine**: SHARED_AUTONOMY → PRE_GRASP → GRASP_APPROACH → GRASP_CLOSE → LIFT → HOLDING
- **Alpha-blending** between human teleop input and autonomous policy (WIP)
- Publishes cartesian references consumed by `main_qp_controller.py`

### 11.1 Goal Set (5 goals)

`Red_Top`, `Red_Side`, `Blue_Top`, `Blue_Side`, `Platform_Place`.

- **Side/Top grasp goals**: dynamic SE(3) manifolds around each cylinder (see `goal_set.py`).
  - **Side-grasp azimuth singularity guard**: the approach direction is the horizontal
    anchor→axis vector, whose direction is undefined when the gripper hovers over the cylinder
    top. Within `_SIDE_AZIMUTH_DEADZONE` (0.04 m) of the axis the azimuth is **frozen** to its
    last committed value (`_last_side_radial`), so crossing the top no longer swings the goal
    around the cylinder. It switches side only once the anchor is unambiguously on the other side
    (≥ deadzone) — a single deterministic switch, never indecision oscillation (blend-safe).
- **`Platform_Place`** (placement manifold): the grasped cylinder must be set down on the
  yellow `placement_area` disk (world center `[1.0, 0.0, 0.701]`, radius `0.15 m`). The ONLY
  hard constraint is **cylinder axis ⊥ platform face** (i.e. vertical). Implementation:
  - At grasp/attach, `GoalSet.set_grasped(color, T_EE)` freezes the cylinder symmetry axis in the
    **gripper frame** (`grasped_axis_local = R_grasp^T @ [0,0,1]`) plus the gripper-vs-cylinder
    height offset.
  - `get_platform_goal_pose` projects the anchor XY onto the disk (clamped `PLATFORM_PLACE_MARGIN`
    inside the rim — user chooses *where* by hovering, so the two cylinders land at different
    spots) and computes the **minimal rotation** of the anchor orientation that brings the held
    axis to vertical. This constrains 2 DOF (tilt) and leaves yaw-about-vertical free → a true
    placement manifold, not a single pose.

### 11.2 Post-grasp lifecycle (the fix for "architecture dies after grasping")

| Phase | Behavior |
|-------|----------|
| `GRASP_CLOSE` → `LIFT` | Sends `ATTACH_*` (re-parents cylinder as a real arm link in the QP collision world) and **clears** the gripper↔cylinder CBF bypass (`ignore_cbf="None"`) so the held cylinder now actively avoids the environment — it is treated as a link of the arm chain, with the handler's 3 s barrier ramp. |
| `LIFT` | Slow blind vertical lift: `LIFT_VELOCITY=0.025 m/s × LIFT_DURATION=2.0 s = 5 cm` clear of the table, then → `HOLDING`. |
| `HOLDING` | Shared autonomy **resumes**: `_holding` passes the outer-loop policy (`pi_max`) straight through, so the user can drive the loaded gripper toward any remaining goal and the belief estimator keeps predicting. **PRE_GRASP is unreachable while holding** (no second grasp with the same gripper). A console banner announces available goals. |
| Release | A trigger pull (or console `OPEN`) in HOLDING → `_release_object()`: opens gripper, detaches the Gazebo plugin weld, **publishes `DETACH_<arm>_<color>_<x>_<y>_<z>`** with the perfect-fall placement pose, **relocates the cylinder in the world model** (`goal_set.relocate_cylinder`) so the re-enabled grasp goals point where it was placed (NOT the spawn), then enters **`RELEASE_LIFT`**. |
| `RELEASE_LIFT` | Dual of the post-CLOSE `LIFT`: a slow vertical lift to move clear of the just-placed object while its CBF barrier ramps in, then → `SHARED_AUTONOMY` (control returned to the user). Belief frozen, Haption yielded, authority handed over — same as the grasp-execution phases. |

**World building on placement** (perfect-fall model, no Gazebo pose reads — ready for real experiments):
- The placed cylinder is assumed to end UPRIGHT resting on the placement surface at the **XY where the EE released it**; Z = `goal_set.platform_rest_z()` (platform top + half height).
- `goal_set.relocate_cylinder(color, pos)` updates the believed cylinder position (and resets its sticky orientation/azimuth memory), so `Color_Top`/`Color_Side` goals re-point to the new location.
- The QP collision obstacle is placed at the same fallen pose via `CollisionManager.detach_object(world_pos=...)`, with the smooth `ATTACH_RAMP_S` barrier ramp re-armed so the gripper can lift clear before the barrier fully engages.

**QP-side detach** (inverse of `ATTACH_`, added in `shared_autonomy_handler` + `collision_manager` + `visualization_engine`):
- `DETACH_<arm>_<color>` → `pending_detach`, processed in the QP loop.
- `CollisionManager.detach_object` re-parents the cylinder geometry back to the **world** (joint 0), frozen at its current world pose (stops following the wrist). Collision pairs are kept (valid for a static obstacle).
- The cylinder is dropped from `attached_objects`/adjacency, and the attach **barrier ramp is re-armed** (`attached_time[cyl_id]=now`) so the re-engaged gripper↔cylinder CBF pair ramps in smoothly over `ATTACH_RAMP_S` instead of spiking (gripper is overlapping the just-released cylinder).
- `VisualizationEngine.restore_object_color` clears the orange Meshcat override (cylinder + gripper revert to original material). RViz auto-reverts (grey rendering is keyed on `attached_objects`).
- **Known limitation**: no Gazebo→twin pose sync, so the QP-twin cylinder freezes at the release pose (matches placement-on-platform; a mid-air drop won't fall in the twin).

### 11.3 Belief exclusion rules

`BeliefEstimator.set_excluded_goals(...)` pins a goal to probability 0, skips it in the cost
update and `blend_policies`, and never returns it from `get_active_goal` — but it stays in
`target_keys` so the UI still shows it (at 0). Its policy is **not evaluated** (zero placeholder).
- **Gripper empty**: `Platform_Place` excluded.
- **Holding `<Color>`**: `<Color>_Top` and `<Color>_Side` excluded; `Platform_Place` enabled.

### 11.4 RViz / visualization

Goal poses are drawn as **belief-opacity gripper markers** (ns `goal_poses`), one per goal,
color-coded by family (Red reddish, Blue bluish, Platform yellow). Opacity is a continuous ramp
`0.2 + 0.6·belief` (→ 0.8 at belief 1, 0.2 at belief 0), so low/zero-belief goals (the
just-grasped cylinder, or the Platform while empty) fade out automatically — no state-machine viz
logic, and nothing goes stale. This **replaced** the per-goal TF frames; only the **active goal**
still broadcasts a precise TF frame (`goal_<active_key>`) for pose debugging. The predictive
trajectory grippers (ns `policy_grippers`) are unchanged.

---

## 12. Current State & Known Issues

| Area | Status | Notes |
|------|--------|-------|
| QP bimanual arm control | ✅ Working | Full 6-DOF tracking with CBF safety |
| Shared autonomy (belief + grasp) | ✅ Working | Refactored from monolithic script |
| Grasp pipeline (full cycle) | ✅ Working | PRE_GRASP → APPROACH → CLOSE → ATTACH all succeed reliably |
| Post-grasp (LIFT + HOLDING + place) | ✅ Working | 5 cm slow lift → HOLDING resumes shared autonomy → Platform placement manifold → release back to start |
| Platform placement goal | 🔧 Active dev | `Platform_Place` manifold (axis ⊥ disk); debugging placement height/orientation live |
| Haption teleoperation (clutch) | 🔧 Active dev | `teleop_triago_clutch.py` + `haptic_force_manager.py` |
| Head control (visual servoing) | 🔧 Active dev | `qp_head_visual_servo.py`: QP-based hand-tracking, independent loop. Starting point — no image processing yet. |
| Mobile base integration | 🔧 Partial | `base_controller.py` exists but not QP-certified |
| Meshcat visualization | ✅ Working | Thread-safe, auto-reloads on grasp coloring |
| Digital twin mode | ✅ Working | `SIMULATE_IDEAL_KINEMATICS` flag in config |
| Dynamic CBF pair removal | ⚠️ Experimental | `DYNAMIC_CBF` flag, used during grasp sequences |
| Open-loop trajectory testing | ✅ Working | `trajectory_generator.py` + `config/trajectory_endpoints.yaml`: YAML-selected quintic reference presets (free space → collision-risk → out-of-workspace) with optional λ-driven `dynamic_trajectory` time scaling |

---

## 13. Build & Run Commands

```bash
# Build
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

# Run QP controller (bimanual arms)
ros2 run triago_control main_qp_controller.py

# Run shared autonomy
ros2 run triago_control main_shared_autonomy.py

# Run head visual servoing (independent, can run alongside arm QP)
ros2 run triago_control qp_head_visual_servo.py

# Run an open-loop robustness trajectory (edit config/trajectory_endpoints.yaml first)
ros2 run triago_control trajectory_generator.py
#   override the endpoint file:
ros2 run triago_control trajectory_generator.py --ros-args -p config_file:=/abs/path/trajectory_endpoints.yaml

# Run plotter dashboard
ros2 run triago_control plotter.py
```

---

## 13.5 Full Simulation Launch Sequence (typical session)

This is the exact, ordered set of commands the user runs for a normal Gazebo
teleoperation session. Two sides: the **robot/control side** (triago_control +
Gazebo + controllers) and the **teleoperation side** (haption_teleoperation).
Each command runs in its own terminal (all need the workspace sourced).

### Robot / control side

```bash
# 1. World: TRIAGo + table + two cylinders + yellow placement zone ("tutorial" world)
ros2 launch triago_gazebo triago_gazebo.launch.py \
    end_effector_right:=pal-pro-gripper \
    end_effector_left:=pal-pro-gripper \
    world_name:=tutorial

# 2. Load the default controllers (joint-space velocity controllers, etc.)
ros2 launch triago_controller_configuration tsid_default_controllers.launch.py \
    use_sim_time:=True

# 3. QP CLF-CBF safety controller (tracks /arm_*/cartesian_reference, owns CBF)
ros2 run triago_control main_qp_controller.py

# 4. Shared autonomy + belief evaluation (intent inference, grasp FSM,
#    publishes /shared_autonomy/* consumed by the haptic force manager)
ros2 run triago_control main_shared_autonomy.py

# 5. RViz visualization (markers, goal grippers, guidance cues)
ros2 launch triago_control visualize.launch.py
```

### Teleoperation side (haption_teleoperation)

```bash
# 6. Haption device server (150 Hz C++ node, talks to the VirtuoseAPI)
ros2 run haption_teleoperation virtuose_server_node

# 7. Clutch-indexing teleop (owns /arm_right/cartesian_reference)
ros2 run haption_teleoperation teleop_triago_clutch.py

# 8. Force feedback to the operator (Virtual-Fixture guidance forces)
ros2 run haption_teleoperation haptic_force_manager_tutorial.py
```

### Active-script note (naming)

The **active** force-feedback node is **`haptic_force_manager_tutorial.py`**
(NOT `haptic_force_manager.py`, which no longer exists). Sibling variants in
`haption_teleoperation/scripts/`:
- `haptic_force_manager_tutorial.py` — ★ active node used in the tutorial-world flow.
- `haptic_force_manager_battery.py` — alternate/experimental variant.

`haptic_force_manager_tutorial.py` currently runs with **`DEBUG_ONLY_GUIDE = True`**,
which means it outputs **only `F_guide`** (the belief-weighted Virtual-Fixture
guidance wrench) — `F_sync`, `F_cbf`, `F_fixture`, clutch-align and global damping
are all bypassed while this debug flag is set.

### main_shared_autonomy ↔ haptic_force_manager_tutorial interface

The shared-autonomy node is the **producer** of the inference state; the force
manager is the **consumer** that turns it into a guidance wrench on the device:

| Topic | Type | Layout / meaning |
|-------|------|------------------|
| `/shared_autonomy/goal_names` | String | comma-joined keys, e.g. `Red_Top,Red_Side,Blue_Top,Blue_Side,Platform_Place` |
| `/shared_autonomy/goal_probabilities` | Float64MultiArray | belief simplex aligned to `goal_names` (excluded goals = 0) |
| `/shared_autonomy/user_policy` | Float64MultiArray | `n_goals × 6` flattened QP-constrained twists, **anchored at the reference pose** (`current_T_user`, from `/arm_right/cartesian_reference`) — "from where the handle/reference is, the velocity toward each goal". This is what F_guide renders. In test mode the reference == real EE, so it matches the EE-anchored policy |
| `/shared_autonomy/active_goal_pose` | Float64MultiArray | `[x,y,z,roll,pitch,yaw,confidence]` in base_footprint (confidence forced 0 during grasp execution) |
| `/shared_autonomy/grasp_active` | Bool | True while the SM autonomously drives the arm (approach/close/lift/release-lift) |

- `F_guide` = `Σ P(k)·π_k` integrated over a lookahead → position spring toward
  that offset, gated by belief confidence (entropy-based). This is the only
  layer active under `DEBUG_ONLY_GUIDE`.
- `F_fixture` = position/orientation spring toward `active_goal_pose`, gated by
  confidence (silent unless `DEBUG_ONLY_GUIDE=False`).
- When `grasp_active=True`, the manager normally switches to a strong pure
  EE-following sync so the operator feels the autonomous grasp (also bypassed
  under `DEBUG_ONLY_GUIDE`).

### Fixes — 2026-06-26 (F_guide guidance, viz, plot layout)

Three issues in the teleop (`POLICY_BELIEF_TEST=False`) + `DEBUG_ONLY_GUIDE=True`
path were diagnosed and fixed:

1. **F_guide too strong / handle drifted instead of being driven to the goal.**
   Root cause: the old `compute_F_guide` turned the tanh-saturated policy
   velocity into a position offset (`pi_blend·lookahead`, capped at 5 cm) → a
   near-constant ~4.5 N push that only faded when the *robot EE* (not the hand)
   reached the goal. With `DEBUG_ONLY_GUIDE` bypassing all damping, a constant
   undamped force just accelerates a lightly-held handle. **Fix:** F_guide is now
   a **velocity-field guidance** force `F = D·(v_field − v_handle)·confidence`,
   where `v_field = map(pi_blend)` (180° Z-flip into the Haption frame). It pushes
   when the handle is still, is intrinsically damped (fades to zero as the handle
   reaches `v_field` → no runaway), lets a passive hand cruise at exactly
   `pi_blend` (so the teleop reference traces the SAME path the test mode commands
   directly), and vanishes at the goal. Gains: `D_guide_lin=28`, `D_guide_ang=0.45`,
   sat `MAX_GUIDE_FORCE=3.5 N` / `MAX_GUIDE_TORQUE=0.25 Nm`.

2. **Green policy gripper marker disappeared in RViz.** Root cause: `timer_callback`
   hard-`return`ed whenever `/collision_constraints` was older than 50 ms, halting
   ALL marker publishing; with the 500 ms marker lifetime any QP-rate jitter
   blinked the marker out. **Fix:** staleness now only WARNs and is folded into
   `valid_matrices` (stale → zero policies = safe halt, which also stops the
   test-mode command) while visualization keeps publishing; marker lifetime raised
   to 1.5 s.

3. **Shared-autonomy frequency plot overlapped / sat beside the twist plot.**
   `PlotManager._build_twist_figure` was a 1×3 row `(radar | diff | freq)` and the
   radar legend bled into the deviation plot. **Fix:** gridspec `2×2`
   (`width_ratios=[1,2]`, `height_ratios=[2,1]`) — radar left full-height, the
   deviation plot top-right, and the **frequency monitor directly under it**;
   radar legend moved below the radar.

4. **User policy re-anchored at the reference pose (this iteration).** Previously
   `user_policies[key]` was a copy of the EE-anchored `ee_policies[key]` (a prior
   attempt had abandoned reference-anchoring). Since F_guide drives the *handle*
   (which maps to the reference), the policy that feeds F_guide is now evaluated
   from `current_T_user` (the `/arm_right/cartesian_reference` pose):
   `T_goal_user = get_dynamic_goal_pose(current_T_user, key, update_memory=False)`
   → `v_geo_user = compute_v_geo(current_T_user, T_goal_user)` →
   `solve_local_policy(v_geo_user, J_c, h_c)`. The belief tiebreaker `pos_costs`
   is likewise anchored at `current_T_user` so the inference frame is consistent
   with the reference twist `current_v_h`. `ee_policies` (which commands the robot
   in test/grasp mode and draws the green gripper) stays anchored at the real EE.
   In test mode `current_T_user == current_T_EE`, so behaviour is unchanged there.

---

## 14. Coding Conventions

- **Config**: every tunable value lives in `qp_controller/config.py`. Never hard-code gains elsewhere.
- **Naming**: snake_case for files and variables, PascalCase for classes.
- **Docstrings**: module-level docstring explaining the "why" and the math. Class/method docstrings for non-obvious logic.
- **No bare `import config`**: always `import triago_control.qp_controller.config as cfg`.
- **No `_refactored`, `_v2`, `_new` in filenames**: that's what git history is for.
- **Entry points**: named `main_*.py` in `scripts/`. Libraries never contain `if __name__ == '__main__'`.

---

## 15. Git Workflow

- **main** branch: stable, runnable code
- Feature/fix branches: `feature/xyz` or `fix/xyz`
- The user pushes from Docker; the AI agent creates branches and PRs for review
- After merging a PR, the user pulls on their machine: `git pull origin main`
