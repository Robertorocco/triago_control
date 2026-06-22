# AI Agent Context — triago_control

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-06-22 (added haption_teleoperation package documentation)

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

Multi-layer force-feedback superposition node. Computes and sums:

| Layer | Symbol | Description |
|-------|--------|-------------|
| Sync | F_sync | Spring-damper (Kp=10, Kd=0) tethering user to robot tracking error |
| CBF | F_cbf | Repulsive force from collision barrier gradient × λ_cbf, tanh-saturated, LPF'd (α=0.15) |
| Guide | F_guide | Belief-weighted blend of all leaf policies (continuous, entropy-gated confidence, viscous B=90 N/(m/s)) |
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

## 5. Entry Point → Library Dependency Map

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
```

---

## 6. Import Convention

All library imports use the **fully-qualified package path**:

```python
import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.shared_autonomy.belief_estimator import BeliefEstimator
```

**Never** use bare `import config` — it collides with system modules. Always anchor to `triago_control.*`.

---

## 7. Critical Hardware Quirks

1. **Corrupted encoder velocities**: TRIAGo's joint_states `velocity` field is unreliable. The controller derives velocity from position differences and filters with a first-order EMA (`ALPHA_FILTER = 0.15`, ~60ms window). Never trust `msg.velocity` directly.

2. **Meshcat thread safety**: Meshcat's WebSocket is NOT thread-safe. ROS callbacks must NEVER call the viewer. Only the dedicated `_run_viz` thread (in `visualization_engine.py`) owns Meshcat WebSocket calls. Callbacks mutate `meshColor` under a `threading.Lock` and set `meshcat_reload_pending = True`.

3. **Controller switching**: TRIAGo requires explicit activation of velocity controllers (`arm_right_joint_space_controller_vel`, `arm_left_joint_space_controller_vel`) and deactivation of conflicting trajectory controllers before the QP can command the arms.

4. **URDF source**: The URDF is fetched at runtime from `/robot_state_publisher/get_parameters` (parameter `robot_description`). A static copy exists at `triago_extracted.urdf` for offline development/testing.

---

## 8. Mathematical Core (QP-CLF-CBF)

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

## 9. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting**: each arm's slack weight drops (toward `BASE_WEIGHT_SLACK=5`) when its shadow price grows, letting the slack absorb more tracking error near obstacles. In free space it rises (toward `MAX_WEIGHT_SLACK=50`) for tighter tracking.
- **Dynamic gamma (CLF)**: the CLF convergence rate γ drops exponentially with the collision Lagrangian λ_col, low-pass filtered (τ=0.125s). This gives tracking priority in free space but yields to safety near obstacles.

---

## 10. Shared Autonomy Architecture

The `main_shared_autonomy.py` node implements:
- **Bayesian belief estimation** over a discrete goal set
- **Local QP policy** (separate from the safety QP) for constrained intent following
- **Grasp state machine**: IDLE → PRE_GRASP → APPROACH → CONTACT → CLOSE → ATTACH
- **Alpha-blending** between human teleop input and autonomous policy (WIP)
- Publishes cartesian references consumed by `main_qp_controller.py`

---

## 11. Current State & Known Issues

| Area | Status | Notes |
|------|--------|-------|
| QP bimanual arm control | ✅ Working | Full 6-DOF tracking with CBF safety |
| Shared autonomy (belief + grasp) | ✅ Working | Refactored from monolithic script |
| Haption teleoperation (clutch) | 🔧 Active dev | `teleop_triago_clutch.py` + `haptic_force_manager.py` |
| Head control | ❌ Not implemented | Planned future addition |
| Mobile base integration | 🔧 Partial | `base_controller.py` exists but not QP-certified |
| Meshcat visualization | ✅ Working | Thread-safe, auto-reloads on grasp coloring |
| Digital twin mode | ✅ Working | `SIMULATE_IDEAL_KINEMATICS` flag in config |
| Dynamic CBF pair removal | ⚠️ Experimental | `DYNAMIC_CBF` flag, used during grasp sequences |
| Open-loop trajectory testing | ✅ Working | `trajectory_generator.py` + `config/trajectory_endpoints.yaml`: YAML-selected quintic reference presets (free space → collision-risk → out-of-workspace) with optional λ-driven `dynamic_trajectory` time scaling |

---

## 12. Build & Run Commands

```bash
# Build
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

# Run QP controller
ros2 run triago_control main_qp_controller.py

# Run shared autonomy
ros2 run triago_control main_shared_autonomy.py

# Run an open-loop robustness trajectory (edit config/trajectory_endpoints.yaml first)
ros2 run triago_control trajectory_generator.py
#   override the endpoint file:
ros2 run triago_control trajectory_generator.py --ros-args -p config_file:=/abs/path/trajectory_endpoints.yaml

# Run plotter dashboard
ros2 run triago_control plotter.py
```

---

## 13. Coding Conventions

- **Config**: every tunable value lives in `qp_controller/config.py`. Never hard-code gains elsewhere.
- **Naming**: snake_case for files and variables, PascalCase for classes.
- **Docstrings**: module-level docstring explaining the "why" and the math. Class/method docstrings for non-obvious logic.
- **No bare `import config`**: always `import triago_control.qp_controller.config as cfg`.
- **No `_refactored`, `_v2`, `_new` in filenames**: that's what git history is for.
- **Entry points**: named `main_*.py` in `scripts/`. Libraries never contain `if __name__ == '__main__'`.

---

## 14. Git Workflow

- **main** branch: stable, runnable code
- Feature/fix branches: `feature/xyz` or `fix/xyz`
- The user pushes from Docker; the AI agent creates branches and PRs for review
- After merging a PR, the user pulls on their machine: `git pull origin main`
