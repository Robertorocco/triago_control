# triago_control

A modular, safety-certified motion control framework for the PAL Robotics TRIAGo bimanual platform. The architecture guarantees constraint satisfaction (collision avoidance, joint limits) regardless of the command source — teleoperation, trajectory planning, or autonomous policy.

The system comprises three independent control subsystems running concurrently:
1. **Bimanual Arm QP Controller** — CLF-CBF safety-critical loop for both arms
2. **Shared Autonomy** — Bayesian intent prediction + haptic guidance
3. **Head Visual Servoing** — Independent vision-based head control to keep hands in camera FOV

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTABLE NODES (scripts/)                               │
├──────────────────────────┬────────────────────────────┬─────────────────────────┤
│  qp_arm_teleop/          │  qp_arm_teleop/            │  head_controller/        │
│  main_qp_controller.py   │  main_shared_autonomy.py   │  qp_head_visual_servo.py │
│  (arm safety QP loop)    │  (intent + blending)       │  (head visual servoing)  │
└────────────┬─────────────┴──────────────┬─────────────┴────────────┬────────────┘
             │                            │                          │
             ▼                            ▼                          │ (independent)
┌──────────────────────────┐   ┌───────────────────────────┐        │
│  triago_control/         │   │  triago_control/          │        │
│  qp_controller/          │   │  shared_autonomy/         │        │
│  ├── config.py           │   │  ├── belief_estimator.py  │        │
│  ├── robot_kinematics.py │   │  ├── goal_set.py          │        │
│  ├── collision_manager.py│   │  ├── grasp_state_machine.py│       │
│  ├── qp_formulator.py    │   │  └── plot_manager.py      │        │
│  ├── shared_autonomy_    │   └───────────────────────────┘        │
│  │   handler.py          │                                         │
│  └── visualization_      │         ┌──────────────────────────────┘
│      engine.py           │         │  Own Pinocchio model + QP
└──────────────────────────┘         │  Own velocity controller
                                     │  No coupling to arm pipeline
                                     ▼
                              /joint_states → FK → pixel projection
                              → QP solve → /arm_head_.../joint_velocity_cmd
```

**Dependency flow:**

```
main_qp_controller  ──→  qp_controller/*   (the math)
                    ──→  shared_autonomy_handler  (grasp/CBF-bypass hooks)

main_shared_autonomy ──→  shared_autonomy/*  (belief, goals, state machine)
                     ──→  publishes references consumed by main_qp_controller

qp_head_visual_servo ──→  own Pinocchio instance (FK + Jacobians)
                     ──→  own quadprog QP (visual servoing)
                     ──→  INDEPENDENT: no imports from qp_controller/ or shared_autonomy/
```

## Key Modules

| Module | Responsibility |
|--------|---------------|
| `qp_controller/config.py` | Single source of truth for every flag, gain, buffer and obstacle |
| `qp_controller/robot_kinematics.py` | Pinocchio model, FK, Jacobians, EMA velocity filter, digital twin |
| `qp_controller/collision_manager.py` | hppfcl geometries, SoftMin CBF aggregation, dynamic safety margin |
| `qp_controller/qp_formulator.py` | CLF-CBF-QP assembly and solving (quadprog), adaptive scheduling |
| `qp_controller/shared_autonomy_handler.py` | Gripper commands, CBF-bypass, cylinder re-parenting |
| `qp_controller/visualization_engine.py` | Thread-safe Meshcat + RViz markers |
| `shared_autonomy/belief_estimator.py` | Bayesian intent inference over goal set |
| `shared_autonomy/goal_set.py` | Dynamic goal pose computation |
| `shared_autonomy/grasp_state_machine.py` | Pick state machine (approach → contact → close → attach) |

### Head Controller (Independent Subsystem)

| Module | Responsibility |
|--------|---------------|
| `scripts/head_controller/qp_head_visual_servo.py` | 2.5D visual servoing QP — keeps both hands centered in camera FOV |

## Prerequisites

- **ROS 2 Humble** (tested on Ubuntu 22.04)
- **Pinocchio** (with hppfcl collision support)
- **Python 3.10+**
- **numpy**, **scipy**, **matplotlib**
- **quadprog** (`pip install quadprog` or `apt install python3-quadprog`)
- **PAL TRIAGo packages** (URDF, controller manager, gripper controllers)
- **IFRA_LinkAttacher** (Gazebo grasp plugin — see below)
- **Meshcat** (optional, for 3D visualization at `http://127.0.0.1:7000/static/`)

## Build

```bash
cd ~/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash
```

## Gazebo Grasp Plugin (IFRA_LinkAttacher)

Required for pick-and-place in simulation. Creates a fixed joint between gripper and object.

```bash
# Install (separate repo, not part of triago_control)
cd ~/ros2-ws/src
git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git
cd ~/ros2-ws
colcon build --packages-up-to ros2_linkattacher
source install/setup.bash
```

Add to your Gazebo world file:
```xml
<plugin name="ros2_linkattacher" filename="libgazebo_link_attacher.so"/>
```

Before launching Gazebo:
```bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/ros2-ws/install/ros2_linkattacher/lib
```

## Run

### QP Safety Controller (bimanual arms)

```bash
ros2 run triago_control main_qp_controller.py
```

Subscribes to `/arm_right/cartesian_reference` and `/arm_left/cartesian_reference` (Float64MultiArray, 12+ floats: pos, rpy, vel, omega).

### Shared Autonomy Node

```bash
ros2 run triago_control main_shared_autonomy.py
```

Publishes cartesian references for the QP controller based on predicted human intent and optimal policy blending.

### Head Visual Servoing (Independent)

```bash
ros2 run triago_control qp_head_visual_servo.py
```

Independently controls the 7-DOF head arm to keep both hands in the camera field-of-view. Uses a QP with 2.5D interaction-matrix-based image servoing (IBVS) when hands are visible, and a 3D rotational look-at (PBVS) when hands are outside FOV. Runs concurrently with the arm QP — no coupling between the two.

### Auxiliary Nodes

```bash
ros2 run triago_control plotter.py              # Live telemetry dashboard
ros2 run triago_control keyboard_teleop.py      # Keyboard-based cartesian jog
ros2 run triago_control base_controller.py      # Mobile base velocity control
ros2 run triago_control drift_evaluator_node.py # Tracking error analysis
```

## Configuration

All tunable parameters live in `triago_control/qp_controller/config.py`:

- **Feature flags**: `DISABLE_CBF`, `DYNAMIC_SLACK_WEIGHT`, `COMPARISON_CLF`, etc.
- **Safety gains**: `ALPHA_SOFTMIN`, `GAMMA_CBF`, `D_SAFE_BASE`, `K_V_SAFE`
- **Control loop**: `CONTROL_FREQ_DEFAULT` (Hz), `PUBLISH_EVERY_N`
- **Workspace geometry**: obstacle positions, capsule radius, wall dimensions

## Simulation vs. Real Hardware (Auto-Detection)

The system **automatically detects** whether it is running on real hardware or in Gazebo simulation — no manual flag or configuration change required.

**Detection method**: At startup, `main_qp_controller.py` fetches the URDF from `robot_state_publisher` and checks for the presence of `gripper_right_grasping_link` and `gripper_left_grasping_link`:

| Condition | Environment | `REAL_HARDWARE` |
|-----------|-------------|-----------------|
| Both frames present in URDF | Gazebo simulation | `False` |
| One or both frames missing | Real TIAGo Pro | `True` |

**Behavioral differences when `REAL_HARDWARE = True`**:

1. **Grasping frame injection**: The missing `gripper_*_grasping_link` frames are injected into the Pinocchio model at runtime (offset: `[0, 0, 0.157]`, Ry(-90°) from `gripper_*_base_link`) and broadcast as static TFs for RViz and other nodes.

2. **Direct joint velocity**: Joint velocities are read directly from `/joint_states` `msg.velocity` instead of being reconstructed via position differentiation + EMA filtering. The real TIAGo Pro velocity sensors are reliable (unlike Gazebo's corrupted encoder simulation).

A colored startup banner in the console announces the detected environment:
```
[ENV] REAL HARDWARE detected (URDF lacks grasping frames). Using direct joint velocities + injecting TCP frames.
[ENV] SIMULATION detected (URDF contains grasping frames). Using EMA-filtered velocity from position differentiation.
```

## ROS 2 Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Hardware joint positions |
| `/arm_right/cartesian_reference` | `std_msgs/Float64MultiArray` | Right arm reference (12+ floats) |
| `/arm_left/cartesian_reference` | `std_msgs/Float64MultiArray` | Left arm reference (12+ floats) |
| `/shared_autonomy/gripper_cmd` | `std_msgs/String` | CLOSE/ORANGE/ATTACH commands |
| `/shared_autonomy/target_ignore` | `std_msgs/String` | +/- CBF bypass protocol |
| `/shared_autonomy/grasp_margin` | `std_msgs/String` | Per-pair negative margins |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/{controller}/joint_velocity_cmd` | `std_msgs/Float64MultiArray` | Safe joint velocities |
| `/qp_debug/safety_margin` | `std_msgs/Float64` | h_soft - d_safe |
| `/qp_debug/min_distance` | `std_msgs/Float64` | Absolute closest distance |
| `/qp_debug/lambda_cbf` | `std_msgs/Float64` | Collision constraint shadow price |
| `/qp_debug/slacks` | `std_msgs/Float64MultiArray` | CLF slack [right, left] |
| `/collision_constraints` | `std_msgs/Float64MultiArray` | Cartesian CBF projection (13 floats) |
| `/qp_debug/head_cartesian_cmd` | `geometry_msgs/TwistStamped` | Head camera velocity command (debug) |
| `/qp_debug/camera_ray` | `visualization_msgs/Marker` | Camera optical axis (RViz) |
| `/qp_debug/target_centroid` | `visualization_msgs/Marker` | Hands centroid sphere (RViz) |

## License

BSD-3-Clause — see [LICENSE](LICENSE).
