# triago_control

A modular, safety-certified motion control framework for the PAL Robotics TRIAGo bimanual platform. The architecture guarantees constraint satisfaction (collision avoidance, joint limits) regardless of the command source — teleoperation, trajectory planning, or autonomous policy.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTABLE NODES (scripts/)                    │
├──────────────────────────────┬──────────────────────────────────┤
│  main_qp_controller.py      │  main_shared_autonomy.py         │
│  (safety-critical QP loop)  │  (intent prediction + blending)  │
└──────────────┬───────────────┴──────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐   ┌──────────────────────────────────┐
│  triago_control/         │   │  triago_control/                 │
│  qp_controller/          │   │  shared_autonomy/                │
│  ├── config.py           │   │  ├── belief_estimator.py         │
│  ├── robot_kinematics.py │   │  ├── goal_set.py                 │
│  ├── collision_manager.py│   │  ├── grasp_state_machine.py      │
│  ├── qp_formulator.py    │   │  └── plot_manager.py             │
│  ├── shared_autonomy_    │   └──────────────────────────────────┘
│  │   handler.py          │
│  └── visualization_      │
│      engine.py           │
└──────────────────────────┘
```

**Dependency flow:**

```
main_qp_controller  ──→  qp_controller/*   (the math)
                    ──→  shared_autonomy_handler  (grasp/CBF-bypass hooks)

main_shared_autonomy ──→  shared_autonomy/*  (belief, goals, state machine)
                     ──→  publishes references consumed by main_qp_controller
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

## License

BSD-3-Clause — see [LICENSE](LICENSE).
