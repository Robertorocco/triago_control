# AI Agent Context — triago_control

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-06-18

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
    ├── triago_control/              ← THIS REPO (git-tracked)
    ├── haption_teleoperation/       ← separate package (teleop hardware interface)
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
│   └── Recording_Rviz.rviz
├── launch/
│   └── visualize.launch.py
├── scripts/                         ← EXECUTABLE ENTRY POINTS (ros2 run targets)
│   ├── qp_arm_teleop/
│   │   ├── main_qp_controller.py       ★ primary: QP-CLF-CBF safety loop
│   │   ├── main_shared_autonomy.py     ★ primary: intent prediction + blending
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

## 4. Entry Point → Library Dependency Map

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
```

---

## 5. Import Convention

All library imports use the **fully-qualified package path**:

```python
import triago_control.qp_controller.config as cfg
from triago_control.qp_controller.robot_kinematics import RobotKinematics
from triago_control.shared_autonomy.belief_estimator import BeliefEstimator
```

**Never** use bare `import config` — it collides with system modules. Always anchor to `triago_control.*`.

---

## 6. Critical Hardware Quirks

1. **Corrupted encoder velocities**: TRIAGo's joint_states `velocity` field is unreliable. The controller derives velocity from position differences and filters with a first-order EMA (`ALPHA_FILTER = 0.15`, ~60ms window). Never trust `msg.velocity` directly.

2. **Meshcat thread safety**: Meshcat's WebSocket is NOT thread-safe. ROS callbacks must NEVER call the viewer. Only the dedicated `_run_viz` thread (in `visualization_engine.py`) owns Meshcat WebSocket calls. Callbacks mutate `meshColor` under a `threading.Lock` and set `meshcat_reload_pending = True`.

3. **Controller switching**: TRIAGo requires explicit activation of velocity controllers (`arm_right_joint_space_controller_vel`, `arm_left_joint_space_controller_vel`) and deactivation of conflicting trajectory controllers before the QP can command the arms.

4. **URDF source**: The URDF is fetched at runtime from `/robot_state_publisher/get_parameters` (parameter `robot_description`). A static copy exists at `triago_extracted.urdf` for offline development/testing.

---

## 7. Mathematical Core (QP-CLF-CBF)

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

## 8. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting**: each arm's slack weight drops (toward `BASE_WEIGHT_SLACK=5`) when its shadow price grows, letting the slack absorb more tracking error near obstacles. In free space it rises (toward `MAX_WEIGHT_SLACK=50`) for tighter tracking.
- **Dynamic gamma (CLF)**: the CLF convergence rate γ drops exponentially with the collision Lagrangian λ_col, low-pass filtered (τ=0.125s). This gives tracking priority in free space but yields to safety near obstacles.

---

## 9. Shared Autonomy Architecture

The `main_shared_autonomy.py` node implements:
- **Bayesian belief estimation** over a discrete goal set
- **Local QP policy** (separate from the safety QP) for constrained intent following
- **Grasp state machine**: IDLE → PRE_GRASP → APPROACH → CONTACT → CLOSE → ATTACH
- **Alpha-blending** between human teleop input and autonomous policy (WIP)
- Publishes cartesian references consumed by `main_qp_controller.py`

---

## 10. Current State & Known Issues

| Area | Status | Notes |
|------|--------|-------|
| QP bimanual arm control | ✅ Working | Full 6-DOF tracking with CBF safety |
| Shared autonomy (belief + grasp) | ✅ Working | Refactored from monolithic script |
| Head control | ❌ Not implemented | Planned future addition |
| Mobile base integration | 🔧 Partial | `base_controller.py` exists but not QP-certified |
| Meshcat visualization | ✅ Working | Thread-safe, auto-reloads on grasp coloring |
| Digital twin mode | ✅ Working | `SIMULATE_IDEAL_KINEMATICS` flag in config |
| Dynamic CBF pair removal | ⚠️ Experimental | `DYNAMIC_CBF` flag, used during grasp sequences |

---

## 11. Build & Run Commands

```bash
# Build
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

# Run QP controller
ros2 run triago_control main_qp_controller.py

# Run shared autonomy
ros2 run triago_control main_shared_autonomy.py

# Run plotter dashboard
ros2 run triago_control plotter.py
```

---

## 12. Coding Conventions

- **Config**: every tunable value lives in `qp_controller/config.py`. Never hard-code gains elsewhere.
- **Naming**: snake_case for files and variables, PascalCase for classes.
- **Docstrings**: module-level docstring explaining the "why" and the math. Class/method docstrings for non-obvious logic.
- **No bare `import config`**: always `import triago_control.qp_controller.config as cfg`.
- **No `_refactored`, `_v2`, `_new` in filenames**: that's what git history is for.
- **Entry points**: named `main_*.py` in `scripts/`. Libraries never contain `if __name__ == '__main__'`.

---

## 13. Git Workflow

- **main** branch: stable, runnable code
- Feature/fix branches: `feature/xyz` or `fix/xyz`
- The user pushes from Docker; the AI agent creates branches and PRs for review
- After merging a PR, the user pulls on their machine: `git pull origin main`
