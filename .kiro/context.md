# AI Agent Context — triago_control

> **This file is maintained by the AI agent.**

## 0. Maintenance Rules

1. **Always share the pull/rebuild command** with the user immediately after pushing any change to this repo (see §11 for the exact sequence). Never wait to be asked.
2. **Keep this file clean.** It must contain only the math formulations and core architectural concepts an AI agent needs to work on this project — no dated changelogs, "Last updated:"/"Earlier:" narratives, tuning history, or bugfix stories. When something changes, update the relevant section **in place**. Historical detail belongs in git commit messages, not here.

---

## 1. Project Identity

- **Package**: `triago_control` — ROS 2 Humble, `ament_cmake` + `ament_cmake_python` (hybrid C++/Python)
- **Robot**: PAL Robotics TRIAGo++ (bimanual, mobile base, lift torso, head)
- **Repository**: https://github.com/Robertorocco/triago_control
- **Runtime**: Dockerized ROS 2 workspace, shared via `~/exchange/` with host
- **Sibling package**: `haption_teleoperation` (haptic device interface, lives inside the same repo) — see its own context.md for the human-side half of the shared-autonomy architecture.

## 2. Package Structure

```
triago_control/
├── config/worlds/*.yaml + *.world      per-scenario obstacle layouts (§6)
├── config/trajectory_endpoints.yaml    open-loop test presets
├── scripts/
│   ├── qp_arm_teleop/
│   │   ├── main_qp_controller.py       ★ QP-CLF-CBF safety loop (arms)
│   │   ├── main_shared_autonomy.py     ★ intent prediction + joystick-mode blending
│   │   ├── trajectory_generator.py     open-loop quintic reference source
│   │   ├── base_controller.py / keyboard_teleop.py   mobile base / keyboard jog
│   │   ├── plotter.py / offline_plotter.py           live / static telemetry dashboards
│   │   └── drift_evaluator_node.py     tracking error analysis
│   └── head_controller/
│       └── qp_head_visual_servo.py     ★ QP-based visual servoing for the head camera
├── haption_teleoperation/              sibling package (haptic device interface)
└── triago_control/                     importable Python library
    ├── qp_controller/                  QP safety math
    │   ├── config.py                       ALL tunable parameters (single source of truth)
    │   ├── robot_kinematics.py             Pinocchio model, FK, EMA filter, digital twin
    │   ├── collision_manager.py            hppfcl geometry, SoftMin CBF, dynamic margin
    │   ├── qp_formulator.py                CLF-CBF-QP: H/g/C/b assembly, quadprog solver
    │   ├── reference_governor.py           pre-CLF reference shaping / local-minima escape
    │   ├── world_loader.py                 YAML world-scene parsing (§6)
    │   ├── shared_autonomy_handler.py      gripper cmds, CBF-bypass, cylinder re-parenting
    │   └── visualization_engine.py         thread-safe Meshcat + RViz markers
    └── shared_autonomy/                 intent prediction
        ├── belief_estimator.py             Bayesian intent inference
        ├── goal_set.py                     dynamic goal-pose computation
        ├── grasp_state_machine.py          pick FSM (approach→contact→close→attach)
        └── plot_manager.py                 shared-autonomy telemetry plotting
```

**Import convention**: always `import triago_control.qp_controller.config as cfg` (fully-qualified). Never bare `import config` — it collides with system modules.

## 3. Mathematical Core: Arm QP-CLF-CBF (`main_qp_controller.py`)

Decision vector: `x = [q̇ (nv), δ_right, δ_left]` (joint velocities + one CLF slack per arm).

**Cost (minimize)**:
- Joint velocity regularization (damping λ=10.0).
- Posture/joint-limit avoidance via a repulsive potential field:
  `v_ref = -K_GRADIENT · dH/dp`, where `H(p) = 1/(1-p)² + 1/(1+p)²` on normalized joint position `p = (q - mid)/half_range` (clamped to ±`V_MAX_POSTURE`). ~0 mid-range, explodes near a limit. Weighted by `W_CENTER`, with a per-arm multiplier `posture_scale_right/left` (used e.g. to reduce posture priority during precision grasp phases, or as a local-minima-escape action).
- Slack penalty, adaptively weighted per arm (§8).

**Constraints (`C'x ≥ b`)**:
- **CLF (task tracking)**: scalar-inequality CLF with diagonal task weights `TASK_WEIGHTS_6D = [pos=10,10,10, ori=0.4,0.4,0.4]` (position weighted 25× orientation — position-first approach, orientation yields near obstacles). During autonomous grasp/release execution the **active** arm's CLF row instead uses `TASK_WEIGHTS_6D_GRASP = [10,10,10, 1,1,1]` (10:1) so the gripper's approach-axis aligns more tightly at the grasp/release pose. This is a **static per-phase swap** gated on `tracking_boost_arm` (`grasp_active`) — not a continuous/adaptive weight update — and covers `GRASP_ALIGN/APPROACH/CLOSE`, `LIFT`, `RELEASE_LIFT` (the placement approach during `HOLDING` keeps the nominal weights).
- **CBF (collision avoidance) — two independent per-arm rows**: `J_soft_R · q̇ ≥ b_R` and `J_soft_L · q̇ ≥ b_L`, each a SoftMin aggregate over the `K_MAX_PAIRS=60` closest collision pairs touching that arm's own geometry (a pair touching both arms' geometry — genuine inter-arm contact, or two held objects — contributes to both rows). Each row uses its own dynamic safety margin `d_safe_dynamic_X = D_SAFE_BASE + K_V_SAFE·‖v_X‖`, computed from only that arm's own joint velocities (this per-arm independence, at both the Jacobian and the margin level, is what prevents the inactive arm from being spuriously recruited to satisfy a barrier that has nothing to do with it).
- **Joint limits**: velocity-aware position buffer (CBF-style): every joint index not in `idx_right ∪ idx_left` (torso, base, gripper fingers, head) is hard-locked to `q̇=0` in the solve — this is also what makes the head chain safe to add as a quasi-static CBF obstacle (below) without ever adding a head joint to the decision vector.

Solver: `quadprog.solve_qp` (active-set method). Shadow prices `λ_cbf_right/left`, `λ_joints_right/left` are exposed as telemetry and consumed by the adaptive scheduler (§8) and the reference governor's local-minima-escape logic (§4).

**Bimanual inactive-arm handling**: the inactive arm is never zero-overwritten (that would discard its own collision-avoidance motion); instead it is frozen at its current EE pose via a zero-velocity CLF, with its slack weight doubled (`INACTIVE_SLACK_FACTOR=2.0`) so it holds position but can still yield if that helps the active arm.

**Head-as-obstacle**: the head chain (`arm_head_1..7_link`) is added to the arm QP's collision model as a quasi-static CBF obstacle via live FK capsules — sound under the assumption the head moves slowly relative to the CBF margin budget (bounded unmodeled term ≈ head's own max commanded speed × one control tick, negligible against `D_SAFE_BASE`). No head joint is added to the decision vector; the pre-existing "everything outside `idx_right/idx_left` is velocity-locked to zero" mechanism makes the barrier satisfiable only through arm motion, by construction.

## 4. Reference Governor (`reference_governor.py`)

An intermediate filter between the raw Cartesian reference (from teleop / trajectory_generator / planner) and the CLF's perceived reference — bounds what the CLF must track, preserving QP feasibility under aggressive/discontinuous commands. One instance per arm. Master switch `cfg.ENABLE_REFERENCE_GOVERNOR`.

| Feature | Mechanism | Config |
|---|---|---|
| Velocity shaping | Clamp reference velocity magnitude, direction preserved | `GOV_V_MAX_LIN`, `GOV_V_MAX_ANG` |
| Position error bounding | Project `x_ref` onto a ball of radius `E_MAX` centered at `x_real` | `GOV_E_MAX_POS` |
| Acceleration limiting | Rate-limit velocity change per tick: `‖Δv‖ ≤ A_MAX·dt` | `GOV_A_MAX_LIN`, `GOV_A_MAX_ANG` |
| Orientation clamping | If `‖log3(R_des·R_real^T)‖ > θ_MAX`, shrink via `exp3` on the same axis | `GOV_E_MAX_ORI` |

**Local minima escape** (extension, `cfg.ENABLE_LOCAL_MINIMA_ESCAPE`, one state machine per arm): detects a stuck 3D position error (error `> LME_ERROR_TRIGGER`, not decreasing over a rolling window), categorizes the cause from the QP's previous-tick shadow prices (`λ_cbf` vs `λ_joints`, obstacle takes priority if both are high), and applies a temporary per-arm posture-weight correction, smoothly ramped (first-order LPF):

| Category | Posture weight | Task dimension |
|---|---|---|
| Obstacle | ×0.2 (more redundancy to slip past) | forced to 3.0 (position-only, orientation relaxed) |
| Joint limit | ×5.0 (push harder away from the limit) | unchanged |

Exits on error recovery (`< LME_ERROR_RECOVERED`) or a max duration timeout, whichever comes first.

## 5. Shared Autonomy & Joystick-Mode Blending (`main_shared_autonomy.py`)

Implements Bayesian belief estimation over a discrete goal set, a local QP policy per goal, a grasp state machine, and (when `cfg.BLENDING`) reference-level blending of the human twist with the autonomous policy. The human side is "Joystick Mode" (`haption_teleoperation`'s context.md §3.2): the Haption handle is spring-centered and its displacement from home is the pure user twist.

### 5.1 Belief & Policy

- **Local policy per goal**: a constrained QP twist `pi_k` toward each goal `k`, computed **from the true EE pose** `current_T_EE`, subject to the same CBF-derived constraints as the safety QP (so guidance never points through an obstacle).
- **Belief estimator**: Bayesian update over goals with goal *exclusion* support (a goal held by either arm, or not currently graspable, is pinned to probability 0). The observed action driving the update is the **EE's actual movement** — a finite-difference + EMA estimate of the realized EE 6D twist (`_update_ee_twist`) compared against the EE-anchored policies, so intent is inferred from where the robot is genuinely going, not from raw handle input. `pos_costs` (10% tie-breaker) are also EE-anchored. (In `POLICY_BELIEF_TEST` the injected fake human twist is used instead, equivalent since `current_T_user == current_T_EE` there.)
- **Policy blend**: `pi_policy = Σ_k belief(k) · pi_k` (convex combination, smooth in belief) — this is the single optimal twist passed into the arbitration.

### 5.2 Joystick-Mode Arbitration (`cfg.BLENDING = True`)

The Cartesian reference sent to the QP is a blend of the user twist and the belief-weighted policy, integrated persistently every tick into a latched reference (`T_blend_ref` via `integrate_twist`, so an idle user yields an absolute hold rather than a reference that chases the drifting EE). `main_shared_autonomy.py` is the sole publisher of `/arm_*/cartesian_reference` while blending is active.

```
v_blend = (1 - alpha) * v_user + alpha * pi_policy
```

**`compute_alpha`** — the authority weight `alpha` (weight on the policy) depends on whether the user is driving and, if so, on the alignment between the user twist and the policy twist:

```
user STILL (v_user == 0, inside the joystick deadband):
    alpha = ALIGN_ALPHA_IDLE                                    # gentle autonomous crawl
user ACTIVE:
    s     = mean per-channel cosine(v_user, pi_policy)  over whichever channel(s) the user commands
    alpha = ALIGN_ALPHA_MIN + (ALIGN_ALPHA_MAX - ALIGN_ALPHA_MIN) · clip(s, 0, 1)
                                                                # (both branches LPF'd, C0-continuous)
```

- User still → `alpha = ALIGN_ALPHA_IDLE = 0.35` → the robot only **gently crawls** toward the inferred goal (deliberately slow when the user isn't driving).
- Misaligned (`s ≤ 0`) → `alpha = ALIGN_ALPHA_MIN = 0.2` → **user keeps 80%** (prioritised whenever they disagree with the policy).
- Aligned and actively moving (`s → 1`) → `alpha = ALIGN_ALPHA_MAX = 0.8` → autonomy leads **fast** toward the goal — the robot only goes fast once the user pushes in a direction the policy agrees with.

Belief still selects *which* goal's policy is `pi_policy`; `alpha` only arbitrates authority. The user twist arrives already deadbanded to zero inside the joystick's home deadband (done teleop-side), so residual handle noise can never creep the arm.

**Blending is active in every user-controlled state** — `SHARED_AUTONOMY`, `PRE_GRASP`, and `HOLDING` — i.e. all states *except* the autonomous grasp-execution ones (`GRASP_ALIGN/APPROACH/CLOSE`, `LIFT`, `RELEASE_LIFT`, `ABORT_RETREAT`), where `grasp_active` suspends the joystick and the state machine drives the arm. So the operator keeps steering while hovering in `PRE_GRASP` (until they trigger the grasp) and while carrying an object in `HOLDING` (until they trigger the release). The persistent `T_blend_ref` latch re-anchors to the live EE on each transition back into a user-controlled state, so control resumes with no jump. This replaces the deprecated belief×distance-gated blend + F_sync force-feedback design, whose force-into-handle path formed an unstable feedback loop.

**Telemetry**: `/shared_autonomy/blend_debug` (19 floats: `[alpha, v_user(6), v_policy(6), v_blend(6)]`), published every tick regardless of `cfg.BLENDING`.

**Operator marker** (`/blended_reference_marker`): the single gripper the operator watches while teleoperating — the pose actually being tracked (`T_virtual_ref`, the reference published to `/arm_*/cartesian_reference`, integrated consistently from the blended twist). Colored by who is currently driving it: **GREEN** = tracking the policy (hands-off autopilot, or the user's twist agrees so the policy dominates, `alpha ≥ 0.5`); **ORANGE** = listening to the user's twist intention (the user is actively overriding). This is the primary teleop cue; the other predictive markers (`/guidance_policy_marker`, `/robot_policy_marker`) can be left disabled in RViz.

### 5.3 Bimanual State

Each arm owns an independent `GraspStateMachine` + `BeliefEstimator` + grasped-color + goal-set placement context; the inactive arm's FSM/belief are simply not stepped. Goal exclusion is the *union* over both arms. A cylinder-vs-cylinder CBF pair prevents two held objects from inter-penetrating.

### 5.4 Goal Set (`goal_set.py`)

Goals are dynamic SE(3) manifolds, not fixed poses:
- **`Top`/`Side`**: approach axis derived from anchor→cylinder-axis geometry; a sticky hysteresis avoids indecision when crossing near-degenerate configurations (e.g. directly above the cylinder for `Side`).
- **`Front`**: approach axis rigidly locked to world +X (used for pure reach/hover tutorial goals with no physical object).
- **`Platform_Place`**: the only hard constraint is *held-object axis ⊥ platform face* (vertical) — 2 DOF constrained (tilt), yaw about vertical left free, so it is a true placement manifold, not a single pose. The held object's symmetry axis is frozen in the gripper frame at grasp time (`grasped_axis_local = R_grasp^T · [0,0,1]`).

### 5.5 Grasp State Machine

`SHARED_AUTONOMY → PRE_GRASP → GRASP_ALIGN → GRASP_APPROACH → GRASP_CLOSE → LIFT → HOLDING` (failure path: `→ ABORT_RETREAT`). On `GRASP_CLOSE → LIFT`, the held object is re-parented as a real link in the QP collision world (its own CBF bypass with the gripper is cleared after a smooth ramp) so it actively avoids the environment from then on. Release reverses this (`DETACH_*`, `RELEASE_LIFT`) and relocates the object's believed pose to where it was actually set down.

**Grasp confirmation is purely geometric** (no force/torque or vision sensing exists on this robot): a signed gripper-box↔object `hppfcl` distance against the object's *believed* (kinematically tracked, not measured) position/axis. The required contact overlap is **per grasp type** (`GraspStateMachine._contact_depth_threshold`): `Top = -0.03 m` (shallow — the arm can't seat fingers deeply on a vertical approach) and `Side = -0.045 m` (deeper seating as the fingers bracket the cylinder wall). More-negative = more overlap required = stricter.

## 6. World Scene Loading (`world_loader.py`, `config/worlds/*.yaml`)

Static obstacle layouts (table, graspable cylinders, walls, shields, etc.) are described declaratively per scenario in `config/worlds/<name>.yaml`, rather than hard-coded in `config.py`. Schema: a list of `ObstacleSpec` (name, role, shape, pose, size, color, collision on/off) + a `grasp_roles: {red: <name>, blue: <name>}` mapping + an optional `platform` field (visual-only placement-target disk, never added to the collision model).

`CollisionManager`, `VisualizationEngine`, and `GoalSet` all consume a `world_scene` generically (building hppfcl geometry / RViz markers / goal cylinders from whatever obstacles are listed) — adding a new world with new obstacles requires **no code changes**, only a new YAML file. `main_qp_controller.py` and `main_shared_autonomy.py` each take a `world_name` ROS parameter (default `no_obstacle`) and must be launched with the **same** value. The YAML filename always matches its companion Gazebo `.world` filename.

An obstacle with `role: "reachable"` and no `<collision>` produces a pure reach/hover goal (`Front` grasp type, §5.4) instead of a graspable one — used for onboarding/movement-tutorial worlds with no real objects.

## 7. Head Controller — Visual Servoing (`qp_head_visual_servo.py`)

Fully independent from the arm QP (own solver instance, own velocity controller, no shared state, does not participate in arm CBF pairs). Keeps both hands in the camera FOV via 2.5D visual servoing.

Decision vector: `x = [dq_head (7), slack (3)]`.

**Two-stage state machine**:
| Stage | Condition | Strategy |
|---|---|---|
| PBVS (Look-At) | hands outside FOV / behind camera | 3D rotational servoing: `cross(z_cam, dir_to_centroid) → ω` via `J_rot` |
| IBVS (Pixel tracking) | both hands inside FOV margin | 2.5D: interaction matrix `L_s` maps pixel+depth error to camera twist |

**Cost**: joint velocity regularization (heavier weight on base joints), slack penalty (`W_SLACK_PIXELS` for u,v; `W_SLACK_DEPTH` for depth, to normalize pixel vs. metric scale), postural centering spring (`K_POSTURE`).

**Equality constraint (CLF-like)**: `J_task · dq − slack = −λ_visual · e`, with `J_task = L_s · J_cam` (IBVS) or `J_rot` (PBVS), `e` the corresponding pixel/rotation error.

**Inequality constraints**: FOV barriers (IBVS only — each hand ≥ `FOV_MARGIN` px from image edges, 4 per hand) + velocity-aware joint-limit buffers (same CBF-style pattern as the arm QP).

Hand positions are currently obtained via Pinocchio FK (kinematic tracking), not image-based detection — this is the acknowledged starting point for future perception work. A separate lightweight collision model (capsules + boxes, head-vs-body/head-vs-wall) exists but its distance constraints are not yet wired into this QP as CBF inequalities.

### 7.1 Active-Arm Tracking (`head_active_arm_tracking.py`)

Teleoperation-specific sibling of the visual servo (same independent-QP / IBVS-PBVS structure, `x = [dq_head (7), slack (3)]`, own velocity controller, head chain not in any arm decision vector). Assumes the teleop invariant that **exactly one arm is active at a time**: it subscribes to `/shared_autonomy/active_arm` (`std_msgs/String` "right"/"left", published by `main_shared_autonomy.py`) and points the camera at that arm's `arm_{side}_tool_link`, tracked purely by FK (no vision).

Two coupled objectives, both handled by the single 2.5D IBVS equality task on the *active hand* (not a two-hand centroid):
- **Centering**: pixel error `[u−CX, v−CY]` → hand driven to image centre (`J_task = L_s · J_cam`).
- **Stand-off**: depth error `Z − TARGET_DISTANCE` (`TARGET_DISTANCE = 0.5 m`, overridable via the `target_distance` ROS param) → camera translates to hold 50 cm.

Falls back to PBVS rotational look-at when the hand is behind the lens / outside the FOV margin. FOV barriers (4 rows) apply to the single tracked hand; same velocity-aware joint-limit CBF buffers as the visual servo.

**Soft roll alignment (up-righting)**: point tracking leaves camera roll about the optical axis unconstrained, so the view could drift upside-down. A *soft* preference keeps the image-right axis (camera X) aligned with world-right (`WORLD_RIGHT = world -Y`): signed roll error `theta = atan2(d_y, d_x)` with `d = R_cam^T · WORLD_RIGHT`, regulated by a roll rate `wz = K_ROLL_ALIGN·theta` about the optical axis (`dtheta/dt = -wz`). Encoded as a **low-weight least-squares term in the QP cost** (`H[:7,:7] += W_ROLL_ALIGN·J_roll J_roll^T`, `g[:7] += W_ROLL_ALIGN·wz·J_roll`, `J_roll = J_cam[5,:]`), NOT an equality row — so it only uses the redundancy left after centering/stand-off and can never override tracking (a hard equality attempt previously broke FOV acquisition). A centering gate fades it out while the hand is off-axis and a gimbal guard disables it when world-right is ~parallel to the optical axis. The optical axis stays free (view the hand from above/below); the image just never rolls past vertical.

A self-contained Matplotlib **plotting thread** (in the same file, guarded so a missing display never stalls control, disable with `-p plot:=false`) shows live centering error (pixel + angular) and the stand-off distance vs. the 0.5 m target. Telemetry also on `/head_active_tracking/{error,qdot,cartesian_cmd}`. Run: `ros2 run triago_control head_active_arm_tracking.py`.

## 8. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting**: each arm's CLF slack weight drops toward `BASE_WEIGHT_SLACK` as its own shadow price grows (letting slack absorb tracking error near obstacles), and rises toward `MAX_WEIGHT_SLACK` in free space (tighter tracking).
- **Dynamic CLF gain**: convergence rate γ drops exponentially with the collision Lagrangian λ_cbf, LPF'd (τ=0.125s) — tracking priority in free space, yields to safety near obstacles.

## 9. Frame Convention (Haption ↔ TRIAGo)

Identical to `haption_teleoperation`'s §7: a pure 180° rotation about Z between the Haption device frame and TRIAGo's `base_footprint` (negate X, negate Y, keep Z — same for force feedback).

## 10. Critical Hardware / Environment Quirks

1. **Simulated joint velocities are unreliable in Gazebo** — derived from position differences + EMA filter (`ALPHA_FILTER=0.5`). On real hardware, velocity is read directly, no differentiation.
2. **`REAL_HARDWARE` auto-detection**: inferred at startup from whether the fetched URDF contains `gripper_{right,left}_grasping_link` natively (Gazebo URDF: yes → simulation; real TIAGo Pro URDF: no → real hardware, frames injected + broadcast as static TFs).
3. **Meshcat is not thread-safe** — only the dedicated `_run_viz` thread may touch the WebSocket; callbacks mutate state under a lock and set a reload-pending flag.
4. **Controller switching** — TRIAGo requires explicit activation of `arm_{right,left}_joint_space_controller_vel` and deactivation of the conflicting trajectory controllers before the QP can command the arms.
5. **No force/torque sensing anywhere on the arm chains**, and no Gazebo ground-truth is ever read — everything is derived from `/joint_states` + FK. This is why grasp confirmation (§5.5) is purely geometric.

## 11. Build & Run

```bash
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

ros2 run triago_control main_qp_controller.py --ros-args -p world_name:=no_obstacle
ros2 run triago_control main_shared_autonomy.py --ros-args -p world_name:=no_obstacle
ros2 run triago_control qp_head_visual_servo.py     # independent, can run alongside
ros2 run triago_control head_active_arm_tracking.py # teleop: head follows the active arm (-p plot:=false to disable dashboard)
ros2 run triago_control trajectory_generator.py     # open-loop test source
ros2 run triago_control plotter.py                  # live dashboard
ros2 run triago_control offline_plotter.py          # static, publication-quality figures
```

**Full simulation launch sequence** (robot side, then teleoperation side):
```bash
ros2 launch triago_gazebo triago_gazebo.launch.py \
    end_effector_right:=pal-pro-gripper end_effector_left:=pal-pro-gripper \
    world_name:=no_obstacle
ros2 launch triago_controller_configuration tsid_default_controllers.launch.py use_sim_time:=True
ros2 run triago_control main_qp_controller.py
ros2 run triago_control main_shared_autonomy.py
ros2 launch triago_control visualize.launch.py

# teleoperation side (haption_teleoperation package) -- teleop + force node pair per cfg.BLENDING:
ros2 run haption_teleoperation virtuose_server_node
#   BLENDING=False (Virtual Fixture):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_tutorial.py
#   BLENDING=True (Joystick Mode):
ros2 run haption_teleoperation teleop_triago_joystick.py
ros2 run haption_teleoperation haptic_force_manager_blending_tutorial.py
```

**Gazebo Link Attacher** (external dependency, kinematic attach/detach during grasping — not part of this repo):
```bash
cd ~/exchange/ros2-ws/src
git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git
cd ~/exchange/ros2-ws && colcon build --packages-up-to ros2_linkattacher && source install/setup.bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/exchange/ros2-ws/install/ros2_linkattacher/lib
```

## 12. Coding Conventions

- Every tunable value lives in `qp_controller/config.py` — never hard-code gains elsewhere.
- snake_case files/variables, PascalCase classes.
- Module-level docstrings explain the "why" and the math; class/method docstrings for non-obvious logic.
- Always `import triago_control.qp_controller.config as cfg` — never bare `import config`.
- No `_refactored`/`_v2`/`_new` suffixes in filenames — git history is for that.
- Entry points are `main_*.py` in `scripts/`; library modules never contain `if __name__ == '__main__'`.

## 13. User Workspace Paths

- **Colcon workspace**: `~/exchange/ros2-ws/`
- **This repo clone location**: `~/exchange/ros2-ws/src/triago_control`

## 14. Git Workflow

- Push directly to `main` (no feature branches / PRs for this repo).
- **After every push**, ALWAYS provide the user with the exact commands to sync their local machine:

```bash
cd ~/exchange/ros2-ws/src/triago_control
git checkout -- .
git pull origin main
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash
```
