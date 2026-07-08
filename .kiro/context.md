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
│   ├── head_controller/
│   │   └── qp_head_visual_servo.py     ★ QP-based visual servoing for the head camera
│   └── analysis/                       user-study data capture + offline analysis (§15)
│       ├── study_config.py                 study/data settings (paths, topics, thresholds)
│       ├── study_recorder.py               Tkinter GUI wrapper around `ros2 bag record`
│       ├── study_metrics.py                bag reader + metric engine (numpy-only)
│       └── analyze_trial.py                per-trial PNG dashboard + metrics summary
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
- **CLF (task tracking)**: scalar-inequality CLF with diagonal task weights `TASK_WEIGHTS_6D = [pos=10,10,10, ori=0.4,0.4,0.4]` (position weighted 25× orientation — position-first approach, orientation yields near obstacles). An arm doing **precision work with the object** instead uses `TASK_WEIGHTS_6D_GRASP = [10,10,10, 2,2,2]` (5:1) so the gripper's approach-axis / placement orientation aligns tightly. This is a **static per-phase swap** (not a continuous/adaptive update), applied per arm via `orient_boost_arms` = the active arm during autonomous grasp/release (`grasp_active`: `GRASP_ALIGN/APPROACH/CLOSE`, `LIFT`, `RELEASE_LIFT`) **∪** any arm currently carrying an attached object (`hri.attached_object_arm`) — so the entire `HOLDING`/placement-approach phase, and thus the release pose, is aligned too. This orientation-weight boost is **decoupled** from `tracking_boost_arm`, which still drives only the slack/gamma boost during `grasp_active`.
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

## 5. Shared Autonomy & Reference-Level Blending (`main_shared_autonomy.py`)

Implements Bayesian belief estimation over a discrete goal set, a local QP policy per goal, a grasp state machine, and (when `cfg.ASSIST_BLENDING`) reference-level blending of the human twist with the autonomous policy. The human side is either "Joystick Mode" (velocity control, `haption_teleoperation`'s context.md §3.2: spring-centered handle, displacement → twist) or "Clutch Mode" (position control, integrates handle twist into a pose reference).

### 5.0 Experiment Condition Selector (fair 2×3 study) — `config.py` §1b

The study is a **fair 2×3 matrix**: two control modes × three assistance levels, selected by three **orthogonal** flags in `config.py` (§1b):

- `CONTROL_MODE ∈ {CLUTCH, JOYSTICK}` — position control (clutch integrates the handle twist to a pose reference) vs velocity control (spring-centered joystick, displacement → twist).
- `ASSIST_FEEDBACK` (bool) — **channel F**: assistive haptic FORCES on the handle (`F_guide` velocity field + `F_fixture` funnel) on top of the always-present `F_sync` tether.
- `ASSIST_BLENDING` (bool) — **channel B**: reference-level arbitration of the user twist with the belief-weighted policy in `main_shared_autonomy` (sole writer of `/arm_*/cartesian_reference`). `cfg.BLENDING` is a backward-compat alias equal to `ASSIST_BLENDING`.

Exposing the two channels independently fixes the previous **unfairness** (the clutch condition only ever closed an assistive-*feedback* loop; the joystick condition only ever closed an assistive-*action/blend* loop).

| CONTROL_MODE | ASSIST_FEEDBACK | ASSIST_BLENDING | Condition | Teleop + force manager |
|---|---|---|---|---|
| CLUTCH | False | False | Sync only (baseline) | `teleop_triago_clutch` + `haptic_force_manager_noguidance_tutorial` |
| CLUTCH | True | False | Guided feedback (VF) | `teleop_triago_clutch` + `haptic_force_manager_tutorial` |
| CLUTCH | True | True | Full guidance | `teleop_triago_clutch` + `haptic_force_manager_full_tutorial` (NEW) |
| JOYSTICK | False | False | Sync only | `teleop_triago_joystick` + `haptic_force_manager_joystick_sync_tutorial` (NEW) |
| JOYSTICK | False | True | Guided blending | `teleop_triago_joystick` + `haptic_force_manager_blending_tutorial` |
| JOYSTICK | True | True | Full guidance | `teleop_triago_joystick` + (joystick full manager, NEW) |

Every teleop / force-manager node calls `cfg.validate_condition(node_name, control_mode=…, feedback=…, blending=…)` at startup and **HARD-ERRORS on mismatch** (teleop nodes constrain only the control mode since they serve all three of their column's cells; force managers pin the full triple), so a mis-launched condition fails loudly instead of silently running the wrong strategy.

> **Ownership**: the CLUTCH column (position control) + this selector/`validate_condition` live with one agent; the JOYSTICK column (velocity control) with another. Both edit `config.py` and `main_shared_autonomy.py`, so pull → edit → push tightly.

### 5.1 Belief & Policy

- **Local policy per goal**: a constrained QP twist `pi_k` toward each goal `k`, computed **from the true EE pose** `current_T_EE`, subject to the same CBF-derived constraints as the safety QP (so guidance never points through an obstacle).
- **Belief estimator**: Bayesian update over goals with goal *exclusion* support (a goal held by either arm, or not currently graspable, is pinned to probability 0). The observed action driving the update is the **user's commanded twist** (`current_v_h`, the user/joystick gripper intent) compared against the **user-frame policies** — so intent reflects what the operator is trying to do, and the autonomy→EE→belief self-confirmation loop is broken (this loop made goals like Side→Top sticky to switch). The twist cost matrix is `W = diag([10,10,10, 2,2,2])`. `pos_costs` (10% tie-breaker) are anchored at `current_T_user` (== `current_T_EE` in joystick mode). (`POLICY_BELIEF_TEST` injects a fake human twist through the same pairing.)
- **Policy blend**: `pi_policy = Σ_k belief(k) · pi_k` (convex combination, smooth in belief) — this is the single optimal twist passed into the arbitration.

### 5.2 Reference-Level Arbitration (`cfg.ASSIST_BLENDING = True`)

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

**CLUTCH control mode** (`CONTROL_MODE=CLUTCH`, the full-guidance cell `F=1,B=1`): the arbitration above is reused **verbatim**, fed by the clutch teleop's twist (message velocity slots). Crucially the user-policies the belief scores the twist against are anchored at the **blended reference gripper** (`self.T_blend_ref`), **not** the clutch's integrated pose — because the operator teleoperates by *watching the blended gripper* and shapes their twist relative to it, so the integrated pose is meaningless as an anchor here (see the `belief_anchor` selection in `timer_callback`). This is the one clutch-specific deviation from the joystick path (which already anchors at the live EE it carries in its pose slots); the 10% position tie-breaker stays anchored at the real EE (≈ the blended gripper). **Idle-crawl hold (Option A, CLUTCH only):** the Side-grasp azimuth is pinned to the *lagging EE*, so when the operator steers the blended gripper to a new side and then goes STILL before the robot has caught up, the idle crawl (`alpha=ALIGN_ALPHA_IDLE`) would drag the reference back toward the old (EE-side) goal. To prevent that, in the CLUTCH cell when the user is still AND the tracking lead `‖T_blend_ref − EE‖ > CLUTCH_CRAWL_HOLD_LEAD` (3 cm), `alpha` is forced to 0 (pure user twist, reference held) — the QP still drives the robot to the commanded pose, the EE-derived azimuth refreshes as the robot arrives, and the crawl resumes only once the lead drops back below the threshold (`_clutch_crawl_hold`). While the **clutch button is held** (`virtuose/button_right` — the operator is indexing/repositioning the handle) blending is **SUSPENDED**: `alpha` is forced to 0 and the alpha-LPF reset, so with the clutch's `v_user=0` the latched `T_blend_ref` holds absolutely still until release. On release the normal alignment blend (including the idle crawl) resumes from the latch. `main_shared_autonomy` subscribes to `virtuose/button_right` for this (unused/harmless in JOYSTICK mode).

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

**Grasp confirmation is purely geometric** (no force/torque or vision sensing exists on this robot): a signed gripper-box↔object `hppfcl` distance against the object's *believed* (kinematically tracked, not measured) position/axis. The required contact overlap is **per grasp type** (`GraspStateMachine._contact_depth_threshold`): `Top = -0.03 m` (shallow — the arm can't seat fingers deeply on a vertical approach) and `Side = -0.05 m` (deeper seating as the fingers bracket the cylinder wall). More-negative = more overlap required = stricter. The straight-line **insertion depth** (advance from standoff along the approach axis) is likewise per type (`_insertion_travel`): `Top = 0.09 m`, `Side = 0.072 m` (20% shallower so a side insertion doesn't shove the cylinder before the fingers close).

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

# teleoperation side (haption_teleoperation) -- launch the teleop + force manager
# pair for the condition selected in config.py §1b (see §5.0). Each node hard-errors
# at startup if it does not match CONTROL_MODE / ASSIST_FEEDBACK / ASSIST_BLENDING.
ros2 run haption_teleoperation virtuose_server_node
#   CLUTCH, sync only        (F=0, B=0):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_noguidance_tutorial.py
#   CLUTCH, guided feedback  (F=1, B=0):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_tutorial.py
#   CLUTCH, full guidance    (F=1, B=1):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_full_tutorial.py   # (NEW)
#   JOYSTICK, guided blending (F=0, B=1):
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


## 15. User Study / Analysis Subsystem (`scripts/analysis/`)

Self-contained tooling to run and record a **human-subject study** comparing feedback strategies on a **cylinder pick-and-place** task, across the declarative worlds (§6), for publication. Lives entirely under `scripts/analysis/` — never mixed with the controller code. Its parameters live in `scripts/analysis/study_config.py`, deliberately **separate** from `qp_controller/config.py` (§12): these are experiment/data-management settings — participant identity, storage paths, topic allowlists, resampling, offline-metric thresholds — not controller gains. Controller behaviour is never tuned here.

### 15.1 Independent Variable — Feedback Conditions

The independent variable is the **fair 2×3 condition matrix** defined authoritatively in §5.0 — a `(CONTROL_MODE, ASSIST_FEEDBACK, ASSIST_BLENDING)` triple, not the single legacy `cfg.BLENDING`. Each cell is a `(control_mode, assist_level)` pair used in trial-folder names and the master table (e.g. `clutch_sync`, `clutch_feedback`, `clutch_full`, `joystick_sync`, `joystick_blend`, `joystick_full`). The recorder snapshots the full triple from `cfg` into `metadata.json` and verifies the launched condition against it (nodes also hard-error on mismatch via `cfg.validate_condition`, §5.0). The experimenter's declaration remains the source of truth for the manual success call; the flag triple resolves the condition unambiguously (unlike the old scheme where `virtual_fixture` and `no_assist` were indistinguishable at `cfg.BLENDING=False`).

> The analysis condition labels/short-codes here are being migrated to the 2×3 scheme; `study_config.py` (analysis agent) owns the canonical label strings.

### 15.2 Recorder — GUI, one launch per trial (`study_recorder.py`)

A tiny **Tkinter GUI** wrapping `ros2 bag record`, launched fresh per trial (`ros2 run triago_control study_recorder.py`). It uses **no rclpy** (pure `subprocess` capture) and is stateless. Because it runs headless-importable, the pure helpers (`build_record_command`, `sanitize_token`, `snapshot_cfg`) can be tested without a display; the tkinter import is guarded. Workflow:
1. `PARTICIPANT_ID` is prefilled from `study_config` (constant per session; `TRIAGO_PARTICIPANT_ID` env / `-p participant:=` overridable) and editable in the form.
2. The experimenter types a **world shortcut** and selects the **feedback strategy** (radio: `virtual_fixture` / `blending` / `no_assist`); a live label shows the `cfg.BLENDING` consistency check (red on mismatch — e.g. `blending` selected while `cfg.BLENDING=False`; `virtual_fixture`/`no_assist` cannot be auto-distinguished so the declaration wins).
3. **START** → spawns `ros2 bag record` for `BAG_TOPICS` in its own process group; an elapsed timer runs.
4. **STOP** → sends SIGINT to the process group so rosbag2 finalises the bag cleanly (falls back to terminate/kill on timeout).
5. The experimenter marks **Success (Yes/No)** + free-text **Notes** and presses **SAVE**, which writes `metadata.json` and resets the form for the next trial (participant/world/strategy stay sticky). The state machine (`IDLE → RECORDING → AWAIT_SAVE → IDLE`) forces classifying every trial before a new one can start.

**Success is always the experimenter's manual call** (no Gazebo ground-truth is read; correct-placement verification is explicitly out of scope).

### 15.3 Capture — rosbag only (post-processed offline)

**The recorder captures a rosbag and nothing else** during a trial (no live subscription/resampling/metrics — an analysis bug can never corrupt or crash a live recording, and the bag is the single authoritative raw record). Storage layout (`study_config` helpers `participant_dir` / `bag_folder_name` / `bag_path`):

```
DATA_ROOT/<participant>/<world_shortcut>_<condition_short>/   # = the `ros2 bag record -o` dir
    <bag>.db3 + metadata.yaml     # written by rosbag2
    metadata.json                 # our provenance sidecar (METADATA_NAME)
```

`condition_short` ∈ {`vf`, `bl`, `na`} (`STRATEGY_SHORTCUTS`). There is **one folder per `(participant, world, condition)` triple — re-recording the same triple OVERWRITES it** (confirmed via a dialog; no repetition index, no timestamp, so folders stay trivially scannable). The bag uses the **curated topic allowlist** (`BAG_TOPICS`; `/joint_states` + `/tf` for standalone replay, head-camera point clouds excluded) and storage backend `BAG_STORAGE_ID` (default `sqlite3`). `metadata.json` holds participant / world / condition (+ short), start/stop timestamps, duration, success, notes, `cfg.BLENDING`, a `CFG_SNAPSHOT_KEYS` snapshot of `cfg`, hostname, and the topic list. The tidy time-series is **derived offline from the bag** (§15.5), not written live.

### 15.4 Storage Split

**Code lives in the git repo; all heavy data lives locally, never on GitHub.** `DATA_ROOT` defaults to `~/exchange/triago_study_data/` (outside the repo, `TRIAGO_STUDY_DATA_ROOT`-overridable); rosbags, timeseries, and figures are written there. A `.gitignore` guard (`triago_study_data/`, `*.bag`, `*.db3`, `*.mcap`, `*.parquet`) covers the case of pointing the root inside the repo.

### 15.5 Offline Analysis

- **`study_metrics.py`** — numpy-only engine (no pandas, matching the repo stack): `load_bag()` reads a trial bag via `rosbag2_py` + `rclpy` deserialization into per-topic numpy `Series` (needs a sourced ROS 2 env); `compute_metrics()` returns a **flat dict** of the metrics below; `format_summary()` renders the human table; `sparc()` is the smoothness metric. All math is pure/unit-testable without ROS. Array layouts of the multi-array telemetry are pinned as topic constants at the top of the file.
- **`analyze_trial.py`** — the runnable per-trial analyzer (`ros2 run triago_control analyze_trial.py [path…]`, or plain `python3`). With no argument it walks `DATA_ROOT`; given a folder it analyzes that trial (or all trials beneath it). Because the QP **decouples the two arms** and only one is teleoperated at a time, it analyzes and plots **each arm separately**, writing into the trial folder: `plot_dashboard_{right,left}.png`, `metrics_summary_{right,left}.txt`, and one `metrics.json` (`{metadata, right, left}`). Each 3×3 dashboard shows that arm's decoupled data: EE path, EE speed (from the **published** `/qp_debug/ee_real` velocity slots — ground truth, never differentiated), measured joint velocity (`qdot_measured`, ground truth) and the **QP solution** (`qdot_cmd`, 7 joints), that arm's own CLF slack + CBF λ (`slacks`/`lambda_cbf` index 0=right/1=left), obstacle clearance (grasp shaded), haptic force + clutch + authority α, plus the **active-arm timeline** and a metrics text panel. Matplotlib headless (`Agg`).
- **Active-hand resolution** (`study_metrics.resolve_active_arm`): prefers `/shared_autonomy/active_arm` when the topic carries data (VF / blending); otherwise **infers the active hand from which arm is actually moving** (published EE speed), exploiting the one-arm-at-a-time invariant. This makes the active hand available even in `no_assist`, where `main_shared_autonomy` does not run — with **no controller change required** (resolved offline from the bag).
- **`build_master_table.py`** (planned) — aggregate every trial's `metrics.json` into the single `trials_summary.csv` (`MASTER_TABLE_NAME`), one row per `(participant, world, condition)` (overwrite semantics ⇒ the latest recording of each triple is the analysed one). Pandas/pyarrow are acceptable here.
- **`study_analysis.py`** (planned) — cross-condition comparison figures + stats tables from the master table.

**Metric families**: (A) *effectiveness* — total/per-phase time (sliced by `/shared_autonomy/grasp_active`), manual success, #retries/#aborts; (B) *motion quality* — EE speed / SPARC smoothness from the **published** `/qp_debug/ee_real` velocity slots (ground truth, per arm) and the QP solution `qdot_cmd` vs measured `qdot_measured`, path efficiency (an earlier flat-zero EE speed was a wrong-arm artifact — the idle arm was plotted — fixed by resolving the active hand and splitting per arm, not by differentiating position); (C) *safety* — min-distance / near-miss stats on `/qp_debug/min_distance` **computed with the autonomous-grasp window excluded** (that topic is the raw *signed* closest distance over all pairs *before* the grasp CBF-bypass, and the graspable cylinder is in the collision set, so the intentional gripper↔cylinder overlap drives it negative during the grasp; a grasp-inclusive raw minimum is kept separately for reference), plus λ_cbf active-time; (D) *human effort* — clutch-press count/duty (`/virtuose/button_right`, VF/no_assist), force impulse from `/virtuose/force_cmd` (**not cross-mode comparable** — different force semantics per condition), handle excursion; (E) *assistance* — α stats and human–policy agreement from `/shared_autonomy/blend_debug`, belief-convergence time from `/shared_autonomy/goal_probabilities`; (F) *subjective* — questionnaire scores (NASA-TLX / trust / preference) stored into `metadata.json`. The dashboard shades the autonomous-grasp window on the authority, CBF and safety panels so the grasp-driven excursions are visually separable.
