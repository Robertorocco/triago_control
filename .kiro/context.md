# AI Agent Context — triago_control

> **This file is maintained by the AI agent.**

## 0. Maintenance Rules

1. **Always share the pull/rebuild command** with the user immediately after pushing any change to this repo (see §11 for the exact sequence). Never wait to be asked.
2. **Keep this file clean and short.** Only math formulations and core architectural concepts an AI agent needs — no dated changelogs, "Last updated:"/"Earlier:" narratives, tuning history, or bugfix stories. Update the relevant section **in place**; historical detail belongs in git commit messages.
3. **Length budget: ~6k words / ~9k tokens total** (check with `wc -w .kiro/context.md` — line count is a poor proxy since paragraphs don't wrap). When adding a new feature's worth of content, cut something else in the same edit (redundant phrasing, resolved caveats, superseded designs) rather than letting the file grow monotonically. One tight paragraph beats three that repeat the same point. State facts once; don't restate the "why" of something already obvious from the math/table next to it.

---

## 1. Project Identity

- **Package**: `triago_control` — ROS 2 Humble, `ament_cmake` + `ament_cmake_python` (hybrid C++/Python)
- **Robot**: PAL Robotics TRIAGo++ (bimanual, mobile base, lift torso, head)
- **Repository**: https://github.com/Robertorocco/triago_control
- **Runtime**: Dockerized ROS 2 workspace, shared via `~/exchange/` with host
- **Sibling package**: `haption_teleoperation` (haptic device interface, same repo) — see its own context.md for the human-side half of the shared-autonomy architecture.

**Branch strategy (from 2026-07-30)**: `real-hw` is a frozen checkpoint of every value tuned on the physical robot (§2, §8.4) — diff against it anytime (`git diff real-hw -- triago_control/qp_controller/config.py`) rather than trusting numbers restated in prose here, which will drift as sim work retunes them. Active development (sim-only, safety/robustness prioritized over tracking performance) continues on `main` / `feature/sim-user-study`.

## 2. Package Structure

```
triago_control/
├── config/worlds/*.yaml + *.world      per-scenario obstacle layouts (§6)
├── config/trajectory_endpoints.yaml    open-loop test presets
├── scripts/
│   ├── qp_arm_teleop/
│   │   ├── main_qp_controller.py       ★ QP-CLF-CBF safety loop (arms)
│   │   ├── main_qp_controller_perceived.py ★ camera-driven variant: CBF world from the head camera (§6.1)
│   │   ├── main_qp_controller_real.py  ★ real-hardware variant: async CBF + overlay threads + staleness watchdog (§8.2)
│   │   ├── main_shared_autonomy.py     ★ intent prediction + joystick-mode blending
│   │   ├── trajectory_generator.py     open-loop quintic reference source
│   │   ├── base_controller.py / keyboard_teleop.py   mobile base / keyboard jog
│   │   ├── plotter.py / offline_plotter.py           live / static telemetry dashboards (latter also writes summary_metrics.json + a `ros2 bag` per trial, §8.3)
│   │   ├── drift_evaluator_node.py     tracking error analysis
│   │   └── freq_oscillation_diagnostic.py  buckets a ripple to CBF/CLF-posture/downstream-of-QP (§8.3)
│   ├── head_controller/
│   │   ├── qp_head_visual_servo.py     ★ QP-based visual servoing for the head camera
│   │   └── main_head.py                ★ RANSAC tabletop perception; publishes the perceived-world snapshot (§6.1)
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
    │   ├── reference_governor.py           pre-CLF reference shaping (vel/pos/accel/orient bounds)
    │   ├── world_loader.py                 YAML world-scene parsing (§6)
    │   ├── perceived_world_builder.py      camera snapshot → WorldScene (§6.1)
    │   ├── shared_autonomy_handler.py      gripper cmds, CBF-bypass, cylinder re-parenting
    │   └── visualization_engine.py         thread-safe Meshcat + RViz markers
    └── shared_autonomy/                 intent prediction
        ├── belief_estimator.py             Bayesian intent inference
        ├── goal_set.py                     dynamic goal-pose computation
        ├── grasp_state_machine.py          pick FSM (approach→contact→close→attach)
        └── plot_manager.py                 shared-autonomy telemetry plotting
```

**Import convention**: always `import triago_control.qp_controller.config as cfg` (fully-qualified). Never bare `import config` — it collides with system modules.

**Offline-recording trigger contract** (`trajectory_generator.py` → `offline_plotter.py`): a single `std_msgs/Bool` on `cfg.OFFLINE_RECORD_TRIGGER_TOPIC` — rising edge at WAITING→TRACKING (t=0), falling edge at TRACKING→REGULATION; the plotter records between them plus `cfg.OFFLINE_PLOT_POST_TRIGGER_S` of settling. Any source can drive the topic. Because the edge is a one-shot VOLATILE message, the generator holds in WAITING after its settle window until a recorder has actually subscribed (`pub.get_subscription_count() > 0`), bounded by `cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S` (`0` = disable) — makes the handshake robust to cross-host DDS discovery latency, not just same-host.

**Multi-waypoint paths** (2026-07-12, `trajectory_generator.py`): presets can set `mode: path` + `waypoints_right`/`waypoints_left` (list of `[x,y,z]`, last = final target) instead of a plain start→end line. `_build_path_spline` threads a `scipy.interpolate.CubicSpline` through `[sampled_start] + waypoints`, arc-length parametrized onto the SAME quintic tau(t)/sigma timing `s`/`s_dot` every other preset uses (`self.path_spline_{r,l}`, `None` for ordinary 2-point presets — zero behavioural change there). First use: `s_curve_right`/`s_curve_left` in `trajectory_endpoints.yaml`, a single-arm 'S' sweeping above the `no_obstacle` world's table, arcing around the red then blue graspable cylinders through the gap between them. Run `home` first (samples a clean start), then switch to one of these — same convention as the LOCAL MINIMA TESTS presets.

**Waypoint clearance rule**: the CBF enforces `h = distance − CAPSULE_RADIUS(0.06) − D_SAFE_BASE(0.015)`, not distance-to-obstacle-surface — budget any hand-authored `mode: path` waypoints against `h`, not raw distance. `s_curve_right_infeasible_regression` is a deliberately infeasible preset kept as an acceptance/stress test (do not "fix" its numbers). If a config edit doesn't move the results at all, diff the on-disk YAML first — editor buffers can silently fail to save (`s_curve_diagnostic.py` prints the commanded reference for exactly this check).

**qdot-oscillation resolution — measured-anchor rate damping**: `ENABLE_RATE_DAMPING=True` + `RATE_DAMPING_VS_MEASURED=True` adds `RATE_WEIGHT·‖q̇ − q̇_measured‖²` (ARM joints only) to the QP cost — tick-to-tick memory anchored to the arm's real state. Why it works: the ripple lived in the right wrist (joints 5/7), near-null-space under 25:1 position:orientation task weights + high slack, resolved by a cost that previously had no memory. Real hardware needed far more of it than sim: current sim defaults are `RATE_WEIGHT=50`/`RATE_WEIGHT_GRASP=20` (confirmed on `feature/sim-user-study` to resolve a sim-only overshoot/oscillation-around-goal regression), a straight 10× cut from the `real-hw` branch's tuned `500`/`200` — the anchor mode (`RATE_DAMPING_VS_MEASURED`) itself is unchanged, only its gain. Expect other real-hw-tuned gains to need the same kind of re-derivation.

**Rejected along the way (mechanisms understood — don't retry)**: commanded-anchor rate damping (`RATE_DAMPING_VS_MEASURED=False`) — self-referential low-pass on the QP's own output, not the plant; command smoother, real motion ~2× worse — a structural problem, not a real-hw-only noise artifact, so don't retry in sim either. Hard slew-rate box on q̇ (removed) — can't relieve the CBF row and breaks feasibility when the row's required direction swings faster than the box. `DYNAMIC_SLACK_WEIGHT=False` / `DAMP` 12→15 — no ripple change, worse tracking either way. Posture field exonerated (worst-ripple joints sat mid-range, not near a limit).

**Two implementation rules (violating either makes quadprog throw "constraints are inconsistent" on a perfectly satisfiable problem)**: (1) every cost gradient MUST stay masked to the active arm joints — locked joints are pinned to `dq=0` by two opposing box rows, and pulling on them (e.g. unmasked measured velocity, which carries head motion + differentiation noise) forces two linearly dependent rows into quadprog's active set, failing its dual method; the same unmasked bug also inflated `E_rate`'s task-authority share before the fix (`self._rate_mask`). (2) Never zero `last_dq_safe` in the infeasible-solve branch — anything anchored to it next tick must see the last successful solve, or one bad tick self-sustains. `_diagnose_infeasibility` (`qp_formulator.py`) prints a throttled post-mortem on every failed solve: live cfg flags (catches stale-process confusion), non-finite entries, inverted per-joint velocity boxes, per-CBF-row reachability within the box.

## 3. Mathematical Core: Arm QP-CLF-CBF (`main_qp_controller.py`)

Decision vector: `x = [q̇ (nv), δ_right, δ_left]` (joint velocities + one CLF slack per arm).

**Cost (minimize)**:
- Joint velocity regularization (damping `DAMP=12.0`).
- Rate damping toward the MEASURED joint velocity (`RATE_WEIGHT`, arm joints only — see the oscillation digest in §2, real-hardware value in §8.4).
- Posture/joint-limit avoidance via a repulsive potential field: `v_ref = -K_GRADIENT · dH/dp`, `H(p) = 1/(1-p)² + 1/(1+p)²` on normalized joint position `p = (q - mid)/half_range` (clamped to ±`V_MAX_POSTURE`). ~0 mid-range, explodes near a limit. Weighted by `W_CENTER`, scaled by a global `posture_scale` that ramps down to `POSTURE_GRASP_SCALE` during autonomous precision-grasp phases and back to 1.0 otherwise.
- Slack penalty, adaptively weighted per arm (§8).

**Constraints (`C'x ≥ b`)**:
- **CLF (task tracking)**: diagonal task weights `TASK_WEIGHTS_6D = [pos=10,10,10, ori=0.4,0.4,0.4]` (position-first, orientation yields near obstacles). An arm doing precision work with the object instead uses `TASK_WEIGHTS_6D_GRASP = [10,10,10, 2,2,2]` (5:1) for tight approach/placement orientation. Static per-phase swap (not continuous), applied per arm via `orient_boost_arms` = active arm during autonomous grasp/release ∪ any arm carrying an attached object — so the whole HOLDING/placement phase, and the release pose, stay aligned too. Decoupled from `tracking_boost_arm`, which drives only the slack/gamma boost during `grasp_active`.
- **CBF (collision avoidance) — two independent per-arm rows**: `J_soft_R · q̇ ≥ b_R` and `J_soft_L · q̇ ≥ b_L`, each a SoftMin aggregate over the `K_MAX_PAIRS=60` closest collision pairs touching that arm's geometry (a pair touching both arms contributes to both rows). Each row has its own dynamic margin `d_safe_dynamic_X = D_SAFE_BASE + K_V_SAFE·‖v_X‖` from only that arm's own joint velocities — this per-arm independence (Jacobian + margin) is what stops the inactive arm being recruited for a barrier that isn't its own.
- **Joint limits**: velocity-aware position buffer (CBF-style); every index outside `idx_right ∪ idx_left` (torso, base, gripper fingers, head) is hard-locked to `q̇=0` — this is also what makes the head chain safe to add as a quasi-static CBF obstacle (below) without adding a head joint to the decision vector.

Solver: `quadprog.solve_qp`. Shadow prices `λ_cbf_right/left`, `λ_joints_right/left` are exposed as telemetry and consumed by the adaptive scheduler (§8).

**Bimanual inactive-arm handling**: the inactive arm is frozen at its current EE pose via a zero-velocity CLF (never zero-overwritten, which would discard its own collision-avoidance motion). While frozen its slack weight is pinned to `MAX_WEIGHT_SLACK`, CLF gain to `GAMMA_MAX`, joint damping doubled (`2·DAMP`) — holds rigidly, independent of the active arm, but can still yield if that helps it.

**Head-as-obstacle**: the head chain (`arm_head_1..7_link`) is added to the arm QP's collision model as a quasi-static CBF obstacle via live FK capsules — sound since the head moves slowly relative to the CBF margin budget. No head joint enters the decision vector; the existing zero-lock on non-arm joints makes the barrier satisfiable only through arm motion.

## 4. Reference Governor (`reference_governor.py`)

Intermediate filter between the raw Cartesian reference (teleop / trajectory_generator / planner) and the CLF's perceived reference — bounds what the CLF must track, preserving QP feasibility under aggressive/discontinuous commands. One instance per arm. Master switch `cfg.ENABLE_REFERENCE_GOVERNOR`.

| Feature | Mechanism | Config |
|---|---|---|
| Velocity shaping | Clamp reference velocity magnitude, direction preserved | `GOV_V_MAX_LIN`, `GOV_V_MAX_ANG` |
| Position error bounding | Project `x_ref` onto a ball of radius `E_MAX` centered at `x_real` | `GOV_E_MAX_POS` |
| Acceleration limiting | Rate-limit velocity change per tick: `‖Δv‖ ≤ A_MAX·dt` | `GOV_A_MAX_LIN`, `GOV_A_MAX_ANG` |
| Orientation clamping | If `‖log3(R_des·R_real^T)‖ > θ_MAX`, shrink via `exp3` on the same axis | `GOV_E_MAX_ORI` |

## 5. Shared Autonomy & Reference-Level Blending (`main_shared_autonomy.py`)

Bayesian belief estimation over a discrete goal set, a local QP policy per goal, a grasp state machine, and (when `cfg.ASSIST_BLENDING`) reference-level blending of the human twist with the autonomous policy. Human side is "Joystick Mode" (velocity control, spring-centered handle → twist) or "Clutch Mode" (position control, integrates handle twist into a pose reference); see `haption_teleoperation`'s context.md §3.2.

### 5.0 Experiment Condition Selector (2×2×2 factorial, 8 cells) — `config.py` §1b

Full 2×2×2 factorial (2 control modes × 4 assistance-channel combinations), selected by three orthogonal flags in `config.py` (§1b):

- `CONTROL_MODE ∈ {CLUTCH, JOYSTICK}` — position control vs velocity control.
- `ASSIST_FEEDBACK` (bool) — channel F: assistive haptic forces (`F_guide` velocity field + `F_fixture` funnel) on top of the always-present `F_sync` tether.
- `ASSIST_BLENDING` (bool) — channel B: reference-level arbitration of user twist with belief-weighted policy in `main_shared_autonomy` (sole writer of `/arm_*/cartesian_reference` while active). `cfg.BLENDING` is a backward-compat alias for `ASSIST_BLENDING`.

| CONTROL_MODE | ASSIST_FEEDBACK | ASSIST_BLENDING | Condition | Teleop + force manager |
|---|---|---|---|---|
| CLUTCH | False | False | Sync only (baseline) | `teleop_triago_clutch` + `haptic_force_manager_C` |
| CLUTCH | True | False | Guided feedback (VF) | `teleop_triago_clutch` + `haptic_force_manager_CF` |
| CLUTCH | False | True | Guided blending | `teleop_triago_clutch` + `haptic_force_manager_CB` |
| CLUTCH | True | True | Full guidance | `teleop_triago_clutch` + `haptic_force_manager_CFB` |
| JOYSTICK | False | False | Sync only | `teleop_triago_joystick` + `haptic_force_manager_J` |
| JOYSTICK | True | False | Guided feedback | `teleop_triago_joystick` + `haptic_force_manager_JF` |
| JOYSTICK | False | True | Guided blending | `teleop_triago_joystick` + `haptic_force_manager_JB` |
| JOYSTICK | True | True | Full guidance | `teleop_triago_joystick` + `haptic_force_manager_JFB` |

**Naming**: `haptic_force_manager_<CELL>`, `<CELL>` = `C`/`J` (mode, always first) + `F` if feedback + `B` if blending → `C/CF/CB/CFB`, `J/JF/JB/JFB`. `CB` (clutch+blending-only) and `JF` (joystick+feedback-only) are the off-diagonal cells completing the factorial (each pairs a mode with its non-native assist channel — conceptually unusual, included for a complete study comparison).

Every teleop/force-manager node calls `cfg.validate_condition(node_name, control_mode=…, feedback=…, blending=…)` at startup and hard-errors on mismatch, so a mis-launched condition fails loudly.

**Unified guidance gating.** `F_guide` in every guidance cell (`JF/JFB/CF/CFB`, + dead copy in `CB`) uses one gate `gain = conf_gate × prox_gate`: proximity full ≤0.10 m / dead >0.60 m (ref→goal distance); confidence dead <0.30 / full ≥0.90 on `b_max` = max-posterior active-goal belief (`BeliefEstimator.get_active_goal()`, published in `/shared_autonomy/active_goal_pose[6]`, 0 during grasp exec) — same signal/thresholds for both control modes. (`F_fixture`'s position virtual fixture in CF/CFB keeps its own `FIX_CONF_*` gate.)

Other fairness unifications live haption-side (context §3.0): clutch sync spring, authority cap, global damping, orientation-alignment torque are identical across all four clutch cells; every B=1 cell renders the same blend-telemetry window. **Accepted residual asymmetry (D3):** during autonomous grasp execution the CLUTCH force managers drag the handle to follow the EE; JOYSTICK managers don't subscribe to `grasp_active` and render nothing extra — documented for the paper, left as-is.

> **Ownership**: CLUTCH column + this selector/`validate_condition` — one agent; JOYSTICK column — another. Both edit `config.py`/`main_shared_autonomy.py`, so pull → edit → push tightly.

### 5.1 Belief & Policy

- **Local policy per goal**: a constrained QP twist `pi_k` toward each goal `k`, computed from the true EE pose `current_T_EE`, under the same CBF constraints as the safety QP (guidance never points through an obstacle).
- **Belief estimator**: Bayesian update over goals with exclusion (a goal held by either arm, or not graspable, is pinned to probability 0). Observed action = user's commanded twist (`current_v_h`) vs. user-frame policies — reflects operator intent and breaks the autonomy→EE→belief self-confirmation loop (was making goals sticky to switch). Twist cost matrix `W = diag([10,10,10, 2,2,2])`; `pos_costs` (10% tie-breaker) anchored at `current_T_user` (== `current_T_EE` in joystick mode).
- **Policy blend**: `pi_policy = Σ_k belief(k) · pi_k` — the single optimal twist passed into arbitration.

### 5.2 Reference-Level Arbitration (`cfg.ASSIST_BLENDING = True`)

Reference sent to the QP = blend of user twist and belief-weighted policy, integrated persistently every tick into a latched reference (`T_blend_ref` via `integrate_twist` — idle user yields an absolute hold, not a reference chasing the drifting EE). `main_shared_autonomy.py` is sole publisher of `/arm_*/cartesian_reference` while blending is active.

```
v_blend = (1 - alpha) * v_user + alpha * pi_policy
```

**`compute_alpha`** — authority weight on the policy:

```
user STILL (v_user == 0, inside deadband):
    alpha = ALIGN_ALPHA_IDLE                                    # gentle autonomous crawl
user ACTIVE:  (two-sided ramp, continuous at s=0 where alpha = ALIGN_ALPHA_MIN)
    s     = mean per-channel cosine(v_user, pi_policy) over commanded channel(s)
    s ≥ 0:  alpha = ALIGN_ALPHA_MIN + (ALIGN_ALPHA_MAX - ALIGN_ALPHA_MIN) · s
    s < 0:  alpha = ALIGN_ALPHA_MIN · (1 + s)                   # ramps MIN -> 0 as s: 0 -> -1
```

Still → `alpha=ALIGN_ALPHA_IDLE=0.35` (gentle crawl). Perpendicular (`s=0`) → `alpha=ALIGN_ALPHA_MIN=0.2` (user keeps 80%). Aligned+active (`s→1`) → `alpha=ALIGN_ALPHA_MAX=0.8` (autonomy leads fast). Opposed (`s<0`) → authority ramps to 0 (`s=-1` → fully yields — no permanent counter-pull).

Belief selects *which* goal's policy is `pi_policy`; `alpha` only arbitrates authority. User twist arrives already deadbanded to zero (teleop-side), so handle noise can't creep the arm.

**CLUTCH mode** reuses the arbitration verbatim, fed by the clutch teleop's twist. Belief scores the twist against user-policies anchored at the **blended reference gripper** (`self.T_blend_ref`), not the clutch's integrated pose (operator watches the blended gripper, not their own integrated pose) — the one clutch-specific deviation from joystick (already anchored at live EE). 10% position tie-breaker stays anchored at real EE.

**"Still → suspend blending"** (both modes): below still thresholds (`STILL_LIN` 5 mm/s, `STILL_ANG` 0.05 rad/s), `alpha` forced to 0 (pure user twist, reference held) so `ALIGN_ALPHA_IDLE` never drives the arm alone. Fixes "steer to new grasp side, stop, reference snaps back to lagging-EE goal." **Clutch button held** (`virtuose/button_right`, indexing): blending SUSPENDED, `alpha=0` + alpha-LPF reset, `T_blend_ref` holds until release then resumes from the latch. `main_shared_autonomy` subscribes to this topic (unused/harmless in JOYSTICK mode).

**Blending active** in every user-controlled state (`SHARED_AUTONOMY`, `PRE_GRASP`, `HOLDING`) — suspended only during autonomous grasp-execution states (`GRASP_ALIGN/APPROACH/CLOSE`, `LIFT`, `RELEASE_LIFT`, `ABORT_RETREAT`), where the state machine drives the arm. `T_blend_ref` re-anchors to live EE on each transition back into a user-controlled state (no jump).

**Telemetry**: `/shared_autonomy/blend_debug` (19 floats: `[alpha, v_user(6), v_policy(6), v_blend(6)]`), published every tick regardless of `cfg.BLENDING`.

**Operator marker** (`/blended_reference_marker`): the gripper the operator watches while teleoperating (`T_virtual_ref`, integrated from the blended twist). GREEN = tracking the policy (`alpha ≥ 0.5`); ORANGE = listening to user override. Primary teleop cue; other predictive markers can stay disabled in RViz.

### 5.3 Bimanual State

Each arm owns an independent `GraspStateMachine` + `BeliefEstimator` + grasped-color + goal-set placement context; inactive arm's FSM/belief simply not stepped. Goal exclusion is the union over both arms. A cylinder-vs-cylinder CBF pair prevents two held objects inter-penetrating.

### 5.4 Goal Set (`goal_set.py`)

Goals are dynamic SE(3) manifolds, not fixed poses:
- **`Top`/`Side`**: approach axis from anchor→cylinder-axis geometry; sticky hysteresis avoids indecision near degenerate configs.
- **`Front`**: approach axis rigidly locked to world +X (reach/hover tutorial goals, no physical object).
- **`Platform_Place`**: only hard constraint is held-object axis ⊥ platform face (2 DOF constrained, yaw free) — a true placement manifold. Object's symmetry axis frozen in gripper frame at grasp time (`grasped_axis_local = R_grasp^T · [0,0,1]`).

### 5.5 Grasp State Machine

`SHARED_AUTONOMY → PRE_GRASP → GRASP_ALIGN → GRASP_APPROACH → GRASP_CLOSE → LIFT → HOLDING` (failure: `→ ABORT_RETREAT`). On `GRASP_CLOSE → LIFT` the held object is re-parented as a real link in the QP collision world (its CBF bypass with the gripper cleared after a smooth ramp), then actively avoids the environment. Release reverses this (`DETACH_*`, `RELEASE_LIFT`) and relocates the object's believed pose to where it was set down.

**Grasp confirmation is purely geometric** (no force/torque or vision sensing): signed gripper-box↔object `hppfcl` distance against the object's believed (kinematic) position/axis. Required overlap per grasp type (`_contact_depth_threshold`): `Top = -0.045 m` (shallow, vertical approach), `Side = -0.05 m` (deeper seating). Insertion depth per type (`_insertion_travel`, straight-line advance from the standoff — `approach_offset=0.05` in `goal_set.py` — along the approach axis): `Top = 0.09 m`, `Side = 0.0918 m`. The `real-hw` branch tunes both deeper — diff against it before assuming either branch's numbers apply to the other.

## 6. World Scene Loading (`world_loader.py`, `config/worlds/*.yaml`)

Static obstacle layouts (table, graspable cylinders, walls, shields, etc.) described declaratively per scenario in `config/worlds/<name>.yaml`, not hard-coded. Schema: list of `ObstacleSpec` (name, role, shape, pose, size, color, collision on/off) + `grasp_roles: {red: <name>, blue: <name>}` + optional `platform` (visual-only placement disk, never in the collision model).

`CollisionManager`, `VisualizationEngine`, `GoalSet` all consume `world_scene` generically — a new world with new obstacles needs only a new YAML file, no code changes. `main_qp_controller.py`/`main_shared_autonomy.py` take a `world_name` ROS param (default `no_obstacle`) and must be launched with the same value. YAML filename matches its companion Gazebo `.world` filename.

An obstacle with `role: "reachable"` and no `<collision>` produces a pure reach/hover goal (`Front` type, §5.4) instead of a graspable one.

### 6.1 Camera-Perceived World (`main_qp_controller_perceived.py`, `perceived_world_builder.py`, `head_control/world_convergence.py`)

Alternative source for the same collision model (§6), built from the **head camera** instead of prior YAML — for when the world is unknown. The head RANSAC tabletop pipeline (`head_control/`, run by `scripts/head_controller/main_head.py`) estimates table + 2 cylinders to ~1 cm in sim; this pipeline freezes that estimate into a `WorldScene` once confident and the arm CBF is built statically on it — no dynamic per-tick obstacle updates (deliberate; a moving obstacle set makes the barrier non-stationary). Downstream (`CollisionManager`, `VisualizationEngine`, Meshcat, RViz) is byte-for-byte the YAML path — only the source of `world_scene` changes.

**Seam**: `SafetyQPController.__init__` calls an overridable `self._build_world_scene()` (default = `load_world(world_name)`). `main_qp_controller_perceived.py` is a thin subclass overriding only that method — the whole CBF/CLF/control-loop/viz stack is inherited verbatim.

**"Confident" concept** (`WorldConvergenceMonitor`, reworked 2026-07-13). Convergence requires, on settled frames only (head-still `INTEGRATE_VEL_THRESH` gate): (a) every expected cylinder (`WORLD_EXPECTED_CYLINDERS=2`) above `WORLD_CONF_MIN` **and** above an arc-coverage floor `WORLD_MIN_ARC_COVERAGE` (an under-observed object can never start the clock), and (b) an **oscillation-amplitude lock**: over a sliding `WORLD_CONVERGE_WINDOW_S` (1 s) of settled frames, every centre within `WORLD_CONVERGE_POS_AMP` (1 mm, peak dev from window mean) and radius/height within `WORLD_CONVERGE_DIM_AMP`. Replaces the old per-frame-drift × `WORLD_STABLE_FRAMES` counter (blind spot: a static viewpoint passes "small per-frame drift" while consistently wrong). Head motion is **perception-led** (`look_at_controller.active_vision_target`, `ENABLE_ACTIVE_VISION`): SWEEP waypoints on coverage-plateau (not timers) → REFINE the weakest object toward its unseen-sector azimuth → HOLD; bounded by `ACTIVE_VISION_TIME_BUDGET_S`. Validated in sim: pos err ≤0.2 cm, radius err −0.13 cm both cylinders, 100% coverage at freeze. On convergence the console prints one final QP-payload block, then goes quiet (rescan re-arms it). Table box is fully camera-derived (percentile bbox of plane inliers, EMA-smoothed). Config in `head_control/config.py` §5c+§16. `offline_plotter.py` now also records head data (fig7 perception / fig8 head kinematics / fig9 active-tracking, emitted only if the head nodes ran; topics in `OFFLINE_BAG_TOPICS`).

**Hand-off**: on convergence `main_head` publishes one latched (`TRANSIENT_LOCAL`) `MarkerArray` on `/perceived_world/snapshot` (`cfg.PERCEIVED_WORLD_TOPIC`): `ns="table"` CUBE + one `ns="objects"` CYLINDER per object (radius inflated by `CYL_RADIUS_INFLATION`, colour = class). A `std_msgs/Empty` on `/perceived_world/rescan` re-arms the monitor. QP side blocks in `perceived_world_builder.wait_for_scene()` (safe inside `__init__`, before `build_collision_model`) until the snapshot arrives, then decodes → `WorldScene` identical to `load_world`'s output.

**Run** (standard Gazebo world; `world_name` ignored by the perceived node):
```bash
ros2 run triago_control main_head.py                       # head perception → converges → latches the snapshot
ros2 run triago_control main_qp_controller_perceived.py    # blocks for the snapshot, then runs the CBF on it
```

**Real hardware**: `main_head.py` itself is unchanged (no `REAL_HARDWARE` branch — the sim/real difference is pure launch-time config, no behavior change). Launch `launch/head_real.launch.py` instead of running the node directly: it starts the RealSense driver (`realsense_camera_cfg`, `camera_name:=head_arm_rgbd`), the `arm_head_tool_link → head_arm_rgbd_link` mount static transform (measured, not the idealized CAD one), and overrides `main_head.py`'s `color_topic`/`depth_topic`/`camera_info_topic`/`depth_center_frame` params for the real camera. Topic name suffixes follow the standard `realsense2_camera` convention but weren't independently confirmed against this specific wrapper — verify with `ros2 topic list | grep head_arm_rgbd` once launched.

## 7. Head Controller — Visual Servoing (`qp_head_visual_servo.py`)

Fully independent from the arm QP (own solver instance, no shared state, not in arm CBF pairs). Keeps both hands in the camera FOV via 2.5D visual servoing.

Decision vector: `x = [dq_head (7), slack (3)]`.

| Stage | Condition | Strategy |
|---|---|---|
| PBVS (Look-At) | hands outside FOV / behind camera | 3D rotational servoing: `cross(z_cam, dir_to_centroid) → ω` via `J_rot` |
| IBVS (Pixel tracking) | both hands inside FOV margin | 2.5D: interaction matrix `L_s` maps pixel+depth error to camera twist |

**Cost**: joint velocity regularization (heavier on base joints), slack penalty (`W_SLACK_PIXELS` for u,v; `W_SLACK_DEPTH` for depth, normalizing pixel vs metric scale), postural centering spring (`K_POSTURE`).

**Equality (CLF-like)**: `J_task · dq − slack = −λ_visual · e`, `J_task = L_s · J_cam` (IBVS) or `J_rot` (PBVS), `e` = pixel/rotation error.

**Inequalities**: FOV barriers (IBVS only — each hand ≥`FOV_MARGIN` px from image edges, 4/hand) + velocity-aware joint-limit buffers (same pattern as arm QP).

Hand positions come from Pinocchio FK (kinematic tracking), not image detection — acknowledged starting point for future perception work. A separate lightweight collision model (capsules+boxes, head-vs-body/wall) exists but isn't yet wired into this QP as CBF inequalities.

### 7.1 Active-Arm Tracking (`head_active_arm_tracking.py`)

Teleop sibling of the visual servo (same independent-QP / IBVS-PBVS structure, `x = [dq_head (7), slack (3)]`). Assumes exactly one arm active at a time: subscribes to `/shared_autonomy/active_arm` (`std_msgs/String`) and points the camera at that arm's `arm_{side}_tool_link` via FK.

Two objectives via one 2.5D IBVS equality task on the active hand: **centering** (pixel error → image centre, `J_task = L_s · J_cam`) and **stand-off** (depth error `Z − TARGET_DISTANCE`, default 0.8 m, `target_distance` ROS param). Falls back to PBVS look-at when the hand is off-FOV. FOV barriers (4 rows) + same joint-limit CBF buffers.

**Soft roll alignment**: keeps image-right (camera X) aligned with world-right (`WORLD_RIGHT = world -Y`) via signed roll error `theta = atan2(d_y, d_x)`, `d = R_cam^T · WORLD_RIGHT`, roll rate `wz = K_ROLL_ALIGN·theta`. Encoded as a low-weight LS term in the QP cost (`H[:7,:7] += W_ROLL_ALIGN·J_roll J_roll^T`, `J_roll = J_cam[5,:]`), not an equality row, so it only uses leftover redundancy and can't override tracking. Gated off while off-axis or near-gimbal.

**Soft approach-axis alignment** (`ENABLE_APPROACH_ALIGN`/`-p approach_align`): aligns optical axis (+Z) with the gripper's approach axis (top grasp → top-down view) — a position/orbit bias, not orientation servo, since centering owns axis direction. `p_cam_des = p_hand - TARGET_DISTANCE·x_gripper`; low-weight LS term on linear velocity toward it (`Jlin = J_cam[:3,:]`). Centering-gated, modest-weighted, subordinate to acquisition.

Matplotlib plotting thread (guarded so a missing display never stalls control, `-p plot:=false` to disable) shows centering/roll/approach error + stand-off distance. Telemetry on `/head_active_tracking/{error,qdot,cartesian_cmd}`. Run: `ros2 run triago_control head_active_arm_tracking.py`.

### 7.2 AprilTag Pose Reconstruction (`head_april_main.py`)

Alternative head-perception path (reuses `HeadKinematics` + `LookAtController`). Localizes one fiducial of known geometry, reconstructs every scene object from a known rigid transform relative to it:

`base_M_obj_k = base_M_cam · c_M_tag · tag_M_obj_k`

`base_M_cam` from TF (`base_footprint`←color optical frame at detection stamp), `c_M_tag` from external `visp_apriltag` ViSP node (`/tracker_apriltag/tags_info`), `tag_M_obj_k = world_M_tag⁻¹ · world_M_obj_k` derived once at startup from `cfg.APRILTAG_SCENE`. Only the tag pose is image-estimated; object geometry is a model prior — legitimate fiducial-based perception (the node never reads any object's absolute world pose at runtime). §14 `GT_*` constants stay diagnostic-only.

**Correct camera TF setup is what makes this accurate** (reaches ~2 mm in sim, the ViSP noise floor). Four requirements, each wrong yields a large systematic offset:

1. **Optical CENTRE frame** (`cfg.APRILTAG_OPTICAL_CENTER_FRAME`): take orientation from the optical frame but position from the true optical centre. In sim, Gazebo's RealSense plugin renders from `camera_link` origin yet tags the image as `..._color_optical_frame` (~2.5 cm offset in the URDF) — so position must come from `gripper_head_camera_rgbd_link`. On real hardware the driver's color-optical-frame origin IS the sensor, so set this equal to `APRILTAG_CAMERA_OPTICAL_FRAME`.
2. **Tag-frame orientation in world** (`cfg.APRILTAG_RPY_WORLD`) must match the rendered/printed tag. ViSP default frame (Z out of tag toward camera, X=image-right, Y=image-up); this sim's OGRE UV mapping gives tag yaw −90° ⇒ `[0,0,−π/2]`. On real robot, read yaw off the live `tracker_apriltag` TF.
3. **Subscriber QoS** must match the publisher (BEST_EFFORT + TRANSIENT_LOCAL) — a default RELIABLE subscriber silently receives nothing.
4. **ViSP pose method** `best_residual_virtual_vs` — lower bias than `homography_virtual_vs` at this tag size/range.

Diagnose with the read-only `[TAG-DIAG]` block (`-p tag_diag`): logs estimated-vs-known tag pose, raw ViSP `c_M_tag`, TF extrinsic (comparison-only, never fed into reconstruction).

Same frame principle applies to `main_head.py` (`cfg.DEPTH_OPTICAL_CENTER_FRAME`/`-p depth_center_frame`) — with that + `ENABLE_SCAN=False` the RGB-D pipeline also reaches <1 cm in sim. Scan is disabled because transforming 30k+ points per frame via TF makes even a few-ms TF lag (during head motion) shift centroids ~5–10 cm; `head_april_main` can safely scan since it transforms a single point (<2 mm error from the same lag). Gains: `MAX_HEAD_VELOCITY=0.10`, `LOOKAT_LAMBDA=1.0`.

Publishes the same topics `head_plotter.py` consumes, so the existing dashboard shows reconstruction quality vs GT with no plotter change. Reconstructs `red_cylinder`, `blue_cylinder`, `work_table`. Run alongside `visp_apriltag_node` (`tag_family:=36h11`, `tag_size:=0.12`). Companion world `config/worlds/apriltag_world.world` = `no_obstacle` + a printable AprilTag plate on the table (visual-only; QP controllers still launch with `world_name:=no_obstacle`).

## 8. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting** (`DYNAMIC_SLACK_WEIGHT`, on by default): each arm's CLF slack weight drops toward `BASE_WEIGHT_SLACK` as its shadow price grows (slack absorbs tracking error near obstacles), rises toward `MAX_WEIGHT_SLACK` in free space (tighter tracking).
- **Dynamic CLF gain** (`DYNAMIC_GAMMA_CLF`, off by default → γ held at `GAMMA_CLF_DEFAULT`): when enabled, γ drops exponentially with collision Lagrangian λ_cbf, LPF'd (τ=0.125s) — tracking priority in free space, yields near obstacles.

### 8.1 Compute-Budget Optimizations + Real-Hardware Load Shedding

`/qp_debug/loop_timing`'s per-phase breakdown (§10) showed `compute_softmin_jacobian` (CBF aggregation) as the largest real-hardware phase, and the RViz witness-marker code was independently re-scanning the same distance results.

- **`compute_softmin_jacobian`** now does its filtering + the witness-line global-closest-pair search in one pass, caches the static `allowed_grasp_ids` set once, and batches per-pair SoftMax weighting/accumulation into 1-2 vectorized numpy calls instead of a Python loop. All SHARED-AUTONOMY hooks and nearest-point/Jacobian extraction stay per-pair, unchanged — only final weighting/summation is batched; verified numerically identical to the original.
- **Real-hardware load shedding** (`self.REAL_HARDWARE`): Meshcat is never initialized on real hardware (pure skip — every Meshcat-touching method already no-ops on `viz_meshcat is None`). `publish_every_n`, the RViz obstacle-marker timer, and the joint-limits telemetry timer all halve their rate on real hardware; sim keeps full cadence. No topic/marker/output is removed, just published less often on real hardware.

### 8.2 Async Execution — `main_qp_controller_real.py` (real-hardware only)

After §8.1, the CBF was still the largest phase sitting inline on the control tick. `main_qp_controller_real.py` is a thin subclass of `SafetyQPController` (same pattern as `main_qp_controller_perceived.py`, §6.1) moving it — and RViz debug overlays — onto worker threads. **Must launch this file, not `main_qp_controller.py`, to get the async path**; a drop-in A/B (falls through to `super()`, byte-identical, when `REAL_HARDWARE` is false or either flag is off).

- **Overridable seams (base class)**: `solve_and_publish` refactored into four pure-extraction seams — `_compute_cbf()` (returns `CbfResult`), `_gate_command(q_dot_safe)` (identity), `_publish_visual_overlays(cbf, q_dot_safe)`, `_process_deferred_topology()` — byte-identical to the old inline code, sim path unchanged. `CbfResult` carries the QP's barrier (`J_soft_*`, `h_soft_*`, `d_safe_*`, `abs_min_distance`) plus witness/top-pairs telemetry and a `fresh` flag.
- **CBF worker**: runs FK+geometry+SoftMin on its own private `pin.Data`/`GeometryData` (never races the main tick's shared `self.data`). Main tick posts a snapshot (`current_q/v` + shallow-copied HRI grasp dicts) and reads the latest result. `compute_softmin_jacobian`/`update_geometry` gained optional `data=`/`cdata=` params (default `self.*`), backward-compatible.
- **Staleness watchdog** (`CBF_STALENESS_MAX_TICKS=3`): `_gate_command` freezes both arms (zero velocity, auto-resume) whenever the worker result hasn't advanced for ≥3 ticks, and during startup before the first result — so the QP can never drive off an unboundedly-old barrier. Edge-triggered `[SAFETY]` log on freeze/resume. The digital twin (`integrate_simulated_state`) uses the gated command too, so a freeze halts it.
- **`cmodel` mutation race**: grasp attach/detach mutates the shared collision model. `CollisionManager.geom_lock` serializes that (held by `_process_deferred_topology`) against the worker's `cmodel` reads/`computeDistances` (worker holds it around its whole compute, rebuilds its private `cdata` on pair-count change). Uncontended in sync/sim.
- **Viz worker**: publishes the collision-witness line + teleop tethers from a small witness snapshot.
- **GIL caveat**: Python threads don't run bytecode in parallel; the speedup only comes from `pinocchio`/`hppfcl`/big-numpy calls releasing the GIL while overlapping the main tick's own C-extension work — net gain is uncertain, validate on hardware via `loop_timing_monitor.py` (`REAL_ASYNC_CBF` makes the A/B a one-line flip). If gain is poor, the remaining lever is reducing CBF cost directly (fewer pairs) — needs explicit sign-off (safety-margin trade-off).

### 8.3 Control Frequency: 150 Hz Default

`CONTROL_FREQ_DEFAULT = 150` (not 300): the real robot can't reliably sustain 300 Hz, and a sim A/B (free-space + fast bimanual-convergence trajectories, via `freq_oscillation_diagnostic.py` + `offline_plotter.py`'s `summary_metrics.json`) found no downside — 150 Hz holds its target rate with far lower jitter than 300 Hz (which measurably misses its own target under CBF load), same safety margin, same tracking accuracy. Ongoing: async CBF (§8.2) alone reaches ~140Hz/~11% tick overruns on real hardware; OS-level RT-priority/CPU-pinning closes the rest of the gap but risks colliding with PAL's own real-time layer — see §10.7.

A ~15-19% `qdot_cmd` ripple (right arm, fast bimanual-convergence trajectories only, frequency-independent) is **resolved** by `DAMP=12.0` (was `10.0`, +20%, isolated single-variable step — a bundled/doubled-damping test had failed) — cuts ripple to ~5-6%, with tracking, slack usage, and governor activity all improving too. Trade-off: `min_observed_distance` tightened ~21% in testing (still safely positive), plausibly from blunting how sharply the CBF can command a sudden brake. Kept as the new default; real-hardware validation still open.

`compute_softmin_jacobian`'s `cfg.ENABLE_INTER_ARM_CLOSING_MARGIN` (default **False**): when true, an inter-arm pair's margin also reflects the OTHER arm's speed, not just the row-owning arm's own (`d_safe_dynamic_r/l`'s normal basis) — correct for two converging grippers, cost-neutral, but didn't fix the ripple above.

`offline_plotter.py` now bags every trial too (`cfg.OFFLINE_BAG_ENABLE`, `cfg.OFFLINE_BAG_TOPICS` — QP-controller telemetry only, no shared-autonomy/teleop topics): `ros2 bag record` starts on the trigger's rising edge and stops at finalize, written to `<trial_dir>/bag` next to the figures/`summary_metrics.json`; the console index is also saved as `trial_summary.txt`. Purpose: sim trials now, matching real-hardware trials later, are both replayable for the 150 Hz rollout push above.

Sim's Gazebo `controller_manager` (hence `/joint_states`) was accidentally running at 100 Hz, not the real robot's 50 Hz — fixed at `update_rate` in `/opt/pal/alum/share/triago_description/ros2_control/gazebo_controller_manager_cfg.yaml` (system path, outside this repo/workspace; the in-repo `pal_sea_arm`/`pal_pro_gripper` copies of that same-named file are NOT the ones actually loaded for the combined robot). Re-validated at the correct 50 Hz: tracking/safety-margin/governor-activity all held up or improved; `ALPHA_FILTER=0.5` still adequate.

`cfg.ENABLE_RATE_DAMPING` defaults **True** with `RATE_DAMPING_VS_MEASURED=True` (measured-anchor, not commanded-anchor — see §2's rejection note).

### 8.4 Real-Hardware Oscillation Retune (accepted defaults)

A persistent real-hardware `qdot_cmd`/EE oscillation (absent in sim) turned out to be frequency- and latency-independent: a controlled off-robot trial (controller on a separate PC, commanding the robot over the network) held 0% tick overruns/1.48% jitter with the ripple unchanged, and `ros2 topic delay /joint_states` measured a negligible, near-constant ~3ms (std dev 0.12ms) — both ruled out. Real hardware's raw `/joint_states` velocity is now always re-derived from position (§10.1) rather than read directly, structurally independent of scheduling/network conditions. The rest was hand-tuned on the robot: raised `RATE_WEIGHT` (anti-oscillation authority), tightened `GOV_V_MAX_ANG`/`GOV_E_MAX_ORI` (reference governor), lowered `W_CENTER` (posture weight), and added a `WEIGHT_SLACK_FILTER_TAU` 2nd-stage LPF directly on `weight_slack_r/l` in `qp_formulator._schedule_weights` (previously only the input shadow price was filtered; `BETA`'s squared-exponential still turned a smoothed but fast-rising lambda, up to ~400 collision / ~40 joint-limit, into a sharp weight jump). Exact values are frozen on the `real-hw` branch — diff its `config.py` rather than trusting numbers restated here.

## 9. Frame Convention (Haption ↔ TRIAGo)

Identical to `haption_teleoperation`'s §7: a pure 180° rotation about Z between the Haption device frame and TRIAGo's `base_footprint` (negate X, negate Y, keep Z — same for force feedback).

## 10. Critical Hardware / Environment Quirks

1. **Joint velocity is always derived, never read directly from `/joint_states`** — position differences + EMA filter (`ALPHA_FILTER=0.5`), same path for sim and real hardware (`robot_kinematics.update_from_joint_state`). Real hardware's raw sensor velocity is unfiltered and was found to inject noise straight into `qp_formulator`'s rate-damping term (`RATE_WEIGHT` tracks it hard) and out into `qdot_cmd` as oscillation — the direct-read `v_direct` path still exists (unused) for a caller that wants to bypass this. Accepted as the current solution — see §8.4.
2. **`REAL_HARDWARE` auto-detection**: inferred at startup from whether the fetched URDF contains `gripper_{right,left}_grasping_link` natively (Gazebo URDF: yes→sim; real TIAGo Pro URDF: no→real, frames injected + broadcast as static TFs).
3. **Meshcat is not thread-safe** — only the dedicated `_run_viz` thread may touch the WebSocket; callbacks mutate state under a lock and set a reload-pending flag.
4. **Controller switching** — TRIAGo requires explicit activation of `arm_{right,left}_joint_space_controller_vel` and deactivation of conflicting trajectory controllers before the QP can command the arms.
5. **No force/torque sensing anywhere on the arm chains**, and no Gazebo ground-truth is ever read — everything derived from `/joint_states` + FK. This is why grasp confirmation (§5.5) is purely geometric.
6. **Clock domain: the control loop runs on wall-clock, always — deliberately, not an oversight.** `main_qp_controller.py`/`_perceived`/`_real`/`trajectory_generator.py` never set `use_sim_time`. A sim-time `create_timer` is driven by `/clock`; when the executor is momentarily starved (e.g. `offline_plotter.py` + its bag recorder sharing the host), it catches up by firing several ticks back-to-back, snapping the arm through motion instead of ramping it (`max_lambda_cbf` spiked from ~24 to 623 in one such trial) — a wall-clock timer fires steadily regardless of executor load. `dt` in the CLF/CBF/governor math is the fixed nominal `1/CONTROL_FREQ_DEFAULT`, not measured — the wall-clock timer already fires at that real rate, and measuring it only reintroduces the same burst-jitter risk. Joint-velocity reconstruction is unaffected either way: `robot_kinematics.update_from_joint_state`'s `dt` comes from the `/joint_states` message's own header stamp, never the node's clock. `offline_plotter.py`/`plotter.py` are the exception — passive, command nothing, so they correctly auto-detect `/clock` and run on sim-time, matching Gazebo's clock for their plotted timestamps; on real hardware (no `/clock`) they fall back to wall-clock automatically, same code path. Diagnostic: `TRIAGO_DEBUG_TICKS=1 ros2 run triago_control main_qp_controller.py` logs ticks/s + bunched-tick count once per second (healthy: ~`CONTROL_FREQ_DEFAULT` ticks/s, zero bunching, even under heavy host load). `/qp_debug/loop_timing` (per-tick phase breakdown, `loop_timing_monitor.py`, §11) and `ps`/`chrt -p PID`/`taskset -pc PID` remain the tools for a real-hardware compute-bound investigation (GIL means idle cores never help a single-threaded bottleneck).
7. **RT-priority/CPU-pinning on the controller process is compute-effective but can collide with PAL's own EtherCAT master.** A trial (`chrt -f -p 50` + `taskset -pc` to 2 cores on the controller process) cut >10ms tick overruns from 10.7% to 0.9% and pulled mean tick_dt onto the 150Hz nominal — but ~10s in, it starved PAL's real-time `ros2_control_node` long enough to trip every slave's EtherCAT sync-manager watchdog at once (`dmesg`: AL status `0x001B` on ~25 slaves simultaneously), a bus-wide fault `pal_startup` auto-recovered from in ~25s. Check `ros2_control_node`'s own `chrt -p`/`taskset -pc` (PID via `ps aux | grep ros2_control_node`) before assigning either to our controller, so priority/cores can't collide with it. Both are per-thread kernel state — live-revertible (`chrt -o -p 0`/`taskset -pc 0-23`) and gone the instant the process dies — so the risk is entirely in what elevated priority does to the live EtherCAT bus, not in the settings themselves persisting.

## 11. Build & Run

```bash
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash

ros2 run triago_control main_qp_controller.py --ros-args -p world_name:=no_obstacle
ros2 run triago_control main_qp_controller_real.py --ros-args -p world_name:=no_obstacle  # ON THE ROBOT: async CBF + overlays (§8.2); sync fallback in sim
ros2 run triago_control main_shared_autonomy.py --ros-args -p world_name:=no_obstacle
ros2 run triago_control qp_head_visual_servo.py     # independent, can run alongside
ros2 run triago_control head_active_arm_tracking.py # teleop: head follows the active arm (-p plot:=false to disable dashboard)
ros2 run triago_control trajectory_generator.py     # open-loop test source
ros2 run triago_control plotter.py                  # live dashboard
ros2 run triago_control offline_plotter.py          # static, publication-quality figures
ros2 run triago_control loop_timing_monitor.py      # real-hardware control-loop jitter / compute-time diagnostic (§10)
ros2 run triago_control freq_oscillation_diagnostic.py  # buckets a qdot_cmd ripple's source, rides the same trigger as offline_plotter.py (§8.3)
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
ros2 run haption_teleoperation haptic_force_manager_C.py
#   CLUTCH, guided feedback  (F=1, B=0):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_CF.py
#   CLUTCH, full guidance    (F=1, B=1):
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_CFB.py
#   JOYSTICK, guided blending (F=0, B=1):
ros2 run haption_teleoperation teleop_triago_joystick.py
ros2 run haption_teleoperation haptic_force_manager_JB.py
```

**Gazebo Link Attacher** (external dependency, kinematic attach/detach during grasping — not part of this repo):
```bash
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

- Commit messages: one line, imperative mood, <72 chars, no body unless asked. Stage only files relevant to the task (never `git add -A`/`.`).

## 15. User Study / Analysis Subsystem (`scripts/analysis/`)

Self-contained tooling to run and record a human-subject study comparing feedback strategies on a cylinder pick-and-place task, across the declarative worlds (§6). Lives entirely under `scripts/analysis/`. Its parameters live in `scripts/analysis/study_config.py`, separate from `qp_controller/config.py` (experiment/data-management settings, not controller gains).

### 15.1 Independent Variable — Feedback Conditions

Full 2×2×2 factorial (8 cells, §5.0) — `(CONTROL_MODE, ASSIST_FEEDBACK, ASSIST_BLENDING)`, not the legacy `cfg.BLENDING` alone. Short codes used in trial-folder names / master table: `C/CF/CB/CFB` (clutch), `J/JF/JB/JFB` (joystick) — see `VALID_CELLS`/`CELL_LABELS` in `study_config.py`. Recorder snapshots the full triple from `cfg` into `metadata.json` and verifies the launched condition against it (nodes also hard-error via `cfg.validate_condition`). Manual success call remains the experimenter's; the flag triple resolves the condition unambiguously.

`study_config.py` owns the canonical label strings and implements all 8 cells via `derive_cell()`.

### 15.2 Recorder — GUI, one launch per trial (`study_recorder.py`)

A Tkinter GUI wrapping `ros2 bag record`, launched fresh per trial. No rclpy (pure `subprocess`), stateless; pure helpers (`build_record_command`, `sanitize_token`, `snapshot_cfg`) are testable headless. Workflow:
1. `PARTICIPANT_ID` prefilled from `study_config` (env/`-p participant:=` overridable), editable.
2. Experimenter types a world shortcut, selects feedback strategy (radio: `virtual_fixture`/`blending`/`no_assist`); a live label flags mismatch against `cfg.BLENDING`.
3. **START** → spawns `ros2 bag record` for `BAG_TOPICS` in its own process group; elapsed timer runs.
4. **STOP** → SIGINT to the process group (falls back to terminate/kill on timeout).
5. Experimenter marks **Success (Yes/No)** + **Notes**, **SAVE** writes `metadata.json` and resets the form (participant/world/strategy stay sticky). State machine `IDLE → RECORDING → AWAIT_SAVE → IDLE` forces classifying every trial before the next starts.

Success is always the experimenter's manual call (no ground-truth correctness check).

### 15.3 Capture — rosbag only (post-processed offline)

Recorder captures a rosbag and nothing else per trial (no live subscription/resampling/metrics). Storage (`study_config` helpers `participant_dir`/`bag_folder_name`/`bag_path`):

```
DATA_ROOT/<participant>/<world_shortcut>_<condition_short>/   # = `ros2 bag record -o` dir
    <bag>.db3 + metadata.yaml     # rosbag2
    metadata.json                 # our provenance sidecar (METADATA_NAME)
```

`condition_short` ∈ {`vf`, `bl`, `na`} (`STRATEGY_SHORTCUTS`). One folder per `(participant, world, condition)` triple — re-recording overwrites it (confirmed via dialog, no repetition index/timestamp). Bag uses the curated `BAG_TOPICS` allowlist (`/joint_states`+`/tf`, head-camera point clouds excluded), backend `BAG_STORAGE_ID` (default `sqlite3`). `metadata.json` holds participant/world/condition(+short), timestamps, duration, success, notes, `cfg.BLENDING`, `CFG_SNAPSHOT_KEYS` cfg snapshot, hostname, topic list. Tidy time-series is derived offline from the bag (§15.5).

### 15.4 Storage Split

Code lives in the git repo; heavy data lives locally, never on GitHub. `DATA_ROOT` defaults to `~/exchange/triago_study_data/` (outside the repo, `TRIAGO_STUDY_DATA_ROOT`-overridable). `.gitignore` guards (`triago_study_data/`, `*.bag`, `*.db3`, `*.mcap`, `*.parquet`) in case the root is pointed inside the repo.

### 15.5 Offline Analysis

- **`study_metrics.py`** — numpy-only engine (no pandas): `load_bag()` reads a trial bag via `rosbag2_py`+`rclpy` into per-topic numpy `Series` (needs a sourced ROS 2 env); `compute_metrics()` returns a flat metrics dict; `format_summary()` renders the human table; `sparc()` is the smoothness metric. Array layouts of multi-array telemetry are pinned as topic constants at the top of the file.
- **`analyze_trial.py`** — runnable per-trial analyzer (`ros2 run triago_control analyze_trial.py [path…]`, or plain `python3`; no arg walks `DATA_ROOT`). Since the QP decouples the two arms and only one is teleoperated at a time, analyzes/plots each arm separately: `plot_dashboard_{right,left}.png`, `metrics_summary_{right,left}.txt`, one `metrics.json` (`{metadata, right, left}`). Each 3×3 dashboard: EE path, EE speed (from published `/qp_debug/ee_real`, ground truth), measured joint velocity (`qdot_measured`) + QP solution (`qdot_cmd`, 7 joints), that arm's CLF slack + CBF λ, obstacle clearance (grasp shaded), haptic force + clutch + authority α, active-arm timeline, metrics text panel. Matplotlib headless (`Agg`).
- **Active-hand resolution** (`study_metrics.resolve_active_arm`): prefers `/shared_autonomy/active_arm` when it carries data (VF/blending); otherwise infers the active hand from which arm is actually moving (published EE speed) — works even in `no_assist` (no controller change needed, resolved offline from the bag).
- **`build_master_table.py`** (planned) — aggregate every trial's `metrics.json` into `trials_summary.csv` (`MASTER_TABLE_NAME`), one row per `(participant, world, condition)` (overwrite semantics). Pandas/pyarrow acceptable here.
- **`study_analysis.py`** (planned) — cross-condition comparison figures + stats from the master table.

**Metric families**: (A) effectiveness — total/per-phase time (sliced by `/shared_autonomy/grasp_active`), manual success, #retries/#aborts; (B) motion quality — EE speed/SPARC smoothness from published `/qp_debug/ee_real` (ground truth, per arm) and QP solution `qdot_cmd` vs measured `qdot_measured`, path efficiency; (C) safety — min-distance/near-miss on `/qp_debug/min_distance` computed with the autonomous-grasp window excluded (that topic is the raw signed closest distance over all pairs before the grasp CBF-bypass, and the graspable cylinder is in the collision set, so the intentional gripper↔cylinder overlap drives it negative during grasp; a grasp-inclusive raw minimum is kept separately), plus λ_cbf active-time; (D) human effort — clutch-press count/duty (`/virtuose/button_right`, VF/no_assist), force impulse from `/virtuose/force_cmd` (not cross-mode comparable), handle excursion; (E) assistance — α stats and human–policy agreement from `/shared_autonomy/blend_debug`, belief-convergence time from `/shared_autonomy/goal_probabilities`; (F) subjective — questionnaire scores (NASA-TLX/trust/preference) in `metadata.json`. Dashboard shades the autonomous-grasp window on authority/CBF/safety panels.
