# AI Agent Context — triago_control

> **This file is maintained by the AI agent.**

## 0. Maintenance Rules

1. **Always share the pull/rebuild command** (§14) right after pushing any change.
2. **Keep this file short.** Only math, core architecture, and non-obvious invariants — no changelogs, tuning history, or bugfix stories (those belong in commit messages). Update sections in place.
3. **Length budget: ~6k words** (`wc -w .kiro/context.md`). When adding content, cut something else in the same edit.

---

## 1. Project Identity

- **Package**: `triago_control` — ROS 2 Humble, `ament_cmake` + `ament_cmake_python` (hybrid C++/Python)
- **Robot**: PAL Robotics TRIAGo++ (bimanual, mobile base, lift torso, head)
- **Repository**: https://github.com/Robertorocco/triago_control
- **Runtime**: Dockerized ROS 2 workspace shared via `~/exchange/` with the host
- **Sibling package**: `haption_teleoperation` (haptic device side of the shared-autonomy pipeline, same branch names) — see its own context.md.

**Official branches** (two; `main` mirrors the first):
- **`feature/sim-user-study`** — latest. Parameters used in the user study and described in the thesis, with the **Haption Desktop 6D Compact**. Also holds `thesis/` (sources, figures, final PDF).
- **`real-hw`** — the code, values, and logic run on the **physical robot** (with the full-size Virtuose 6D). Functionally frozen; only documentation may change. Diff it to see what a sim retune changed: `git diff real-hw -- triago_control/qp_controller/config.py`. Do not restate its numbers here.
- **`main`** — kept fast-forwarded to `feature/sim-user-study`.

## 2. Package Structure

```
triago_control/
├── config/worlds/*.yaml + *.world      per-scenario obstacle layouts (§6)
├── config/trajectory_endpoints.yaml    open-loop test presets
├── scripts/
│   ├── qp_arm_teleop/
│   │   ├── main_qp_controller.py       ★ QP-CLF-CBF safety loop (arms)
│   │   ├── main_qp_controller_perceived.py  camera-driven CBF world (§6.1)
│   │   ├── main_qp_controller_real.py  real-hardware variant: async CBF, staleness watchdog (§8.2)
│   │   ├── main_shared_autonomy.py     ★ intent prediction + blending
│   │   ├── trajectory_generator.py     open-loop quintic reference source
│   │   ├── plotter.py / offline_plotter.py   live / static telemetry (offline also bags each trial)
│   │   └── base_controller.py, keyboard_teleop.py, drift_evaluator_node.py, freq_oscillation_diagnostic.py
│   ├── head_controller/
│   │   ├── qp_head_visual_servo.py     QP visual servoing for the head camera
│   │   └── main_head.py                RANSAC tabletop perception (§6.1)
│   └── analysis/                       user-study capture + offline analysis (§15)
├── haption_teleoperation/              sibling package
└── triago_control/                     importable Python library
    ├── qp_controller/  config.py (ALL tunables), robot_kinematics.py, collision_manager.py,
    │                   qp_formulator.py, reference_governor.py, world_loader.py,
    │                   perceived_world_builder.py, shared_autonomy_handler.py, visualization_engine.py
    └── shared_autonomy/ belief_estimator.py, goal_set.py, grasp_state_machine.py, plot_manager.py
```

**Import convention**: always `import triago_control.qp_controller.config as cfg`; never bare `import config` (collides with system modules).

**Offline-recording trigger** (`trajectory_generator.py` → `offline_plotter.py`): one `std_msgs/Bool` on `cfg.OFFLINE_RECORD_TRIGGER_TOPIC`; rising edge at WAITING→TRACKING, falling at TRACKING→REGULATION. The edge is a one-shot message, so the generator waits for a subscriber (`cfg.OFFLINE_RECORD_WAIT_TIMEOUT_S`, 0 = off).

**Multi-waypoint paths**: presets with `mode: path` + `waypoints_*` are threaded by a cubic spline, arc-length parametrized onto the usual quintic timing. The CBF enforces `h = distance − CAPSULE_RADIUS − D_SAFE_BASE`, so budget waypoints against `h`. `s_curve_right_infeasible_regression` is intentionally infeasible; do not fix it.

**Rate damping** (`ENABLE_RATE_DAMPING`, `RATE_DAMPING_VS_MEASURED=True`): adds `RATE_WEIGHT·‖q̇ − q̇_measured‖²` (arm joints only) to the cost. The measured anchor is essential: anchoring on the commanded value is a self-referential low-pass on the QP's own output and was rejected. Real hardware needs a much larger weight than sim.

**Two implementation rules** (violating either makes quadprog throw "constraints are inconsistent" on a satisfiable problem): (1) every cost gradient stays masked to the active arm joints, because locked joints are pinned by two opposing box rows; (2) never zero `last_dq_safe` in the infeasible branch. `_diagnose_infeasibility` prints a throttled post-mortem.

## 3. Mathematical Core: Arm QP-CLF-CBF (`main_qp_controller.py`)

Decision vector `x = [q̇ (nv), δ_right, δ_left]` (joint velocities + one CLF slack per arm).

**Cost**: joint-velocity damping (`DAMP`); rate damping toward measured velocity (`RATE_WEIGHT`); posture repulsive field `v_ref = −K_GRADIENT·dH/dp`, `H(p)=1/(1−p)²+1/(1+p)²` on normalized joint position `p`, weighted by `W_CENTER` and scaled by `posture_scale` (ramps to `POSTURE_GRASP_SCALE` in precision grasp phases); the field's center may be retargeted per world (`POSTURE_TARGET_BY_WORLD`), each side normalized by its own half-width so `p` still reaches ±1 at both real limits; slack penalty, adaptively weighted per arm (§8).

**Wrist-branch escape** (`WRIST_BRANCH_MIN_BY_WORLD`): `arm_*_6_joint` below about 0 sits in a disconnected IK branch the task row cannot leave. A posture recovery pull acts until the joint crosses the floor, then a permanent joint-limit CBF floor latches. Arming the floor against a pose already below it makes the barrier demand a velocity past the joint limit and the solve fails.

**Constraints (`C'x ≥ b`)**:
- **CLF (task tracking)**: weights `TASK_WEIGHTS_6D` (position-first); precision phases use `TASK_WEIGHTS_6D_GRASP` per arm (`orient_boost_arms`).
- **CBF**: two independent per-arm rows `J_soft_X·q̇ ≥ b_X`, each a SoftMin over the `K_MAX_PAIRS` closest pairs touching that arm, with its own margin `d_safe_dynamic_X = D_SAFE_BASE + K_V_SAFE·‖v_X‖`. Per-arm independence stops the inactive arm being recruited. A high `ALPHA_SOFTMIN` keeps the SoftMin on the truly nearest pair in clutter.
- **Joint limits**: velocity-aware position buffer; every index outside the two arms (torso, base, fingers, head) is locked to `q̇ = 0`.

Solver `quadprog.solve_qp`. Shadow prices `λ_cbf_*`, `λ_joints_*` are telemetry and drive the scheduler (§8).

**Inactive arm**: frozen at its current pose by a zero-velocity CLF; slack weight pinned to `MAX_WEIGHT_SLACK`, gain to `GAMMA_MAX`, damping doubled; it can still yield.

**Head-as-obstacle**: the head chain is a quasi-static CBF obstacle (live FK capsules); no head joint is in the decision vector. `d_safe_dynamic` sees only arm speed, so head motion is an unmodelled disturbance; the symmetric half lives head-side (§7.1).

## 4. Reference Governor (`reference_governor.py`)

Per-arm limiter between the raw Cartesian reference and the CLF target (master switch `ENABLE_REFERENCE_GOVERNOR`): velocity cap (`GOV_V_MAX_LIN/ANG`), position-error ball (`GOV_E_MAX_POS`), acceleration limit (`GOV_A_MAX_*`), orientation-error clamp (`GOV_E_MAX_ORI`). It limits; it does not test admissibility.

## 5. Shared Autonomy & Reference-Level Blending (`main_shared_autonomy.py`)

Belief over a discrete goal set, a local QP policy per goal, a grasp state machine, and (when `ASSIST_BLENDING`) blending of the human twist with the policy. `-p plot:=false` disables the live dashboard.

### 5.0 Condition selector — 2×3 study design (`config.py` §1b)

Flags: `CONTROL_MODE ∈ {CLUTCH, JOYSTICK}`, `ASSIST_FEEDBACK` (channel F: `F_guide` + `F_fixture` on top of the `F_sync` tether), `ASSIST_BLENDING` (channel B: arbitration in `main_shared_autonomy`, sole writer of `/arm_*/cartesian_reference` while active). Cells: `CF/CB/CFB`, `JF/JB/JFB` (study); `C/J` (no-assist baseline, works in code, excluded from the study). Force managers are `haptic_force_manager_<CELL>`; every teleop and force-manager node calls `cfg.validate_condition(...)` at startup and hard-errors on a mismatch.

**Guidance gate**: `gain = conf_gate × prox_gate`, proximity full ≤0.10 m / dead >0.60 m, confidence dead <0.30 / full ≥0.90 on `b_max` (max-posterior active-goal belief, `/shared_autonomy/active_goal_pose`). Same thresholds in both modes. **Grasp-phase asymmetry**: during autonomous grasp the CLUTCH managers render no force (vibration cue only); JOYSTICK managers keep the centring spring plus the cue.

### 5.1 Belief & policy

- **Local policy**: constrained QP twist `π_k` per goal under the same CBF constraints as the safety QP.
- **Belief**: discounted competition-normalised score (not a posterior) over goals; goals held by either arm or not graspable are pinned to zero. The observed action is the user's commanded twist against user-frame policies. Twist metric `W = diag(10,10,10,2,2,2)`; a 10% position tie-breaker is anchored at `current_T_user`.
- **Blend**: `π_policy = Σ_k belief(k)·π_k`. `get_object_belief` sums belief over an object's grasp types, and the grasp trigger uses that object-level mass.

### 5.2 Arbitration (`ASSIST_BLENDING`)

The reference is `v_blend = (1−α) v_user + α π_policy`, integrated every tick into a latched reference (`T_blend_ref`), so an idle user yields an absolute hold.

```
user STILL:   α = ALIGN_ALPHA_IDLE
user ACTIVE:  s = mean per-channel cosine(v_user, π_policy)
              s ≥ 0: α = MIN + (MAX − MIN)·s       s < 0: α = MIN·(1 + s)
```

Below the stillness thresholds (`STILL_LIN`, `STILL_ANG`) α is forced to 0. Clutch button held: blending suspended, `T_blend_ref` holds. In CLUTCH mode belief is scored against policies anchored at the blended reference, not the integrated handle pose. Blending is active in user-controlled states and suspended during autonomous grasp states; `T_blend_ref` re-anchors to the live EE on re-entry. Telemetry: `/shared_autonomy/blend_debug` (19 floats: α, v_user, v_policy, v_blend). The operator marker `/blended_reference_marker` shows the literal pose on `/arm_*/cartesian_reference`.

### 5.3 Bimanual state

Each arm owns a `GraspStateMachine`, `BeliefEstimator`, and goal-set context; goal exclusion is the union over both arms; a cylinder-vs-cylinder pair prevents two held objects interpenetrating.

### 5.4 Goal set (`goal_set.py`)

Goals are SE(3) manifolds. **Side**: approach azimuth is a confidence-weighted mix of the radial anchor→axis direction and the gripper's own heading (their degenerate cases are disjoint), plus a discrete fingers-up/down candidate with sticky hysteresis. **Top**: position above the axis, approach axis −Z, roll about the axis free, resolved in closed form to the circle point nearest the anchor, `θ* = atan2(M01−M10, M00+M11)`, `M = R_TOP_DOWN·R_anchorᵀ`; its angular error is exactly the approach-axis tilt. **Front**: approach axis locked to +X (reach goals). **Platform_Place**: only the held-object axis ⊥ platform face is constrained. The two free DOFs are stored as unit 2-vectors and low-passed by a normalized EMA (`GOAL_DIRECTION_FILTER_TAU`), which bounds how fast a goal can slide around its manifold; `update_memory=False` queries do not advance the filter.

### 5.5 Grasp state machine

`SHARED_AUTONOMY → PRE_GRASP → GRASP_ALIGN → GRASP_APPROACH → GRASP_CLOSE → LIFT → HOLDING` (failure: `ABORT_RETREAT`). On `GRASP_CLOSE → LIFT` the held object is re-parented as a link of the collision world; release reverses it (`RELEASE_LIFT`) and relocates the object's believed pose. `RELEASE_LIFT` is distance-gated (`RELEASE_RETREAT_DISTANCE` along the reverse approach axis, capped by `RELEASE_RETREAT_MAX_S`) so clearance is real before handoff.

Grasp confirmation is purely geometric (no force sensing): signed gripper-box↔object distance against the believed pose, with a per-type contact depth and insertion travel that are coupled (a shallower insertion caps how deep contact can register). `real-hw` tunes both deeper; diff before assuming either branch's numbers apply to the other. The `PRE_GRASP → GRASP_ALIGN` gate has asymmetric ENTER/STAY tolerances (`POS/ANG_ERR_ENTER/STAY`); `/shared_autonomy/grasp_gate` (5 Hz) reports which gate is blocking — read it before retuning.

## 6. World Scene Loading (`world_loader.py`, `config/worlds/*.yaml`)

Static obstacles, graspable cylinders, and placement platforms are declared per scenario in `config/worlds/<name>.yaml` (`ObstacleSpec` list, `grasp_roles`, `platform(s)`; platforms are visual-only). `CollisionManager`, `VisualizationEngine`, and `GoalSet` consume `world_scene` generically, so a new world needs only a YAML whose name matches its Gazebo `.world`. `main_qp_controller.py` and `main_shared_autonomy.py` take the same `world_name` parameter. A `role: "reachable"` obstacle without collision gives a `Front` goal.

**`rack_world`**: a shelf over the table forces SIDE grasps; a conveyor `Final` platform is reachable only by the right arm, so the left arm stages on the `Handover` disk. Decorative scenery is only in the `.world`. After editing, copy the `.world` by hand to `pal_gazebo_worlds/worlds/` (plain copy, not a symlink) and check the XML parses.

### 6.1 Camera-perceived world

`main_qp_controller_perceived.py` overrides only `_build_world_scene()` to build the collision model from the head camera instead of YAML. The head RANSAC pipeline (`main_head.py`, `head_control/`) freezes its estimate once confident: expected cylinders above `WORLD_CONF_MIN` and `WORLD_MIN_ARC_COVERAGE`, plus an amplitude lock (centres within `WORLD_CONVERGE_POS_AMP`, dimensions within `WORLD_CONVERGE_DIM_AMP` over `WORLD_CONVERGE_WINDOW_S` of settled frames). Head motion is perception-led (`ENABLE_ACTIVE_VISION`). The snapshot is a latched `MarkerArray` on `/perceived_world/snapshot`; the QP side blocks in `wait_for_scene()`; `/perceived_world/rescan` re-arms it. No dynamic per-tick updates (a moving obstacle set makes the barrier non-stationary). Real hardware: launch `launch/head_real.launch.py` (RealSense driver, measured mount transform, camera topic overrides).

## 7. Head Controller — Visual Servoing (`qp_head_visual_servo.py`)

Fully independent from the arm QP (own solver, no shared state). Decision `x = [dq_head (7), slack (3)]`. PBVS look-at when hands are outside the FOV, 2.5D IBVS when inside. Cost: joint regularization, slack (`W_SLACK_PIXELS/DEPTH`), posture. Equality `J_task·dq − slack = −λ_visual·e`. Inequalities: FOV barriers (IBVS only) and joint-limit buffers. Hand positions come from FK, not detection.

### 7.1 Active-arm tracking (`head_active_arm_tracking.py`)

Points the camera at the active arm's TCP (`gripper_{side}_grasping_link`, injected on real hardware if missing) from `/shared_autonomy/active_arm`. One IBVS task: centring plus stand-off `TARGET_DISTANCE` (0.7 m). Speed is `min(MAX_VELOCITY, LAMBDA_VISUAL·e)`: the cap owns the far field, the gain the near field. `JOINT_WEIGHTS` is uniform (a proximal-to-distal ramp prices the shoulder out of reach of every soft term).

- **Arm-switch homing** (`ENABLE_SWITCH_HOMING`): a switch suspends the visual task and servos the head to a compact home pose (`meq=0`; ends on `HOME_TOL` or `HOME_TIMEOUT_S`), Y-biased per side by `HOME_Y_BIAS`.
- **Head-vs-arm CBF** (`ENABLE_HEAD_CBF`): the arms' own SoftMin barrier restricted to head pairs, `J_soft[head]·dq ≥ −GAMMA_CBF_HEAD·(h_soft − (d_safe_head + D_SAFE_HEAD_EXTRA))`, with `d_safe_head` driven by the head's own last commanded velocity. The head brakes earlier and softer than the arm by design. An infeasible solve commands zero velocity (the head freezes rather than colliding).
- **PBVS slack** must be re-priced (`W_SLACK_PBVS`): in PBVS the slack is an angular velocity, not pixels; any new cost term must be checked against both branches' units.
- **Soft terms** (all LS terms, never equalities): posture field (`K_POSTURE_GRADIENT`, clamp `V_MAX_POSTURE`), roll alignment to world-right, approach-axis alignment (orbit toward the approach axis, blended toward −Z by `top_view_bias`), and chain lean on the elbow (`LEAN_FRACTION`, `K_LEAN`; joints 5–7 cannot absorb it). Telemetry on `/head_active_tracking/{error,qdot,cartesian_cmd}`.

### 7.2 AprilTag pose reconstruction (`head_april_main.py`)

Alternative perception: localize one fiducial and reconstruct objects from a known rigid transform, `base_M_obj = base_M_cam · c_M_tag · tag_M_obj`. Accuracy depends on: the optical-centre frame for position (`APRILTAG_OPTICAL_CENTER_FRAME`), the tag orientation in the world (`APRILTAG_RPY_WORLD`), BEST_EFFORT + TRANSIENT_LOCAL subscriber QoS, and the ViSP method `best_residual_virtual_vs`. `[TAG-DIAG]` (`-p tag_diag`) logs estimated against known tag pose. Run next to `visp_apriltag_node` (`tag_family:=36h11`, `tag_size:=0.12`) in `apriltag_world`.

## 8. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting** (`DYNAMIC_SLACK_WEIGHT`, on): each arm's CLF slack weight falls from `MAX_WEIGHT_SLACK` toward `BASE_WEIGHT_SLACK` as its low-pass-filtered shadow price grows (`BETA`, `SLACK_FILTER_TAU`), then passes through a second filter `WEIGHT_SLACK_FILTER_TAU`. Thesis §3.2.6 (`sec:sched`) is the reference derivation.
- **Dynamic CLF gain** (`DYNAMIC_GAMMA_CLF`, off) and **posture weight** (`DYNAMIC_POSTURE_WEIGHT`, off): same λ-driven pattern; the trajectory generator also time-scales its reference from λ. Tested in development, absent from reported trials.

### 8.1 Compute budget

`compute_softmin_jacobian` is the largest phase; it filters, searches the witness pair, and weights pairs with batched numpy. On real hardware Meshcat is never started and telemetry/marker rates halve.

### 8.2 Async execution — `main_qp_controller_real.py` (real hardware only)

A subclass of `SafetyQPController` that moves the CBF and RViz overlays to worker threads; launch this file on the robot. Seams in the base class: `_compute_cbf()`, `_gate_command()`, `_publish_visual_overlays()`, `_process_deferred_topology()`. The worker uses private `pin.Data`/`GeometryData`. A staleness watchdog (`CBF_STALENESS_MAX_TICKS`) freezes both arms when the barrier result is stale; `CollisionManager.geom_lock` serializes grasp attach/detach against the worker. Python threads gain only where C extensions release the GIL; validate with `loop_timing_monitor.py`.

### 8.3 Control frequency

`CONTROL_FREQ_DEFAULT = 150` Hz (300 Hz misses its own target under CBF load). `ENABLE_INTER_ARM_CLOSING_MARGIN` (default False) makes an inter-arm pair's margin also reflect the other arm's speed. The control loop runs on wall-clock deliberately (§10.6).

## 9. Frame Convention (Haption ↔ TRIAGo)

A pure 180° rotation about Z, `Φ = diag(−1,−1,1)`, between the device frame and `base_footprint`; it applies to velocities and, identically, to forces. It is our convention in the Python teleop layer, not a device property.

## 10. Critical Hardware / Environment Quirks

1. **Joint velocity is always derived** from position differences + EMA (`ALPHA_FILTER`) in `robot_kinematics.update_from_joint_state`, on sim and real alike; raw sensor velocity injects noise through the rate-damping term.
2. **`REAL_HARDWARE` auto-detection**: the URDF lacks `gripper_{right,left}_grasping_link` on the real robot, so the frames are injected and broadcast as static TFs.
3. **Meshcat is not thread-safe**: only the `_run_viz` thread touches the WebSocket.
4. **Controller switching**: activate `arm_{right,left}_joint_space_controller_vel` and deactivate conflicting trajectory controllers before commanding.
5. **No force/torque sensing** and no Gazebo ground truth anywhere; hence geometric grasp confirmation (§5.5).
6. **Wall-clock control loop, on purpose**: the controllers never set `use_sim_time`; a sim-time timer catches up in bursts under executor starvation and snaps the arm. `dt` in the math is the nominal `1/CONTROL_FREQ_DEFAULT`; joint-velocity reconstruction takes `dt` from the message stamp. Plotters auto-detect `/clock`. Debug with `TRIAGO_DEBUG_TICKS=1` and `/qp_debug/loop_timing`.
7. **RT priority / CPU pinning on the controller can starve PAL's EtherCAT master** and trip a bus-wide watchdog. Check `ros2_control_node`'s own `chrt`/`taskset` first.

## 11. Build & Run

```bash
cd ~/exchange/ros2-ws
colcon build --symlink-install            # config/launch/Python edits then need no rebuild
source install/setup.bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/exchange/ros2-ws/src/pal-packages/pal_gazebo_worlds

ros2 launch triago_control simulation_bringup.launch.py world:=rack_world   # Gazebo, controllers, 3 RViz stations, robot-model filter
ros2 run triago_control main_qp_controller.py --ros-args -p world_name:=rack_world
ros2 run triago_control main_shared_autonomy.py --ros-args -p world_name:=rack_world -p plot:=false
ros2 run triago_control head_active_arm_tracking.py --ros-args -p plot:=false
ros2 run triago_control study_recorder.py
# teleoperation side, pair matching config.py §1b:
ros2 run haption_teleoperation virtuose_server_node
ros2 run haption_teleoperation teleop_triago_clutch.py
ros2 run haption_teleoperation haptic_force_manager_CFB.py
```

`world_name` must match on every node. On the robot use `main_qp_controller_real.py` (or `_perceived`) and `launch/head_real.launch.py`. The launch file does not start the QP, shared-autonomy, or recorder nodes, so they restart independently. Gazebo needs the external IFRA_LinkAttacher plugin (`GAZEBO_PLUGIN_PATH` to its `lib`). Other nodes: `trajectory_generator.py`, `plotter.py`, `offline_plotter.py`, `loop_timing_monitor.py`, `qp_head_visual_servo.py`.

## 12. Coding Conventions

- Every tunable lives in `qp_controller/config.py`; no hard-coded gains elsewhere.
- snake_case files and variables, PascalCase classes; no `_v2`/`_new` suffixes.
- Entry points are `main_*.py` in `scripts/`; library modules have no `__main__`.
- Comments: one line, no history, TRIAGo spelled as such.
- Diagnosing a regression: change one `config.py` gain at a time, read the values fresh from the file, and use the plotter's Task Authority and Posture Weight panels to see which term moved.

## 13. Workspace paths

Colcon workspace `~/exchange/ros2-ws/`; this repo at `~/exchange/ros2-ws/src/triago_control`.

## 14. Git Workflow

- Branches as in §1. Commit messages: one line, imperative, <72 chars. Stage only the task's files.
- Videos and raw trial data are gitignored and never published.
- **After every push**, give the user this sync block:

```bash
cd ~/exchange/ros2-ws/src/triago_control
git checkout -- .
git pull origin <branch>
cd ~/exchange/ros2-ws
colcon build --packages-select triago_control
source install/setup.bash
```

## 15. User Study / Analysis Subsystem (`scripts/analysis/`)

Tooling to run and record the human-subject study (24 participants P01–P24, 288 trials; P00 pilot excluded) and analyze it offline. Parameters live in `scripts/analysis/study_config.py`, separate from controller gains.

- **Design (§5.0)**: `(CONTROL_MODE, ASSIST_FEEDBACK, ASSIST_BLENDING)`; cell codes `CF/CB/CFB/JF/JB/JFB`; `derive_cell()` also covers the `C/J` baseline.
- **Recorder** (`study_recorder.py`): Tkinter GUI over `ros2 bag record`, one launch per trial. START spawns the bag, STOP sends SIGINT, the experimenter marks success and notes, SAVE writes `metadata.json` (including a `cfg` snapshot). The cell is auto-detected from `cfg`. On SAVE it spawns `analyze_trial.py` (niced) and, when a participant's full grid is complete, `export_to_matlab.py`. Storage: `DATA_ROOT/<participant>/<world>_<cell>/`; re-recording overwrites.
- **Storage split**: code in git; data in `~/exchange/triago_study_data/` (44 GB, `TRIAGO_STUDY_DATA_ROOT`-overridable, gitignored). Only the bags are irreplaceable.
- **Offline analysis**: `study_metrics.py` (numpy-only engine; metric families: effectiveness, motion quality, safety, human effort, assistance, subjective), `analyze_trial.py` (per-arm dashboards), `build_master_table.py`, `check_study_data.py` (single source of truth for "is a trial usable"; folder name is ground truth for world and cell).
- **MATLAB** (`matlab/`): `export_to_matlab.py` writes `matlab_export/` (`savemat(..., long_field_names=True, oned_as='column')` is mandatory); `run_study_analysis.m` runs the group statistics; `build_paper_figures.m` and `build_report.m` make the working documents; `fig_thesis_panel.m` and `fig_thesis_learning.m` draw the print panels into `thesis/.../figures/results/`. A `matlab_export/` created before a `study_metrics.py` change keeps old values silently: rescore with `export_to_matlab.py --from-mats`, then repoint `analysis_results/latest.txt`.
- **Metrics (2026-09-22 definitions)**: SPARC is scored per active-driving segment; `T_λ` counts `λ > 10⁻³` over operator-control time; ρ is the per-channel cosine the controller uses; φ_α and Δ are taken over the active-driving set. Clearance comes from `h_soft` (carried and bypassed objects excluded). Force and clutch metrics are never compared across modes; `human_effort` is `qdot_cmd_rms` alone.
- **Subjective analysis** (`subjective/run_subjective_report.py`): reads the Google Forms export (six 1–7 items in six repeated blocks, one per condition; demand items reversed), rank-based tests, repeated-measures correlation. Item polarity is an inference, not documented. `thesis_figs.py` draws the thesis panels with `text.usetex` + `lmodern`.
- **Hardware-trial figures (thesis §4.1)**: `export_offline_bag.py <bag_dir>` converts an `offline_plotter.py` bag to `.mat`; `fig_thesis_hw.m` draws one PDF per panel (registry-based, broken axes per trial, truncated at the deadman release). The raw bags and videos stay local. Six `hw_*`/`tel_*` panels remain in the old face because their rounded axis limits cannot be reproduced; there is no committed driver recording the exact `fig_thesis_hw` calls.
- **Handover to another machine**: copy `triago_study_data/` (or at minimum `triago_matlab_bundle.zip` for the group statistics, plus `matlab_export/mat/` for per-trial series). Pipeline order, each step idempotent: `check_study_data.py` → `export_to_matlab.py` → MATLAB `run_study_analysis` → `build_paper_figures` → `make_matlab_bundle.sh`. For the thesis: MATLAB `build_thesis_figures`, `python3 scripts/analysis/subjective/run_subjective_report.py`, then `latexmk -pdf main.tex`. To drive MATLAB from a script, use `matlab.engine.shareEngine('TRIAGO')` and `connect_matlab`; prefer explicit path arguments over `eng.run`. The questionnaire CSV carries personal data and stays local.

### 15.1 Thesis (`thesis/roberto_rocco_master_thesis/`)

UNINA template (`extbook`, 12 pt, oneside); sources are `main.tex`, `0_frontespizio.tex`, `abstract.tex`, `chapters/*.tex`, `references.bib`; `\graphicspath{{figures/}{img/}}`. Style rules are in `thesis/master_thesis_rules.md`. Build with `latexmk -pdf main.tex` in that directory (baseline: 0 undefined references, 19 Overfull warnings), then copy `main.pdf` to `thesis/roberto_rocco_master_thesis.pdf`; the PDF is committed with the release, `main.pdf` is not. Never invent bib entries.

Facts to keep the text consistent: the CLF row is the quadratic candidate's condition divided by ‖W e‖; `K_V_SAFE` is a conservative closing-speed margin with horizon `k_v/ℓ_max ≈ 14–29 ms`; the QP enforces a nominal barrier condition, not certified invariance (limitations in Conclusions §6.2); hardware ran the async CBF (≤2 ticks stale); the grasp lifecycle is a hybrid heuristic; the belief is a score, not a posterior; the hardware trials are the home-to-crossed run and a CFB teleoperated bimanual grasp whose failed retreat stalled until an emergency stop. §3.2.6 frames the KKT slack schedule as the main contribution, with time scaling, γ, and posture as supporting hints. Ringraziamenti is the last page (QR code only). A hidden NOTE in §4.1 flags that some telworld values reuse earlier recordings.

### 15.2 Figure typography

Every thesis panel is set with the real LaTeX interpreter in Latin Modern (MATLAB: `DefaultTextInterpreter` = `latex`, no `FontName`; Python: `text.usetex` with `lmodern`). Literal strings are LaTeX source, so `&` is `\&` and `%` is `\%`; p-value labels are wrapped in `$…$`. The working documents (`study_paper_figures.pdf`, `subjective_report.pdf`) keep their own look.
