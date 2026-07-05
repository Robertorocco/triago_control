# AI Agent Context — triago_control

> **This file is maintained by the AI agent. Do not edit manually.**
> Last updated: 2026-07-04 (§9.12 follow-up: added the `platform` field to
> the world scene schema -- the yellow `placement_area` disk used by
> shared-autonomy's Platform_Place goal. NOT an obstacle (no collision, not
> in `static_obstacles`) -- a separate top-level `WorldScene.platform`
> field, since the user pointed out this is purely a visualization/
> reference-pose aid for the operator, not something the CBF should ever
> see. See §9.12 below for the full design.)
> Earlier: 2026-07-04 (§9.11 follow-up #3: `fig6_reference_governor`
> styling fix -- the "commanded" (raw) curve is now DASHED instead of
> solid+alpha for both arms, while "governed" stays SOLID. Rationale: with
> both curves solid+alpha, overlapping Right/Left commanded lines
> alpha-blended into an ambiguous purple smear; dash gaps keep pure
> red/blue visible even where the two arms' commanded curves coincide.
> Governed curves are unchanged (solid, drawn on top) since they're the
> primary signal of interest. `offline_plotter.py` only.)
> Earlier: 2026-07-04 (§9.11 follow-up #2: `offline_plotter.py`'s
> `fig6_reference_governor` was rebuilt from a raw-minus-governed DIFFERENCE
> plot into a raw-VS-governed ABSOLUTE plot -- 4 rows (linear velocity,
> angular velocity, position tracking error, orientation tracking error),
> each showing the commanded ("raw") and governed curve per arm PLUS a
> dashed grey horizontal line at that row's governor-configured ceiling
> (`cfg.GOV_V_MAX_LIN/ANG`, `GOV_E_MAX_POS/ORI`) -- directly answers "how did
> the governor modify the trajectory", reconstructed from data ALREADY on
> the wire (`/qp_debug/governor`'s existing diff payload + the already-
> tracked raw reference), zero new topics/publishers needed. Only emitted
> when `cfg.ENABLE_REFERENCE_GOVERNOR=True`.)
> Earlier: 2026-07-04 (§9.11 follow-up: `offline_plotter.py` gained a
> 3D commanded-vs-executed gripper trajectory figure -- solid=commanded
> reference / dashed=executed EE pose, red=Right / blue=Left, matching every
> other per-arm color convention in this codebase. Saved as PDF+PNG always;
> ALSO as a browser-navigable HTML (free rotate/zoom/pan) if the optional
> `plotly` package is installed -- gracefully skipped otherwise, zero hard
> dependency added. Also: `config/trajectory_endpoints.yaml`'s
> `local_minima_behind` preset (Case 1, BEHIND THE ROBOT) was removed per
> instruction -- Case 2 (THROUGH THE TABLE) is untouched; `active_preset`
> reset to `"home"`.)
> Earlier: 2026-07-04 (§9.11 NEW: `offline_plotter.py` -- static,
> publication-quality figures for the QP-CLF-CBF pipeline, companion to the
> LIVE `plotter.py`. Driven by a generic, source-agnostic Bool trigger topic
> (`cfg.OFFLINE_RECORD_TRIGGER_TOPIC`) so it can be wired to
> `trajectory_generator.py` today and to a future teleoperation-side trigger
> later with zero code changes. See §9.11.)
> Earlier: 2026-07-03 (§11.9 NEW: orientation symmetry fix for the
> authority gates. Operator report: orientation stayed "almost frozen" in
> BLENDING mode even while position clearly responded to user steering, and
> the plotted `ALPHA_MAX` line looked "deprecated" since alpha visibly
> exceeded it near a goal. Root cause of the orientation issue: BOTH the
> user-effort gate (§11.7) and the position-divergence override (§11.8) read
> ONLY linear/position quantities (`v_user[0:3]`, `||pos_user - pos_EE||`) --
> spinning the handle or holding the reference rotated away produced ZERO
> effort/divergence signal, so alpha stayed belief-driven and the policy's
> angular twist dominated `v_blend` completely unopposed. Fix: `compute_alpha`
> now also reads `v_user[3:6]` (angular effort, gated by new
> `cfg.ALPHA_EFFORT_ANG_THRESHOLD=1.0 rad/s`) and a new `ang_divergence`
> parameter (geodesic rotation gap via `pin.log3`, gated by new
> `cfg.ALPHA_DIVERGENCE_ANG_NEAR/FAR=0.15/0.60 rad`, matching the existing
> catch-up deadband exactly) -- each combined with its linear counterpart via
> `max()`, so EITHER channel alone hands the user authority, mirroring how
> position already worked. Per explicit operator instruction,
> `cfg.TASK_WEIGHTS_6D` (the CLF's own position:orientation cost ratio) was
> NOT touched -- this fix operates purely at the alpha/blend level. Also
> fixed the plotted alpha ceiling line, which showed the stale `ALPHA_MAX`
> instead of the true near-goal ceiling `ALPHA_PROXIMITY_CAP`. See §11.9.)
> Earlier: 2026-07-03 (RRT-Connect abandoned: an RRT-Connect joint-space
> planner was attempted (2026-07-01 -- 2026-07-03) as a fallback local-minima-
> escape strategy alongside the existing posture-weight correction. The
> approach was ultimately unsuccessful and has been fully removed from the
> codebase (`rrt_planner.py` deleted; `reference_governor.py`,
> `main_qp_controller.py`, and `plotter.py` stripped of all RRT integration
> -- background planning thread, Cartesian waypoint queue, `/rrt_planned_*`
> RViz topics, `/qp_debug/rrt_planner` telemetry, the RRT Planner plot
> window, and the `cfg.LME_ESCAPE_STRATEGY` selector). `cfg.
> ENABLE_LOCAL_MINIMA_ESCAPE` now offers ONLY the original posture-weight +
> task_dim=3 correction and is set to `False` by default. `cfg.BLENDING` is
> now `True` by default (shared-autonomy twist blending, §11.5-§11.8, is the
> active teleoperation mode).)
> Earlier: 2026-07-03 (§11.8 NEW: position-divergence authority override +
> bounded reference catch-up. Operator report on §11.7: the velocity-effort
> gate only reacts while the hand is ACTIVELY MOVING -- the instant the user
> decelerates and HOLDS their hand displaced from the EE, ||v_user|| -> 0, the
> gate relaxes, and the belief-driven policy dominates again, so the robot
> barely follows through to where the hand is resting (exactly the mechanism
> the operator wanted to use to escape local minima via reference POSITION,
> not just twist) -- while fighting F_sync's restoring force the whole time.
> Root cause: v_blend has no memory of current_T_user, only its derivative.
> Fix (two purely-geometric mechanisms, built only from current_T_user vs
> current_T_EE -- no QP Lagrangian/shadow-price anywhere, per explicit
> operator constraint): (1) a SUSTAINED position-divergence override on
> compute_alpha (`ALPHA_DIVERGENCE_OVERRIDE=0.6`, does not decay with
> velocity); (2) a new `compute_reference_catchup` -- a gentle, CAPPED P-pull
> (`V_CATCHUP_MAX_LIN=0.06 m/s`) ADDED onto the blended twist toward the
> user's held pose, gated by a deadband, that still passes entirely through
> the downstream QP CLF-CBF (cannot force the arm through an obstacle). See
> §11.8.)
> Earlier: 2026-07-03 (§11.7 NEW: user-effort authority gating. Operator
> reported the arm was almost "blind" to their own hand twist in TWIST
> BLENDING mode -- `pi_policy` is a large, saturated velocity while comfortable
> hand motion is much smaller, so even a moderate `alpha(belief)` let the
> policy dominate `v_blend`; the user could only ever pick a different goal,
> never meaningfully resist/steer once one was inferred. Fix: `compute_alpha`
> now also takes the raw human twist (`current_v_h`) and scales alpha DOWN by
> how hard the user is ACTIVELY moving the handle --
> `effort = clip(||v_user_lin|| / ALPHA_EFFORT_THRESHOLD, 0, 1)`,
> `alpha *= (1 - effort * ALPHA_EFFORT_OVERRIDE)` -- LPF'd, fully smooth,
> deliberately NOT sourced from any QP Lagrangian/shadow-price (avoids the
> discontinuity/filtering-delay tradeoff explicitly flagged by the operator).
> `ALPHA_EFFORT_THRESHOLD=0.4 m/s` (fast hand motion), `ALPHA_EFFORT_OVERRIDE=
> 0.5` (half of alpha displaced at full effort -- the rest of the "follow"
> reduction already comes from fast motion naturally lowering the belief
> estimate). Still handle -> unaffected (full belief-driven assistance, exactly
> where it's wanted: near obstacles / when intent changes). See §11.7.)
> Earlier: 2026-07-03 (§11.6 NEW: three usability fixes to the TWIST
> BLENDING architecture (§11.5), based on operator hands-on feedback: (1) new
> `/blended_reference_marker` RViz gripper (light-blue, same style as the
> existing pure-user-intent `/guidance_policy_marker`) showing the LITERAL pose
> integrated from the blended twist and sent to the QP -- both light-blue
> markers stay live simultaneously so the operator can A/B which is more
> intuitive; (2) `cfg.ALPHA_MAX` 0.80 -> 0.60 (user's guaranteed authority floor
> raised from 20% -> 40%) since once a goal was picked the autonomy had too
> much authority for the user to meaningfully override; (3) a new smooth,
> distance-based proximity boost (`cfg.ALPHA_PROXIMITY_*`) on `compute_alpha`
> so the assistive twist doesn't fade to uselessness near the goal --
> compensating for `pi_policy`'s own natural CLF-style falloff so the task can
> actually be CONCLUDED, still capped below 100% authority via
> `ALPHA_PROXIMITY_CAP`. See §11.6.)
> Earlier: 2026-07-03 (§11.5 NEW: config-driven shared-autonomy TWIST
> BLENDING architecture. `cfg.BLENDING` in `qp_controller/config.py` is now the
> SINGLE source of truth for the flag (removed the old local, always-false,
> never-wired `self.BLENDING` on `SharedControlNode` — `compute_alpha` used to
> `raise NotImplementedError` and was dead code even when the local flag was
> flipped, since the blended twist was only ever PUBLISHED during grasp
> execution / `POLICY_BELIEF_TEST`, never in normal teleop). `teleop_triago_
> clutch.py` (haption_teleoperation) reads the SAME flag to decide whether it
> publishes the pure user reference on `/arm_*/cartesian_reference` (BLENDING=
> False, legacy, unchanged) or on a NEW topic `/arm_*/user_cartesian_reference`
> (BLENDING=True) — freeing `main_shared_autonomy.py` to become the SOLE,
> PERSISTENT publisher of the real `/arm_*/cartesian_reference` at all times
> when blending is on, so the two nodes never race over the same topic. New
> telemetry topic `/shared_autonomy/blend_debug` (19 floats:
> `[alpha, v_user(6), v_policy(6), v_blend(6)]`) is the single source of truth
> for "who is commanding what" — consumed verbatim by the rewritten
> `haptic_force_manager_blending_tutorial.py` (haption_teleoperation) for its
> new "Authority Share" plot, rather than recomputing the blend independently.
> See §11.5 for the full design.)
> Also 2026-07-02 (§5.10: head cylinder-perception accuracy pass —
> found and fixed the dominant error source in `main_head.py`'s geometric
> pipeline: the circle fit ran on the disk INTERIOR, not just the boundary,
> systematically biasing the radius small by ~-4 to -5mm; no amount of
> scanning could fix that. Added rim extraction + a Hyper circle fit, a
> top-slice-median height estimate (was a biased-high `z_max`), shrank
> `VOXEL_SIZE` 10mm->3mm (was destroying the rim-fit gain), moved
> `HEAD_POSTURE_TARGET` closer to the table (FK-verified reachable, ~40%
> depth-noise-variance reduction), closed a distortion-correctness gap in
> `camera_interface.py`, and switched `ObjectTracker`'s dimension fusion from
> grow-only-max to EMA (grow-only was quietly compensating for the
> since-fixed bias; on an unbiased signal it just drifts upward). All
> verified numerically against the real project code with synthetic,
> realistic-noise point clouds — no access to the real robot/Gazebo in this
> session. Reviewed `feature/head-sweep-compute-track`, did not merge it —
> incompatible with the current pipeline interface and re-enables a
> point-level accumulation approach `main` already tried and disabled.)
> Earlier: 2026-07-02 (§9.10: RRT-Connect planner fixed — the always-fail
> `samples=0` goal finder (uniform random rejection sampling, which cannot hit a
> 3cm Cartesian ball in 7D) was replaced with damped least-squares position IK
> (`_find_goal_config_ik`, random restarts on collision/stall); the per-tick
> re-launch spam on failure was replaced with ONE planning attempt per escape
> episode; `abort()` is now fully NON-BLOCKING (no `join` in the 300Hz loop) and
> in-flight threads are fenced by a monotonic epoch; the planner thread is wrapped
> so no failure can propagate to the QP; an in-flight plan is aborted the moment
> the reference resumes moving. On planner failure the controller simply HOLDS the
> obstacle-escape posture correction (lowered posture weight + task_dim=3) and
> tracks nothing — the QP-CLF-CBF is fully insulated. See §9.10.)
> Earlier: 2026-07-01 (§9.7: head chain added to the ARM QP as a quasi-static
> CBF obstacle — the arms now avoid the head chain via live-FK capsules, WITHOUT
> adding any head joint to the QP's decision vector; `arm_right_1`/`arm_left_1`
> excluded from head pairs per instruction; head capsules tinted yellow in Meshcat.
> No change was needed in qp_formulator — the pre-existing "everything outside
> idx_right/idx_left is velocity-locked to zero" joint-limit mechanism already
> makes this correct by construction; see §9.7 for the full math-soundness
> discussion (quasi-static assumption, bounded guarantee degradation).
> Earlier this day: fixed an RViz gripper-lag regression caused by extra
> per-tick publishes (§9.4 follow-up); slider-GUI polish pass (§9.6:
> MAX_WEIGHT_SLACK 60→100, label/value overlap fix, gripper row de-aligned,
> 2-decimal value readout, gripper-lag topic-rate diagnosis + EMA mitigation);
> per-arm coupling fix CONFIRMED working by the operator; RViz visualizer fixed
> to draw BOTH grippers blue when both arms are actively driven (§9.4);
> "Slack Weight" plot split into per-arm R/L traces (§9.4); "Joint Data"
> position line-plots replaced by the slider-panel GUI using REAL joint limits
> from the live Pinocchio model (§9.5); per-arm dynamic CBF safety-margin split
> (§9.3); fixed a `NameError` on a stale `b_col` reference (PR #8); replaced
> plotter.py's broken-in-sim raw-encoder plot with the QP-solved joint velocity.
> STABLE v1 checkpoint tagged 2026-06-30 at commit "STABLE v1: teleoperation + grasping backup
> checkpoint" — roll back there if this or a later change regresses the system)

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

### 5.10 Cylinder-perception accuracy pass (2026-07-02) — `main_head.py` / `head_control/*`

Context: the user reported the cylinder pose/radius estimate from
`main_head.py`'s geometric pipeline (RANSAC plane -> cluster -> upright-
cylinder fit) had a few-cm error and asked to diagnose and push it under
1cm, since the eventual goal is feeding this into the arm QP-CLF-CBF as a
live obstacle. All fixes below were verified NUMERICALLY (synthetic point
clouds with realistic ~3mm RealSense-like depth noise, run against the
ACTUAL project code, not just a standalone re-implementation) before being
applied — see the session's tool history for the raw sweep numbers. No
access to the real robot/Gazebo was available in this environment.

**Root cause identified (the dominant one, ~-4 to -5mm of the reported
error): the circle fit ran on the wrong point set.** A cylinder viewed
top-down is not just a ring — the camera also sees the entire SOLID TOP
FACE (a filled disk). Fitting any algebraic circle (Kasa or otherwise)
directly to a disk-interior + partial-arc cluster is systematically biased
toward a SMALLER radius (disk interior points outnumber and sit closer to
centre than the true boundary). Critically, **more scan viewpoints do NOT
fix this** — they just re-confirm the same biased fit with more points,
which is why the existing `ENABLE_SCAN` sweep wasn't closing the gap.

**Fixes applied, all in `triago_control/head_control/`:**

1. **Rim extraction before fitting** (`object_detector.py`,
   `_extract_rim`): bins the cluster by angle about a rough centroid and
   keeps, per bin, points near the LOCAL 93rd-percentile radius (a
   percentile, not the raw max — verified more robust to RGB-D "flying
   pixel" outliers). Collapses the disk interior away, leaving an
   approximately uniform ring for the circle fit.
2. **Kasa -> Hyper circle fit** (`_fit_circle_hyper`, Al-Sharadqah &
   Chernov's algebraic fit; Kasa kept as a fallback for degenerate rims):
   removes Kasa's own small residual bias on top of the rim fix.
3. **Height: top-slice MEDIAN, not `z_max`** — `z_max` is a maximum-
   statistic, systematically biased HIGH under noise (verified: +7 to
   +9mm at realistic point counts/noise). Median of the top slice
   (`CYL_TOP_SLICE`) removes almost all of this.
4. **`VOXEL_SIZE` 10mm -> 3mm** (`config.py`): the old 10mm downsample leaf
   was roughly HALF the cylinder's diameter (r=2cm) and was destroying
   most of the rim-extraction gain by pre-collapsing near-boundary points
   before the rim fit ever saw them (verified: 10mm leaf -> end-to-end
   bias ~-3.4mm; 3mm leaf -> ~-1.4mm, matching the un-downsampled case).
5. **Distortion correctness gap closed** (`camera_interface.py`):
   `CameraInfo.d` was received and silently discarded — deprojection never
   checked or corrected for it. RealSense depth streams are typically
   firmware-rectified (`D=[0]*5`, confirmed from vendor docs), so this was
   likely NOT the dominant error source for the actual hardware, but it
   was a real correctness gap (would silently mishandle any camera/config
   with non-zero `D`, e.g. the COLOR stream). Now applies iterative
   Brown-Conrady undistortion (5 fixed-point iterations, verified to
   converge to machine precision) whenever `D` is non-negligible, with a
   one-time warning log.
6. **`HEAD_POSTURE_TARGET` moved closer** (`config.py`): searched via
   Pinocchio FK against the real URDF for a reachable pose with similar
   "character" (elevation angle, joint-limit margins) to the previous
   target but closer to the table — 0.81m -> 0.63m camera-to-table-centre
   distance (kept a safety margin above the RealSense D455's rated 0.4m
   minimum depth range). Stereo depth noise scales roughly with
   distance^2, so this is a free ~40% reduction in depth-noise variance.
   Also raised `DEPTH_MIN` 0.20 -> 0.35m to reject near-range returns
   below the sensor's rated floor.
7. **`ObjectTracker` fusion policy: grow-only-max -> EMA**
   (`object_tracker.py`): the old grow-only rule ("radius/height can only
   ever increase") was a reasonable patch for a per-frame estimator KNOWN
   to be biased small — it isn't anymore. Grow-only on an unbiased signal
   drifts upward over many frames (verified: +0.5 to +0.9mm after just
   5-50 frames, then keeps climbing). Switched to the same EMA already
   used for position (`TRACK_POS_ALPHA`); `TRACK_DIM_DECAY` removed as
   dead config.
8. **`ENABLE_SCAN` kept True but re-justified**: no longer needed to fix
   bias (a single rim-corrected view is already sub-cm in simulation), but
   still legitimately useful for closing angular COVERAGE gaps (self-
   occlusion on the far side of the cylinder from any single viewpoint).
   Safe to disable if startup latency matters more than full-circumference
   arc coverage.

**Verified end-to-end result** (real project code, synthetic ~3mm-noise
point clouds, r=2cm/h=15cm cylinder, matching the Gazebo world's ground
truth): single-view radius bias improved from roughly -4 to -5mm (baseline)
to about -1.3 to -2.5mm depending on the visible arc width; after a 5-
waypoint scan through the real `ObjectTracker` EMA fusion, radius bias
settled around -2 to -4mm and height bias to roughly ±1mm — comfortably
under the 1cm target requested, though not yet at the sub-millimetre level
the isolated rim-fit unit test showed (the gap is the realistic combination
of voxel downsampling + Euclidean clustering + the arc width actually
visible per waypoint; further gains would require either a wider per-view
arc, more waypoints, or an even finer voxel leaf, each with a cost/latency
trade this pass did not take).

**`feature/head-sweep-compute-track` branch reviewed but not merged**: adds
a SWEEP -> COMPUTE -> TRACK state machine (dense one-shot detection at fixed
waypoints, then live-tracking a single target colour while freezing the
rest as CBF obstacles) plus a multi-threaded executor so perception can't
stall the control loop. Not directly usable as-is — it was written against
`dynamic_plane`/`target_only`/`radius_bounds` kwargs on
`perception_pipeline.py`/`object_detector.py` that don't exist on `main`,
and it re-enables raw point-level `VoxelMap` accumulation, which `main`
already tried and explicitly disabled (smears/breaks plane RANSAC — see
§5b in `config.py`). Two ideas from it are worth revisiting later once
there's a concrete need: (a) separate control/perception ROS callback
groups/threads (a real robustness improvement, independent of accuracy),
and (b) freezing a one-shot dense obstacle model while live-tracking only
the actively-manipulated target, now that the underlying single-frame
estimator is actually trustworthy.
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
           triago_control.qp_controller.reference_governor.ReferenceGovernor

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

5. **Bimanual inactive-arm handling (Option B, 2026-06-29)**: the QP-computed velocity is ALWAYS sent to TSID for BOTH arms — the old zero-overwrite for a reference-less arm is removed (it discarded the inactive arm's collision-avoidance motion, causing silent inter-arm penetration). The INACTIVE arm (the one not currently teleoperated) is FROZEN at its current EE pose via a zero-velocity CLF (`_freeze_arm` in `main_qp_controller`, triggered on `/shared_autonomy/active_arm` switch, on watchdog-stale reference, and once at startup for both arms). Its slack weight is doubled (`INACTIVE_SLACK_FACTOR=2.0`) so it holds position firmly yet can still bend if that helps the active arm satisfy task + CBF safety. `build_and_solve(..., inactive_arm=...)` applies the doubling.

---

## 9. Mathematical Core (QP-CLF-CBF)

Decision vector: `x = [q_dot (nv), delta_right, delta_left]`

**Cost** (minimize):
- Joint velocity regularization (damping λ = 10.0)
- Posture / joint-limit avoidance: **repulsive potential-field** reference (replaced the
  old q_neutral spring + Chan&Dubey ramp). `v_ref = -K_GRADIENT·dH/dp` where
  `H(p)=1/(1-p)²+1/(1+p)²` on the normalized position `p=(q-mid)/half_range`; clamped to
  ±`V_MAX_POSTURE`, denominators guarded inside (-1,1). ~0 in mid-range (CLF keeps tracking
  priority), explodes near a limit (cost-only, never overrides constraints). Weighted by `W_CENTER`.
  During autonomous precision phases (`grasp_active`: align/approach/close/lift/abort/release-lift)
  the posture weight is smoothly scaled down to `POSTURE_GRASP_SCALE`×`W_CENTER` (default 0.05×) so
  the QP spends the redundancy on precise tracking instead of posture. `main_qp_controller`
  subscribes to `/shared_autonomy/grasp_active` and ramps `qp.posture_scale` (τ=`POSTURE_SCALE_TAU`).
- Slack penalty (adaptive per-arm weighting)
- Telemetry: the QP publishes its soft-task cost decomposition `[E_damp, E_posture, E_slack]`
  on `/qp_debug/task_authority`; the plotter's "Task Authority" window shows each one's
  normalized share (hard-constraint authority = the λ_CBF / λ_Joints shadow prices).

**Constraints** (C'x >= b):
- **CLF (task tracking)**: Perfect Scalar Inequality CLF with diagonal task weights [pos=10, ori=1]. Two formulations available (`COMPARISON_CLF` flag): normalized (unit-error) or raw.
- **CBF (collision avoidance) — TWO INDEPENDENT PER-ARM ROWS (2026-07-01)**: replaced the single
  combined SoftMin row with `J_soft_R·q̇ ≥ b_R` and `J_soft_L·q̇ ≥ b_L`, aggregated separately over
  K_MAX_PAIRS=60 closest pairs that touch each arm's own geometry (`CollisionManager._arm_membership`
  — a pair touching BOTH arms, e.g. a genuine inter-arm contact or two held cylinders, contributes
  to BOTH rows; a pair touching only one arm's geometry vs. a static obstacle appears in ONLY that
  arm's row). Dynamic margin = d_safe_base + k_v_safe * ||v|| (shared by both rows). Fixes the
  spurious coupling where the inactive arm twitched/oscillated whenever the active arm neared an
  UNRELATED obstacle — see §10.1.
- **Joint limits**: velocity-aware position buffer (CBF-style).

Solver: `quadprog.solve_qp` (active-set method).

### 9.2 Per-arm SoftMin CBF split (2026-07-01) — the coupling fix

**Problem** (diagnosed and confirmed correct by design review): a single scalar SoftMin CBF row
mixes ALL active pairs via one softmax weighting. Its gradient can have nonzero columns in an
arm's joints even when NONE of that arm's own pairs are actually close to binding — merely because
some OTHER pair (possibly involving only the other arm) was among the K closest. The QP then
legitimately recruits the "innocent" arm's joints to help satisfy a barrier that has nothing to do
with it, causing oscillation / unwanted motion of the inactive arm whenever the active arm nears
ANY obstacle (not just the other arm).

**Fix**: `CollisionManager.compute_softmin_jacobian` now builds **two independent aggregates**,
routed per-pair by `_arm_membership(geom_id, attached_object_arm)`: a geometry belongs to arm X if
it is one of X's own capsules/gripper box, OR a cylinder X currently holds (`attached_object_arm:
{cyl_id: 'right'/'left'}`, threaded through from `SharedAutonomyHandler`). A pair contributes to
arm X's SoftMin iff **at least one** of its two geometries belongs to X:
- A pair touching **both** arms (genuine inter-arm contact, or two held cylinders nearing each
  other) contributes to **both** rows → **preserves** "arm A may yield to help arm B reduce its
  tracking error" (the desired coupling).
- A pair touching **only** arm A's geometry vs. a static obstacle (table, wall, un-held cylinder,
  body) **never** appears in arm B's row → **eliminates** the spurious oscillation (the unwanted
  coupling).

`qp_formulator.build_and_solve` now takes `(J_soft_r, h_soft_r, J_soft_l, h_soft_l, ...)` and
assembles TWO CBF rows (rows 0 and 1; joint-limit rows shifted +1 to indices 2..2+2*n_joints).
Returns `b_col_pair = (b_col_r, b_col_l)` instead of a scalar. Two independent shadow prices are
tracked: `qp.last_lambda_cbf_right` / `qp.last_lambda_cbf_left` (their max is kept as
`qp.last_lambda_col` for backward-compat with the slack scheduler).

**Downstream propagation** (all updated together, matching payload layouts):
- `/qp_debug/lambda_cbf`: `Float64` → **`Float64MultiArray`** `[lambda_cbf_R, lambda_cbf_L]`.
- `/collision_constraints`: 13 floats → **14 floats** `[b_col_r, b_col_l, J_c_cart_R(6), J_c_cart_L(6)]`.
  Each arm's own cartesian gradient now comes from its OWN SoftMin (`J_soft_r`/`J_soft_l`)
  instead of both projecting the same combined `J_soft` — this also fixed a latent bug where
  `main_shared_autonomy.collision_data_callback` read the SAME `b_col`/gradient regardless of
  which arm was active.
- Plotter: "QP Data" window's CBF-price row now plots **both** `λ_CBF,R` (red) and `λ_CBF,L`
  (blue) on the same axes (`lambda_cbf_callback` now expects a 2-element array).
- `haption_teleoperation/haptic_force_manager_tutorial.py`: `cbf_gradient_cb` reads the shifted
  14-float layout (indices 2:8 for the right gradient, was 1:7); `lambda_cb` takes `msg.data[0]`
  (lambda_cbf_R) since that node always drives the right gripper.

### 9.3 Per-arm dynamic safety margin split (2026-07-01) — the SECOND coupling channel

After §9.2's Jacobian split, the operator still observed idle-arm oscillation when the
active arm moved fast near an obstacle. Full mathematical re-audit of the CBF math found
a second, independent coupling channel that the Jacobian split did not touch:
`d_safe_dynamic = D_SAFE_BASE + K_V_SAFE * ||v_norm||`, where `v_norm` was computed from
the **combined** velocity of BOTH arms (`concat(current_v[idx_right], current_v[idx_left])`),
and this ONE scalar was used identically in both CBF rows:
`b_col_X = -GAMMA_CBF * (h_soft_X - d_safe_dynamic)`.

A fast active arm inflates the combined `v_norm`, which shrinks the idle arm's own margin
against its (nearly always finite, just usually harmless) own `h_soft_L` — tightening
`b_col_L` and forcing idle-arm joint motion **purely because the other arm sped up**, with
no change in the idle arm's own geometry or proximity. This is a *threshold*-level coupling
(shifts the constraint's RHS), distinct from the Jacobian-level coupling fixed in §9.2, and
it survived that fix entirely.

**Fix**: `CollisionManager.compute_softmin_jacobian` now computes `d_safe_dynamic_r` and
`d_safe_dynamic_l` independently, each from only that arm's own `current_v[idx_right]` /
`current_v[idx_left]` norm. `QPFormulator.build_and_solve` takes both and applies each to
its own row. The grasp-margin SoftMin shift (per-pair negative margin during a controlled
grasp contact) now looks up which arm actually owns the gripper side of that pair
(`_arm_membership`) and uses that arm's own dynamic margin, instead of the old combined
value (falls back to `max(d_safe_dynamic_r, d_safe_dynamic_l)` if membership is ambiguous —
should not occur in practice). `/qp_debug/d_safe_dynamic` is now a 2-element
`Float64MultiArray` `[d_safe_R, d_safe_L]` (was a single `Float64`); the plotter's "Dynamic
Safety Margin" row now plots both.

**A third, currently-inert channel was found and documented but NOT changed**: in
`QPFormulator._schedule_weights`, `weight_slack_l`'s schedule uses
`max(self._lam_col_f, self._lam_jl_f)`, where `_lam_col_f` is the LPF of
`last_lambda_col = max(lambda_cbf_right, lambda_cbf_left)` — i.e. the ACTIVE arm's CBF
shadow price can affect the IDLE arm's slack-weight schedule. This is currently masked in
the reported "one active, one idle" scenario because a frozen arm's slack weight is pinned
unconditionally to `MAX_WEIGHT_SLACK` (see `right_frozen`/`left_frozen` in
`build_and_solve`), bypassing the dynamic schedule entirely. It only becomes live if BOTH
arms are simultaneously active (neither frozen) — not the reported bug's scenario. Flagged
here for whoever eventually revisits simultaneous-bimanual-teleop tuning; not touched now to
keep this fix minimal and testable.

**New diagnostic tool (plotter.py, 2026-07-01)**: row 2 of the "Joint Data" window
("Raw Encoder vel") is REMOVED — Gazebo's simulated joint encoder velocities are
known-broken (see §8.1) and added no diagnostic value. Replaced with "QP Solution
(q_dot_safe)": the EXACT joint velocity vector the QP solved and sent to the TSID
controllers this tick (sourced from the existing `/qp_debug/qdot_cmd` topic, previously
only used for the row-3 servo-error computation). This lets the next diagnosis step
directly distinguish "the QP itself commands nonzero idle-arm velocity" (a further
QP-side coupling bug) from "the QP commands ~0 for the idle arm but the simulator/
robot moves it anyway" (a simulator/inertia/PID-tuning issue, out of the QP's scope).

### 9.4 Per-arm gripper visualization + per-arm slack telemetry (2026-07-01)

**Bug reported**: running `trajectory_generator.py` (which publishes references to
BOTH arms simultaneously) showed only ONE blue commanded gripper in RViz, with the
other arm rendered grey (as if frozen), even though the QP was correctly tracking
both arms.

**Root cause**: `qp_visualizer_tutorial.QPVisualizer` was built for the single-
active-arm teleop paradigm — it held ONE `cmd_pos`/`cmd_rot_matrix` (blue) and ONE
`frozen_pos`/`frozen_rot_matrix` (grey), gated by a single `self.active_arm`
("right"/"left") that only ever changed on a `/shared_autonomy/active_arm`
message. `trajectory_generator.py` never publishes that topic, so `active_arm`
stayed at its default "right": the right arm's reference always went to the blue
slot, and the left arm's reference was *always* routed to the grey "frozen" slot
regardless of whether it was actually frozen.

**Fix**: `QPVisualizer` now tracks each arm's reference pose and frozen flag
**independently** — `ref_pos_right/left`, `ref_rot_right/left`, `frozen_right/left`
(no more `active_arm`/`cmd_pos`/`frozen_pos`). The frozen flags are populated from
a NEW topic, `/qp_debug/arm_frozen` (`[right_frozen, left_frozen]` as 0.0/1.0
floats), published by `main_qp_controller.py` from its own ground-truth
`self.right_frozen`/`self.left_frozen` state — the same state that actually drives
the QP's cost decoupling (§8.5) — rather than re-derived/guessed in the visualizer.
Each arm now renders its own gripper marker on its own pair of namespaces
(`qp_debug_gripper_{right,left}` for blue, `frozen_gripper_{right,left}` for grey);
whichever state isn't active is explicitly `DELETE`d so switching never leaves a
ghost. Result: BOTH grippers render blue when both arms are actively driven
(trajectory_generator, dual teleop), while the classic single-arm teleop paradigm
(one arm frozen by the CLF hold) still shows exactly one blue + one grey, as
before. `publish_teleop_tether` was similarly split into two independent
tethers (ns `teleop_tether`, ids 0=right/1=left).

**Slack weight telemetry (confirmed per-arm, now shown as such)**: confirmed by
reading `qp_formulator.build_and_solve`'s Hessian slack block —
`weight_slack_r` weights ONLY `delta_r` (the right-arm CLF slack) and
`weight_slack_l` weights ONLY `delta_l`, fully independently; they were already
computed independently by the per-arm dynamic scheduler (§10). The bug was
telemetry-only: `/qp_debug/dynamic_weights` published just their AVERAGE
(`[weight_slack_avg, gamma_clf]`). Now publishes `[weight_slack_r, weight_slack_l,
gamma_clf]`; `QPFormulator` also stores `self.weight_slack_r`/`self.weight_slack_l`
alongside the legacy averaged `self.weight_slack`. The plotter's "Slack Weight"
row in the "Task Error & Adaptation" window now plots both (red=R, blue=L),
matching the convention used for every other per-arm quantity in the dashboard
(λ_CBF, λ_Joints, d_safe_dynamic).

### 9.5 Plotter: joint-position slider GUI (replaces the position line plots)

The scrolling 14-line position plot in the "Joint Data" window (Figure 1) was
reported hard to read at a glance. Removed the "L/R- Position" rows entirely;
Figure 1 is now a 3×2 grid (velocity / QP solution / servo error only — see
§9.2's diagnostic-plot note for the QP-solution row).

**New Window 6, "Joint Positions"**: a read-only slider-panel GUI matching the
layout of a reference control-panel image — one `matplotlib.widgets.Slider` per
joint, arranged in a grid: col 0 = left arm (`arm_left_1..7_joint`), col 1 = head
(`arm_head_1..7_joint`), col 2 = right arm (`arm_right_1..7_joint`). The grid
(`cfg.SLIDER_LAYOUT`, list-of-lists) lives in `config.py` as the single source of
truth for both the plotter's layout and any future consumer. **Per instruction,
the reference image's `torso_lift_joint` slider and joystick widget are NOT
encoded.**

Sliders are **display-only** (`eventson=False`; the handle position is set
programmatically from live `/joint_states` each animation frame — dragging them
does nothing). Slider **ranges use the REAL joint limits** read from the live
Pinocchio model (`RobotKinematics.get_joint_limits`, the exact same limits the
joint-limit CBF rows in `qp_formulator` enforce), published once (latched via a
self-cancelling 2 s timer) on a new topic `/qp_debug/joint_limits` as a
semicolon/colon-encoded `String` (`"name:lower:upper;..."` — chosen over a new
custom message type since this is a low-rate, non-critical debug topic). Falls
back to a placeholder `[-3.15, 3.15]` range until that message arrives, then
snaps to the real limits.

### 9.6 Slider GUI polish + gripper-lag topic-sanity diagnosis (2026-07-01)

Follow-up pass after the operator tried the new slider GUI:

1. **`MAX_WEIGHT_SLACK` 60 → 100**: operator-tested value from experimentation,
   persisted so their tuned free-space tracking behaviour survives a fresh pull.
   (Still the ceiling of the dynamic slack schedule AND the fixed value pinned on
   a frozen/inactive arm — see §3 of `config.py`.)

2. **Label overlap fixed**: the original 4-column layout (arm/head/arm/gripper)
   put the two gripper sliders in only rows 5–6 of column 3, leaving 5 empty
   cells and making that column look unbalanced — and `matplotlib.widgets.Slider`
   labels default to sitting immediately LEFT of the track, which overlapped the
   value readout at this GUI's compact per-cell size. Fix: `slider.label` is
   repositioned to `(0.5, 1.6)` in axes-fraction coordinates with
   `ha='center', va='bottom'` — i.e. **centered, above the slider track** — while
   `slider.valtext` (the numeric readout) stays in its default position, on the
   RIGHT of the track, unchanged.

3. **Gripper column removed; dedicated de-aligned gripper row added**: the 4th
   ("gripper") column is gone from `cfg.SLIDER_LAYOUT` (now a clean 7×3 arm/head
   grid). The two gripper finger sliders (`cfg.GRIPPER_SLIDER_ROW`) render in
   their OWN row below the grid, via a nested `GridSpecFromSubplotSpec`
   (`subgridspec`) that centers them with blank gap columns on both sides and
   between them — by construction this makes them NOT column-aligned with the
   3-column arm/head grid above (per instruction), and a genuinely blank
   height-0.3 spacer row separates the two sections visually.

4. **Value display truncated to 2 decimals**: `Slider(..., valfmt='%.2f')` on
   every slider (was matplotlib's default `%.3g`-ish formatting, e.g. `2.258`).

5. **Gripper slider lag — root-caused and mitigated**: the reported "laggy"
   behaviour (smooth with one gripper shown, visibly steppy with two) is a
   **topic-rate mismatch, not a plotting bug**. `gripper_{left,right}_finger_joint`
   are confirmed ABSENT from the robot's own URDF finger kinematic tree (which
   only has the real underactuated `gripper_{side}_finger_{1,2,3}_*flexor*` /
   `*rotatory*` / `*tip*` joints, plus `_coupler_joint`/`_palm_joint`/`_tc_joint`
   — see the URDF grep in this session). These two names are **virtual joints
   fed into `/joint_states` by the gripper's own controller** (commanded via
   `SharedAutonomyHandler`'s `FollowJointTrajectory` action clients to
   `/gripper_{side}_controller/follow_joint_trajectory`, joint name
   `gripper_{side}_finger_joint` — see `close_gripper`), whose JointTrajectory
   controller state broadcaster can run at a materially different (often much
   lower / event-driven) rate than the main arm `/joint_states` stream — hence
   the visible step/lag once BOTH grippers were being read (a single gripper's
   own lower rate is less noticeable in isolation; watching two side-by-side
   made the discrete steps obvious).
   - **Diagnostic added**: `TriagoDashboard._check_topic_sanity` (a one-shot
     6 s-after-startup timer) computes the median `/joint_states` inter-arrival
     interval for a representative arm joint vs. each gripper finger joint and
     logs a `[TOPIC SANITY]` WARN if the ratio exceeds 3× (or if a gripper name
     never appeared at all), naming the exact joint and the measured rates.
   - **Mitigation added** (cosmetic, does not change ground truth): a short EMA
     (`alpha=0.35`) is applied ONLY to the two gripper finger joints' slider
     display value (`self.slider_display`, separate from the raw
     `self.slider_positions` used by any future numeric consumer) to visually
     smooth the steps between the controller's own lower-rate updates. Arm/head
     joints are passed straight through, unfiltered (no lag ever reported
     there — they DO share the main joint-state stream).
   - **If a truly smoother physical readout is wanted** (not just a smoothed
     display), the real fix is on the gripper controller side: raise
     `/gripper_{left,right}_controller`'s own joint-state-broadcast rate — this
     is outside `triago_control`'s control (PAL vendor controller
     configuration), so it is not something this package can fix directly.

### 9.7 Head chain as a quasi-static CBF obstacle for the arms (2026-07-01)

The arms were previously unaware of the head chain (`arm_head_1..7_link`, same
7-DOF hardware as the L/R arms, driven by its own future vision-based
controller, `qp_head_visual_servo.py`) — arm-vs-head collisions could occur
silently. Per instruction, added the head as a **quasi-static geometric
obstacle** for the arm QP, WITHOUT adding a single head joint to the QP's
decision vector (`idx_right`/`idx_left` are untouched; `RobotKinematics` never
maps `HEAD_JOINTS` into the actuated velocity indices).

**Math soundness / guarantee trade-off (asked by the operator, answered before
implementing)**: modeling the head as static is sound under the explicit
assumption that the head moves slowly relative to the CBF's margin budget.
The head's geometry is refreshed from LIVE FK every tick (not stale — `current_q`
already contains every joint found in `/joint_states`, arms and head alike), so
positional accuracy is exact. What is given up is *exact* forward-invariance:
the barrier's `ḣ` is computed assuming `q̇_head = 0` (enforced by the pre-existing
joint-limit box constraint — see below), so if the head is ACTUALLY moving at
that instant (driven by its own separate controller), the true `ḣ` differs from
the assumed one by an unmodeled term `∇h_head · q̇_head_actual`. This is bounded
by the head's own commanded speed (`MAX_VELOCITY=0.15 rad/s` in
`qp_head_visual_servo.py`) times one control tick (`dt≈3.3ms @ 300Hz`) — on the
order of `10⁻⁴ m`, negligible against `D_SAFE_BASE=0.015 m`. Guarantee degrades
from "exact" to "quasi-static-valid" — acceptable given the stated assumption,
but worth remembering if the head is ever driven fast or the safety margin is
tightened.

**Implementation** (deliberately minimal — the existing architecture handles
this by construction, confirmed by design review before coding):
- `CollisionManager.build_collision_model` gained an optional `head_offsets`
  parameter; reuses `calculate_offsets`/the SAME dominant-axis capsule
  construction as the arms (the head is literally the same hardware) to build
  `self.head_geom_ids`, kept **separate** from `right_geom_ids`/`left_geom_ids`
  (so `_arm_membership` never classifies a head capsule as belonging to either
  arm — it belongs to neither, by design, since it's not held/grasped).
- `CollisionManager.define_collision_pairs` §2d: adds an arm-vs-head
  `CollisionPair` for every (arm capsule, head capsule) combination, SKIPPING
  any pair touching `arm_right_1` or `arm_left_1` (per instruction — that link
  cannot collide with the head chain).
- `main_qp_controller.py`: builds `head_offsets` via
  `calculate_offsets(cfg.HEAD_CHAIN, cfg.HEAD_TOOL_LINK)` (new config constants;
  `HEAD_TOOL_LINK='arm_head_tool_link'` mirrors `gripper_{side}_base_link`'s
  role for the arms — the fixed frame past the last real link used to size that
  link's capsule length) and passes it through; gracefully skips (with a
  warning) if the head chain is absent from the URDF.
- `visualization_engine.color_collision_model`: head capsules tinted **yellow**
  in Meshcat (distinct from red=right/blue=left) so the operator can visually
  confirm the new obstacle geometry.

**Why no change was needed in `qp_formulator.py` or `compute_softmin_jacobian`'s
per-arm routing** (verified before coding, not discovered after a bug):
- An arm-vs-head pair's two geometries have `_arm_membership` = `{'right'}` (or
  `{'left'}`) for the arm side and `{}` (empty) for the head side — so
  `touched = {'right'}` (never both), meaning the pair contributes to **exactly
  one** arm's SoftMin aggregate, exactly like an arm-vs-table pair. This is the
  SAME per-arm routing mechanism from §9.2, unmodified.
- `J_soft_r`/`J_soft_l` are full `model.nv`-length vectors. For an arm-vs-head
  pair, the per-point Jacobian construction (`get_point_jacobian`, keyed on
  each geometry's OWN parent joint) DOES populate nonzero entries at the head's
  joint columns of that pair's distance-rate row. However, in
  `qp_formulator.build_and_solve`'s joint-limit block (§D), EVERY joint whose
  velocity index is not in `kin.idx_right + kin.idx_left` (`active_indices`)
  has `dq_max_safe = dq_min_safe = 0` — i.e. is HARD-LOCKED to exactly zero
  velocity in the solve (this pre-existing mechanism already applies to torso,
  mobile base, and gripper fingers; the head is just one more member of that
  set). Since the CBF row is `J_soft_X · q̇ ≥ b_col_X` and `q̇` at every head
  index is constrained to exactly `0`, the head's nonzero Jacobian *columns*
  contribute exactly `0` to the row's value regardless of their magnitude — the
  barrier is satisfied ENTIRELY through the arm's own joints bending to
  maintain distance, which is precisely the required behavior (avoid via arm
  motion, never expect head motion). No masking, slicing, or QPFormulator
  change was needed; the existing "everything not in idx_right/idx_left is
  locked" invariant already generalizes correctly to the head.

### 9.8 Reference Governor — CLF-safety intermediate layer (2026-07-01)

An intermediate filter between the raw cartesian reference (`/arm_{right,left}/
cartesian_reference` — from teleop / trajectory_generator / planner) and the
CLF's actual perceived reference inside `extract_task_errors`. Bounds the
position/orientation error and reference velocity/acceleration that the CLF must
handle, preserving QP feasibility guarantees even under aggressive, discontinuous,
or far-away commands.

**New module**: `triago_control/qp_controller/reference_governor.py`, class
`ReferenceGovernor`. One instance per arm (`gov_right`/`gov_left` in
`main_qp_controller.py`), each with its own velocity memory for acceleration
limiting. Master switch: `cfg.ENABLE_REFERENCE_GOVERNOR` (default `True`).

**Four active features** (all independently tunable via config.py §3b):

| Feature | Mechanism | Config |
|---------|-----------|--------|
| Velocity shaping | Clamp reference velocity magnitude (direction preserved) | `GOV_V_MAX_LIN=0.20 m/s`, `GOV_V_MAX_ANG=1.2 rad/s` |
| Position error bounding | Project x_ref onto ball of radius E_MAX centered at x_real | `GOV_E_MAX_POS=0.30 m` (30 cm) |
| Acceleration limiting | Rate-limit velocity change per tick: `‖Δv‖ ≤ A_MAX·dt` | `GOV_A_MAX_LIN=2.0 m/s²`, `GOV_A_MAX_ANG=8.0 rad/s²` |
| Orientation clamping | If `‖log3(R_des·R_real^T)‖ > THETA_MAX`, shrink via exp3 on the same axis | `GOV_E_MAX_ORI=0.524 rad` (~30°) |

**Integration point**: inside `solve_and_publish`, right before the CLF task
errors are computed. The RAW references (`self.x_ref_right`, etc.) are PRESERVED
(for future plotting / other consumers); the governed versions are used ONLY for
the CLF task-error computation and the feedforward velocity passed into
`build_and_solve`. Governor is reset (velocity memory cleared) on `_freeze_arm`
(arm switch / watchdog stale / startup hold).

**Telemetry**: `/qp_debug/governor` (`Float64MultiArray`, 24 floats) publishes
the raw-minus-governed difference each downsampled tick. Layout:
`[pos_diff_R(3), ori_diff_R(3), vel_diff_R(3), wvel_diff_R(3),
  pos_diff_L(3), ori_diff_L(3), vel_diff_L(3), wvel_diff_L(3)]`.
All zeros when the governor is off or all bounds are satisfied (passthrough).

**Plotter**: dedicated Window 7 ("Reference Governor: raw − governed") with 4
stacked subplots (position diff / orientation diff / linear velocity diff /
angular velocity diff), each showing 3 components per arm (R = red shades,
L = blue shades/dashed). Lets the operator see WHICH bound is active, on WHICH
arm, at any instant.

**Waypoint injection interface (STUB, for future planning)**: `set_waypoint(pos,
rpy, priority)` / `clear_waypoint()` / `waypoint_active` property are defined
but NOT yet wired into `govern()`'s output. When a high-level planner (future
RRT/PRM/potential-field escape) calls `set_waypoint`, the governor will blend the
raw reference toward the waypoint while respecting all velocity/error/accel
bounds — so the QP is transparently steered out of local minima without ever
receiving an infeasible/discontinuous command. The blending logic will be
implemented when the planning module is built.

### 9.9 Local Minima Escape (governor extension, 2026-07-01)

Extends the Reference Governor (§9.8) with a state machine that detects a
possible QP-CLF-CBF local minimum and applies a temporary, PER-ARM posture-
weight correction to help escape it. Implemented in
`reference_governor.ReferenceGovernor.update_local_minima_escape` (one state
machine per arm, same as the rest of the governor).

**Two known local-minima causes** (identified before implementing, confirmed
by the operator via manual teleoperation trigger — not encoded as fixed test
scenarios):
1. A CBF obstacle blocks the direct path to the reference (high `lambda_cbf`).
2. A joint-limit barrier blocks the required joint rotation (high
   `lambda_joints`).
3. Both simultaneously — **obstacle takes priority** per instruction.

**Detection** (3D position error only, per instruction — orientation/velocity
error not handled yet): the error norm `||x_ref_governed − x_real||` is
tracked over a rolling `LME_ERROR_STUCK_WINDOW=2.0s` window. "Stuck" = error
`> LME_ERROR_TRIGGER=0.15m` AND has varied by less than
`LME_ERROR_STUCK_TOLERANCE=0.02m` across the whole window (not just moving
slowly — genuinely not decreasing).

**Categorization**: read from the shadow prices produced by the QP's
**PREVIOUS** solve (`qp.last_lambda_cbf_right/left`, `qp.last_lambda_joints_
right/left` — the current tick hasn't solved yet when the governor runs).
Thresholds (`LME_LAMBDA_CBF_THRESHOLD=10.0`, `LME_LAMBDA_JOINT_THRESHOLD=1.0`)
are **operator-tuned for the CURRENT parameter set** — if `GAMMA_CBF`,
`D_SAFE_BASE`, `P_GAIN_LIMITS`, `MAX_WEIGHT_SLACK`, etc. are retuned, these
may need to be revisited (explicitly flagged by the operator).

**Escape action** (posture task ONLY — verified no other module writes to
`qp.posture_scale_right`/`_left`, so no conflict):

| Category | Posture weight | Task dimension |
|----------|----------------|-----------------|
| Obstacle | `x0.2` (`LME_POSTURE_SCALE_OBSTACLE`) — more redundancy to slip past | forced to `3.0` (position-only, orientation fully relaxed — `LME_TASK_DIM_OBSTACLE`) |
| Joint limit | `x5.0` (`LME_POSTURE_SCALE_JOINT`) — push harder away from the limit | unchanged |
| Unknown (stuck but neither threshold exceeded) | unchanged (ramped back to `1.0`) | unchanged |

The multiplier is **smoothly ramped** (first-order low-pass, `LME_RAMP_TAU=
0.3s` — same technique as the existing grasp-phase `POSTURE_SCALE_TAU` ramp)
rather than stepped, so the QP cost function never sees a discontinuity.

**Per-arm posture weight (`qp_formulator.py` extension)**: `QPFormulator`
previously had only a single GLOBAL `posture_scale` (used for the grasp-phase
ramp, shared by both arms). Added `posture_scale_right`/`posture_scale_left`
(default `1.0`), which multiply the GLOBAL scale **on that arm's joints only**
— composed as `w_center_vec[idx_right] = W_CENTER * posture_scale * posture_
scale_right`, i.e. a per-joint weight vector instead of a single scalar. The
soft-task energy telemetry (`task_energies[1]`, the posture share on the
"Task Authority" plot) was updated to use a proper weighted quadratic form
(`np.dot(w_center_vec, dq_post**2)`) since the weight is no longer uniform.

**Exit conditions**: the escape ends (state → `'normal'`) when the error drops
below `LME_ERROR_RECOVERED=0.10m` (success) **or** `LME_MAX_ESCAPE_DURATION=
10.0s` elapses (give up — avoid holding a distorted posture weight forever if
the correction didn't work), whichever comes first. On exit, the governor
immediately resumes checking for a NEW local minimum (no cooldown) — this is
what "back to the normal execution state" means per instruction.

**Console reporting** (non-spam, per instruction): a colored one-shot message
on DETECTION (categorized: obstacle / joint limit / unknown), a colored
one-shot message on EXIT (recovered vs. timed out), and a **throttled status
line every `LME_CONSOLE_PERIOD=3.0s`** while an escape is in progress (posture
scale value, current error, elapsed/max duration). No plot was added for this
weight per instruction — console only.

**Master switch**: `cfg.ENABLE_LOCAL_MINIMA_ESCAPE` (independent of
`cfg.ENABLE_REFERENCE_GOVERNOR` — the escape mechanism does not depend on the
velocity/error/acceleration bounding features being active, though in
practice it operates on the GOVERNED reference `x_gov_r`/`x_gov_l`, so if the
main governor is off, `x_gov == raw reference` and the escape still works
correctly against the raw error).

**Integration point** (`main_qp_controller.py`): computed right after the
governor's `govern()` call, using `x_gov_r`/`x_gov_l` (the governed reference)
vs. the real EE position for the error norm. The resulting
`task_dim_eff_right`/`_left` (raw `task_dim` unless overridden to `3.0`) feeds
`_arm_task_error` for THIS tick only — `self.task_dim_right`/`_left` (the raw
value from the cartesian-reference topic) is never mutated, so the escape is
fully transparent to any other consumer of that state.

### 9.10 [REMOVED] RRT-Connect planner attempt (2026-07-02 -- 2026-07-03)

An RRT-Connect joint-space planner was built as a local-minima-escape fallback
for §9.9 (background-thread planning, damped-least-squares goal-IK with
null-space obstacle avoidance, an escape-strategy selector choosing between
the posture nudge and an RRT-only mode, and supporting RViz/telemetry). After
several rounds of debugging (goal-IK sampling failures, starved time budgets,
an always-converges-but-always-collides bug, a task_dim coupling bug), the
approach was ultimately judged unsuccessful and abandoned on 2026-07-03. All
of that code has been fully removed: `rrt_planner.py` deleted;
`reference_governor.py` stripped of the RRT state/waypoint-queue/trigger
logic (keeping only the original posture-weight + task_dim correction from
§9.9); `main_qp_controller.py` and `plotter.py` stripped of the RRT
publishers, telemetry, and plot window; `config.py`'s §3d RRT block and the
`LME_ESCAPE_STRATEGY` selector removed. `cfg.ENABLE_LOCAL_MINIMA_ESCAPE`
defaults to `False`; if re-enabled, it now offers ONLY the §9.9 posture
correction.

### 9.1 Sensing constraints (real-hardware honesty, 2026-06-29)

**No Gazebo ground-truth is ever read** (`/gazebo/model_states`, `GetEntityState`, etc. — none
exist in this codebase, confirmed by design review). The system relies **only** on what real
hardware provides: joint position/velocity (`/joint_states`) and, for future perception work,
camera (`head_control/object_detector.py` — upright-cylinder fit, built but not yet wired into
the grasp pipeline). **There is no force/torque sensing on the arm chains anywhere**:
`TickInput.current_force_mag` / `current_force_local` are hardcoded to `0.0` / `zeros(3)` —
present in the dataclass for a future sensor, never populated. Grasp confirmation is therefore
**purely geometric**: `grasp_contact` (signed gripper-box↔cylinder `hppfcl` distance) is computed
from the robot's own FK against the cylinder's *believed* position — a kinematic value tracked in
`GoalSet.cylinders[...]['pos']`, updated only by `relocate_cylinder` after a placement, never
measured from the object itself. `GoalSet.set_grasped` similarly **assumes** the grasped
cylinder's axis is world `+Z` at the attach instant rather than measuring it — if the object was
tilted when grasped (plausible with the realistic, lower-friction world physics), the frozen axis
is wrong and the `Platform_Place` orientation goal inherits that error. **Next-agent TODO**: wire
`object_detector.py` (or an equivalent vision check) as an independent confirmation signal, and to
measure the real grasped-object axis instead of assuming it.

Because there is no independent (force/vision) confirmation, the **geometric** grasp-success gates
were tightened ~10% (2026-06-29) after observing false-positive `GRASP_CLOSE` triggers (fingers
not actually well-seated): `GRASP_CONTACT_DEPTH` −0.038→−0.0418 m, `APPROACH_ANG_TOL` 0.15→0.135
rad, `APPROACH_POS_TOL` 0.01→0.009 m (all in `grasp_state_machine.py`, `_grasp_approach`).

---

### 9.11 `offline_plotter.py` -- static publication-quality figures (2026-07-04)

New node, `scripts/qp_arm_teleop/offline_plotter.py`, companion to the LIVE
`plotter.py`. Where `plotter.py` is a scrolling 50 s-window dashboard meant
to be watched while the robot moves, `offline_plotter.py` records exactly
ONE trial from a clean t=0 and, once the trial ends, saves the SAME
telemetry as a fixed set of formal, print-quality figures (PDF + PNG,
300 dpi, serif font, formal titles/subtitles) suitable for a paper. It never
opens a live window (`matplotlib.use('Agg')`) -- it only ever writes files.

**Recording trigger -- deliberately generic/source-agnostic.** The node is
driven entirely by one boolean topic, `cfg.OFFLINE_RECORD_TRIGGER_TOPIC`
(default `/offline_plotter/record_trigger`, `std_msgs/Bool`): `True` starts
(or continues) recording -- on the rising edge, t=0 is anchored to that
EXACT instant (never node startup), guaranteeing the saved time axis always
starts at 0; `False` means "the commanded motion has concluded" and begins a
`cfg.OFFLINE_PLOT_POST_TRIGGER_S=10.0s` post-roll window (captures the
REGULATION/settling phase on the SAME time axis) before finalizing. A plain,
UNLABELED vertical dashed grey line (`_draw_trigger_line`, no legend entry,
per instruction) is drawn on every time-series subplot at the exact instant
of the falling edge. If the trigger goes `True` again during the post-roll
window (motion resumed), the SAME trial is extended rather than truncated.

`offline_plotter.py` has ZERO knowledge of what produces this signal.
Today, `trajectory_generator.py` drives it (`update_phase`: `True` on
WAITING->TRACKING -- the quintic motion actually starts; `False` on
TRACKING->REGULATION -- the motion has concluded). Per instruction, this
generic contract is what makes it trivial to later wire a teleoperation-side
trigger onto the SAME topic (e.g. "the operator is holding the handle and
the clutch is released") without touching this file at all -- only the
publisher-side node changes.

**Ctrl-C**: whatever was recorded so far (even mid-trial, even before the
post-roll window elapses) is finalized and saved before the process exits
(`main`'s `except KeyboardInterrupt` calls `_finalize_and_save`).

**Data sources**: reuses the EXACT topics `plotter.py` already subscribes
to (`/joint_states`, `/qp_debug/qdot_cmd`, `/qp_debug/slacks`,
`/qp_debug/lambda_cbf`, `/qp_debug/lambda_joints`, `/qp_debug/loop_freq`,
`/qp_debug/safety_margin`, `/qp_debug/min_distance`,
`/qp_debug/dynamic_weights`, `/qp_debug/d_safe_dynamic`,
`/qp_debug/task_authority`, `/qp_debug/governor`, `/arm_*/cartesian_
reference`, `/qp_debug/ee_real`), PLUS one NEW topic:
`/qp_debug/qdot_measured` (`main_qp_controller.py`, 14 floats R7+L7) --
publishes `kin.current_v` at the arm indices, i.e. the SAME
environment-resolved velocity signal the QP itself already consumes
internally (EMA-filtered differentiated velocity in Gazebo, direct sensor
reading on real hardware -- see `robot_kinematics.update_from_joint_state`'s
existing branch). This ALSO fixes a live-dashboard inaccuracy: `plotter.py`'s
"Velocity from driver" row plots the RAW, known-broken-in-sim `/joint_states`
velocity (see §8.1) -- the new topic is the trustworthy one, used only by
`offline_plotter.py` for now (per instruction, `plotter.py` itself was left
unchanged in this pass). Named generically ("measured", not "filtered")
since it means something different in sim vs. real hardware, by design.

**Figures produced** (mirrors `plotter.py`'s content; the live-only CBF
active-pairs debug view and the joint-position slider GUI are dropped --
both are meaningless as a static artifact):

| File | Content |
|------|---------|
| `fig1_joint_kinematics` | 3x2 grid: Position / Velocity (`qdot_measured`) / QP solution (`qdot_cmd`) -- **col 0 = Left arm, col 1 = Right arm** (the one deliberate layout change vs. `plotter.py`'s L/R row pairing), 7-color `jet` legend (J1-J7) shared across the figure, same convention as `plotter.py` |
| `fig2_qp_data` | 7 stacked rows: slacks (R/L), CBF shadow prices, joint-limit shadow prices, loop frequency, safety margin, min. distance |
| `fig3_task_error_adaptation` | Cartesian position/velocity tracking error + whichever dynamic-weight rows are active per `cfg.DYNAMIC_*` flags |
| `fig4_task_authority` | Normalized soft-task cost shares (damping/posture/slack) |
| `fig5_3d_trajectory` | **(2026-07-04)** 3D commanded-vs-executed gripper path, both arms. Solid line = the Cartesian reference commanded on `/arm_*/cartesian_reference` (source-agnostic -- today `trajectory_generator.py`, but works identically for any future publisher on that same contract, e.g. teleoperation); dashed line = the REAL EE pose from `/qp_debug/ee_real`. Red=Right, Blue=Left (same convention as everywhere else). Circle marker = trial start, X marker = trial end (helps orient a static, non-rotatable PDF/PNG reader). Sampled at the exact same tick/anchor as the existing Cartesian tracking-error computation (`cb_real`), so it is free -- no new subscription was needed. Saved as PDF+PNG always (`_build_fig_3d_trajectory`, `mpl_toolkits.mplot3d`); ALSO saved as a fully browser-navigable HTML (`_save_3d_trajectory_html`, free rotate/zoom/pan, `plotly.graph_objects.Scatter3d`) **only if `plotly` is installed** -- checked once at import time (`_HAS_PLOTLY`), silently skipped with a log line otherwise. `plotly` is intentionally NOT added to `package.xml`/rosdep since it is optional and purely additive: `pip install plotly` in the same Python environment enables it, no code change needed. |
| `fig6_reference_governor` | **(2026-07-04, revised)** Only emitted if `cfg.ENABLE_REFERENCE_GOVERNOR` -- 4 rows: Linear Velocity, Angular Velocity, Position Tracking Error, Orientation Tracking Error. Each row plots the COMMANDED ("raw") and GOVERNED curve per arm (Red=Right, Blue=Left; commanded=lighter/thinner, governed=solid/bold) PLUS a dashed grey horizontal line at that row's governor-configured ceiling (`cfg.GOV_V_MAX_LIN`, `GOV_V_MAX_ANG`, `GOV_E_MAX_POS`, `GOV_E_MAX_ORI`). Answers "how did the governor reshape the input trajectory" directly -- whenever the raw curve pokes above the dashed ceiling while the governed curve stays clipped at/below it, that is the governor visibly acting. Reconstructed ENTIRELY from data already on the wire: `gov_callback` (in `offline_plotter.py`) recovers `governed = raw - diff` algebraically from the EXISTING `/qp_debug/governor` diff payload (`main_qp_controller.py`'s publisher was NOT touched) combined with the already-tracked raw reference (`self.ref_right/left`) and the latest cached real EE pose/orientation from `/qp_debug/ee_real` (`cb_real` now unconditionally caches `last_real_pos_r/l`, `last_real_rpy_r/l` -- even outside a recording trial, so the very first governor sample of a new trial has a valid anchor). Orientation tracking error uses the SAME geodesic construction as `ReferenceGovernor._clamp_orientation_error` (`_geodesic_angle` static helper, `pin.log3` with a near-pi singularity guard) -- read-only, never fed back into control. Zero new ROS topics or publishers were added for this. Previous version (raw-minus-governed DIFFERENCE only) superseded -- the absolute view was requested as clearer for a reader unfamiliar with the internals. |

**Output location**: `cfg.OFFLINE_PLOT_ROOT_DIR` (default
`~/exchange/ros2-ws/triago_offline_plots/`), one timestamped subfolder per
trial (`trial_YYYYMMDD_HHMMSS/`), each containing both a `.pdf` (vector, for
LaTeX) and a `.png` (quick preview) per figure.

**New config (`config.py` §7)**: `OFFLINE_RECORD_TRIGGER_TOPIC`,
`OFFLINE_PLOT_ROOT_DIR`, `OFFLINE_PLOT_POST_TRIGGER_S` -- single source of
truth shared by `trajectory_generator.py` (publisher) and
`offline_plotter.py` (subscriber) so neither hard-codes the topic name.

**Registered as a `ros2 run` entry point** in `CMakeLists.txt`, alongside
`plotter.py`: `ros2 run triago_control offline_plotter.py`.

---

### 9.12 World-scene YAML loading (2026-07-04) -- replaces hard-coded obstacles

**Problem**: the workspace obstacles (table + red/blue graspable cylinders +
optional virtual wall) used to be hard-coded numeric constants in
`config.py` §6 (`TABLE_POS`, `RED_CYLINDER_POS`, `BLUE_CYLINDER_POS`,
`CYLINDER_SIZE`, `TABLE_SIZE`, `WALL_POS`, `WALL_SIZE`), read DIRECTLY by
`collision_manager.py` and `visualization_engine.py`. The SAME numbers were
ALSO independently hard-coded in THREE more places that had to be kept in
sync by hand: `goal_set.py`'s `GoalSet.__init__` default `cylinders` dict,
`head_control/config.py`'s `GT_RED_CENTER`/`TABLE_CENTER_WORLD`/etc
(ground-truth for the perception pipeline's diagnostics), and this node's
OWN `self.cylinder_model` dict for the Gazebo LinkAttacher plugin in
`main_shared_autonomy.py`. Testing a new Gazebo world (different table pose,
different cylinder size/count) meant hand-editing 3-4 files with no
guarantee they'd agree -- and no way to keep multiple world variants around
at once. There was even a dead, never-wired flag (`cfg.PINHOLE_TASK`) left
over from an earlier, unfinished attempt at exactly this idea.

**Design**: one interchange format -- a small YAML file per world scenario,
under `config/worlds/<world_name>.yaml` -- fully describing that world's
static obstacle layout (shape/pose/size/color/collision-on-off) plus which
named obstacle plays the "red"/"blue" grasp role. See
`triago_control/qp_controller/world_loader.py`'s module docstring for the
full schema, and `config/worlds/no_obstacle.yaml` (renamed 2026-07-04 from
`bimanual_default.yaml` -- see §9.13 below) for the reference world
(reproduces the OLD hard-coded values exactly -- verified numerically
byte-for-byte against the previous constants, this refactor changes NO
runtime behavior for the default world).

**New module**: `world_loader.py` -- `ObstacleSpec` (one obstacle: name,
role, shape, pose, size, color, collision) + `WorldScene` (the parsed YAML:
a list of `ObstacleSpec` + a `grasp_roles: {red: <name>, blue: <name>}`
dict) + `load_world(world_name)` (resolves via ament share dir first, then a
source-tree fallback, mirroring `trajectory_generator.py`'s existing
`config_file` resolution convention).

**Wiring** (every consumer takes an OPTIONAL `world_scene` -- omitting it
falls back to the exact old hard-coded-constants code path, so any caller
not yet updated keeps working unchanged):
- `collision_manager.CollisionManager.build_collision_model(..., world_scene=None)`:
  loops `world_scene.static_obstacles` generically, creating one hppfcl Box/
  Cylinder per entry (replaces the old table/red/blue/wall bespoke blocks).
  Resolves `self.red_cyl_id`/`self.blue_cyl_id` (relied on everywhere --
  grasp state machine, shared_autonomy_handler, visualization_engine) from
  `world_scene.grasp_roles`, and `self.table_id` from the obstacle whose
  `role=="table"` (previously `detach_object` assumed the table was always
  `workspace_obstacle_ids[0]` -- now explicit and name/role-based, robust to
  a future world listing obstacles in a different order).
- `visualization_engine.VisualizationEngine(..., world_scene=None)`:
  `publish_obstacle_marker`/`publish_wall_marker`/`color_collision_model`/
  `restore_object_color` all iterate/color from `world_scene` generically
  (replaces the old `"red" in name` / `"blue" in name` string-matching
  heuristic with a direct YAML color lookup).
- `main_qp_controller.py` / `main_shared_autonomy.py`: both declare a
  `world_name` ROS parameter (default `'no_obstacle'`, see §9.13) and call
  `load_world(world_name)` at startup, logging which scene + how many
  obstacles were loaded. **Both nodes must be given the SAME `world_name`**
  (they must agree on where the cylinders are) -- e.g.:
  ```bash
  ros2 run triago_control main_qp_controller.py --ros-args -p world_name:=no_obstacle
  ros2 run triago_control main_shared_autonomy.py --ros-args -p world_name:=no_obstacle
  ```
- `main_shared_autonomy.py`'s `GoalSet` is now constructed from a new
  `_cylinders_from_world_scene(world_scene)` static helper (maps
  `world_scene.obstacle_for_role('red'/'blue')` to GoalSet's expected
  `{'pos', 'height', 'radius', 'cbf_name'}` shape; falls back to `GoalSet`'s
  own original hard-coded table if the scene doesn't define both roles) --
  verified numerically to produce the IDENTICAL dict `GoalSet.__init__`'s
  old hard-coded default used to build. Its `self.cylinder_model` dict (the
  Gazebo LinkAttacher plugin's model names) is now `world_scene.grasp_roles`
  directly, replacing a second independent hard-coded `{'red':
  'red_cylinder', 'blue': 'blue_cylinder'}` dict.

**Explicitly OUT of scope for this pass** (per instruction):
- The Gazebo launch command is UNCHANGED
  (`ros2 launch triago_gazebo ... world_name:=tutorial`) -- this module has
  NO connection to Gazebo; it only describes, on the Pinocchio/hppfcl/RViz
  side, the SAME layout the chosen `.world` file already spawns. Keeping a
  new world's YAML in sync with its matching `.world` file's `<pose>`/
  `<geometry>` blocks is a manual, single-file bookkeeping step (see the
  `gazebo_world_file` field, informational only, not read by this loader).
- `trajectory_endpoints.yaml` (open-loop test presets) is untouched -- these
  new worlds are teleoperation-driven, not open-loop-preset-driven; some
  presets (e.g. `local_minima_table`) assume the DEFAULT table pose and
  would need a per-world variant if ever reused against a world that moves
  the table -- not addressed here.
- `head_control/config.py`'s OWN duplicated ground-truth constants
  (`GT_RED_CENTER` etc.) are NOT yet sourced from the world scene -- still a
  4th, separate hard-coded copy, flagged as a known follow-up, not done in
  this pass (the head/perception subsystem was out of scope for this
  request).
- Camera-fed (perception-driven) obstacles: `ObstacleSpec.source` /
  `CollisionManager.update_dynamic_obstacle(...)` (a live-pose-mutation
  entry point generalizing the pattern already used by
  `detach_object`/`add_attached_object_pairs`) were proposed in the planning
  discussion but NOT implemented -- deferred to a future request once
  camera-driven obstacles are actually needed.

**Bugfix #2 (same day, reported by operator via Meshcat -- the gripper mesh
"blinks" orange/purple): pre-existing, unrelated to the world-scene
refactor** (last touched in commit `71bcb00`, well before this session).
Two-part root cause, found in two passes:

1. (Padding pass, insufficient alone) `collision_manager.py`'s gripper
   COLLISION box and `visualization_engine.add_gripper_visual_boxes`'s
   gripper VISUAL box shared the exact same parent joint, placement, AND
   size -- perfectly coincident surfaces. Fixed by inflating the visual box
   by a symmetric `GRIPPER_VISUAL_BOX_PADDING = 0.005` m margin.
2. (Actual root cause) The visual box was set to a TRANSLUCENT orange
   (`alpha=0.4`) UNCONDITIONALLY at creation time -- i.e. permanently
   visible even in the DEFAULT (no grasp) state, not just as the
   grasp-intent indicator `paint_grasp_intent` intends it to be. Sitting
   permanently over the always-visible red/blue collision box, this created
   a PERMANENT overlap of two semi-transparent meshes. WebGL/three.js (what
   Meshcat draws with) does not reliably depth-sort multiple overlapping
   TRANSPARENT objects -- unlike fully OPAQUE ones, which DO depth-sort
   correctly -- so draw order between the two near-coincident translucent
   meshes flipped frame-to-frame regardless of the small padding gap from
   fix #1. Orange-over-red (right gripper) and orange-over-blue (left
   gripper, where red+blue channels blend under the translucent overlay --
   the reported "purple") explains both reported colors. A SECOND, related
   bug in the same code path: `restore_object_color` (called on grasp
   release) only cleared `overrideMaterial`, which has no effect on a
   synthetic (non-URDF) box -- so after any grasp+release the box stayed
   stuck fully-opaque orange for the rest of the session.

   Fixed by making the visual box start fully TRANSPARENT (`alpha=0.0`,
   invisible) instead of translucent -- `paint_grasp_intent` already sets
   it to fully OPAQUE (`alpha=1.0`) when signaling an actual grasp, which
   DOES depth-sort correctly against the collision box (no flicker during a
   real grasp either) -- and `restore_object_color` now explicitly sets it
   back to `alpha=0.0` (matched by exact name `gripper_{side}_visual_box`,
   not just the `overrideMaterial` flag) on release, closing the "stuck
   orange forever" bug too.

**New diagnostic tool (same day, operator report: "the capsules are not
perfectly aligned to the link... even if the cylinder radius is bigger than
the link radius, the link mesh goes outside")**: `scripts/qp_arm_teleop/
capsule_alignment_audit.py`. This is the signature of a LATERAL
MISALIGNMENT, not an undersized radius -- `calculate_offsets` builds each
capsule as a straight joint-to-joint segment (dominant-axis snapped), but
the real CAD mesh's centerline is not always collinear with that line, so a
bigger radius helps uniformly around the (possibly off-center) axis but
cannot fix a genuine sideways offset. The audit script quantifies this
PRECISELY instead of eyeballing it: it calls the REAL
`CollisionManager.calculate_offsets()` (zero drift vs. what
`main_qp_controller.py` actually builds), extracts every visual mesh
VERTEX for each link (transformed into the same joint-relative frame the
capsule lives in), and computes exact point-to-segment distance minus the
capsule radius -- vertex-only checking is exact here (not approximate)
because point-to-line distance is convex, so its max over a flat triangle
is always attained at a vertex. Reports, per link: worst-case protrusion in
mm, and WHERE along the capsule's own axis (t=0..1) it occurs -- a
protrusion concentrated near one end (t near 0 or 1) points at a
translational/kink fix near that joint; a protrusion spread evenly across
t points at a genuinely undersized radius. Pure offline Pinocchio/hppfcl
audit against a URDF file (defaults to the repo's own
`triago_extracted.urdf`, or a freshly-dumped live URDF via `ros2 param get
/robot_state_publisher robot_description`) -- no ROS/Gazebo needs to be
running. Registered in `CMakeLists.txt` alongside the other scripts.

**Bugfix #3 (same day, operator report: "now cylinder blinks... red and
blue")**: SAME root mechanism as #2, on a DIFFERENT pair of meshes.
`color_collision_model` painted the arm collision capsules (`right_geom_ids`/
`left_geom_ids` -- visually cylinder-like, rounded-cap tubes; also the
ground plane and the table/red/blue workspace obstacles) at `alpha=0.8`
(translucent). These sit directly under the ALWAYS-rendered green "ghost"
skin mesh (`self.vmodel`, `alpha=0.3` -- the real URDF visual geometry, see
`__init__`) -- another permanent translucent-over-translucent overlap, same
WebGL non-deterministic-depth-sort instability as bugfix #2, this time
showing as the right arm (red) / left arm (blue) capsules flickering.
Fixed by making EVERY color in `color_collision_model` fully OPAQUE
(`alpha=1.0` -- capsules, head capsules, ground, and every
`workspace_obstacle_ids` entry including the table/red/blue cylinders,
regardless of whether the alpha came from the world-scene YAML's `color`
field or the hard-coded fallback) -- opaque-vs-translucent pairs sort
deterministically, so there is no longer any translucent-vs-translucent
pair anywhere in the collision model. `restore_object_color`'s `default`
color (used when a grasp is released) is likewise now forced to
`alpha=1.0` for the same reason. Net effect: every mesh in `cmodel` is now
either fully opaque or fully invisible, never partially transparent --
eliminating this entire class of flicker bug at its source rather than
patching it mesh-by-mesh.

**Bugfix #1 (same day, reported by operator via Meshcat -- a solid black wall
was visible even though the default world's `virtual_wall` has
`collision: false`)**: the world_scene branch of `build_collision_model`
originally created EVERY obstacle's hppfcl geometry unconditionally, and
only used `obs.collision` to skip the pair/id bookkeeping AFTER creation --
unlike the old `if cfg.WALL_COLLIDER:` gate, which never created the wall's
geometry at all when disabled. Since `cmodel` is exactly what Meshcat
renders (`displayCollisions(True)`), a "disabled" wall still existed in the
scene -- and rendered BLACK because `color_collision_model` only colors
`workspace_obstacle_ids`/`wall_id`, neither of which a skipped-bookkeeping
wall was ever added to, so it kept hppfcl's uncolored default. Fixed by
moving the `if not obs.collision: continue` check to BEFORE geometry
creation, restoring "collision: false means this geometry does not exist"
semantics exactly like the original flag.

**Legacy/deprecated constants**: `config.py` §1 (`WALL_COLLIDER`,
`PINHOLE_TASK` -- the latter marked `[DEAD FLAG]`) and §6
(`TABLE_POS`/`TABLE_SIZE`/`RED_CYLINDER_POS`/`BLUE_CYLINDER_POS`/
`CYLINDER_SIZE`/`WALL_POS`/`WALL_SIZE`) are KEPT (not deleted) purely as the
legacy fallback path consumed when `world_scene=None`. New obstacles should
be added to a world YAML, not to `config.py`.

**New file**: `config/worlds/no_obstacle.yaml` (renamed 2026-07-04, see
§9.13). `package.xml` gained a
`python3-yaml` dependency (already transitively required by
`trajectory_generator.py`'s pre-existing `import yaml`, now made explicit
since `world_loader.py` is a second load-bearing import site). No
`CMakeLists.txt` change needed -- `config/worlds/` installs automatically
under the existing `install(DIRECTORY config ...)` rule.

**Follow-up (same day): the `platform` field.** The provided Gazebo
`tutorial.world` SDF revealed a FOURTH duplicated-truth object this initial
pass had missed: `placement_area`, a static, visual-only (no `<collision>`
block) yellow disk at `pose 1.000 0.0 0.701`, `radius 0.15`, `length 0.002`
-- the surface shared-autonomy's `Platform_Place` goal asks the operator to
set a grasped cylinder down on (see `goal_set.py`'s `PLATFORM_POSE`/
`PLATFORM_RADIUS`/`PLATFORM_THICKNESS`/`PLATFORM_PLACE_MARGIN`, previously
hard-coded and independent of everything else).

Per explicit instruction, this is **NOT an obstacle** -- it has no collision
geometry, is never added to `CollisionManager`'s `cmodel`, and the operator
already sees the actual disk directly in Gazebo/RViz (rendered by the
`.world` file itself), so nothing needed to be added on the RViz-marker
side. It was therefore deliberately kept OUT of `static_obstacles` (which
exists specifically for "things the collision model builds geometry for")
and added as a separate top-level YAML field instead:
```yaml
platform:
  pose: [1.000, 0.0, 0.701]
  radius: 0.15
  thickness: 0.002
  place_margin: 0.03
```
New `world_loader.PlatformSpec` dataclass (`pose`, `radius`, `thickness`,
`place_margin`) + `WorldScene.platform: Optional[PlatformSpec]`.
`GoalSet.__init__` gained an optional `platform=None` parameter that, when
given, overrides the class-level `PLATFORM_POSE`/`PLATFORM_RADIUS`/
`PLATFORM_THICKNESS`/`PLATFORM_PLACE_MARGIN` defaults with the loaded
world's values (verified numerically identical to the old hard-coded
defaults for `bimanual_default.yaml`, i.e. behavior-unchanged); `None`
(the default) keeps the original hard-coded constants for any caller not
yet passing a world scene. `main_shared_autonomy.py` now passes
`platform=self.world_scene.platform` alongside its existing `cylinders=`
argument when constructing `GoalSet`.

---

### 9.13 World rename: `bimanual_default` -> `no_obstacle` (2026-07-04)

**Context**: kicking off a multi-world user-study plan (novice teleoperation
under 3 assistive-feedback conditions, across ~5 world scenarios of
increasing difficulty). `bimanual_default` is World 1 -- the free-space
baseline, no extra obstacles -- and is being renamed to `no_obstacle` to
read clearly as "the easy/control scenario" once World 2-5 (`forced_side_
grasp`, `narrow_corridor`, `ambiguous_targets`, `cluttered_bimanual`, ...)
exist alongside it. Pure rename -- NO obstacle poses/sizes/colors changed;
still byte-for-byte the same scene as the original hard-coded
TABLE_POS/RED_CYLINDER_POS/BLUE_CYLINDER_POS/CYLINDER_SIZE constants.

**Naming convention established for every future world in the study set**:
the YAML filename ALWAYS matches the Gazebo `.world` filename (minus
extension) -- e.g. `no_obstacle.yaml` <-> `no_obstacle.world`. This removes
any ambiguity once several worlds exist side by side.

**Changed** (YAML-side, this repo):
- `config/worlds/bimanual_default.yaml` deleted, replaced by
  `config/worlds/no_obstacle.yaml` (`world_name: "no_obstacle"`,
  `gazebo_world_file: "no_obstacle.world"`).
- `main_qp_controller.py` / `main_shared_autonomy.py`: `world_name` ROS
  parameter default changed `'bimanual_default'` -> `'no_obstacle'`.
- `world_loader.py`, `config.py`, `README.md`: every docstring/comment
  example updated to reference `no_obstacle` instead of `bimanual_default`/
  `tutorial`.

**NOT changed by this repo** (lives in the `triago_gazebo` package, not
shared with the assistant -- see the corresponding operator-side
instructions given alongside this commit):
- The actual Gazebo `.world` SDF file itself must be renamed
  `tutorial.world` -> `no_obstacle.world` on disk (`<world name="...">`
  INSIDE the SDF does not need to change -- Gazebo's `world_name:=` launch
  argument matches the FILE name, not the SDF's internal `<world name=...>`
  attribute, which is free to stay `bimanual_manipulation_world` or
  whatever it already is).
- Any `triago_gazebo` launch file default value / world-file search path
  that hard-codes `tutorial` needs updating to `no_obstacle` too, if such a
  default exists (operator-side, package not in this repo).

**Launch command changes from**:
```bash
ros2 launch triago_gazebo triago_gazebo.launch.py ... world_name:=tutorial
```
**to**:
```bash
ros2 launch triago_gazebo triago_gazebo.launch.py ... world_name:=no_obstacle
```
with the two ROS-side nodes' `world_name` parameter following suit (now
defaulted, no explicit `-p world_name:=...` needed unless overriding).

---

## 10. Adaptive Scheduling (shadow-price feedback)

- **Decoupled slack weighting**: each arm's slack weight drops (toward `BASE_WEIGHT_SLACK=5`) when its shadow price grows, letting the slack absorb more tracking error near obstacles. In free space it rises (toward `MAX_WEIGHT_SLACK=50`) for tighter tracking.
- **Dynamic gamma (CLF)**: the CLF convergence rate γ drops exponentially with the collision Lagrangian λ_col, low-pass filtered (τ=0.125s). This gives tracking priority in free space but yields to safety near obstacles.

---

## 11. Shared Autonomy Architecture

The `main_shared_autonomy.py` node implements:
- **Bayesian belief estimation** over a discrete goal set (with goal **exclusion** support)
- **Local QP policy** (separate from the safety QP) for constrained intent following
- **Grasp state machine**: SHARED_AUTONOMY → PRE_GRASP → GRASP_ALIGN → GRASP_APPROACH → GRASP_CLOSE → LIFT → HOLDING (abort path: → ABORT_RETREAT)
- **Alpha-blending** between human teleop input and autonomous policy (WIP)
- Publishes cartesian references consumed by `main_qp_controller.py`

### 11.0 Bimanual: TWO independent state machines (2026-06-29)

Each arm has its OWN `GraspStateMachine` + `BeliefEstimator` + grasped-color +
active-goal + goal_set placement context (`self._sm['right'/'left']`,
`self._be[...]`, `self._ctx_*`). `self.grasp_sm` / `self.belief_estimator` always
POINT at the ACTIVE arm's instance, so `timer_callback` is unchanged; the inactive
arm's FSM/belief are simply never stepped (frozen) until reactivated.
`_switch_active_arm(new_arm)` (called on the left-button double-click) saves the
leaving arm's context and restores the entering arm's — so e.g. *grasp Red with
right → switch to left → grasp Blue* works: the right FSM stays HOLDING(Red) while
the left independently runs SHARED_AUTONOMY→grasp. Goal exclusion is the **union**
(`_update_goal_exclusions`): a color held by EITHER arm is excluded from both arms'
beliefs; Platform is demandable for an arm only if THAT arm holds something. The
collision world declares a **cylinder-vs-cylinder CBF pair** so two held cylinders
cannot inter-penetrate. The node DRIVES the switch and PUBLISHES
`/shared_autonomy/active_arm` (it no longer subscribes to it — that self-echo was
removed). Failed grasps (align/approach timeout) → **ABORT_RETREAT**: back out
along the reverse approach axis (gripper open) while the cylinder CBF bypass stays
active, then restore CBF.

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

**Single-publish marker consolidation (2026-06-29)**: ALL markers on `/shared_policy_markers`
(green `policy_grippers`, `goal_poses`, `grasp_ready_cue`, grasp-guidance arrows) are collected
into ONE `MarkerArray` and published in a SINGLE `pub_markers.publish()` per timer tick. Before,
each marker type was a separate publish → ~100 msg/s on a queue-10 topic → RViz subscriber queue
starvation made the green gripper look stale while goal markers refreshed fine. Internal `_build_*`
methods return marker lists; legacy `publish_*` wrappers are kept but the loop uses the
consolidated path. **Do not revert to separate publishes.**

**Ghost-marker fixes (2026-06-29, round 2)**: two root causes of markers "stuck forever" on
screen were found and fixed:
1. **Stale excluded-goal position** — `_build_goal_pose_markers` drew a faded (not hidden) gripper
   for EXCLUDED goals (e.g. the just-grasped cylinder's own Top/Side), anchored at
   `GoalSet.cylinders[color]['pos']` — the cylinder's ORIGINAL TABLE SPAWN position, only updated
   by `relocate_cylinder()` at release. While held, that anchor is stale (the object moved with the
   gripper), so the "faded" marker looked like a ghost frozen at the pick-up spot. Fix: excluded
   goals are now explicitly `Marker.DELETE`d (`_delete_gripper_markers`) instead of faded.
2. **Shifting marker ID in `qp_visualizer_tutorial.QPVisualizer.publish_debug`** — the blue
   commanded gripper and grey frozen gripper used `id=idx`, where `idx` accumulates from
   CONDITIONAL markers earlier in the same tick (collision line/text, joint-limit sphere/text —
   each appears only sometimes). RViz identifies a marker by `(ns, id)`; since the id shifted tick
   to tick even though the ns never changed, every tick created a NEW marker instead of overwriting
   the previous one, and the old (ns, id) was never revisited (no lifetime, no DELETE) — permanent
   ghost, independent of arm switching. Fix: both grippers now use a FIXED `id=0` (unique
   per-namespace, cannot collide with the `"qp_debug"` ns markers). Also explicitly `DELETE`s each
   gripper's markers when its pose becomes `None` (e.g. right after an arm switch, before the first
   new-arm reference arrives).
3. Both `main_shared_autonomy` (3 topics) and `QPVisualizer` (`/qp_debug_visualization`,
   `/teleop_debug_visualization`) now run an independent periodic `Marker.DELETEALL` sweep every
   `MARKER_CLEANUP_PERIOD_S=3.0` s as defense-in-depth against any other orphaning path.

**Guidance / robot-policy gripper topics (updated 2026-06-29)**: BOTH predictive grippers now live
on their OWN dedicated topics (previously the robot-policy gripper was mixed into
`/shared_policy_markers`):
- `/guidance_policy_marker` (ns `guidance_policy`, **light blue** — was yellow) — the human-side
  counterpart. **Reference-anchored** (`current_T_user`), integrates the belief-weighted
  **user-policy** blend `pi_blend = Σ belief[k]·user_policies[k]` — exactly the velocity field the
  haptic `F_guide` renders onto the handle.
- `/robot_policy_marker` (ns `policy_grippers`, **light green**) — the robot-side counterpart.
  **EE-anchored** (`current_T_EE`), integrates the **ee-policy** blend (`pi_max`).

In test mode (`current_T_user == current_T_EE`) the two coincide. `/shared_policy_markers` now
only carries: belief-opacity goal grippers + the PRE_GRASP pulsing cue (grasp-guidance move/rotate
arrows were removed — they cluttered the view without adding clarity; the builder methods
`_build_grasp_guidance` / `_build_clear_grasp_guidance` remain in the code, unused, in case
they're wanted again). All three marker topics receive a periodic `Marker.DELETEALL` sweep every
`MARKER_CLEANUP_PERIOD_S=3.0` s (`main_shared_autonomy._sweep_all_markers`) so any RViz marker that
got stuck (lifetime never re-triggered, or a namespace changed between versions) self-heals
instead of lingering forever; the very next control tick republishes what should actually be
visible, so at most one frame is dropped.

### 11.5 Shared-autonomy TWIST BLENDING architecture (2026-07-03)

**Motivation**: the operator wanted a shared-autonomy mode where the robot's
Cartesian reference is a continuous BLEND of the user's own hand motion and a
belief-weighted assistive policy toward the most likely goal — instead of the
existing "haptic Virtual Fixture" mode (§13.5) where the user's raw reference
is fed to the QP unmodified and ALL assistance is rendered purely as force at
the Haption handle.

**Bug history (why this took multiple passes to land correctly)**:
1. A first attempt computed the blend entirely inside a NEW haptic-side script
   (`haptic_force_manager_blending_tutorial.py`) and published it to
   `/arm_right/blended_cartesian_reference` — a topic `main_qp_controller.py`
   never subscribed to. The blend was computed but had ZERO effect on the
   robot (silently orphaned topic).
2. That same attempt also blended at the POSE level (`ref_blended = (1-alpha)*
   pos_user + alpha*(pos_user + policy_twist*dt)`), recomputed from scratch
   every tick — since it was never integrated persistently, a stationary user
   never accumulated any real motion toward the goal no matter how high alpha
   got.
3. **Root cause once actually investigated**: `main_shared_autonomy.py`
   ALREADY had the correct twist-level blend formula
   (`target_twist = (1-alpha)*current_v_h + alpha*tick_output.target_twist`,
   inside `timer_callback`) and already PERSISTENTLY integrates its output
   every tick via `integrate_twist` (the exact mechanism trusted for grasp
   execution / `POLICY_BELIEF_TEST`) — but (a) `compute_alpha` was a stub that
   unconditionally `raise NotImplementedError`, and (b) even with alpha
   implemented, the blended `target_twist` was only ever PUBLISHED to
   `/arm_*/cartesian_reference` when `POLICY_BELIEF_TEST or grasp_exec` — in
   ordinary teleop the blend was computed and thrown away every tick, and
   `teleop_triago_clutch.py` remained the sole publisher of the user's raw pose.

**Final architecture** (config-flag-driven, per operator instruction — no
new script for `teleop_triago_clutch.py`, and no local `self.BLENDING` flag
on `SharedControlNode`):

```
                          cfg.BLENDING (qp_controller/config.py)
                     SINGLE SOURCE OF TRUTH, read by BOTH nodes below
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
        teleop_triago_clutch.py                     main_shared_autonomy.py
        (haption_teleoperation)                      (triago_control)
                     │                                           │
   BLENDING=False:                              BLENDING=False:
     publishes user pose on                       only subscribes (belief
     /arm_*/cartesian_reference                    inference); only PUBLISHES
     (unchanged legacy behavior)                    during grasp_exec / TEST
                     │                                           │
   BLENDING=True:                                BLENDING=True:
     publishes user pose on                        subscribes on
     /arm_*/user_cartesian_reference                /arm_*/user_cartesian_reference;
     INSTEAD (so the two nodes                      alpha = compute_alpha(b_max);
     never race for the same topic)                 v_blend = (1-alpha)*v_user
                                                       + alpha*pi_policy;
                                                     PERSISTENTLY integrates v_blend
                                                       every tick (same integrate_twist
                                                       used for grasp exec);
                                                     becomes the SOLE, ALWAYS-ON
                                                       publisher of the real
                                                       /arm_*/cartesian_reference
                                                     also publishes
                                                       /shared_autonomy/blend_debug
                                                       every tick (19 floats:
                                                       [alpha, v_user(6), v_policy(6),
                                                        v_blend(6)]) -- single source
                                                       of truth for "who commanded what"
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                            ▼
                             /arm_*/cartesian_reference
                                            ▼
                              main_qp_controller.py (QP CLF-CBF)
                                     [UNCHANGED -- still only ever
                                      reads /arm_*/cartesian_reference;
                                      no topic-routing awareness needed here]
```

**`cfg.BLENDING` and its tuning constants** (`qp_controller/config.py`, §1):
```python
BLENDING = False          # master switch (see topic-routing table above)
ALPHA_MAX = 0.80          # hard cap on autonomy authority (user retains >= 20%)
ALPHA_GAMMA = 0.5         # <1 = alpha ramps toward ALPHA_MAX quickly once belief
                          #   is "sufficiently high" (not just near-certainty)
ALPHA_LPF_COEFF = 0.08    # LPF on alpha -- guarantees C0 continuity in the
                          #   blended reference even if belief itself jumps
```

**`compute_alpha(b_max)` formula** (`main_shared_autonomy.py`):
```
x = 0                                                 if b_max <= 1/N_active_goals (uniform)
x = clip((b_max - 1/N_active_goals) / (1 - 1/N_active_goals), 0, 1)   otherwise
alpha_raw = ALPHA_MAX * x**ALPHA_GAMMA
alpha = LPF(alpha_raw, coeff=ALPHA_LPF_COEFF)     # self.alpha_lpf, persistent state
```
`N_active_goals` excludes goals currently pinned to 0 by `BeliefEstimator.
get_excluded_goals()` (e.g. the already-grasped color's own goals), so the
"uniform belief" baseline always reflects the TRUE number of live candidates.

**Why F_sync (haptic side) must read the PURE user pose, not the blended
one**: `haptic_force_manager_blending_tutorial.py`'s `target_cb`/`target_cb_
left` subscribe to `/arm_*/user_cartesian_reference` when `cfg.BLENDING=True`
(the SAME conditional topic teleop_triago_clutch.py now publishes to) — NOT
the now-blended `/arm_*/cartesian_reference`. If it read the blended pose,
F_sync would tether the handle toward a target that already includes the
user's own contribution, hiding exactly the divergence (autonomy pulling the
real EE away from the user's raw intent) the operator wants to FEEL through
the handle.

**Why the debug topic exists**: `/shared_autonomy/blend_debug` is published
EVERY tick (alpha=0, v_policy=tick_output.target_twist even when `cfg.
BLENDING=False`) so the haptic-side "Authority Share" plot always has a
well-defined, non-recomputed signal — the exact numbers that did (or, if
disabled, WOULD) command the robot, never a second independently-drifting
copy of the same math.

**Cross-package dependency**: `haption_teleoperation/package.xml` now
`<depend>triago_control</depend>` (both `teleop_triago_clutch.py` and
`haptic_force_manager_blending_tutorial.py` `import triago_control.qp_
controller.config as cfg`).

**Known limitation (flagged, not solved in this pass)**: the pure-user
integrator (`teleop_triago_clutch.py`, still owns its own `ref_pos`/`ref_rot`
state regardless of which topic it publishes to) keeps integrating from the
Haption twist independently of where the robot actually ends up. At
sustained high alpha the user's own pure pose and the real EE can drift apart
over time (bounded only by F_sync's spring pulling the HANDLE, never the
user's internal integrator state). Re-anchoring logic (e.g. periodically
snapping `ref_pos`/`ref_rot` toward the real EE when divergence exceeds a
threshold) is a natural next step if this proves disorienting in practice —
not implemented now since it wasn't part of the agreed plan for this pass.

### 11.6 TWIST BLENDING usability fixes (2026-07-03)

Follow-up on §11.5, based on operator hands-on feedback after testing the
architecture: (a) confusion about what the two light-blue-ish RViz grippers
actually represent, (b) the autonomy having too much authority once a goal
was locked in ("almost anything can be done" — not desirable as the primary
feel), and (c) the robot being unable to CONCLUDE the approach near the goal.

**1. New `/blended_reference_marker` RViz gripper.** Renders the LITERAL pose
integrated from `target_twist` (the blended twist actually sent to the QP)
and about to be published on `/arm_*/cartesian_reference` — i.e. exactly what
the robot is tracking this tick, not a speculative lookahead. Built at the
same point in `timer_callback` where `T_virtual_ref` (the 20 ms lookahead
pose) is computed, right before it's packed into the outgoing
`Float64MultiArray`. Same style/color as the existing `/guidance_policy_marker`
(light blue, `rgb=(0.3, 0.7, 1.0)`, `create_gripper_markers(..., ns=
"blended_reference", ...)`) so the two can be visually compared side-by-side:
- `/guidance_policy_marker` — pure USER intent (reference-anchored, integrates
  `pi_blend_user` — the belief-weighted user-policy blend, NOT the actual
  blended twist).
- `/blended_reference_marker` — what is ACTUALLY being sent to the QP
  (EE-anchored, integrates the true `target_twist`).

Both topics are published independently and BOTH stay live simultaneously
(the operator manually toggles visibility in RViz to judge which is more
intuitive to show — no code-level exclusivity was added). Included in the
existing `_sweep_all_markers` DELETEALL self-heal pass (every
`MARKER_CLEANUP_PERIOD_S=3.0s`) alongside the other three marker topics.

**2. `cfg.ALPHA_MAX` 0.80 → 0.60 — user authority floor raised 20%→40%.**
Operator report: once a goal was confidently picked, the blend gave the
autonomy so much authority that the user could effectively only "change the
robot's mind about the goal" but couldn't meaningfully steer once committed —
not a desirable default feel. Lowering the ceiling by 20 percentage points
raises the user's GUARANTEED floor from `1-0.80=20%` to `1-0.60=40%` at every
belief level (the `compute_alpha` formula and `ALPHA_GAMMA` ramp shape are
unchanged — only the ceiling moved).

**3. Smooth proximity boost — fixes "can't conclude the task near the
goal".** Root cause: `pi_policy` (the QP-constrained policy twist, `tick_
output.target_twist` in `_shared_autonomy`) is itself a CLF-style
proportional term that naturally shrinks toward zero as the EE approaches the
goal (by design — no overshoot). Combined with `ALPHA_MAX` capping the blend
weight, the ALREADY-SMALL near-goal policy twist got a bounded fraction
applied to it, and the resulting `v_blend` could be too weak to close the
final few centimeters. New mechanism (`compute_alpha`, now takes an optional
`pos_error` argument — the real EE-to-active-goal distance, already computed
earlier in the same tick):

```
gain = 1.0 + smoothstep(FAR - pos_error, 0, FAR - NEAR) * (MAX_GAIN - 1.0)
alpha_boosted = min(alpha_raw * gain, ALPHA_PROXIMITY_CAP)
```
- `cfg.ALPHA_PROXIMITY_FAR=0.20m` / `cfg.ALPHA_PROXIMITY_NEAR=0.05m`: the gain
  ramps smoothly (C1-continuous smoothstep, `_smoothstep` static helper) from
  `1.0` (no boost, beyond 20cm) to `cfg.ALPHA_PROXIMITY_MAX_GAIN=1.5` (at/inside
  5cm), so there is no discontinuity in the blended reference as the EE
  approaches.
- `cfg.ALPHA_PROXIMITY_CAP=0.90`: hard ceiling on the BOOSTED alpha — HIGHER
  than the away-from-goal `ALPHA_MAX=0.60` (deliberately, since this is
  specifically meant to help finish the task), but still `< 1.0`, so the user
  retains at least 10% authority even at the exact moment of task completion.
- The boosted value is computed BEFORE the existing LPF (`cfg.
  ALPHA_LPF_COEFF`), so temporal smoothness is preserved exactly as before —
  no new discontinuity source was introduced.

The proximity boost is unconditional whenever `pos_error` is supplied (which
it always is, from the call site in `timer_callback` — `compute_alpha(b_max,
pos_error=pos_error)`); it has no effect (`gain=1.0`) far from any goal, so
free-space blending behavior away from a goal is unchanged by this fix.

### 11.7 User-effort authority gating (2026-07-03)

Follow-up on §11.5/§11.6. After testing with the raised `ALPHA_MAX=0.60`
floor, the operator reported the deeper issue was NOT the belief-driven
ceiling itself but that **the arm was almost blind to the user's own hand
twist**: `F_sync` (haptic side, correctly tuned, NOT to be changed) keeps the
handle near the real EE, so the user always FEELS where the robot is, but the
blended REFERENCE the robot tracks barely reflects the user's own motion —
only their inferred GOAL (via belief) had any real leverage. Root cause:
`pi_policy` (`tick_output.target_twist`) is a large, tanh-saturated velocity
(`v_max_lin=0.1 m/s`), while comfortable Haption hand motion is much smaller
(`v_max_lin_user=0.04 m/s` is the CEILING used for the user-anchored policy
solve, but typical relaxed motion is well below that) — so even at a modest
`alpha`, `alpha·pi_policy` can dominate `(1-alpha)·v_user` in the blend by a
wide margin. The user effectively could only "change the robot's mind about
the goal" (via belief), never resist or steer once a goal was inferred.

**Explicit constraint from the operator**: do NOT introduce dynamic weighting
sourced from the QP's own feedback (Lagrangian multipliers / shadow prices)
on the FORCE side — those are known to be discontinuous tick-to-tick, or need
heavy filtering that reintroduces lag. The fix below is deliberately built
from a signal that is neither of those things: the user's OWN commanded twist
norm, which is inherently smooth (integrated from a human hand) and entirely
independent of the QP solve.

**Fix — user-effort gate on `compute_alpha`** (NOT on the force/haptic side
at all; still purely a REFERENCE-level blending weight):

```
effort_raw = clip(||v_user[:3]|| / ALPHA_EFFORT_THRESHOLD, 0, 1)
effort     = LPF(effort_raw, coeff=ALPHA_EFFORT_LPF_COEFF)     # self.alpha_effort_lpf
alpha     *= (1 - effort * ALPHA_EFFORT_OVERRIDE)
```
applied to `alpha_raw` AFTER the proximity boost (§11.6) and BEFORE the final
`alpha_lpf` low-pass — so the existing continuity guarantee is preserved (no
new discontinuity source; effort is just one more smooth multiplicative term
folded into the same filtered pipeline).

`compute_alpha(b_max, pos_error=None, v_user=None)` gained the `v_user`
parameter; the call site in `timer_callback` passes `self.current_v_h` (the
raw human twist, already used everywhere else as "what the user is doing
right now").

**Tuned values (operator-selected, first trial)**:
- `ALPHA_EFFORT_THRESHOLD = 0.4` m/s — chosen as "a credible fast hand
  movement" ceiling; effort saturates to 1.0 at/above this linear speed.
- `ALPHA_EFFORT_OVERRIDE = 0.5` — at full effort, alpha is HALVED (not
  zeroed): the operator's reasoning is that fast hand motion already lowers
  the belief estimate itself (via `BeliefEstimator`'s `engagement` term and
  the twist-cost update), so roughly half of the total "policy stops
  following" effect comes from THAT channel already; this gate supplies the
  other half directly at the blend-weight level.
- `ALPHA_EFFORT_LPF_COEFF = 0.15` — independent LPF coefficient from
  `ALPHA_LPF_COEFF` (which smooths the FINAL alpha); this one smooths the
  effort signal itself before it multiplies in.

**Why this satisfies all the stated requirements**:
- Near an obstacle: the user is not typically pushing hard INTO the obstacle
  (their own twist stays small), so `effort≈0` and belief-driven assistance
  (including the CBF-aware component baked into `pi_policy`) stays at full
  strength — safety behavior is unchanged.
- When intent changes: same — turning the hand toward a new goal is
  initially "still learning" (low `b_max` -> low `alpha_raw` regardless), and
  once the user commits and moves briskly, effort rises and hands them
  authority directly, on top of the belief update happening in parallel.
  Both channels reinforce "let the user redirect".
  - When idle/slow: `effort≈0` -> assistance behaves exactly as sized by
  belief alone (§11.5/§11.6, unchanged) — this is deliberately where the
  system should be MOST helpful, per the operator's own framing ("guidance
  is helpful... while... almost negligible when low twist is being expressed"
  refers to the OPPOSITE case, fast motion — see below).
- When the user expresses a fast, deliberate twist: `effort→1`, `alpha`
  drops toward `0.5·alpha_belief` — the user's own motion now carries
  proportionally more weight in `v_blend`, directly addressing "the arm is
  blind to user movement."

**Not touched**: `F_sync` gains (per explicit instruction), the proximity
boost (§11.6, still applies independently), `ALPHA_MAX`/`ALPHA_GAMMA` shape
(§11.5/§11.6, unchanged), and nothing on the haptic/force side of the
architecture (§4.3 in `haption_teleoperation`'s context.md) was touched at
all — this fix is 100% contained to `compute_alpha` in `main_shared_
autonomy.py` plus the four new `cfg.ALPHA_EFFORT_*` constants.

### 11.8 Position-divergence override + bounded reference catch-up (2026-07-03)

Follow-up on §11.7. Operator feedback after testing the velocity-effort gate:
fast hand motion nudges the gripper, but the moment the hand decelerates and
is HELD at a displaced position, the robot barely continues moving there —
the operator ends up fighting `F_sync`'s restoring force (proportional to the
still-open EE↔user position gap) without the robot ever closing it. The
operator's own diagnosis, confirmed correct: this is architectural, and human
REFERENCE POSITION (not just twist) is specifically valuable as an
escape-from-local-minima mechanism — the whole point of shared autonomy
getting stuck is that the belief-driven policy alone cannot always find its
way out, and the user's hand position is exactly the extra information that
can redirect it, even (especially) once they've stopped actively steering.

**Root cause, precisely**: the published reference is built as
```
target_twist = (1-alpha)*current_v_h + alpha*pi_policy      # blend at TWIST level
T_virtual_ref = integrate_twist(current_T_EE, target_twist, dt_virtual=0.02)  # ALWAYS from current EE
```
`current_T_user` (the user's own persistently-integrated hand pose, maintained
independently by `teleop_triago_clutch.py`) never appears as a POSITION target
anywhere in this pipeline — only its derivative `current_v_h` does. The
instant `current_v_h → 0` (hand held still), `target_twist → alpha*pi_policy`
alone, and `T_virtual_ref` collapses right back onto wherever the EE currently
is — regardless of how far away the user is holding their hand. §11.7's
effort gate reacts to the SAME derivative, so it also relaxes at exactly this
moment; it could never fix this on its own.

**Fix — two complementary, purely-geometric mechanisms** (both built only
from `current_T_user` and `current_T_EE`, i.e. plain forward-kinematics-level
poses already available every tick; explicitly NOT sourced from any QP
Lagrangian multiplier or shadow price, per the operator's standing constraint
that those are discontinuous tick-to-tick or need lag-inducing filtering):

**1. Position-divergence alpha override** (`compute_alpha`, new
`pos_divergence` parameter — sustained, does NOT decay with velocity, unlike
§11.7's effort gate):
```python
div_effort = smoothstep(pos_divergence, ALPHA_DIVERGENCE_NEAR=0.05, ALPHA_DIVERGENCE_FAR=0.20)  # meters
alpha *= (1 - div_effort * ALPHA_DIVERGENCE_OVERRIDE)   # ALPHA_DIVERGENCE_OVERRIDE = 0.6
```
LPF'd independently (`ALPHA_DIVERGENCE_LPF_COEFF=0.15`, `self.
alpha_divergence_lpf`) before folding into the same filtered `alpha_lpf`
pipeline as every other gate — no new discontinuity source. Deliberately the
STRONGEST override in the pipeline (`0.6` vs. the velocity gate's `0.5`)
since it is meant to be the dominant, PERSISTENT signal — it stays active
exactly as long as the user keeps holding their hand away, not just while
it's in motion.

**2. Bounded reference catch-up** (`compute_reference_catchup`, new method —
the mechanism that actually answers "the robot must follow through to where
the hand is resting"): a gentle, CAPPED P-control pull ADDED directly onto
`target_twist` after the blend, toward `current_T_user`:
```python
pos_gap = T_user.pos - T_EE.pos
v_lin   = clip(K_CATCHUP_LIN * pos_gap * smoothstep(||pos_gap||, DEADBAND_POS=0.03, FULL_POS=0.15),
               max_norm=V_CATCHUP_MAX_LIN=0.06 m/s)
# + analogous orientation term via pin.log3, capped at V_CATCHUP_MAX_ANG=0.15 rad/s
target_twist += concat(v_lin, w_ang)
```
- The deadband (`CATCHUP_DEADBAND_POS/ANG`) means ordinary small tracking
  gaps contribute nothing — this only activates once the user has genuinely
  moved their hand away and is holding it there.
- The velocity is HARD-CAPPED (deliberately gentle — a slow, steady pull, not
  a snap toward the hand).
- Crucially, this term is **added to `target_twist`, which still flows
  through the exact same downstream integration and QP CLF-CBF as before** —
  it never bypasses safety. If an obstacle genuinely blocks the path, the CBF
  still constrains the actually-executed motion; the catch-up term can only
  ever nudge the REFERENCE the QP is asked to track, never force a collision.

**Call site** (`timer_callback`, inside the `cfg.BLENDING and new_state ==
"SHARED_AUTONOMY"` branch):
```python
pos_divergence = ||current_T_user.pos - current_T_EE.pos||
alpha = compute_alpha(b_max, pos_error=pos_error, v_user=current_v_h,
                       pos_divergence=pos_divergence)
target_twist = (1-alpha)*current_v_h + alpha*tick_output.target_twist
target_twist = target_twist + compute_reference_catchup(current_T_user, current_T_EE)
```

**New config constants** (`config.py`): `ALPHA_DIVERGENCE_NEAR/FAR/OVERRIDE/
LPF_COEFF`, `CATCHUP_DEADBAND_POS/FULL_POS`, `K_CATCHUP_LIN`,
`V_CATCHUP_MAX_LIN`, `CATCHUP_DEADBAND_ANG/FULL_ANG`, `K_CATCHUP_ANG`,
`V_CATCHUP_MAX_ANG`.

**Expected effect on the reported "fighting F_sync" feeling**: as the catch-up
term steadily closes the EE↔user gap, `F_sync`'s own restoring force (which is
proportional to that same gap) should relax in step — the two are coupled by
construction, not independently tuned. `F_sync`'s own gains were NOT touched
(per explicit instruction).

**First trial — untested, pending operator feedback**: all listed constants
are first-pass values, not yet validated hands-on. Expect follow-up tuning.

### 11.9 Orientation symmetry fix for the authority gates (2026-07-03)

Follow-up on §11.7/§11.8. Operator feedback after testing position-divergence
+ catch-up: overall the strategy felt close to satisfying, EXCEPT orientation
was almost completely ignored -- twisting the handle barely rotated the
gripper, and even the light-blue guidance gripper in RViz stayed nearly fixed
in orientation while its position was clearly pulled toward the user. Also
flagged: the plotted `ALPHA_MAX` line looked stale/wrong since `alpha`
visibly climbed above it near a goal.

**Root cause -- both authority gates were position/linear-only by
construction, not by any deliberate choice to exclude orientation:**
```python
# §11.7 effort gate (BEFORE):
v_user_lin_norm = ||v_user[0:3]||              # only ever reads LINEAR velocity
# §11.8 divergence override (BEFORE):
pos_divergence = ||pos_user - pos_EE||         # only ever reads POSITION gap
```
Neither gate ever inspected `v_user[3:6]` (the user's angular twist) or the
rotation gap between `current_T_user` and `current_T_EE`. So no matter how
hard the user turned the handle, or how far their reference orientation
drifted from the gripper's, `alpha` stayed exactly wherever belief put it --
`pi_policy`'s angular twist dominated the blend completely unopposed. This is
the exact same architectural gap already fixed for position in §11.7/§11.8,
simply never extended to the angular half of the twist.

**Fix -- extend both gates to also read orientation, combined via `max()`
with their existing linear/position counterpart (so EITHER channel alone
is sufficient to hand the user authority):**

```python
# Effort gate (compute_alpha, v_user param):
lin_effort = clip(||v_user[0:3]|| / ALPHA_EFFORT_THRESHOLD, 0, 1)          # unchanged
ang_effort = clip(||v_user[3:6]|| / ALPHA_EFFORT_ANG_THRESHOLD, 0, 1)      # NEW
effort = max(lin_effort, ang_effort)
alpha *= (1 - effort * ALPHA_EFFORT_OVERRIDE)

# Divergence override (compute_alpha, new ang_divergence param):
pos_div_t = smoothstep(pos_divergence, ALPHA_DIVERGENCE_NEAR, ALPHA_DIVERGENCE_FAR)       # unchanged
ang_div_t = smoothstep(ang_divergence, ALPHA_DIVERGENCE_ANG_NEAR, ALPHA_DIVERGENCE_ANG_FAR)  # NEW
div_effort = max(pos_div_t, ang_div_t)
alpha *= (1 - div_effort * ALPHA_DIVERGENCE_OVERRIDE)
```

`ang_divergence` is computed at the call site (`timer_callback`) as the
geodesic rotation gap between the user's held reference orientation and the
real EE's, via the same `pin.log3` / near-π-singularity guard pattern already
used elsewhere in this file (e.g. `_arm_task_error`):
```python
R_gap = current_T_user[:3,:3] @ current_T_EE[:3,:3].T
ang_divergence = pi if trace(R_gap) <= -1+eps else ||pin.log3(R_gap)||
```

**New config constants** (`config.py`):
- `ALPHA_EFFORT_ANG_THRESHOLD = 1.0 rad/s` -- scaled the same way as the
  existing linear threshold (`ALPHA_EFFORT_THRESHOLD=0.4 m/s` is ~10x the
  comfortable teleop rate `v_max_lin_user=0.04 m/s`; `1.0 rad/s` is ~10x
  `w_max_ang_user=0.10 rad/s`).
- `ALPHA_DIVERGENCE_ANG_NEAR = 0.15 rad` / `ALPHA_DIVERGENCE_ANG_FAR = 0.60 rad`
  -- deliberately set equal to `CATCHUP_DEADBAND_ANG` / `CATCHUP_FULL_ANG`
  (§11.8), so the SAME physical rotation gap triggers both the alpha override
  and the reference catch-up's angular pull -- the two mechanisms now agree on
  what counts as "the user has rotated the reference away."

**Explicitly NOT touched, per operator instruction**: `cfg.TASK_WEIGHTS_6D`
(the CLF's own position:orientation cost ratio, `[1,1,1,0.04,0.04,0.04]*10`)
was left exactly as-is. This fix operates entirely upstream of the CLF, at
the alpha/blend-weight level -- it changes how much authority the USER's
orientation gets in the blended twist, not how the QP itself weighs
orientation error once that twist is handed to it.

**Plot fix** (`haptic_force_manager_blending_tutorial.py`, Window 3): the
"Blending Factor alpha" subplot's dashed ceiling line was `cfg.ALPHA_MAX`
(the away-from-goal ceiling only), but `alpha` can legitimately exceed it
near a goal via the §11.6 proximity boost, whose true ceiling is
`cfg.ALPHA_PROXIMITY_CAP=0.90`. This looked like a bug ("the line is
deprecated") but was actually the boost working as designed. Both lines are
now drawn (orange dashed = `ALPHA_MAX`, red dotted = `ALPHA_PROXIMITY_CAP`)
so the operator can see which regime `alpha` is currently in.

**First trial -- untested, pending operator feedback**: `ALPHA_EFFORT_ANG_
THRESHOLD` and `ALPHA_DIVERGENCE_ANG_NEAR/FAR` are first-pass values (chosen
by direct analogy to the existing linear constants and the catch-up
deadband), not yet validated hands-on.

---

## 12. Current State & Known Issues

| Area | Status | Notes |
|------|--------|-------|
| QP bimanual arm control | ✅ Working | Full 6-DOF tracking with CBF safety; per-arm cost decoupling (inactive arm: 2×DAMP, MAX slack, GAMMA_MAX); grasp-boost (active arm pinned to MAX during align/approach). Posture task: gradient potential field (not a home spring), now with a PER-ARM multiplier (`posture_scale_right/left`, §9.9). **Reference Governor** (§9.8): velocity/error/acceleration/orientation bounds between the raw reference and the CLF, preserving feasibility guarantees under aggressive commands. **Local Minima Escape** (§9.9, 🔧 new/untested by operator): detects a stuck 3D position error + categorizes via shadow prices (obstacle vs. joint-limit) + applies a temporary per-arm posture-weight correction. |
| Bimanual arm switching | ✅ Working | Double-click Haption left button → switch active arm. Per-arm FSM + belief (frozen inactive arm). QP always publishes qdot for both arms (no zero-overwrite). Inactive arm CLF-held at frozen EE pose. |
| Shared autonomy (belief + grasp) | ✅ Working | Two independent GraspStateMachine + BeliefEstimator per arm. Union goal exclusion. GRASP_ALIGN tolerances relaxed 10%. Side-grasp manifold 3 cm from cylinder centre. |
| Grasp pipeline (full cycle) | ✅ Working | PRE_GRASP → ALIGN → APPROACH → CLOSE → LIFT → HOLDING. Tracking boost (MAX gamma+slack) during align. Align timeout 12s. |
| Grasp failure | ✅ Working | Clear `[GRASP FAILED]` log. ABORT_RETREAT: backs out along reverse approach axis (gripper open, CBF bypass active during retreat, then restore). |
| Post-grasp (LIFT + HOLDING + place) | ✅ Working | 9 cm slow lift → HOLDING resumes shared autonomy → Platform placement manifold → release → RELEASE_LIFT → SHARED_AUTONOMY |
| RViz visualization | ✅ Working | Light-green=EE robot policy (own topic `/robot_policy_marker`), Light-blue=reference guidance (own topic `/guidance_policy_marker`). Commanded gripper is now PER-ARM (2026-07-01, §9.4): each arm independently Blue=actively tracking / Grey=frozen (`right_frozen`/`left_frozen` ground truth via `/qp_debug/arm_frozen`) — both render blue simultaneously under `trajectory_generator.py` or dual teleop, exactly one blue+one grey in single-arm teleop. Belief-opacity goal grippers + grasp-ready cue on `/shared_policy_markers`. Grasp-guidance move/rotate arrows REMOVED (cluttered the view). Periodic DELETEALL sweep every 3s on all 3 marker topics (self-heals stuck/ghost markers). Dual belief subplot (active colored, inactive greyscale). Task-authority soft-cost plot (`/qp_debug/task_authority`). Plotter dashboard also has a NEW "Joint Positions" slider-panel GUI window (§9.5, real Pinocchio joint limits) and per-arm "Slack Weight" (R/L) traces. |
| Haption teleoperation (Virtual Fixture mode) | ✅ Working | `teleop_triago_clutch.py` + `haptic_force_manager_tutorial.py`. Force layers: F_guide (velocity-field, v_max_user 0.04/0.10) + F_fixture (position spring near goal). Handle drag during autonomous phases (KP+KD velocity-following). Active when `cfg.BLENDING=False` (default). |
| Shared-autonomy TWIST BLENDING mode | 🔧 New (2026-07-03), untested on real hardware | `cfg.BLENDING=True` (§11.5): `main_shared_autonomy.py` becomes the sole persistent publisher of `/arm_*/cartesian_reference`, integrating `v_blend=(1-alpha)*v_user+alpha*pi_policy` every tick; `teleop_triago_clutch.py` redirects to `/arm_*/user_cartesian_reference`; `haptic_force_manager_blending_tutorial.py` renders ONLY F_sync (tethered to the pure user pose) + a new "Authority Share" plot sourced from `/shared_autonomy/blend_debug`. Usability pass (§11.6): `ALPHA_MAX` 0.80→0.60 (user floor 20%→40%), smooth distance-based proximity boost so the task can be concluded near the goal (capped `ALPHA_PROXIMITY_CAP=0.90`), new `/blended_reference_marker` RViz gripper (light-blue, literal QP-bound pose) shown alongside the existing pure-user-intent `/guidance_policy_marker` for A/B comparison. User-effort gating (§11.7, first trial values): `compute_alpha` scales alpha down by how briskly the user is moving the handle (`ALPHA_EFFORT_THRESHOLD=0.4 m/s`, `ALPHA_EFFORT_OVERRIDE=0.5`), fixing the "arm blind to user twist" report — no QP Lagrangian/shadow-price feedback involved. Position-divergence override + bounded reference catch-up (§11.8, untested first trial): a SUSTAINED alpha override (`ALPHA_DIVERGENCE_OVERRIDE=0.6`) plus a new capped P-pull (`compute_reference_catchup`, `V_CATCHUP_MAX_LIN=0.06 m/s`) toward the user's held reference pose, fixing "robot doesn't follow through once the hand stops moving / user fights F_sync". |
| Head control (visual servoing) | 🔧 Active dev | `qp_head_visual_servo.py`: QP-based hand-tracking, independent loop. Starting point — no image processing yet. |
| Head cylinder perception (`main_head.py`) | ✅ Improved (2026-07-02, §5.10) | Rim-extraction + Hyper circle fit (removed a ~-4 to -5mm disk-interior bias in the radius fit), top-slice-median height (removed a `z_max` high-bias), 3mm voxel leaf (was 10mm — too coarse vs. a 2cm cylinder radius), closer/verified-reachable `HEAD_POSTURE_TARGET`, distortion-correctness gap closed in `camera_interface.py`, tracker fusion switched grow-only→EMA. Verified numerically (not on real hardware) to land single-view/scanned radius+height error at roughly -1 to -4mm, comfortably under the 1cm target. `feature/head-sweep-compute-track` reviewed, not merged (incompatible kwargs, re-enables the already-disabled point-level VoxelMap) — see §5.10 for salvageable ideas. |
| Arm-vs-head collision avoidance | ✅ Working | Head chain modeled as a quasi-static CBF obstacle in the ARM QP (§9.7): live FK-driven capsules (yellow in Meshcat), zero head joints added to the decision vector. `arm_right_1`/`arm_left_1` excluded from head pairs per instruction. Head's OWN separate controller/collision model (`qp_head_visual_servo.py`) is unchanged. |
| Mobile base integration | 🔧 Partial | `base_controller.py` exists but not QP-certified |
| Meshcat visualization | ✅ Working | Thread-safe, auto-reloads on grasp coloring |
| Digital twin mode | ✅ Working | `SIMULATE_IDEAL_KINEMATICS` flag in config |
| Open-loop trajectory testing | ✅ Working | `trajectory_generator.py` + `config/trajectory_endpoints.yaml` |

---

## 12.1 Known Issues / Next Steps for the Next Agent

| Issue | Description | Proposed Fix |
|-------|-------------|-------------|
| **Residual inactive-arm motion** | ✅ **FIXED, CONFIRMED by the operator (2026-07-01)**. Step 1 (§9.2): per-arm SoftMin CBF Jacobian split. Step 2 (§9.3): per-arm dynamic safety-margin split (`d_safe_dynamic_r`/`d_safe_dynamic_l`) — this was the channel that actually mattered; a single combined scalar was still inflating the idle arm's threshold from the active arm's speed alone. Operator confirmed: "the hands moves independently, and the shaking of one hand do not invoke oscillation in the others." The QP-solution plotter row (§9.2) was added as the diagnostic tool for this but wasn't ultimately needed — the math fix alone resolved it. | Done — confirmed working. |
| **Gazebo dual attach** | The IFRA_LinkAttacher plugin has a global `IsAttached` boolean allowing only ONE attachment. A patched `gazebo_link_attacher.cpp` was provided (per-pair gating, vector erase in Detach) — confirmed working by the operator. | Done. |
| **No independent grasp confirmation** | Grasp success is decided PURELY geometrically (FK-derived contact distance/angle vs. the robot's own *believed* cylinder pose) — there is no force/torque sensing on the arm chains and no vision confirmation wired in yet, so a "successful" grasp can still be a miss if the geometric gates are satisfied by coincidence. Gates were tightened ~10% (2026-06-29) as a stopgap. | Wire `head_control/object_detector.py` (or similar) as an independent post-close confirmation (does the detected cylinder pose match "in the gripper"?), and/or measure the grasped cylinder's real axis at attach time instead of assuming world +Z (see §9.1). |
| **Platform placement (dual arm)** | `GoalSet.set_grasped` / `clear_grasped` tracks one grasped color at a time. The per-arm context save/restore (`_ctx_goalset`) handles this for one-arm-active-at-a-time, but a true simultaneous dual-arm placement (both arms holding and both wanting to place concurrently) is NOT supported. | For the current "one active at a time" policy this is fine. If needed: extend GoalSet to hold per-arm grasped state (two axes, two z-offsets). |
| **Orientation choice** | The sticky hysteresis can commit to the "wrong" 180° candidate early; the user can't override without getting very close. | Consider resetting the sticky memory on clutch-engage, or using the reference velocity direction to break the tie. |

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

# Run the offline (static, publication-quality) plotter -- see §9.11.
# Leave this running; it records a trial whenever the active trigger source
# (e.g. trajectory_generator.py) toggles /offline_plotter/record_trigger.
ros2 run triago_control offline_plotter.py
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
(NOT `haptic_force_manager.py`, which no longer exists) when `cfg.BLENDING=
False` (default, Virtual Fixture mode). Sibling variants in
`haption_teleoperation/scripts/`:
- `haptic_force_manager_tutorial.py` — ★ active node, Virtual Fixture mode (`cfg.BLENDING=False`).
- `haptic_force_manager_blending_tutorial.py` — ★ active node, TWIST BLENDING mode (`cfg.BLENDING=True`, §11.5). Renders ONLY F_sync + an "Authority Share" plot; no blending math of its own.
- `haptic_force_manager_battery.py` — alternate/experimental variant.

**Switching modes**: flip `BLENDING` in `triago_control/qp_controller/config.py`
and restart BOTH `main_shared_autonomy.py` and `teleop_triago_clutch.py` (they
both read the flag once at their own startup — there is no live-toggle). Run
`haptic_force_manager_tutorial.py` for `BLENDING=False`, or
`haptic_force_manager_blending_tutorial.py` for `BLENDING=True`.

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

5. **Goal manifold anchored at the reference pose (this iteration).** The dynamic
   goal pose `get_dynamic_goal_pose(T_anchor, key)` resolves the point on each
   goal *manifold* from `T_anchor` (Side: approach azimuth + grasp height; Top:
   roll; Platform: placement XY + yaw). All non-speculative callsites now pass
   `current_T_user` instead of `current_T_EE`: the policy-loop resolution (one
   shared `T_goal` per key, `update_memory=True`, feeding BOTH `ee_policies` and
   `user_policies`), the active goal `T_active_goal` (`update_memory=False`), and
   the RViz belief-opacity goal markers. Rationale: in teleop the QP-CLF tracks
   the reference, so the reference — not the lagging EE — is the robot's intended
   pose and the right anchor for selecting the manifold point. The robot policy's
   *velocity* is still taken FROM the real EE (`compute_v_geo(current_T_EE, T_goal)`)
   so it still commands the actual robot; only the manifold POINT is reference-
   selected. The speculative one-step-lookahead used to draw the green gripper
   (`get_dynamic_goal_pose(sim_T_EE, ...)`, `update_memory=False`) is intentionally
   left EE-anchored. Test mode unchanged (`current_T_user == current_T_EE`).

   **Grasp condition is intentionally split across two anchors:** the goal is
   reference-defined (user intent), but `pos_error`/`ang_error` (which gate
   `PRE_GRASP` and therefore the grasp trigger) are computed from the REAL EE
   (`current_T_EE`) vs that goal — so a grasp can only be triggered once the
   real robot has actually been steered into the target config. Do not unify
   these two anchors.

6. **Tame the "exploding" guidance far from the goal (this iteration).** Two
   complementary changes, since far away the goal pose is ill-determined and the
   velocity-field policy is large (tanh-saturated), so F_guide would chase a
   swinging target:
   - **Proximity gate on F_guide** (`haptic_force_manager_tutorial.py`):
     `guidance = belief_confidence × proximity`, where proximity is a smoothstep
     of the distance from the reference (`pos_target`) to the active goal
     (`fix_goal_pos`): **0 beyond `GUIDE_PROX_FAR=0.50 m` (device free)**, full by
     `GUIDE_PROX_NEAR=0.10 m`. Also bumped the guidance gains 1.2× (`D_guide_lin
     33.6`, `D_guide_ang 0.54`, `MAX_GUIDE_FORCE 4.2`, `MAX_GUIDE_TORQUE 0.30`)
     because near the goal the policy twist is small and the handle was hard to move.
   - **Distance-locked orientation choice** (`goal_set._pick_orientation` /
     `_orientation_hysteresis`): the 180°-apart Top/Side candidate switch margin is
     now distance-scaled — nominal `0.05 rad` at/below `0.12 m` anchor→cylinder
     distance, ramping to an effectively infinite `10 rad` (locked) by `0.30 m`.
     This stops the goal orientation flipping 180° while the user is still far and
     uncommitted (the spike that whipped the guidance), while still letting the
     user choose/change the approach side up close. Benefits guidance, belief, and
     the green-gripper marker together.

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
