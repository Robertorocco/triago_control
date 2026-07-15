# Project context

Before doing any work in this repo, read `.kiro/context.md` — it holds the current state of the project (architecture, in-progress work, decisions) and is the primary source of truth for Claude sessions here, alongside README.md for user-facing info.

It is Claude's responsibility to keep `.kiro/context.md` accurate and up to date as work happens in this repo: update it when architecture changes, features land, or decisions are made, so a new chat can pick up full context from this file alone. **Keep it short** — see its own §0 for the length rule; don't let entries grow back into essays.

## Commit & Push Discipline

- Commit messages: **one line, imperative mood, <72 chars**. No body unless the user asks for one.
- Do not narrate the commit/push process step by step — run it, then report the result in ≤2 sentences.
- Never re-read a file immediately after editing/writing it "to verify" — trust the tool; Edit/Write already error loudly on failure.
- Stage only the files actually changed for the task at hand — never `git add -A`/`git add .`.
- After push, give the sync command block (§14 of context.md) and stop — no summary of the diff unless asked.

## Code Comment Style

- Max **2 lines per comment**. State the constraint or the why, not the history.
- No story: no dates, no "previously X now Y", no reference to past bugs, commits, or sessions. That belongs in the commit message, not the file.
- Comment only the non-obvious (a hidden invariant, a subtle workaround, a surprising constraint) — well-named code doesn't need a comment restating what it does.

## Sibling repo: haption_teleoperation

`~/exchange/ros2-ws/src/haption_teleoperation` drives the Haption haptic device and is the teleoperation half of the same pipeline this repo's QP controller/shared-autonomy stack serves. They run together at runtime and share a live ROS2 topic interface — a change on one side routinely requires a matching change on the other.

**If a task touches the teleop loop, grasp state machine, blending, or force feedback, also read `../haption_teleoperation/.kiro/context.md` before making changes** — don't rely on the table below alone for anything beyond a quick topic lookup.

### Runtime interface (verified 2026-07-10)

**haption_teleoperation → triago_control** (user commands)

| Topic | Type | Published by (haption_teleoperation) | Consumed by (this repo) |
|---|---|---|---|
| `/arm_right/cartesian_reference`, `/arm_left/cartesian_reference` | `Float64MultiArray` | `teleop_triago_clutch.py` / `teleop_triago_joystick.py` (only when `cfg.BLENDING`/`ASSIST_BLENDING=False`) | `scripts/qp_arm_teleop/main_qp_controller.py` (`ref_cb_right/left`) |
| `/arm_right/user_cartesian_reference`, `/arm_left/user_cartesian_reference` | `Float64MultiArray` | same scripts, redirected here when blending is on | `scripts/qp_arm_teleop/main_shared_autonomy.py` (`sub_human_reference_right/left`) |
| `virtuose/button_left` | `std_msgs/Bool` | `virtuose_server_node.cpp` | `main_shared_autonomy.py` (`sub_trigger` — grasp/switch-arm) |
| `virtuose/button_right` | `std_msgs/Bool` | `virtuose_server_node.cpp` | `main_shared_autonomy.py` (`sub_clutch` — suspend blending while indexing) |

Note: `main_shared_autonomy.py` itself also publishes `/arm_*/cartesian_reference` (`pub_blend_right/left`) and becomes the sole writer once `ASSIST_BLENDING=True`.

**triago_control → haption_teleoperation** (state / force-feedback context)

| Topic | Type | Published by (this repo) | Consumed by (haption_teleoperation) |
|---|---|---|---|
| `/qp_debug/ee_real` | `Float64MultiArray` | `main_qp_controller.py` (`pub_ee_state`) | `teleop_triago_{clutch,joystick}.py`, `haptic_force_manager_{C,CF,CFB}.py` |
| `/collision_constraints` | `Float64MultiArray` | `main_qp_controller.py` (`pub_shared_col`) | `haptic_force_manager_{CF,CFB,CB}.py` (`cbf_gradient_cb`) |
| `/qp_debug/lambda_cbf` | `Float64MultiArray` | `main_qp_controller.py` (`pub_lambda_cbf`) | `haptic_force_manager_{CF,CFB,CB}.py` (`lambda_cb`) |
| `/shared_autonomy/grasp_active` | `std_msgs/Bool` | `main_shared_autonomy.py` (`pub_grasp_active`) | all teleop + force-manager scripts |
| `/shared_autonomy/active_arm` | `std_msgs/String` | `main_shared_autonomy.py` (`pub_active_arm`) | all teleop + force-manager scripts |
| `/shared_autonomy/goal_names`, `goal_probabilities`, `user_policy`, `active_goal_pose` | `String`/`Float64MultiArray` | `main_shared_autonomy.py` | `haptic_force_manager_{J,C}F/JFB/CFB/CB/JB.py` |
| `/shared_autonomy/blend_debug` | `Float64MultiArray` (19 floats: `alpha, v_user(6), v_policy(6), v_blend(6)`) | `main_shared_autonomy.py` (`pub_blend_debug`) | `haptic_force_manager_{CB,CFB}.py` |

Force feedback to the operator (`virtuose/force_cmd`, `geometry_msgs/Wrench`, → `virtuose_server_node.cpp`) is synthesized locally inside haption_teleoperation's `haptic_force_manager_*.py` scripts from the telemetry above — this repo does not publish a dedicated force/state topic for that purpose.

**Import-level coupling (not a topic):** haption_teleoperation depends on this repo's `triago_control.qp_controller.config` module (`cfg.BLENDING`, `cfg.ASSIST_*`, `cfg.validate_condition(...)`) — declared as a `<depend>` in haption_teleoperation's `package.xml`. Changing the condition-selector shape in `qp_controller/config.py` breaks haption_teleoperation's teleop and force-manager scripts.

**Key entrypoints in this repo:** `scripts/qp_arm_teleop/main_qp_controller.py` (QP-CLF-CBF safety loop), `scripts/qp_arm_teleop/main_shared_autonomy.py` (belief estimation, grasp state machine, reference blending), `triago_control/qp_controller/config.py` (single source of truth for the 2×2×2 experiment condition matrix).
