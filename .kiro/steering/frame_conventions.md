# Frame Conventions — TIAGo Dual-Arm Platform

## base_footprint (world / reference frame)

The QP controller's `REF_FRAME`. All cartesian references, targets, and collision
geometry are expressed in this frame.

| Axis | Direction |
|------|-----------|
| **X** | Forward (in front of TIAGo) |
| **Y** | Left |
| **Z** | Up |

**Origin**: center of TIAGo's mobile base (ground plane).

## gripper_right_grasping_link / gripper_left_grasping_link

Both grippers share the **same** axis convention (not mirrored):

| Axis | Direction |
|------|-----------|
| **X** | Approach axis (points outward from the gripper, i.e., the insertion/reach direction) |
| **Y** | Right (finger-spread direction) |
| **Z** | Down (palm-bottom direction) |

## RPY in Cartesian References

The `/arm_{right,left}/cartesian_reference` messages carry orientation as
**RPY (roll, pitch, yaw)** expressed as the rotation of the gripper frame
relative to `base_footprint`, using the Pinocchio convention:

```
R_desired = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

The controller converts this via `pin.rpy.rpyToMatrix(r, p, y)` and computes
the SO(3) error as `pin.log3(R_des @ R_real.T)`.

## Common Orientation Presets (RPY in radians)

| Name | RPY | Description |
|------|-----|-------------|
| forward-down | `[π, 0, 0]` | Gripper pointing forward, palm facing down |
| top-grasp | `[0, π/2, 0]` | Gripper pointing straight down, palm facing forward |
| approach-left | `[π, 0, π/2]` | Gripper pointing left (+Y), palm down |
| approach-right | `[π, 0, -π/2]` | Gripper pointing right (-Y), palm down |

## ee_real Layout (/qp_debug/ee_real — 18 floats)

```
[p_r(3), v_r(3), p_l(3), v_l(3), rpy_r(3), rpy_l(3)]
 0:3     3:6     6:9     9:12    12:15     15:18
```

All expressed in `base_footprint`.



# Console Output Preferences

## Rules

- **No spam**: never print continuous data streams to the console (e.g., per-tick values at 100+ Hz).
- **Metrics only**: periodic summaries are acceptable at low frequency (≤ 0.2 Hz / every 5+ seconds).
- **Warnings & errors**: always print immediately with `[WARN]` or `[ERROR]` prefix.
- **Phase transitions**: print once per state change (e.g., `[PHASE] Switched to TRACKING`).
- **Startup banners**: one compact banner at node init showing configuration is fine.

## Preferred debugging approach

1. **Plots** (matplotlib / rqt_plot) for continuous data — superior for human interpretation.
2. **ROS topics** for machine-readable diagnostics (e.g., `/debug/...` topics).
3. **Console** only for rare events, compact summaries, and errors.

## Anti-patterns (do NOT do)

- Printing every loop iteration
- Printing raw float arrays per tick
- Flooding stdout with data that should be a topic or a plot
