# Frame Conventions — TIAGo Dual-Arm Platform

## base_footprint (world / reference frame)

The QP controller's `REF_FRAME`. All cartesian references, targets, and collision
geometry are expressed in this frame.

| Axis | Direction |
|------|-----------|
| **X** | Forward (in front of TIAGo) |
| **Y** | Left |
| **Z** | Up |

**Origin**: center of TIAGo's torso.

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
