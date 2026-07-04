# world_loader.py
"""
World scene loading: the SINGLE place a new Gazebo world's obstacle layout is
described to the QP-CLF-CBF stack.

WHY this module exists
-----------------------------------------------------------------------------
Before this module, the workspace obstacles (table + red/blue cylinders +
optional wall) were hard-coded as numeric constants in config.py (TABLE_POS,
RED_CYLINDER_POS, BLUE_CYLINDER_POS, CYLINDER_SIZE, TABLE_SIZE, WALL_POS,
WALL_SIZE) and read DIRECTLY by collision_manager.py and
visualization_engine.py. Testing a new Gazebo world (different table pose,
different cylinder size, an extra obstacle for a harder task) meant hand-
editing config.py -- and, separately, head_control/config.py's OWN duplicated
copy of the same numbers (GT_RED_CENTER, TABLE_CENTER_WORLD, ...) -- with no
guarantee the two ever agreed, and no way to keep several world variants
around at once.

This module introduces ONE interchange format -- a small YAML file under
config/worlds/<name>.yaml -- that fully describes a world's static obstacle
layout (shape/pose/size/color) plus which named obstacle plays the "red"/
"blue" grasp role (see WorldScene.grasp_roles). `load_world(world_name)`
parses it into a plain, dependency-free WorldScene object that every
downstream module (CollisionManager, VisualizationEngine, GoalSet,
head_control/config.py) consumes generically -- no per-obstacle bespoke code,
no duplicated numbers.

WHAT THIS DOES NOT CHANGE
-----------------------------------------------------------------------------
* The Gazebo launch command is UNCHANGED (`ros2 launch triago_gazebo ...
  world_name:=tutorial`). This module has NO connection to Gazebo -- it only
  describes, on the Pinocchio/hppfcl/RViz side, the SAME obstacle layout that
  the chosen Gazebo .world file already spawns. Keeping the two in sync when
  authoring a NEW world is a manual (but now single-file, single-place)
  bookkeeping step -- see the `gazebo_world_file` field below.
* No CBF/CLF math changes. CollisionManager still builds the exact same
  hppfcl geometry types (Box, Cylinder) at the exact same poses; only WHERE
  those numbers come from changed (YAML instead of Python constants).
* `trajectory_endpoints.yaml` (test presets) is untouched, per instruction --
  these worlds are teleoperation-driven, not open-loop-preset-driven.

SCHEMA (see config/worlds/bimanual_default.yaml for a full worked example)
-----------------------------------------------------------------------------
world_name: str                     -- must match the YAML's own file, informational
gazebo_world_file: str               -- bookkeeping pointer to the matching .world file
static_obstacles: list of:
    name: str                        -- unique geometry name (also the hppfcl GeometryObject name)
    role: "table" | "graspable" | "wall" | "obstacle"   -- informational + drives default RViz color
    shape: "box" | "cylinder"
    pose: [x, y, z, roll, pitch, yaw]  -- roll/pitch/yaw currently unused (all current
                                          obstacles are axis-aligned); kept for forward
                                          compatibility, see _pose_to_se3 below.
    size: box -> [sx, sy, sz]; cylinder -> [radius, length]
    color: [r, g, b, a]               -- used by RViz + Meshcat; NOT physics
    collision: bool                   -- if False, geometry is created but not added to
                                          workspace_obstacle_ids (mirrors WALL_COLLIDER's
                                          old on/off behavior, generalized)
grasp_roles: {red: <name>, blue: <name>}   -- resolves today's grasp state machine's
                                          hard-coded "red"/"blue" concepts to whichever
                                          named obstacle plays that role in THIS world.
platform: (optional)                -- NOT an obstacle (no collision geometry, never
                                          added to CollisionManager's cmodel) -- purely
                                          the reference pose for shared_autonomy's
                                          Platform_Place goal, and a visual aid the
                                          operator sees directly in Gazebo (e.g. the
                                          yellow `placement_area` disk). Deliberately
                                          a SEPARATE top-level field from
                                          `static_obstacles`, not another entry in that
                                          list, to keep that list's semantics ("things
                                          the collision model builds geometry for")
                                          unambiguous.
    pose: [x, y, z]                  -- world center
    radius: float                    -- [m] disk radius
    thickness: float                 -- [m] disk thickness
    place_margin: float (optional, default 0.03)  -- [m] keep the placed
                                          footprint this far inside the rim
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - ament always present under ROS 2
    get_package_share_directory = None


@dataclass
class ObstacleSpec:
    """One static obstacle, as described in a world scene YAML."""
    name: str
    role: str
    shape: str                      # "box" | "cylinder"
    pose: np.ndarray                # (6,) [x, y, z, roll, pitch, yaw]
    size: np.ndarray                # box: (3,) [sx,sy,sz]; cylinder: (2,) [radius, length]
    color: np.ndarray               # (4,) [r, g, b, a]
    collision: bool = True

    @property
    def position(self):
        """(3,) [x, y, z] -- the only part of `pose` any current consumer uses."""
        return self.pose[:3]


@dataclass
class PlatformSpec:
    """The placement/goal disk (e.g. Gazebo's `placement_area` model).

    NOT an obstacle: it has no collision geometry and is never added to
    CollisionManager's cmodel -- it exists purely as a REFERENCE POSE for
    shared_autonomy's Platform_Place goal (see goal_set.GoalSet.
    get_platform_goal_pose) and as a visual aid for the human operator
    (rendered by Gazebo itself; nothing on the RViz/Meshcat side needs to
    draw it, since the operator already sees it directly in the sim view).
    """
    pose: np.ndarray                # (3,) [x, y, z] world center
    radius: float                   # [m] disk radius
    thickness: float                # [m] disk thickness
    place_margin: float = 0.03      # [m] keep the placed footprint this far inside the rim


@dataclass
class WorldScene:
    """Parsed world scene: every static obstacle + the red/blue grasp-role mapping."""
    world_name: str
    gazebo_world_file: str
    static_obstacles: List[ObstacleSpec] = field(default_factory=list)
    grasp_roles: Dict[str, str] = field(default_factory=dict)
    platform: Optional[PlatformSpec] = None

    def get_obstacle(self, name) -> Optional[ObstacleSpec]:
        for obs in self.static_obstacles:
            if obs.name == name:
                return obs
        return None

    def obstacle_for_role(self, color) -> Optional[ObstacleSpec]:
        """Resolve 'red'/'blue' (case-insensitive) to its ObstacleSpec in THIS world."""
        name = self.grasp_roles.get(color.lower())
        return self.get_obstacle(name) if name else None

    def get_obstacle_by_role(self, role) -> Optional[ObstacleSpec]:
        """First obstacle whose `role` field matches (e.g. 'table', 'wall').

        Used where exactly one obstacle of that role is expected (the table,
        the optional wall) -- unlike grasp_roles/obstacle_for_role, this does
        NOT require an explicit name mapping in the YAML, just the `role:`
        tag on the obstacle itself.
        """
        for obs in self.static_obstacles:
            if obs.role == role:
                return obs
        return None


def _find_world_yaml(world_name):
    """Resolve a world name to its YAML path.

    Search order mirrors trajectory_generator.py's config-file resolution
    convention (installed ament share dir first, then a source-tree fallback
    for `colcon build --symlink-install` / running from source).
    """
    fname = f"{world_name}.yaml"

    if get_package_share_directory is not None:
        try:
            share = get_package_share_directory('triago_control')
            candidate = os.path.join(share, 'config', 'worlds', fname)
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass

    # Fallback: relative to this file's location in the source tree
    # (.../triago_control/triago_control/qp_controller/world_loader.py
    #   -> .../triago_control/config/worlds/<fname>)
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, '..', '..', 'config', 'worlds', fname)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"[world_loader] Could not find world scene '{fname}' in the installed "
        f"share directory or the source tree's config/worlds/. Checked ament "
        f"share (triago_control/config/worlds/) and {candidate}.")


def load_world(world_name) -> WorldScene:
    """Load and parse a world scene YAML by name (no '.yaml' extension, no path).

    Example: load_world('bimanual_default') reads
             config/worlds/bimanual_default.yaml
    """
    path = _find_world_yaml(world_name)
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    obstacles = []
    for o in raw.get('static_obstacles', []):
        obstacles.append(ObstacleSpec(
            name=o['name'],
            role=o.get('role', 'obstacle'),
            shape=o['shape'],
            pose=np.array(o.get('pose', [0, 0, 0, 0, 0, 0]), dtype=float),
            size=np.array(o['size'], dtype=float),
            color=np.array(o.get('color', [0.7, 0.7, 0.7, 0.8]), dtype=float),
            collision=bool(o.get('collision', True)),
        ))

    platform = None
    p_raw = raw.get('platform')
    if p_raw is not None:
        platform = PlatformSpec(
            pose=np.array(p_raw['pose'], dtype=float),
            radius=float(p_raw['radius']),
            thickness=float(p_raw['thickness']),
            place_margin=float(p_raw.get('place_margin', 0.03)),
        )

    return WorldScene(
        world_name=raw.get('world_name', world_name),
        gazebo_world_file=raw.get('gazebo_world_file', ''),
        static_obstacles=obstacles,
        grasp_roles={k.lower(): v for k, v in raw.get('grasp_roles', {}).items()},
        platform=platform,
    )
