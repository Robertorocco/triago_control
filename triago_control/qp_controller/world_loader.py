# world_loader.py
"""Declarative world loading: parses config/worlds/<name>.yaml into a WorldScene consumed
generically by CollisionManager, VisualizationEngine and GoalSet -- one file per world, no
hard-coded obstacle numbers. No Gazebo connection: it mirrors the layout the matching .world
file spawns (keeping the two in sync is a manual, single-file step via gazebo_world_file).

Schema: world_name, gazebo_world_file, static_obstacles (name/role/shape/pose/size/color/
collision), grasp_roles {red,blue}, and optional platform(s) (placement disks: pure goal
references, never collision geometry)."""

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
    name: str = "Place"             # goal-key suffix -> 'Platform_<name>' (default keeps
                                    # the legacy single-platform key 'Platform_Place')


@dataclass
class WorldScene:
    """Parsed world scene: every static obstacle + the red/blue grasp-role mapping."""
    world_name: str
    gazebo_world_file: str
    static_obstacles: List[ObstacleSpec] = field(default_factory=list)
    grasp_roles: Dict[str, str] = field(default_factory=dict)
    # `platform` = the FIRST placement disk (kept for legacy readers / single-
    # platform worlds); `platforms` = the full list (one entry per disk). A world
    # A single `platform:` mapping yields platforms=[that one].
    platform: Optional[PlatformSpec] = None
    platforms: List[PlatformSpec] = field(default_factory=list)

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

    Example: load_world('no_obstacle') reads
             config/worlds/no_obstacle.yaml
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

    # Placement platform(s). Backward compatible:
    #   * a single top-level `platform:` mapping -> exactly one platform whose
    #     name defaults to "Place" (goal key 'Platform_Place'), unchanged;
    #   * a `platforms:` list -> one PlatformSpec per entry (each may set `name`).
    # `platform` (singular) is kept pointing at the FIRST entry for legacy readers.
    def _parse_platform(pd, default_name):
        return PlatformSpec(
            pose=np.array(pd['pose'], dtype=float),
            radius=float(pd['radius']),
            thickness=float(pd['thickness']),
            place_margin=float(pd.get('place_margin', 0.03)),
            name=str(pd.get('name', default_name)),
        )

    platforms = []
    if raw.get('platforms'):
        for i, pd in enumerate(raw['platforms']):
            platforms.append(_parse_platform(pd, default_name=f"P{i}"))
    elif raw.get('platform') is not None:
        platforms.append(_parse_platform(raw['platform'], default_name="Place"))
    platform = platforms[0] if platforms else None

    return WorldScene(
        world_name=raw.get('world_name', world_name),
        gazebo_world_file=raw.get('gazebo_world_file', ''),
        static_obstacles=obstacles,
        grasp_roles={k.lower(): v for k, v in raw.get('grasp_roles', {}).items()},
        platform=platform,
        platforms=platforms,
    )
