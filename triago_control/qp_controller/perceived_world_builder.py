"""
perceived_world_builder.py — turn the head camera's latched perceived-world
snapshot into a WorldScene the QP-CLF-CBF stack can build a collision model from.

This is the QP-side half of the camera-driven pipeline (the head-side half is
main_head.py + head_control/world_convergence.py). The head perception node, once
its estimate is confident + stable, publishes ONE latched MarkerArray on
cfg_head.PERCEIVED_WORLD_TOPIC describing the table + cylinders. This module:

    * subscribes to that latched topic (TRANSIENT_LOCAL, so a QP node started
      AFTER convergence still receives the last snapshot immediately on
      subscribe), and
    * decodes the MarkerArray into ObstacleSpec/WorldScene — the EXACT same
      dependency-free contract world_loader.load_world produces from YAML — so
      everything downstream (CollisionManager.build_collision_model,
      VisualizationEngine, Meshcat, RViz) consumes it unchanged.

The decode mirrors head_plotter.py's marker decode: cylinders are ns="objects"
CYLINDER markers (radius = scale.x/2, height = scale.z, class by the r/b colour
channel); the table is the ns="table" CUBE (full box). No custom message type is
needed — the marker encoding IS the interchange format, and it doubles as a
directly-viewable RViz display.

WHY the world is built ONCE, statically: a CBF barrier over a moving obstacle set
would be non-stationary (its zero-level set shifts every tick), which is exactly
what we do not want for a safety-critical constraint. So the QP node blocks until
one confident snapshot arrives (wait_for_scene), builds the collision model from
it, and never updates the static obstacles again — matching the "confident ->
static load" design.
"""

import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

# The head perception config owns the canonical topic name; import it so the two
# sides can never drift apart. (Cross-subpackage import within triago_control.)
import triago_control.head_control.config as cfg_head
from triago_control.qp_controller.world_loader import ObstacleSpec, WorldScene


def latched_qos():
    """QoS matching main_head's latched perceived-world publisher."""
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


def markers_to_world_scene(msg: MarkerArray) -> WorldScene:
    """Decode a perceived-world snapshot MarkerArray into a WorldScene.

    Cylinders -> role='graspable' ObstacleSpec named 'red_cylinder'/'blue_cylinder'
    (+ grasp_roles so CollisionManager resolves red_cyl_id/blue_cyl_id, which the
    grasp state machine and viz key on). Table CUBE -> role='table' box.
    """
    obstacles = []
    grasp_roles = {}
    for m in msg.markers:
        if m.action != Marker.ADD:
            continue

        if m.ns == "objects" and m.type == Marker.CYLINDER:
            cx, cy, cz = m.pose.position.x, m.pose.position.y, m.pose.position.z
            radius = m.scale.x / 2.0
            height = m.scale.z
            if m.color.r > 0.5:
                name, role_color = "red_cylinder", "red"
            elif m.color.b > 0.5:
                name, role_color = "blue_cylinder", "blue"
            else:
                continue                                # unclassified -> skip
            obstacles.append(ObstacleSpec(
                name=name,
                role="graspable",
                shape="cylinder",
                pose=np.array([cx, cy, cz, 0.0, 0.0, 0.0], dtype=float),
                size=np.array([radius, height], dtype=float),   # [radius, length]
                color=np.array([m.color.r, m.color.g, m.color.b, m.color.a], dtype=float),
                collision=True,
            ))
            grasp_roles[role_color] = name

        elif m.ns == "table" and m.type == Marker.CUBE:
            cx, cy, cz = m.pose.position.x, m.pose.position.y, m.pose.position.z
            obstacles.append(ObstacleSpec(
                name="work_table",
                role="table",
                shape="box",
                pose=np.array([cx, cy, cz, 0.0, 0.0, 0.0], dtype=float),
                size=np.array([m.scale.x, m.scale.y, m.scale.z], dtype=float),
                color=np.array([m.color.r, m.color.g, m.color.b, m.color.a], dtype=float),
                collision=True,
            ))

    return WorldScene(
        world_name="perceived",
        gazebo_world_file="",
        static_obstacles=obstacles,
        grasp_roles=grasp_roles,
        platform=None,
        platforms=[],
    )


def wait_for_scene(node, timeout=None) -> WorldScene:
    """Block until the latched perceived-world snapshot arrives; return a WorldScene.

    Safe to call from inside a Node's __init__ (before rclpy.spin): the node is
    already constructed, rclpy.init() has run, and we pump rclpy.spin_once
    ourselves. A TRANSIENT_LOCAL subscriber receives the last latched message
    immediately if the head has already converged; otherwise this waits for it.

    Returns None on timeout (if `timeout` seconds is given), else blocks until a
    snapshot arrives or rclpy shuts down.
    """
    holder = {}

    def _cb(msg):
        holder["msg"] = msg

    sub = node.create_subscription(
        MarkerArray, cfg_head.PERCEIVED_WORLD_TOPIC, _cb, latched_qos()
    )
    node.get_logger().info(
        f"\033[96m[PerceivedWorld] Waiting for a latched world snapshot on "
        f"'{cfg_head.PERCEIVED_WORLD_TOPIC}' (run the head perception node and let "
        f"its estimate converge)...\033[0m")

    start = time.time()
    try:
        while "msg" not in holder and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if timeout is not None and (time.time() - start) > timeout:
                node.get_logger().error(
                    f"[PerceivedWorld] Timed out after {timeout:.0f}s waiting for a "
                    f"snapshot on '{cfg_head.PERCEIVED_WORLD_TOPIC}'.")
                return None
    finally:
        node.destroy_subscription(sub)

    scene = markers_to_world_scene(holder["msg"])
    names = [o.name for o in scene.static_obstacles]
    node.get_logger().info(
        f"\033[92m[PerceivedWorld] Snapshot received: {len(scene.static_obstacles)} "
        f"obstacles {names}, grasp_roles={scene.grasp_roles}.\033[0m")
    return scene
