#!/usr/bin/env python3
# main_qp_controller_perceived.py
"""
Camera-driven QP-CLF-CBF controller — builds the collision world from the HEAD
CAMERA instead of a static YAML.

This is a THIN SUBCLASS of SafetyQPController (main_qp_controller.py). It is
IDENTICAL in every respect — same CBF/CLF math, same real-time control loop, same
Meshcat/RViz visualization — EXCEPT for one thing: where the obstacle world comes
from. Instead of load_world(<yaml>), it blocks at startup until the head perception
node publishes a confident, latched /perceived_world/snapshot, and builds the
WorldScene from that (see qp_controller/perceived_world_builder.py). Everything
downstream is inherited verbatim, so this file stays ~1 method + a main().

The world is built ONCE, statically, from the confident snapshot — no dynamic
per-tick CBF updates (a moving obstacle set would make the barrier non-stationary,
which we deliberately avoid for a safety constraint).

Run (after the head perception node has been started and its estimate converges):
    ros2 run triago_control main_qp_controller_perceived.py

The `world_name` ROS param is ignored in this mode (the world is perceived, not
loaded from YAML) — launch the standard Gazebo world as usual.
"""

import os
import sys

import rclpy

# The QP entrypoints install flat into lib/triago_control/ alongside each other;
# add this file's dir so the sibling `main_qp_controller` module imports cleanly
# whether run from the install tree or source (mirrors the analysis scripts).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_qp_controller import SafetyQPController                       # noqa: E402
from triago_control.qp_controller.perceived_world_builder import wait_for_scene  # noqa: E402


class PerceivedQPController(SafetyQPController):
    """SafetyQPController whose collision world is BUILT FROM THE HEAD CAMERA."""

    def _build_world_scene(self):
        # Overrides the base YAML seam: block until the head publishes a
        # confident latched snapshot, then build the WorldScene from it. Runs at
        # base __init__ BEFORE build_collision_model, so the static CBF build uses
        # the confident world by construction.
        self.get_logger().info(
            "\033[96m[PerceivedWorld] Camera-driven mode: the collision world will "
            "be built from the head camera's confident snapshot (world_name param "
            "is ignored).\033[0m")
        return wait_for_scene(self)


def main():
    rclpy.init()
    node = PerceivedQPController()

    # --- PHASE 1: wait for TF, then verify controller state ---
    node.get_logger().info("[Main] Waiting for TF...")
    node.wait_for_tf()

    node.get_logger().info("[Main] Verifying Controller State...")
    if node.check_and_switch_controllers():
        print("------------------------------------------------")
        print("PERCEIVED-WORLD SAFETY CONTROLLER RUNNING (Velocity Mode)")
        print("------------------------------------------------")
    else:
        print("[Error] Could not switch controllers. Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- PHASE 2: visualization + diagnostics ---
    node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    node.start_control_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
