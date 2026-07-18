#!/usr/bin/env python3
# main_qp_controller_perceived.py
"""
Camera-driven QP-CLF-CBF controller — builds the collision world from the HEAD
CAMERA instead of a static YAML.

This is a THIN SUBCLASS of RealQPController (main_qp_controller_real.py), which is
itself a thin subclass of SafetyQPController. It is IDENTICAL in every respect —
same CBF/CLF math, same real-time control loop, same async-CBF/RViz-overlay worker
threads on real hardware — EXCEPT for one thing: where the obstacle world comes
from. Instead of load_world(<yaml>), it blocks at startup until the head perception
node publishes a confident, latched /perceived_world/snapshot, and builds the
WorldScene from that (see qp_controller/perceived_world_builder.py). Everything
downstream is inherited verbatim, so this file stays ~1 method + a main().

Subclassing RealQPController (not SafetyQPController directly) matters on real hardware:
the in-line SoftMin CBF alone exceeds the control-loop budget there, collapsing the publish
rate below what the plotters and the collision-staleness watchdog expect. RealQPController
moves that cost to a worker thread; in sim it falls back to fully synchronous behavior.

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
from rclpy.signals import SignalHandlerOptions

# The QP entrypoints install flat into lib/triago_control/ alongside each other;
# add this file's dir so the sibling `main_qp_controller` module imports cleanly
# whether run from the install tree or source (mirrors the analysis scripts).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_qp_controller_real import RealQPController                    # noqa: E402
from triago_control.qp_controller.perceived_world_builder import wait_for_scene  # noqa: E402


class PerceivedQPController(RealQPController):
    """RealQPController whose collision world is BUILT FROM THE HEAD CAMERA."""

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
    # NO signal handlers: rclpy's default SIGINT handler shuts down the whole
    # context BEFORE our except/finally below runs, so restore_controllers()
    # would find an already-dead context on Ctrl-C. Disabling it means Ctrl-C
    # just raises a plain KeyboardInterrupt and WE control shutdown ordering.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PerceivedQPController()

    # Wall-clock control loop by design -- do not set use_sim_time here.

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
        node.shutdown_workers()
        node.destroy_node()
        rclpy.shutdown()
        return

    # --- PHASE 2: visualization + diagnostics (Meshcat suspended on real HW) ---
    if node.REAL_HARDWARE:
        node.get_logger().info(
            "\033[93m[Viz] REAL HARDWARE: Meshcat suspended (RViz/plotters unaffected).\033[0m")
    else:
        node.viz.init_meshcat(lambda: node.kin.current_q, node.col)
    node.kin.print_joint_limits_table(node.get_logger())

    # --- PHASE 3: engage the real-time loop ---
    # spin_once(timeout_sec=0.1) in a loop, NOT rclpy.spin(node): with the
    # signal handler disabled above, an indefinitely-blocking spin() has
    # nothing to wake it on Ctrl-C -- this returns to Python every 0.1s so
    # KeyboardInterrupt actually gets raised promptly (same pattern already
    # used by trajectory_generator.py's main()).
    node.start_control_loop()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_workers()
        node.restore_controllers()
        if os.path.exists(node.urdf_path):
            os.remove(node.urdf_path)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
