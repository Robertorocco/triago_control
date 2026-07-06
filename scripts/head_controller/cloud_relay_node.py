#!/usr/bin/env python3
"""
cloud_relay_node.py — republish a real camera PointCloud2 onto the topic
tabletop_perception_node.py subscribes to, throttled.

WHY THIS EXISTS (root cause, found 2026-07-06)
    tabletop_perception_node.py subscribes to `/filtered_cloud`, matching its
    upstream C++ original 1:1 (see that node's own module docstring). But
    NOTHING in the sibling chiarapanagrosso/Triago_Project repository
    actually publishes `/filtered_cloud`:
        * perception.launch.py's own `topic_tools/throttle` node republishes
          the RealSense cloud to `/throttle_filtering_points/filtered_points`
          -- a DIFFERENT topic name.
        * confirmed independently from a captured sim log: MoveIt's
          `pointcloud_octomap_updater` explicitly listens to
          `/throttle_filtering_points/filtered_points`, not `/filtered_cloud`.
        * the only other mention of `/filtered_cloud` in that repo is
          octomap_server's `cloud_in` REMAP -- i.e. it also just expects
          something else to publish it.
    This is a genuine topic-name mismatch/unfinished wiring in the SOURCE
    project, not a bug introduced by porting the node. Without a publisher
    on `/filtered_cloud`, tabletop_perception_node's `_cloud_callback` never
    fires -- which is exactly what the reported symptom (zero log output
    past the startup banner) shows.

WHAT THIS NODE DOES
    Subscribes to a real camera point cloud topic (default: this project's
    own head camera, `/gripper_head_camera_rgbd/depth/color/points` --
    already the exact topic your own Recording_Rviz.rviz config points at)
    and republishes it, rate-limited, on `/filtered_cloud`. Pure pass-
    through -- no filtering/cropping/resampling of the cloud CONTENT, only a
    republish rate cap (mirrors what `topic_tools/throttle` would have done
    upstream, without requiring that package as a dependency).

RUN (three terminals):
    ros2 run triago_control cloud_relay_node.py
    ros2 run triago_control head_look_at_table.py
    ros2 run triago_control tabletop_perception_node.py

OVERRIDE the source topic if your real camera cloud has a different name:
    ros2 run triago_control cloud_relay_node.py --ros-args \
        -p source_topic:=/your/camera/depth/color/points
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

import triago_control.head_control.config as cfg


class CloudRelayNode(Node):
    def __init__(self):
        super().__init__("cloud_relay_node")

        source_topic = self.declare_parameter(
            "source_topic", cfg.TP_RELAY_SOURCE_TOPIC).value
        target_topic = self.declare_parameter(
            "target_topic", cfg.TP_RELAY_TARGET_TOPIC).value
        rate_hz = self.declare_parameter(
            "rate_hz", cfg.TP_RELAY_RATE_HZ).value

        self._min_period_s = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._last_publish_time = None
        self._n_received = 0
        self._n_relayed = 0

        self.publisher = self.create_publisher(PointCloud2, target_topic, 10)
        self.create_subscription(
            PointCloud2, source_topic, self._cloud_cb, qos_profile_sensor_data)

        self.create_timer(5.0, self._watchdog_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " cloud_relay_node — bridging a real camera cloud to /filtered_cloud\n"
            "------------------------------------------------------------------\n"
            f"  source_topic : {source_topic}\n"
            f"  target_topic : {target_topic}\n"
            f"  rate_hz      : {rate_hz}\n"
            "==================================================================")

    def _cloud_cb(self, msg: PointCloud2):
        self._n_received += 1
        now = self.get_clock().now()
        if self._min_period_s > 0.0 and self._last_publish_time is not None:
            elapsed = (now - self._last_publish_time).nanoseconds * 1e-9
            if elapsed < self._min_period_s:
                return
        self._last_publish_time = now
        self._n_relayed += 1
        self.publisher.publish(msg)

    def _watchdog_tick(self):
        if self._n_received == 0:
            self.get_logger().warn(
                "No messages received yet on the source topic — check "
                "'ros2 topic list | grep -i point' for the real camera cloud "
                "topic name and override with -p source_topic:=<name> if "
                "different.")


def main():
    rclpy.init()
    node = CloudRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
