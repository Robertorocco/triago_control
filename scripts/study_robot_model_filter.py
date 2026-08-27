#!/usr/bin/env python3
"""study_robot_model_filter.py -- publishes a modified robot_description for the
study RViz stations, since RViz's own per-link Alpha/Value overrides do not
render for this robot's Collada meshes (confirmed: neither a saved-config edit
nor a live GUI change to a link's Alpha takes any visual effect -- a known
class of Ogre material-sharing bug, not something fixable from a .rviz file).

Two modes for every link outside the two arm/gripper chains (`dim_mode` param):
  'remove'   -- strip <visual> geometry outright. Guaranteed to render (there
                is nothing left to submit to Ogre), immune to any alpha/
                material bug by construction.
  'material' -- keep the geometry but replace its <material> with a translucent
                grey, under a link-unique material name (never reusing the
                link's original name). This sidesteps the shared-material
                write-collision suspected in the live-override failures (many
                links referencing one Ogre material by name, so one link's
                alpha silently overwrites another's) -- baking a UNIQUE name
                per link means no two links can collide here even if the
                original mesh export shared one. Unverified against the actual
                mesh files (they live inside the container, not this repo);
                try it, fall back to 'remove' if it doesn't render either.
<collision>/<inertial>/joints are untouched either way -- this is a
display-only copy, never touched by the real robot_state_publisher/TF/planning
pipeline.

Publishes on /robot_description_filtered (NOT /robot_description -- the real
topic is left alone since other consumers need the complete kinematic tree).
Each study .rviz's RobotModel display points its Description Topic here
instead. Republishes for REPUBLISH_WINDOW_S after startup (not forever): the
existing RobotModel displays subscribe with Volatile durability (no historical
replay), so a burst of early republishes reaches a late-starting RViz
regardless of QoS durability on either side -- but publishing forever makes
RViz tear down and rebuild its entire Links property tree on every message
(it can't know it's the same robot), which collapses any tree the operator
has expanded to inspect. TRANSIENT_LOCAL still covers any later-joining
subscriber once the burst ends.

RUN:
    ros2 run triago_control study_robot_model_filter.py
    ros2 run triago_control study_robot_model_filter.py --ros-args -p dim_mode:=remove
"""

import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from rcl_interfaces.srv import GetParameters

# Links kept fully intact: the true kinematic chain per side, shoulder to
# fingertip (numbered arm links + tool link, wrist F/T sensor group, full
# gripper/end-effector assembly). Everything else outside these prefixes is
# processed per `dim_mode`.
KEEP_PREFIXES = ('arm_left_', 'arm_right_', 'wrist_left_', 'wrist_right_',
                  'gripper_left_', 'gripper_right_')

FILTERED_TOPIC = '/robot_description_filtered'
DIM_MATERIAL_RGBA = '0.5 0.5 0.5 0.3'
REPUBLISH_PERIOD_S = 1.0
REPUBLISH_WINDOW_S = 20.0   # comfortably past RVIZ_DELAY_S (14s) in simulation_bringup.launch.py


class RobotModelFilter(Node):
    def __init__(self):
        super().__init__('study_robot_model_filter')
        self.declare_parameter('dim_mode', 'material')
        self.dim_mode = self.get_parameter('dim_mode').value
        if self.dim_mode not in ('material', 'remove'):
            self.get_logger().warn(f"Unknown dim_mode '{self.dim_mode}', defaulting to 'material'.")
            self.dim_mode = 'material'

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(String, FILTERED_TOPIC, qos)
        self.filtered_urdf = None
        self._elapsed_s = 0.0
        self._republish_timer = None

    def get_urdf(self):
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("robot_state_publisher not available after 30s!")
            return None
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value

    def _dim_link_material(self, link):
        name = link.get('name', '')
        for i, visual in enumerate(link.findall('visual')):
            for old_mat in visual.findall('material'):
                visual.remove(old_mat)
            mat = ET.SubElement(visual, 'material')
            mat.set('name', f'__study_dim_{name}_{i}')
            color = ET.SubElement(mat, 'color')
            color.set('rgba', DIM_MATERIAL_RGBA)

    def build_filtered_urdf(self, urdf_str):
        root = ET.fromstring(urdf_str)
        n_touched = 0
        for link in root.findall('link'):
            name = link.get('name', '')
            if name.startswith(KEEP_PREFIXES):
                continue
            if self.dim_mode == 'remove':
                for visual in link.findall('visual'):
                    link.remove(visual)
                    n_touched += 1
            else:
                visuals = link.findall('visual')
                if visuals:
                    self._dim_link_material(link)
                    n_touched += len(visuals)
        self.get_logger().info(
            f"[Filter] dim_mode='{self.dim_mode}': touched {n_touched} <visual> "
            f"element(s) outside the arm/gripper chains; publishing on {FILTERED_TOPIC}.")
        return ET.tostring(root, encoding='unicode')

    def publish_once(self):
        if self.filtered_urdf is not None:
            self.pub.publish(String(data=self.filtered_urdf))

    def _republish_tick(self):
        self.publish_once()
        self._elapsed_s += REPUBLISH_PERIOD_S
        if self._elapsed_s >= REPUBLISH_WINDOW_S:
            self._republish_timer.cancel()
            self.get_logger().info(
                f"[Filter] Republish burst finished after {REPUBLISH_WINDOW_S}s; "
                f"relying on TRANSIENT_LOCAL for any later subscriber.")


def main():
    rclpy.init()
    node = RobotModelFilter()

    print("[Main] Fetching URDF from robot_state_publisher...")
    urdf_str = node.get_urdf()
    if urdf_str is None:
        print("[Error] Could not fetch URDF. Exiting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.filtered_urdf = node.build_filtered_urdf(urdf_str)
    node.publish_once()
    node._republish_timer = node.create_timer(REPUBLISH_PERIOD_S, node._republish_tick)
    print(f"[Main] Publishing filtered robot_description (dim_mode={node.dim_mode}) "
          f"on {FILTERED_TOPIC}, republishing every {REPUBLISH_PERIOD_S}s for "
          f"{REPUBLISH_WINDOW_S}s then stopping.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
