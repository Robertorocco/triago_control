#!/usr/bin/env python3
"""
dump_live_urdf.py — fetch the LIVE robot_description straight from
robot_state_publisher via the GetParameters service (the SAME code path
HeadKinematics.fetch_urdf() uses, already proven to work — both
main_head.py and calibration_audit.py build a valid Pinocchio model from
it) and write it to a clean file on disk.

WHY THIS EXISTS
    `ros2 param get /robot_state_publisher robot_description` has proven
    fragile at the command line across multiple attempts in this session
    (a `String value is:` header line breaking XML parsers; then, even
    after stripping that, an "Error document empty" from check_urdf,
    suggesting the CLI is ALSO wrapping/quoting/truncating the value in
    some other way on this ROS 2 distro). Rather than keep guessing CLI
    flags, this script fetches the value the same way the actual head
    perception code does (a direct GetParameters service call — no CLI
    text-formatting layer involved at all) and writes the RAW string
    straight to a file with no intermediate reformatting.

USAGE
    ros2 run triago_control dump_live_urdf.py [output_path]
    (output_path defaults to /tmp/live_robot.urdf)

    Then, as before:
        check_urdf /tmp/live_robot.urdf
        urdf_to_graphviz /tmp/live_robot.urdf /tmp/triago_live

WHAT ELSE IT PRINTS
    Also greps the parsed URDF for every joint whose name or parent/child
    link contains "camera" or "gripper_head", and prints its type,
    parent, child, and xyz/rpy origin — this is the exact extrinsic
    offset chain from the last real link to the optical frames, which is
    NOT present in the checked-in (stale) triago_extracted.urdf. No
    ground-truth/scene knowledge is used anywhere in this script — it is
    a pure structural dump of the live kinematic model, same category as
    calibration_audit.py's read-only checks.
"""

import sys
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters


class UrdfDumpNode(Node):
    def __init__(self):
        super().__init__("dump_live_urdf")

    def fetch_urdf(self):
        client = self.create_client(GetParameters, "/robot_state_publisher/get_parameters")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("robot_state_publisher not available -- cannot fetch URDF.")
            return None
        req = GetParameters.Request()
        req.names = ["robot_description"]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().error("Timed out fetching robot_description.")
            return None
        return future.result().values[0].string_value


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/live_robot.urdf"

    rclpy.init()
    node = UrdfDumpNode()
    urdf_str = node.fetch_urdf()
    node.destroy_node()
    rclpy.shutdown()

    if not urdf_str:
        print("[ERROR] Got an empty/None robot_description. Is robot_state_publisher running?")
        sys.exit(1)

    with open(out_path, "w") as f:
        f.write(urdf_str)
    print(f"[OK] Wrote {len(urdf_str)} bytes to {out_path}")
    print(f"     Run:  check_urdf {out_path}")
    print(f"     Run:  urdf_to_graphviz {out_path} /tmp/triago_live")

    # --- Parse and print the camera extrinsic chain -----------------------
    try:
        root = ET.fromstring(urdf_str)
    except ET.ParseError as e:
        print(f"[ERROR] robot_description did not parse as XML: {e}")
        sys.exit(1)

    print("\n=== JOINTS touching 'camera' or 'gripper_head' (name/parent/child) ===")
    found = False
    for j in root.findall("joint"):
        name = j.get("name", "")
        parent_el = j.find("parent")
        child_el = j.find("child")
        parent = parent_el.get("link") if parent_el is not None else "?"
        child = child_el.get("link") if child_el is not None else "?"
        haystack = f"{name} {parent} {child}".lower()
        if "camera" in haystack or "gripper_head" in haystack:
            found = True
            o = j.find("origin")
            xyz = o.get("xyz") if o is not None else "0 0 0"
            rpy = o.get("rpy") if o is not None else "0 0 0"
            jtype = j.get("type", "?")
            print(f"  {name:55s} type={jtype:10s}")
            print(f"      parent={parent}")
            print(f"      child ={child}")
            print(f"      xyz={xyz}  rpy={rpy}")
    if not found:
        print("  (none found -- unexpected; the camera frames may be injected by a "
              "mechanism other than a <joint> tag, e.g. a Gazebo sensor plugin that "
              "publishes its own TF directly without a URDF joint at all)")

    # --- Also list every link with 'camera' or 'optical' in its name ------
    print("\n=== LINKS containing 'camera' or 'optical' ===")
    cam_links = [l.get("name") for l in root.findall("link")
                 if "camera" in l.get("name", "").lower() or "optical" in l.get("name", "").lower()]
    if cam_links:
        for name in cam_links:
            print(f"  {name}")
    else:
        print("  (none -- confirms the *_optical_frame frames are NOT URDF links at all; "
              "they must be published as extra TF frames by a Gazebo sensor plugin, "
              "outside the URDF/robot_state_publisher entirely)")


if __name__ == "__main__":
    main()
