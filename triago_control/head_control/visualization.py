"""
Visualisation helpers for RViz.

Publishes, all in the base_footprint frame so RViz (with robot_state_publisher
TF) places them correctly:
    * /head_perception/cloud        PointCloud2 (XYZRGB) of the cropped cloud
    * /head_perception/markers      MarkerArray:
          - table bounding box (obstacle volume) + highlighted top plane
          - one CYLINDER per detected object, coloured by class
          - a text label above each object (class + dimensions)
          - the camera optical-axis ray (where the head is looking)

PointCloud2 packing note: RViz expects XYZ as float32 and colour packed into a
single float32 'rgb' field as 0x00RRGGBB reinterpreted as float. We build that
with a numpy structured array and ship the raw bytes.
"""

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import triago_control.head_control.config as cfg


# ---------------------------------------------------------------------- #
# PointCloud2                                                             #
# ---------------------------------------------------------------------- #
def make_pointcloud2(points, colors_uint8, frame_id, stamp):
    """Build an XYZRGB PointCloud2 from (N,3) points + (N,3) uint8 colours."""
    n = len(points)
    data = np.zeros(
        n, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)]
    )
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    r = colors_uint8[:, 0].astype(np.uint32)
    g = colors_uint8[:, 1].astype(np.uint32)
    b = colors_uint8[:, 2].astype(np.uint32)
    rgb_uint32 = (r << 16) | (g << 8) | b
    data["rgb"] = rgb_uint32.view(np.float32)       # reinterpret bits as float

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


# ---------------------------------------------------------------------- #
# Markers                                                                 #
# ---------------------------------------------------------------------- #
class PerceptionVisualizer:
    def __init__(self, frame_id=cfg.BASE_FRAME):
        self.frame_id = frame_id
        # Track how many object/label markers we published last frame so we can
        # DELETE only the now-unused IDs. This avoids the DELETEALL-every-frame
        # pattern that makes markers blink in RViz.
        self._prev_obj_count = 0

    def _base_marker(self, ns, mid, mtype, stamp):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def build(self, result, target_base, cam_pos_base, stamp):
        """Return a MarkerArray for a PerceptionResult.

        target_base : current look-at point (base frame)
        cam_pos_base: camera origin in base frame (ray start)
        """
        arr = MarkerArray()

        # NOTE: we deliberately do NOT use Marker.DELETEALL here — that makes
        # every marker blink each frame. Instead we reuse stable IDs and emit
        # targeted DELETEs only for object slots that disappeared this frame.

        # --- Table top plane (highlighted thin slab) -------------------
        if result.plane is not None:
            top = self._base_marker("table_top", 0, Marker.CUBE, stamp)
            top.pose.position.x = float(cfg.TABLE_CENTER_BASE[0])
            top.pose.position.y = float(cfg.TABLE_CENTER_BASE[1])
            top.pose.position.z = float(result.plane.height)
            top.scale.x = float(cfg.TABLE_SIZE[0])
            top.scale.y = float(cfg.TABLE_SIZE[1])
            top.scale.z = 0.005
            top.color = ColorRGBA(r=0.1, g=0.9, b=0.3, a=0.6)
            arr.markers.append(top)

            # Table body (obstacle volume) — semi-transparent box.
            body = self._base_marker("table_body", 1, Marker.CUBE, stamp)
            body.pose.position.x = float(cfg.TABLE_CENTER_BASE[0])
            body.pose.position.y = float(cfg.TABLE_CENTER_BASE[1])
            body.pose.position.z = float(result.plane.height - cfg.TABLE_SIZE[2] / 2.0)
            body.scale.x = float(cfg.TABLE_SIZE[0])
            body.scale.y = float(cfg.TABLE_SIZE[1])
            body.scale.z = float(cfg.TABLE_SIZE[2])
            body.color = ColorRGBA(r=0.6, g=0.4, b=0.2, a=0.15)
            arr.markers.append(body)

        # --- Detected cylinders ----------------------------------------
        for i, obj in enumerate(result.objects):
            cyl = self._base_marker("objects", 10 + i, Marker.CYLINDER, stamp)
            cyl.pose.position.x = float(obj.center[0])
            cyl.pose.position.y = float(obj.center[1])
            cyl.pose.position.z = float(obj.center[2])
            cyl.scale.x = float(2.0 * obj.radius)
            cyl.scale.y = float(2.0 * obj.radius)
            cyl.scale.z = float(obj.height)
            cyl.color = self._class_color(obj.color_name)
            arr.markers.append(cyl)

            txt = self._base_marker("labels", 100 + i, Marker.TEXT_VIEW_FACING, stamp)
            txt.pose.position.x = float(obj.center[0])
            txt.pose.position.y = float(obj.center[1])
            txt.pose.position.z = float(obj.center[2] + obj.height / 2.0 + 0.05)
            txt.scale.z = 0.04
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f"{obj.label}\nr={obj.radius*100:.1f}cm h={obj.height*100:.1f}cm\nconf={obj.confidence*100:.0f}% (cov {obj.arc_coverage*100:.0f}%)"
            arr.markers.append(txt)

        # --- Camera look-at ray ----------------------------------------
        ray = self._base_marker("look_ray", 200, Marker.ARROW, stamp)
        ray.points = [
            Point(x=float(cam_pos_base[0]), y=float(cam_pos_base[1]), z=float(cam_pos_base[2])),
            Point(x=float(target_base[0]), y=float(target_base[1]), z=float(target_base[2])),
        ]
        ray.scale.x = 0.01      # shaft diameter
        ray.scale.y = 0.03      # head diameter
        ray.scale.z = 0.05      # head length
        # --- Stale-slot cleanup: DELETE object/label IDs no longer used -----
        n_now = len(result.objects)
        for i in range(n_now, self._prev_obj_count):
            for ns, base_id in (("objects", 10), ("labels", 100)):
                d = Marker()
                d.header.frame_id = self.frame_id
                d.header.stamp = stamp
                d.ns = ns
                d.id = base_id + i
                d.action = Marker.DELETE
                arr.markers.append(d)
        self._prev_obj_count = n_now

        return arr

    @staticmethod
    def _class_color(color_name):
        if color_name == "red":
            return ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)
        if color_name == "blue":
            return ColorRGBA(r=0.1, g=0.3, b=1.0, a=0.9)
        return ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.9)    # unknown -> grey
