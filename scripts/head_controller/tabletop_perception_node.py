#!/usr/bin/env python3
"""
tabletop_perception_node.py — generic OBB tabletop perception with memory
tracking, ported from the sibling ROS 2 package `triago_perception`
(repository chiarapanagrosso/Triago_Project, src/tabletop_perception_node.cpp,
C++/PCL). This is a 1:1 behavioural port to this project's pure numpy/scipy
stack (project convention — see camera_interface.py's module docstring for
why cv_bridge/PCL-style extra dependencies are avoided here).

WHAT THIS NODE IS, AND HOW IT DIFFERS FROM THE EXISTING head_control PIPELINE
    main_head.py's own pipeline (table_segmenter.py + object_detector.py +
    object_tracker.py) assumes UPRIGHT CYLINDERS and classifies red/blue by
    colour, with a known-table-height gate and EMA-smoothed dimension fusion.
    This node makes NONE of those assumptions:
        * table plane RANSAC is UNBIASED (no verticality filter, no known-
          height gate) — it finds whichever dominant plane best explains the
          cloud, exactly like the upstream C++ node.
        * every above/below-plane cluster gets a Z-ALIGNED ORIENTED bounding
          box (2D PCA in the XY-plane, axis fixed to the plane's own "up"),
          not a cylinder fit — works for arbitrary object shapes.
        * temporal fusion is GROW-ONLY (matched dimensions can only increase,
          never shrink) with a hard N-frame persistence/deletion window —
          this is an intentional, faithful port of the ORIGINAL node's own
          design, not an endorsement over the EMA policy in
          head_control/object_tracker.py (see that module's docstring for why
          EMA was chosen for the cylinder pipeline instead).
        * objects are classified obstacle vs. target purely from
          `position.z < table_surface_z` (below the table's own top surface
          => sitting on the floor/pedestal => obstacle; at/above it =>
          candidate manipulation target) and the first stable ("frames_unseen
          == 0") target's pose is published as the ProDMP attractor, exactly
          mirroring the upstream node.
    Runs fully independently of head_control/perception_pipeline.py: no
    shared state, own tracked-object memory, own topics. Safe to run
    alongside main_head.py.

INPUT TOPIC — `/filtered_cloud` (overridable via the `cloud_topic` ROS param):
    This is NOT produced anywhere in this repository. In the upstream stack
    (chiarapanagrosso/Triago_Project's triago_perception/launch/
    perception.launch.py) it is a `topic_tools/throttle` of the RealSense's
    own `/{camera}/depth/color/points`, throttled to ~10 Hz. This node keeps
    the same interface (subscribe, don't produce, that cloud) so it can be
    launched against whatever provides `/filtered_cloud` in a given setup.

FAITHFUL-PORT NOTE (intentional, not a bug fix): the original node's matching
loop searches ALL tracked objects for the nearest one within threshold for
EVERY new cluster, without excluding tracks already matched earlier in the
same frame — so two nearby clusters in one frame COULD both match the same
track. This is preserved exactly as in the source node (task is to duplicate/
adapt, not to redesign the tracking logic).

TF: exact-time-only lookup at the cloud's own header stamp (no "latest
available" fallback) — matches the original node's
`tf2::durationFromSec(0.05)` timeout with no secondary attempt.

HEAD_PLOTTER BRIDGE (additive — does not change anything above)
    head_plotter.py (the existing dashboard for main_head.py's cylinder
    pipeline) expects a specific schema:
        /head_perception/markers    MarkerArray with ns="objects" CYLINDER
                                     markers (colour red/blue) + ns="table_top"
                                     CUBE.
        /head_perception/telemetry  Float64MultiArray
                                     [n_raw, n_crop, plane_z, look_err_deg,
                                      slack, proc_ms, red_conf, blue_conf,
                                      map_size]
    This node has neither cylinders nor colour, so a like-for-like feed is
    impossible; instead this bridge re-expresses what this node DOES have
    (OBB targets/obstacles, table plane) in that same schema so the SAME
    dashboard can display this pipeline's output without modification:
        * every tracked target/obstacle -> one CYLINDER marker (ns="objects")
          sized from the OBB's footprint diameter/height. Targets are tinted
          red (r>0.5) so the plotter's "red vs blue" classifier bins them as
          "red"; obstacles are tinted blue (b>0.5) so they land in the "blue"
          series. This is a display convenience only — it does NOT mean this
          node performs colour classification, and it has zero effect on the
          real /perception/* outputs above.
        * the table AABB -> one ns="table_top" CUBE at the detected surface Z.
        * telemetry fields that have no equivalent here (look_err_deg, slack,
          red_conf/blue_conf) are published as 0.0 rather than invented.
    Fields NOT read by head_plotter.py's GT/error columns still render (the
    plotter's GT_RED/GT_BLUE circles are unrelated to this generic pipeline);
    this bridge is about making live detections visible, not about matching
    the cylinder ground truth.
"""

import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float64MultiArray

import triago_control.head_control.config as cfg


# ========================================================================== #
# PointCloud2 encode/decode (manual — no sensor_msgs_py / ros_numpy dep)     #
# ========================================================================== #
_PCL_DATATYPE_TO_NUMPY = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _decode_pointcloud2_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode any PointCloud2 into an (N, 3) float64 XYZ array.

    Builds a numpy structured dtype directly from `msg.fields` (works for any
    field layout/order — XYZ-only, XYZRGB, XYZI, ...) and reads out just x/y/z.
    Non-finite rows (NaN/Inf — "no return") are dropped, mirroring PCL's own
    handling of invalid points.
    """
    field_names = {f.name for f in msg.fields}
    if not {"x", "y", "z"} <= field_names:
        return np.empty((0, 3), dtype=np.float64)

    names, formats, offsets = [], [], []
    for f in msg.fields:
        np_type = _PCL_DATATYPE_TO_NUMPY.get(f.datatype)
        if np_type is None:
            continue
        names.append(f.name)
        formats.append(np_type)
        offsets.append(f.offset)

    if msg.point_step <= 0:
        return np.empty((0, 3), dtype=np.float64)

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                       "itemsize": msg.point_step})
    n_pts = len(msg.data) // msg.point_step
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float64)

    arr = np.frombuffer(msg.data, dtype=dtype, count=n_pts)
    pts = np.stack(
        [arr["x"].astype(np.float64), arr["y"].astype(np.float64), arr["z"].astype(np.float64)],
        axis=-1,
    )
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def _make_xyz_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Encode an (N, 3) array as a plain XYZ PointCloud2 (no colour) — matches
    the original node's `pcl::PointXYZ` table/objects cloud output."""
    n = len(points)
    data = np.zeros(n, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
    if n:
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = data.tobytes()
    return msg


# ========================================================================== #
# Geometry: plane RANSAC, Euclidean clustering, Z-aligned OBB fit            #
# ========================================================================== #
def _segment_plane(points: np.ndarray, iters: int, dist_thresh: float, rng):
    """Unbiased plane RANSAC (any orientation/height).

    Deliberately NOT head_control/table_segmenter.py's TableSegmenter, which
    is biased toward horizontal planes near a KNOWN table height — a
    scene-specific prior this generic node must not carry (see config.py's
    §13 header comment). Mirrors `pcl::SACSegmentation<PointXYZ>` with
    `SACMODEL_PLANE` + `setOptimizeCoefficients(true)`: RANSAC for the
    candidate with the most inliers, then an SVD least-squares refit on the
    winning inlier set.

    Returns a boolean inlier mask, or None if no plane was found.
    """
    n = len(points)
    if n < 3:
        return None

    best_count = 0
    best_normal = None
    best_d = 0.0
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ p0
        dist = np.abs(points @ normal + d)
        count = int(np.count_nonzero(dist < dist_thresh))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d

    if best_normal is None or best_count == 0:
        return None

    inlier_mask = np.abs(points @ best_normal + best_d) < dist_thresh
    inliers = points[inlier_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    refined_normal = vh[-1]
    refined_d = -refined_normal @ centroid
    return np.abs(points @ refined_normal + refined_d) < dist_thresh


def _euclidean_cluster(points: np.ndarray, tolerance: float, min_size: int, max_size: int):
    """KD-tree region growing — same approach as
    head_control/object_detector.py's `_euclidean_cluster`, parameterised
    independently here (own tolerance/size bounds, no cylinder-specific
    downstream assumptions)."""
    n = len(points)
    if n == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for seed in range(n):
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        comp = [seed]
        while queue:
            j = queue.pop()
            neighbours = tree.query_ball_point(points[j], tolerance)
            for k in neighbours:
                if not visited[k]:
                    visited[k] = True
                    queue.append(k)
                    comp.append(k)
        if min_size <= len(comp) <= max_size:
            clusters.append(np.array(comp))
    return clusters


def _fit_z_aligned_obb(cluster_pts: np.ndarray):
    """Z-aligned oriented bounding box via 2D PCA — direct port of the
    original node's Eigen-based fit.

    2D PCA (points flattened to z=0) gives the two horizontal principal axes;
    the third axis is forced to +Z, then the second horizontal axis is
    re-derived as cross(z, axis0) so the resulting frame is always a proper
    (det=+1) right-handed rotation regardless of the PCA eigenvectors' own
    handedness (cross(z, axis0) . cross(cross(z,axis0), z) algebra guarantees
    this — see the reasoning in the C++ node this was ported from).

    Returns (position (3,), orientation_xyzw (4,), dimensions (3,)).
    """
    centroid = cluster_pts.mean(axis=0)

    pts_2d = cluster_pts.copy()
    pts_2d[:, 2] = 0.0
    centroid_2d = pts_2d.mean(axis=0)
    diffs_2d = pts_2d - centroid_2d
    cov_3d = (diffs_2d.T @ diffs_2d) / len(diffs_2d)
    cov_2d = cov_3d[:2, :2]

    # Ascending eigenvalues, eigenvectors as columns — same convention as
    # Eigen::SelfAdjointEigenSolver used upstream.
    _, eigvecs_2d = np.linalg.eigh(cov_2d)

    eigen_vectors_pca = np.eye(3)
    eigen_vectors_pca[:2, :2] = eigvecs_2d
    col0 = eigen_vectors_pca[:, 0]
    col2 = eigen_vectors_pca[:, 2]                 # fixed [0, 0, 1]
    col1 = np.cross(col2, col0)
    col1_norm = np.linalg.norm(col1)
    if col1_norm < 1e-9:
        # Degenerate (col0 parallel to z, shouldn't happen since col0 is
        # purely horizontal by construction) — fall back to a perpendicular
        # horizontal axis.
        col1 = np.array([-col0[1], col0[0], 0.0])
        col1_norm = np.linalg.norm(col1)
    eigen_vectors_pca[:, 1] = col1 / col1_norm

    projected = (cluster_pts - centroid) @ eigen_vectors_pca
    min_pt = projected.min(axis=0)
    max_pt = projected.max(axis=0)
    dimensions = max_pt - min_pt

    mean_diagonal = 0.5 * (max_pt + min_pt)
    position = eigen_vectors_pca @ mean_diagonal + centroid
    orientation_xyzw = Rotation.from_matrix(eigen_vectors_pca).as_quat()

    return position, orientation_xyzw, dimensions


# ========================================================================== #
# Memory tracking                                                            #
# ========================================================================== #
@dataclass
class TrackedObject:
    id: int
    position: np.ndarray
    orientation: np.ndarray     # quaternion, xyzw
    dimensions: np.ndarray
    frames_unseen: int = 0
    matched_this_frame: bool = False
    is_obstacle: bool = False   # remembered classification, for safe deletion


# ========================================================================== #
# Marker helpers                                                             #
# ========================================================================== #
def _make_aabb_marker(min_pt, max_pt, ns, marker_id, rgb, frame_id, stamp, padding=0.0):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float((max_pt[0] + min_pt[0]) / 2.0)
    m.pose.position.y = float((max_pt[1] + min_pt[1]) / 2.0)
    m.pose.position.z = float((max_pt[2] + min_pt[2]) / 2.0)
    m.pose.orientation.w = 1.0
    m.scale.x = float(max_pt[0] - min_pt[0] + padding)
    m.scale.y = float(max_pt[1] - min_pt[1] + padding)
    m.scale.z = float(max_pt[2] - min_pt[2] + padding)
    m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.5)
    return m


def _make_obb_marker(position, orientation_xyzw, dimensions, ns, marker_id, rgb,
                      frame_id, stamp, padding=0.0, min_scale=0.01):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2])
    m.pose.orientation.x = float(orientation_xyzw[0])
    m.pose.orientation.y = float(orientation_xyzw[1])
    m.pose.orientation.z = float(orientation_xyzw[2])
    m.pose.orientation.w = float(orientation_xyzw[3])
    m.scale.x = max(float(dimensions[0] + padding * 2.0), min_scale)
    m.scale.y = max(float(dimensions[1] + padding * 2.0), min_scale)
    m.scale.z = max(float(dimensions[2] + padding * 2.0), min_scale)
    m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=0.5)
    return m


def _make_delete_marker(ns, marker_id, frame_id, stamp):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.action = Marker.DELETE
    return m


# ========================================================================== #
# Node                                                                       #
# ========================================================================== #
class TabletopPerceptionNode(Node):
    def __init__(self):
        super().__init__("tabletop_perception_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.target_frame = self.declare_parameter(
            "target_frame", cfg.TP_TARGET_FRAME).value
        cloud_topic = self.declare_parameter(
            "cloud_topic", cfg.TP_CLOUD_TOPIC).value

        self.cloud_subscriber = self.create_subscription(
            PointCloud2, cloud_topic, self._cloud_callback, qos_profile_sensor_data
        )

        self.table_publisher = self.create_publisher(
            PointCloud2, "perception/table_cloud", 10)
        self.objects_publisher = self.create_publisher(
            PointCloud2, "perception/objects_cloud", 10)
        self.bbox_publisher = self.create_publisher(
            MarkerArray, "perception/bounding_boxes", 10)
        self.target_pose_publisher = self.create_publisher(
            PoseStamped, "perception/target_pose", 10)
        self.table_pose_publisher = self.create_publisher(
            PoseStamped, "perception/table_pose", 10)

        # --- head_plotter.py bridge (additive, see module docstring) -------
        publish_to_head_plotter = self.declare_parameter(
            "publish_to_head_plotter", True).value
        self._head_plotter_markers_pub = None
        self._head_plotter_telemetry_pub = None
        if publish_to_head_plotter:
            self._head_plotter_markers_pub = self.create_publisher(
                MarkerArray, "/head_perception/markers", 1)
            self._head_plotter_telemetry_pub = self.create_publisher(
                Float64MultiArray, "/head_perception/telemetry", 10)

        # --- Memory tracking state ----------------------------------------
        self._tracked_objects = []
        self._next_marker_id = 10   # avoid clashing with the table marker (id 0)
        self._rng = np.random.default_rng(0)

        # --- No-data watchdog -----------------------------------------------
        # Diagnoses the single most common failure mode of this node: if
        # `cloud_topic` is never actually published (see module docstring —
        # the upstream project's own launch file has a topic-name mismatch
        # between its throttle output and what this node subscribes to),
        # _cloud_callback never runs and EVERY downstream topic (including
        # the head_plotter bridge) stays empty forever, with no other signal
        # that anything is wrong. This makes that failure loud instead of
        # silent.
        self._n_clouds_received = 0
        self._cloud_topic_name = cloud_topic
        self.create_timer(5.0, self._watchdog_tick)

        self.get_logger().info(
            "\n"
            "==================================================================\n"
            " Tabletop Perception Node initialized with Memory Tracking.\n"
            "------------------------------------------------------------------\n"
            f"  cloud_topic  : {cloud_topic}\n"
            f"  target_frame : {self.target_frame}\n"
            "==================================================================")

    def _watchdog_tick(self):
        if self._n_clouds_received > 0:
            return   # data is flowing, nothing to report
        self.get_logger().warn(
            f"No messages received yet on '{self._cloud_topic_name}' — "
            "_cloud_callback has never run, so table_cloud/objects_cloud/"
            "bounding_boxes/target_pose and the head_plotter bridge "
            "(/head_perception/*) are all EMPTY. This is almost always why "
            "head_plotter.py shows nothing. Check:\n"
            f"    ros2 topic list | grep -i point\n"
            f"    ros2 topic hz {self._cloud_topic_name}\n"
            "If nothing publishes that topic, run cloud_relay_node.py to "
            "bridge your real depth-camera cloud topic to it, or override "
            "with --ros-args -p cloud_topic:=<your real topic>.")

    # ------------------------------------------------------------------ #
    def _cloud_callback(self, msg: PointCloud2):
        self._n_clouds_received += 1
        _t_start = time.perf_counter()

        # 1. Exact-time TF lookup + transform into target_frame. No "latest
        #    available" fallback — matches the original node's single
        #    exact-stamp attempt with a 0.05s tf2 timeout.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=cfg.TP_TF_TIMEOUT_S)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as ex:
            self.get_logger().warn(
                f"Could not transform cloud: {ex}", throttle_duration_sec=2.0)
            return

        q = transform.transform.rotation
        t = transform.transform.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        translation = np.array([t.x, t.y, t.z])

        points_src = _decode_pointcloud2_xyz(msg)
        if len(points_src) == 0:
            return
        points = points_src @ R.T + translation

        frame_id = self.target_frame     # tf2::doTransform sets this on the result
        stamp = msg.header.stamp

        # 2. RANSAC planar segmentation (unbiased).
        inlier_mask = _segment_plane(
            points, cfg.TP_PLANE_RANSAC_ITERS, cfg.TP_PLANE_DIST_THRESH, self._rng)
        if inlier_mask is None or not inlier_mask.any():
            self.get_logger().warn("Table not found.", throttle_duration_sec=2.0)
            return

        # 3. Extract table and objects.
        cloud_table = points[inlier_mask]
        cloud_objects = points[~inlier_mask]

        # 4. Table AABB ("monolithic extension" down to the floor).
        min_pt_table = cloud_table.min(axis=0).copy()
        max_pt_table = cloud_table.max(axis=0).copy()
        min_pt_table[2] = 0.0
        table_surface_z = float(max_pt_table[2])

        marker_array = MarkerArray()
        marker_array.markers.append(_make_aabb_marker(
            min_pt_table, max_pt_table, "table_bbox", 0, (0.0, 1.0, 0.0),
            frame_id, stamp, padding=cfg.TP_TABLE_BBOX_PADDING))

        table_pose_msg = PoseStamped()
        table_pose_msg.header.frame_id = frame_id
        table_pose_msg.header.stamp = stamp
        table_pose_msg.pose.position.x = float((max_pt_table[0] + min_pt_table[0]) / 2.0)
        table_pose_msg.pose.position.y = float((max_pt_table[1] + min_pt_table[1]) / 2.0)
        table_pose_msg.pose.position.z = table_surface_z
        table_pose_msg.pose.orientation.w = 1.0
        self.table_pose_publisher.publish(table_pose_msg)

        # 5. Euclidean cluster extraction on the non-table residue.
        clusters = _euclidean_cluster(
            cloud_objects, cfg.TP_CLUSTER_TOLERANCE,
            cfg.TP_CLUSTER_MIN_POINTS, cfg.TP_CLUSTER_MAX_POINTS)

        # --- Reset matching flags for memory --------------------------------
        for tracked in self._tracked_objects:
            tracked.matched_this_frame = False

        # 6. Process new clusters & match to memory.
        for idx in clusters:
            cluster_pts = cloud_objects[idx]
            position, orientation_xyzw, dimensions = _fit_z_aligned_obb(cluster_pts)

            # --- Hysteresis & matching (faithful port — see module
            # docstring: intentionally does NOT exclude tracks already
            # matched earlier in this same frame). --------------------------
            min_dist = float("inf")
            best_match = None
            for tracked in self._tracked_objects:
                dist = float(np.linalg.norm(tracked.position - position))
                if dist < min_dist and dist < cfg.TP_MATCH_DIST_THRESH:
                    min_dist = dist
                    best_match = tracked

            if best_match is not None:
                best_match.position = position
                best_match.orientation = orientation_xyzw
                # HYSTERESIS: dimensions can only grow, never shrink from occlusion.
                best_match.dimensions = np.maximum(best_match.dimensions, dimensions)
                best_match.frames_unseen = 0
                best_match.matched_this_frame = True
            else:
                self._tracked_objects.append(TrackedObject(
                    id=self._next_marker_id,
                    position=position,
                    orientation=orientation_xyzw,
                    dimensions=dimensions,
                    frames_unseen=0,
                    matched_this_frame=True,
                ))
                self._next_marker_id += 1

        # 7. Publish tracked objects & handle persistence.
        primary_target_published = False
        surviving = []
        for tracked in self._tracked_objects:
            if not tracked.matched_this_frame:
                tracked.frames_unseen += 1

            # Occluded/missing too long -> remove and emit a DELETE marker.
            if tracked.frames_unseen >= cfg.TP_MAX_UNSEEN_FRAMES:
                ns = "obstacle_obb" if tracked.is_obstacle else "target_obb"
                marker_array.markers.append(
                    _make_delete_marker(ns, tracked.id, frame_id, stamp))
                continue

            # Classify current object based on Z height vs. table.
            tracked.is_obstacle = tracked.position[2] < table_surface_z

            if tracked.is_obstacle:
                marker_array.markers.append(_make_obb_marker(
                    tracked.position, tracked.orientation, tracked.dimensions,
                    "obstacle_obb", tracked.id, (1.0, 0.5, 0.0), frame_id, stamp,
                    padding=cfg.TP_OBSTACLE_PADDING, min_scale=cfg.TP_MIN_MARKER_SCALE))
            else:
                marker_array.markers.append(_make_obb_marker(
                    tracked.position, tracked.orientation, tracked.dimensions,
                    "target_obb", tracked.id, (1.0, 0.0, 0.0), frame_id, stamp,
                    padding=cfg.TP_TARGET_PADDING, min_scale=cfg.TP_MIN_MARKER_SCALE))

                # Publish the ProDMP attractor for the most reliable target.
                if not primary_target_published and tracked.frames_unseen == 0:
                    target_pose_msg = PoseStamped()
                    target_pose_msg.header.frame_id = frame_id
                    target_pose_msg.header.stamp = stamp
                    target_pose_msg.pose.position.x = float(tracked.position[0])
                    target_pose_msg.pose.position.y = float(tracked.position[1])
                    target_pose_msg.pose.position.z = float(tracked.position[2])
                    target_pose_msg.pose.orientation.x = float(tracked.orientation[0])
                    target_pose_msg.pose.orientation.y = float(tracked.orientation[1])
                    target_pose_msg.pose.orientation.z = float(tracked.orientation[2])
                    target_pose_msg.pose.orientation.w = float(tracked.orientation[3])
                    self.target_pose_publisher.publish(target_pose_msg)
                    primary_target_published = True

            surviving.append(tracked)
        self._tracked_objects = surviving

        # 8. Convert and publish clouds + markers.
        self.table_publisher.publish(_make_xyz_pointcloud2(cloud_table, frame_id, stamp))
        self.objects_publisher.publish(_make_xyz_pointcloud2(cloud_objects, frame_id, stamp))
        self.bbox_publisher.publish(marker_array)

        # 9. head_plotter.py bridge (additive — see module docstring).
        if self._head_plotter_markers_pub is not None:
            proc_ms = (time.perf_counter() - _t_start) * 1000.0
            self._publish_head_plotter_bridge(
                frame_id, stamp, table_surface_z, len(points), len(cloud_objects), proc_ms)

    def _publish_head_plotter_bridge(self, frame_id, stamp, table_surface_z,
                                      n_raw, n_crop, proc_ms):
        """Re-express this node's OBB tracks in head_plotter.py's expected
        schema (ns="objects" CYLINDER markers + ns="table_top" CUBE +
        9-element telemetry array). See module docstring for the exact
        mapping and its limits (no colour/GT here — display only)."""
        markers = MarkerArray()

        table_marker = Marker()
        table_marker.header.frame_id = frame_id
        table_marker.header.stamp = stamp
        table_marker.ns = "table_top"
        table_marker.id = 0
        table_marker.type = Marker.CUBE
        table_marker.action = Marker.ADD
        table_marker.pose.position.z = table_surface_z
        table_marker.pose.orientation.w = 1.0
        table_marker.scale.x = float(cfg.TABLE_SIZE[0])
        table_marker.scale.y = float(cfg.TABLE_SIZE[1])
        table_marker.scale.z = 0.005
        table_marker.color = ColorRGBA(r=0.1, g=0.9, b=0.3, a=0.6)
        markers.markers.append(table_marker)

        for i, tracked in enumerate(self._tracked_objects):
            cyl = Marker()
            cyl.header.frame_id = frame_id
            cyl.header.stamp = stamp
            cyl.ns = "objects"
            cyl.id = 10 + i
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = float(tracked.position[0])
            cyl.pose.position.y = float(tracked.position[1])
            cyl.pose.position.z = float(tracked.position[2])
            cyl.pose.orientation.w = 1.0
            footprint_diameter = float(max(tracked.dimensions[0], tracked.dimensions[1]))
            cyl.scale.x = max(footprint_diameter, cfg.TP_MIN_MARKER_SCALE)
            cyl.scale.y = max(footprint_diameter, cfg.TP_MIN_MARKER_SCALE)
            cyl.scale.z = max(float(tracked.dimensions[2]), cfg.TP_MIN_MARKER_SCALE)
            # Display-only tint so head_plotter's red/blue classifier can bin
            # target vs. obstacle — NOT a colour classification of the object.
            if tracked.is_obstacle:
                cyl.color = ColorRGBA(r=0.1, g=0.3, b=1.0, a=0.9)   # -> "blue" bin
            else:
                cyl.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9)   # -> "red" bin
            markers.markers.append(cyl)

        self._head_plotter_markers_pub.publish(markers)

        telemetry = Float64MultiArray()
        # [n_raw, n_crop, plane_z, look_err_deg, slack, proc_ms, red_conf,
        #  blue_conf, map_size] — this node has no look-at controller or
        # colour confidence, so those fields are honestly reported as 0.0
        # rather than invented (see module docstring).
        telemetry.data = [
            float(n_raw), float(n_crop), float(table_surface_z),
            0.0, 0.0, float(proc_ms), 0.0, 0.0, float(len(self._tracked_objects)),
        ]
        self._head_plotter_telemetry_pub.publish(telemetry)


def main():
    rclpy.init()
    node = TabletopPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
