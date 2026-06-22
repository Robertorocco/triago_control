# visualization_engine.py
"""
The UI / Telemetry engine.

Owns all NON-critical rendering so the real-time control loop never blocks on it:
    * builds and serves the Meshcat web visualizer (collision + visual models),
    * publishes RViz markers for the workspace obstacles and the virtual wall,
    * applies grasp-intent coloring (orange) on demand.

----------------------------------------------------------------------------
THREAD-SAFETY RULE (PRESERVED EXACTLY):
    Meshcat's WebSocket state machine is NOT thread-safe. ROS callbacks must
    NEVER touch the viewer directly. They only mutate pure-Python data
    (`meshColor`, `overrideMaterial`) under `meshcat_lock` and raise the
    `meshcat_reload_pending` flag. The dedicated `_run_viz` thread is the SINGLE
    owner of every Meshcat WebSocket call (loadViewerModel / display) and is the
    only place the viewer is reloaded / displayed.
----------------------------------------------------------------------------
"""

import threading
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from visualization_msgs.msg import Marker
try:
    import hppfcl
except ImportError:
    import pinocchio.hppfcl as hppfcl
import triago_control.qp_controller.config as cfg

# Optional external telemetry helper (debug tethers / RViz overlays). Kept optional
# so the controller still runs when the triago_control package is absent.
try:
    from triago_control.qp_controller.qp_visualizer_tutorial import QPVisualizer
    _HAS_QP_VISUALIZER = True
except Exception:
    QPVisualizer = None
    _HAS_QP_VISUALIZER = False


class VisualizationEngine:
    """Manages Meshcat, RViz markers and grasp-intent coloring (thread-safe)."""

    def __init__(self, node, model, cmodel, urdf_path):
        self.node = node
        self.model = model
        self.cmodel = cmodel
        self.urdf_path = urdf_path

        # Build the visual ("skin") model from the URDF, tinted as a green ghost
        try:
            self.vmodel = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.VISUAL, package_dirs=cfg.MESH_PATHS)
        except Exception as e:
            print(f"[Viz] Could not load meshes: {e}")
            self.vmodel = pin.GeometryModel()
        for geom in self.vmodel.geometryObjects:
            geom.meshColor = np.array([0.0, 1.0, 0.0, 0.3])  # green, transparent ghost
            geom.overrideMaterial = True

        # --- MESHCAT THREAD-SAFETY STATE ---
        self.meshcat_lock = threading.Lock()
        self.meshcat_reload_pending = False
        self.viz_meshcat = None
        self._q_provider = None  # callable returning the live joint configuration

        # Optional external telemetry visualizer (debug tethers etc.)
        self.qp_viz = QPVisualizer(self.node, ref_frame=cfg.REF_FRAME) if _HAS_QP_VISUALIZER else None

        # --- RViz MARKER PUBLISHERS ---
        self.pub_wall_marker = self.node.create_publisher(Marker, '/qp_debug/virtual_wall_marker', 10)
        self.pub_cyl_obs_marker = self.node.create_publisher(Marker, '/cylinder_obstacle_marker', 10)
        self.pub_fly_obs_marker = self.node.create_publisher(Marker, '/flying_obstacle_marker', 10)

    def add_gripper_visual_boxes(self, col_manager):
        # Mirror the collision gripper boxes into the visual model (semi-transparent orange).
        for side in ('right', 'left'):
            base_link = f'gripper_{side}_base_link'
            if self.model.existFrame(base_link):
                frame_id = self.model.getFrameId(base_link)
                parent_joint = self.model.frames[frame_id].parent
                placement = self.model.frames[frame_id].placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.05]))
                vis_obj = pin.GeometryObject(f"gripper_{side}_visual_box", parent_joint, placement, hppfcl.Box(0.05, 0.08, 0.25))
                vis_obj.meshColor = np.array([1.0, 0.5, 0.0, 0.4])
                self.vmodel.addGeometryObject(vis_obj)

    def color_collision_model(self, col_manager):
        # Tint the collision capsules/obstacles: right=red, left=blue, ground=grey, workspace.
        for geom_id in col_manager.right_geom_ids:
            if geom_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[geom_id].meshColor = np.array([1.0, 0.0, 0.0, 0.8])
                self.cmodel.geometryObjects[geom_id].overrideMaterial = True
        for geom_id in col_manager.left_geom_ids:
            if geom_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[geom_id].meshColor = np.array([0.0, 0.0, 1.0, 0.8])
                self.cmodel.geometryObjects[geom_id].overrideMaterial = True
        if hasattr(col_manager, 'ground_id') and col_manager.ground_id < len(self.cmodel.geometryObjects):
            self.cmodel.geometryObjects[col_manager.ground_id].meshColor = np.array([0.5, 0.5, 0.5, 0.5])
            self.cmodel.geometryObjects[col_manager.ground_id].overrideMaterial = True
        for obs_id in getattr(col_manager, 'workspace_obstacle_ids', []):
            if obs_id < len(self.cmodel.geometryObjects):
                name = self.cmodel.geometryObjects[obs_id].name
                if "red" in name:
                    color = [1.0, 0.0, 0.0, 0.8]
                elif "blue" in name:
                    color = [0.0, 0.0, 1.0, 0.8]
                else:  # table
                    color = [0.6, 0.4, 0.2, 0.8]
                self.cmodel.geometryObjects[obs_id].meshColor = np.array(color)
                self.cmodel.geometryObjects[obs_id].overrideMaterial = True

    def init_meshcat(self, q_provider, col_manager=None):
        # Initialize the Meshcat viewer and spawn the single viewer-owning thread.
        self._q_provider = q_provider
        if col_manager is not None:
            self.color_collision_model(col_manager)
        try:
            # Pass all three models: (physics, collision, visuals)
            self.viz_meshcat = MeshcatVisualizer(self.model, self.cmodel, self.vmodel)
            self.viz_meshcat.initViewer(open=False)
            self.viz_meshcat.loadViewerModel()
            self.viz_meshcat.displayCollisions(True)  # show capsules
            self.viz_meshcat.displayVisuals(True)     # show skin
            q0 = self._q_provider()
            if q0 is not None:
                self.viz_meshcat.display(q0)
            threading.Thread(target=self._run_viz, daemon=True).start()
        except Exception as e:
            print(f"[Viz Error] Meshcat failed to start: {e}")
            self.viz_meshcat = None

    def paint_grasp_intent(self, arm_side, color, col_manager, opaque=False):
        # Color the cylinder + gripper orange to signal a grasp in progress (thread-safe).
        self.node.get_logger().info(
            f"\033[93m[MESHCAT] Painting {color} + {arm_side} gripper orange (grasp intent).\033[0m")
        if self.viz_meshcat is None:
            return
        alpha = 1.0  # opaque for both intent and attach in the original
        orange = np.array([1.0, 0.5, 0.0, alpha])
        cyl_id = col_manager.red_cyl_id if color == "red" else col_manager.blue_cyl_id

        # Mutate only pure-Python meshColor under the lock; the run_viz thread reloads.
        with self.meshcat_lock:
            if cyl_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[cyl_id].meshColor = orange
                self.cmodel.geometryObjects[cyl_id].overrideMaterial = True
            for geom in self.vmodel.geometryObjects:
                if f"gripper_{arm_side}" in geom.name:
                    geom.meshColor = orange
                    geom.overrideMaterial = True
            # Hide the red gripper collision box (fully transparent) so it stops overlapping
            # the orange mesh, while the CBF/distance machinery keeps working untouched.
            if not opaque:
                box_id = col_manager.gripper_box_ids.get(arm_side)
                if box_id is not None and box_id < len(self.cmodel.geometryObjects):
                    self.cmodel.geometryObjects[box_id].meshColor = np.array([1.0, 0.0, 0.0, 0.0])
                    self.cmodel.geometryObjects[box_id].overrideMaterial = True
            self.meshcat_reload_pending = True

    def _run_viz(self):
        # SOLE owner of Meshcat WebSocket calls: reload on demand, then display LIVE q.
        while True:
            q = self._q_provider() if self._q_provider is not None else None
            if self.viz_meshcat is not None and q is not None:
                try:
                    with self.meshcat_lock:
                        if self.meshcat_reload_pending:
                            self.viz_meshcat.loadViewerModel()
                            self.viz_meshcat.displayCollisions(True)
                            self.viz_meshcat.displayVisuals(True)
                            self.meshcat_reload_pending = False
                        self.viz_meshcat.display(q)  # live configuration, refreshed each tick
                except Exception:
                    pass
            time.sleep(0.2)

    def publish_obstacle_marker(self, hri=None):
        # Publish the workspace obstacles (table + cylinders) to RViz, with grasp coloring.
        marker_table = Marker()
        marker_table.header.frame_id = "base_footprint"
        marker_table.header.stamp = self.node.get_clock().now().to_msg()
        marker_table.ns = "workspace"
        marker_table.id = 0
        marker_table.type = Marker.CUBE
        marker_table.action = Marker.ADD
        marker_table.pose.position.x, marker_table.pose.position.y, marker_table.pose.position.z = cfg.TABLE_POS
        marker_table.pose.orientation.w = 1.0
        marker_table.scale.x, marker_table.scale.y, marker_table.scale.z = cfg.TABLE_SIZE
        marker_table.color.r, marker_table.color.g, marker_table.color.b, marker_table.color.a = 0.6, 0.4, 0.2, 0.8
        self.pub_cyl_obs_marker.publish(marker_table)

        # Build the grasp-active id set so cylinders turn orange while grasped/attached
        grasp_active_ids = set()
        if hri is not None:
            grasp_active_ids = set(hri.grasp_margin_targets.keys()) | set(hri.attached_objects)

        def create_cyl_marker(m_id, pos, name, default_color, cyl_id_for_name):
            m = Marker()
            m.header = marker_table.header
            m.ns = "workspace"
            m.id = m_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            # Use the LIVE collision-geometry pose so the cylinder follows the
            # gripper once it has been re-parented (grasped). Before grasp this
            # equals the static workspace pose; after grasp it tracks the wrist.
            attached = cyl_id_for_name in set(hri.attached_objects) if hri is not None else False
            live_pose = None
            if cyl_id_for_name is not None and hasattr(self.node, 'col') \
                    and hasattr(self.node.col, 'cdata') \
                    and cyl_id_for_name < len(self.node.col.cdata.oMg):
                live_pose = self.node.col.cdata.oMg[cyl_id_for_name]

            if live_pose is not None:
                p = live_pose.translation
                quat = pin.Quaternion(live_pose.rotation)
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
                m.pose.orientation.x = float(quat.x); m.pose.orientation.y = float(quat.y)
                m.pose.orientation.z = float(quat.z); m.pose.orientation.w = float(quat.w)
            else:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
                m.pose.orientation.w = 1.0

            m.scale.x, m.scale.y, m.scale.z = float(cfg.CYLINDER_SIZE[0] * 2), float(cfg.CYLINDER_SIZE[0] * 2), float(cfg.CYLINDER_SIZE[1])
            if attached:
                # Grasped object rendered grey, at its live (gripper-following) pose
                m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.5, 1.0
            elif cfg.DYNAMIC_CBF and cyl_id_for_name in grasp_active_ids:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = default_color
            return m

        red_id = getattr(self.node.col, 'red_cyl_id', None) if hasattr(self.node, 'col') else None
        blue_id = getattr(self.node.col, 'blue_cyl_id', None) if hasattr(self.node, 'col') else None
        self.pub_cyl_obs_marker.publish(create_cyl_marker(1, cfg.RED_CYLINDER_POS, "red_cylinder", (1.0, 0.0, 0.0, 1.0), red_id))
        self.pub_cyl_obs_marker.publish(create_cyl_marker(2, cfg.BLUE_CYLINDER_POS, "blue_cylinder", (0.0, 0.0, 1.0, 1.0), blue_id))

    def publish_wall_marker(self):
        # Publish the virtual wall as a transparent orange cube in RViz.
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = "environment"
        m.id = 99
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(cfg.WALL_POS[0])
        m.pose.position.y = float(cfg.WALL_POS[1])
        m.pose.position.z = float(cfg.WALL_POS[2])
        m.pose.orientation.w = 1.0
        m.scale.x = float(cfg.WALL_SIZE[0])
        m.scale.y = float(cfg.WALL_SIZE[1])
        m.scale.z = float(cfg.WALL_SIZE[2])
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.5
        self.pub_wall_marker.publish(m)

    def publish_debug(self, *args, **kwargs):
        # Forward optional debug telemetry (tethers etc.) to the external visualizer if present.
        if self.qp_viz is not None:
            self.qp_viz.publish_debug(*args, **kwargs)

    def publish_teleop_tether(self):
        # Forward the teleop tether publish to the external visualizer if present.
        if self.qp_viz is not None:
            self.qp_viz.publish_teleop_tether()
