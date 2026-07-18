# visualization_engine.py
"""Non-critical rendering (Meshcat + RViz markers + grasp coloring). Meshcat is NOT thread-safe:
callbacks only mutate colors under meshcat_lock and set a reload flag; the _run_viz thread is the
sole owner of every WebSocket call."""

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

    def __init__(self, node, model, cmodel, urdf_path, world_scene=None):
        self.node = node
        self.model = model
        self.cmodel = cmodel
        self.urdf_path = urdf_path
        # Optional WorldScene driving the marker/color methods; None falls back to the legacy cfg constants.
        self.world_scene = world_scene

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
        self.qp_viz = QPVisualizer(self.node, ref_frame=cfg.REF_FRAME,
                                   world_scene=self.world_scene) if _HAS_QP_VISUALIZER else None

        # --- RViz MARKER PUBLISHERS ---
        self.pub_wall_marker = self.node.create_publisher(Marker, '/qp_debug/virtual_wall_marker', 10)
        self.pub_cyl_obs_marker = self.node.create_publisher(Marker, '/cylinder_obstacle_marker', 10)

    # Visual gripper box inflated vs its coincident collision box: identical surfaces z-fight
    # in the GPU depth test (pixel flicker), a strict size margin guarantees zero coincidence.
    GRIPPER_VISUAL_BOX_PADDING = 0.005  # [m] per-dimension

    def add_gripper_visual_boxes(self, col_manager):
        """Mirrors the gripper collision boxes as grasp-intent indicators, padded and invisible by default."""
        # Invisible (alpha=0) until a grasp is signaled: WebGL cannot depth-sort two overlapping
        # TRANSLUCENT meshes, so translucent-over-translucent flickers; invisible/opaque never does.
        pad = self.GRIPPER_VISUAL_BOX_PADDING
        for side in ('right', 'left'):
            base_link = f'gripper_{side}_base_link'
            if self.model.existFrame(base_link):
                frame_id = self.model.getFrameId(base_link)
                parent_joint = self.model.frames[frame_id].parentJoint
                placement = self.model.frames[frame_id].placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.05]))
                vis_obj = pin.GeometryObject(f"gripper_{side}_visual_box", parent_joint, placement,
                                             hppfcl.Box(0.05 + pad, 0.08 + pad, 0.25 + pad))
                vis_obj.meshColor = np.array([1.0, 0.5, 0.0, 0.0])  # invisible until a grasp is signaled
                vis_obj.overrideMaterial = True
                self.vmodel.addGeometryObject(vis_obj)

    def color_collision_model(self, col_manager):
        """Tints the collision geometry: right=red, left=blue, head=yellow, ground=grey, plus obstacles."""
        # All colors fully opaque: a translucent capsule under the translucent skin mesh z-fights in WebGL.
        for geom_id in col_manager.right_geom_ids:
            if geom_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[geom_id].meshColor = np.array([1.0, 0.0, 0.0, 1.0])
                self.cmodel.geometryObjects[geom_id].overrideMaterial = True
        for geom_id in col_manager.left_geom_ids:
            if geom_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[geom_id].meshColor = np.array([0.0, 0.0, 1.0, 1.0])
                self.cmodel.geometryObjects[geom_id].overrideMaterial = True
        # Head capsules in distinct yellow: identifiable as the quasi-static obstacle.
        for geom_id in getattr(col_manager, 'head_geom_ids', []):
            if geom_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[geom_id].meshColor = np.array([1.0, 1.0, 0.0, 1.0])
                self.cmodel.geometryObjects[geom_id].overrideMaterial = True
        if hasattr(col_manager, 'ground_id') and col_manager.ground_id < len(self.cmodel.geometryObjects):
            self.cmodel.geometryObjects[col_manager.ground_id].meshColor = np.array([0.5, 0.5, 0.5, 1.0])
            self.cmodel.geometryObjects[col_manager.ground_id].overrideMaterial = True
        for obs_id in getattr(col_manager, 'workspace_obstacle_ids', []):
            if obs_id >= len(self.cmodel.geometryObjects):
                continue
            name = self.cmodel.geometryObjects[obs_id].name
            # Color from the YAML's own field; legacy name-string heuristic only without a world_scene.
            color = None
            if self.world_scene is not None:
                spec = self.world_scene.get_obstacle(name)
                if spec is not None:
                    color = [float(c) for c in spec.color]  # [r, g, b, a] straight from YAML
            if color is None:
                if "red" in name:
                    color = [1.0, 0.0, 0.0, 1.0]
                elif "blue" in name:
                    color = [0.0, 0.0, 1.0, 1.0]
                else:  # table
                    color = [0.6, 0.4, 0.2, 1.0]
            # Force full opacity regardless of source (anti-flicker: keep every color opaque).
            color = [color[0], color[1], color[2], 1.0]
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

    def restore_object_color(self, arm_side, color, col_manager):
        # Inverse of paint_grasp_intent: drop the orange override so the cylinder
        # and gripper revert to their original materials (thread-safe).
        self.node.get_logger().info(
            f"\033[92m[MESHCAT] Restoring {color} + {arm_side} gripper original color (released).\033[0m")
        if self.viz_meshcat is None:
            return
        cyl_id = col_manager.red_cyl_id if color == "red" else col_manager.blue_cyl_id
        default = None
        if self.world_scene is not None:
            spec = self.world_scene.obstacle_for_role(color)
            if spec is not None:
                default = np.array([float(c) for c in spec.color])
        if default is None:
            default = np.array([1.0, 0.0, 0.0, 1.0]) if color == "red" else np.array([0.0, 0.0, 1.0, 1.0])
        # Force full opacity (anti-flicker) regardless of the color source.
        default = np.array([default[0], default[1], default[2], 1.0])

        with self.meshcat_lock:
            if cyl_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[cyl_id].meshColor = default
                self.cmodel.geometryObjects[cyl_id].overrideMaterial = False
            for geom in self.vmodel.geometryObjects:
                if f"gripper_{arm_side}_visual_box" == geom.name:
                    # Must go back to invisible explicitly: a synthetic box has no default material,
                    # so overrideMaterial=False alone would leave it stuck opaque orange.
                    geom.meshColor = np.array([1.0, 0.5, 0.0, 0.0])
                    geom.overrideMaterial = True
                elif f"gripper_{arm_side}" in geom.name:
                    geom.overrideMaterial = False
            # Restore the gripper collision box to its normal (visible) state.
            box_id = col_manager.gripper_box_ids.get(arm_side)
            if box_id is not None and box_id < len(self.cmodel.geometryObjects):
                self.cmodel.geometryObjects[box_id].overrideMaterial = False
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
        """Publishes the workspace obstacles to RViz with grasp coloring, iterating world_scene generically."""
        header = Marker()
        header.header.frame_id = "base_footprint"
        header.header.stamp = self.node.get_clock().now().to_msg()

        # Build the grasp-active id set so cylinders turn orange while grasped/attached
        grasp_active_ids = set()
        if hri is not None:
            grasp_active_ids = set(hri.grasp_margin_targets.keys()) | set(hri.attached_objects)

        col_manager = getattr(self.node, 'col', None) if hasattr(self.node, 'col') else None

        def geom_id_for_name(name):
            if col_manager is None:
                return None
            by_name = getattr(col_manager, '_geom_id_by_obstacle_name', None)
            if by_name and name in by_name:
                return by_name[name]
            return None

        def create_marker(m_id, obs, cyl_id_for_name):
            m = Marker()
            m.header = header.header
            m.ns = "workspace"
            m.id = m_id
            m.type = Marker.CYLINDER if obs.shape == 'cylinder' else Marker.CUBE
            m.action = Marker.ADD

            # Use the LIVE collision-geometry pose so a graspable obstacle
            # follows the gripper once it has been re-parented (grasped).
            # Before grasp this equals the static workspace pose; after grasp
            # it tracks the wrist. Static-only obstacles (table, wall) simply
            # never appear in hri.attached_objects, so this is always a no-op
            # fallback to the YAML pose for them.
            attached = (hri is not None and cyl_id_for_name is not None
                       and cyl_id_for_name in set(hri.attached_objects))
            live_pose = None
            if cyl_id_for_name is not None and col_manager is not None \
                    and hasattr(col_manager, 'cdata') \
                    and cyl_id_for_name < len(col_manager.cdata.oMg):
                live_pose = col_manager.cdata.oMg[cyl_id_for_name]

            if live_pose is not None:
                p = live_pose.translation
                quat = pin.Quaternion(live_pose.rotation)
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
                m.pose.orientation.x = float(quat.x); m.pose.orientation.y = float(quat.y)
                m.pose.orientation.z = float(quat.z); m.pose.orientation.w = float(quat.w)
            else:
                m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in obs.position]
                m.pose.orientation.w = 1.0

            if obs.shape == 'cylinder':
                m.scale.x, m.scale.y, m.scale.z = float(obs.size[0] * 2), float(obs.size[0] * 2), float(obs.size[1])
            else:
                m.scale.x, m.scale.y, m.scale.z = [float(v) for v in obs.size]

            if attached:
                # Grasped object rendered grey, at its live (gripper-following) pose
                m.color.r, m.color.g, m.color.b, m.color.a = 0.5, 0.5, 0.5, 1.0
            elif cfg.DYNAMIC_CBF and cyl_id_for_name in grasp_active_ids:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in obs.color]
            return m

        if self.world_scene is not None:
            for i, obs in enumerate(self.world_scene.static_obstacles):
                if obs.role == 'wall':
                    continue  # the wall keeps its own dedicated publish_wall_marker
                if not obs.collision and obs.role != 'table':
                    continue  # non-colliding, non-table extras: skip (nothing to show yet)
                cyl_id = geom_id_for_name(obs.name)
                self.pub_cyl_obs_marker.publish(create_marker(i, obs, cyl_id))
        else:
            # --- Legacy path (no world_scene loaded): unchanged from before. ---
            from types import SimpleNamespace
            legacy_table = SimpleNamespace(shape='box', position=cfg.TABLE_POS, size=cfg.TABLE_SIZE,
                                           color=(0.6, 0.4, 0.2, 0.8), role='table')
            legacy_red = SimpleNamespace(shape='cylinder', position=cfg.RED_CYLINDER_POS,
                                         size=cfg.CYLINDER_SIZE, color=(1.0, 0.0, 0.0, 1.0), role='graspable')
            legacy_blue = SimpleNamespace(shape='cylinder', position=cfg.BLUE_CYLINDER_POS,
                                          size=cfg.CYLINDER_SIZE, color=(0.0, 0.0, 1.0, 1.0), role='graspable')
            red_id = getattr(col_manager, 'red_cyl_id', None) if col_manager is not None else None
            blue_id = getattr(col_manager, 'blue_cyl_id', None) if col_manager is not None else None
            self.pub_cyl_obs_marker.publish(create_marker(0, legacy_table, None))
            self.pub_cyl_obs_marker.publish(create_marker(1, legacy_red, red_id))
            self.pub_cyl_obs_marker.publish(create_marker(2, legacy_blue, blue_id))

    def publish_wall_marker(self):
        """Publishes the virtual wall cube in RViz from the world_scene (fallback: legacy cfg constants)."""
        wall = self.world_scene.get_obstacle_by_role('wall') if self.world_scene is not None else None
        if wall is not None:
            pos, size, color = wall.position, wall.size, wall.color
        else:
            pos, size, color = cfg.WALL_POS, cfg.WALL_SIZE, (1.0, 0.5, 0.0, 0.5)

        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = "environment"
        m.id = 99
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.x = float(size[0])
        m.scale.y = float(size[1])
        m.scale.z = float(size[2])
        m.color.r, m.color.g, m.color.b, m.color.a = [float(c) for c in color]
        self.pub_wall_marker.publish(m)

    def publish_debug(self, *args, **kwargs):
        # Forward optional debug telemetry (tethers etc.) to the external visualizer if present.
        if self.qp_viz is not None:
            self.qp_viz.publish_debug(*args, **kwargs)

    def publish_teleop_tether(self):
        # Forward the teleop tether publish to the external visualizer if present.
        if self.qp_viz is not None:
            self.qp_viz.publish_teleop_tether()
