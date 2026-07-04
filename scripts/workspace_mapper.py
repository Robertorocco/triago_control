import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading
import sys
import os
import tempfile
from rcl_interfaces.srv import GetParameters

# --- CONFIGURATION ---
EE_FRAME_NAME = "arm_right_tool_link" 
BASE_FRAME_NAME = "base_footprint"
N_SAMPLES = 50000
DOWNSAMPLE_PLOT = 2000

class WorkspaceMapper(Node):
    def __init__(self):
        super().__init__('triago_workspace_mapper')
        
        # 1. Fetch Live URDF and Initialize Pinocchio Model
        self.get_logger().info("Fetching URDF from /robot_state_publisher...")
        urdf_str = self.get_urdf()
        if urdf_str is None:
            self.get_logger().error("Failed to get URDF from robot_state_publisher.")
            sys.exit(1)
            
        # Create a temporary file to hold the URDF (exactly as done in the controller)
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
            f.write(urdf_str)
            self.urdf_path = f.name
            
        self.get_logger().info(f"Loading Pinocchio model from temp file: {self.urdf_path}")
        try:
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            self.data = self.model.createData()
        except Exception as e:
            self.get_logger().error(f"Failed to load URDF: {e}")
            sys.exit(1)

        # Retrieve frame IDs
        if self.model.existFrame(EE_FRAME_NAME):
            self.ee_frame_id = self.model.getFrameId(EE_FRAME_NAME)
        else:
            self.get_logger().error(f"Frame {EE_FRAME_NAME} not found in URDF.")
            sys.exit(1)

        # Real-time state vector q
        self.q_real = pin.neutral(self.model)
        self.joint_name_to_id = {name: i for i, name in enumerate(self.model.names)}

        # 2. Run Offline Monte Carlo Analysis
        self.get_logger().info(f"Running Monte Carlo mapping ({N_SAMPLES} samples)...")
        self.cloud_pts, self.aabb_min, self.aabb_max = self.run_monte_carlo()
        
        self.get_logger().info(f"--- WORKSPACE AABB LIMITS (relative to base) ---")
        self.get_logger().info(f"X: [{self.aabb_min[0]:.3f}, {self.aabb_max[0]:.3f}] m")
        self.get_logger().info(f"Y: [{self.aabb_min[1]:.3f}, {self.aabb_max[1]:.3f}] m")
        self.get_logger().info(f"Z: [{self.aabb_min[2]:.3f}, {self.aabb_max[2]:.3f}] m")
        self.get_logger().info(f"Scaling Factor K calculation ready.")

        # 3. Setup ROS Subscriber for live teleoperation feedback
        self.sub_joint_states = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )

        # 4. Initialize Matplotlib
        self.setup_plots()

    def get_urdf(self):
        """Fetches the live URDF string from the robot_state_publisher."""
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=5.0): 
            return None
        request = GetParameters.Request()
        request.names = ['robot_description']
        future = client.call_async(request)
        # It is safe to spin here because the main background thread hasn't started yet
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value
    
    def run_monte_carlo(self):
        """Samples the configuration space and computes the AABB."""
        q_min = self.model.lowerPositionLimit
        q_max = self.model.upperPositionLimit
        
        valid_points = []
        
        for _ in range(N_SAMPLES):
            # Uniform sampling within joint limits
            q_rand = np.random.uniform(q_min, q_max)
            
            pin.forwardKinematics(self.model, self.data, q_rand)
            pin.updateFramePlacements(self.model, self.data)
            p_ee = self.data.oMf[self.ee_frame_id].translation
            
            # Simple Heuristic Collision Filter
            # Example: Restrict to frontal workspace (X > 0.1) and above the mobile base (Z > 0.2)
            if p_ee[0] > 0.1 and p_ee[2] > 0.2:
                valid_points.append(p_ee.copy())
                
        points_arr = np.array(valid_points)
        
        aabb_min = np.min(points_arr, axis=0)
        aabb_max = np.max(points_arr, axis=0)
        
        return points_arr, aabb_min, aabb_max

    def joint_state_callback(self, msg):
        """Mathematically maps ROS 2 joint states into Pinocchio's q vector using strictly valid indices."""
        for i, name in enumerate(msg.name):
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                idx_q = self.model.joints[joint_id].idx_q
                
                # Only update standard 1-DoF joints (revolute/prismatic). 
                # This safely ignores complex root joints (like a planar mobile base)
                if self.model.joints[joint_id].nq == 1:
                    self.q_real[idx_q] = msg.position[i]

    def setup_plots(self):
        """Initializes the live Matplotlib workspace visualizer."""
        self.fig = plt.figure(figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('Absolute Teleoperation: Workspace Mapper')

        # Subplot 1: Workspace Point Cloud & AABB
        self.ax_space = self.fig.add_subplot(121, projection='3d')
        self.ax_space.set_title("Reachable Operational Space & AABB")
        self.ax_space.set_xlabel('X [m]'); self.ax_space.set_ylabel('Y [m]'); self.ax_space.set_zlabel('Z [m]')
        
        # Downsample for rendering speed
        idx = np.random.choice(self.cloud_pts.shape[0], min(DOWNSAMPLE_PLOT, self.cloud_pts.shape[0]), replace=False)
        ds_points = self.cloud_pts[idx]
        self.ax_space.scatter(ds_points[:, 0], ds_points[:, 1], ds_points[:, 2], c='c', s=1, alpha=0.3)
        
        # Draw Glass Box
        self.draw_glass_box(self.ax_space, self.aabb_min, self.aabb_max)
        
        # Initialize Live Marker (zorder=10 forces it to render OVER the point cloud)
        self.live_marker, = self.ax_space.plot([], [], [], 'ro', markersize=12, zorder=10, label="TRIAGo EE")
        self.ax_space.legend()

        # Set limits based on AABB
        self.ax_space.set_xlim([self.aabb_min[0]-0.2, self.aabb_max[0]+0.2])
        self.ax_space.set_ylim([self.aabb_min[1]-0.2, self.aabb_max[1]+0.2])
        self.ax_space.set_zlim([self.aabb_min[2]-0.2, self.aabb_max[2]+0.2])

        # Subplot 2: Orientation Triad
        self.ax_rot = self.fig.add_subplot(122, projection='3d')
        self.ax_rot.set_title("End-Effector Orientation (SO(3))")
        
        # The animation engine
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False)

    def draw_glass_box(self, ax, min_b, max_b):
        """Renders the AABB as a semi-transparent Poly3DCollection."""
        x0, y0, z0 = min_b
        x1, y1, z1 = max_b

        # 8 vertices of the box
        verts = [
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], # Bottom
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], # Top
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], # Front
            [[x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]], # Back
            [[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]], # Left
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y1, z1]]  # Right
        ]
        
        box = Poly3DCollection(verts, alpha=0.1, linewidths=1, edgecolors='b', facecolors='cyan')
        ax.add_collection3d(box)

    def update_plot(self, frame):
        """Animation loop: Evaluates live FK and updates visualization with crash protection."""
        try:
            # 1. Compute Live Forward Kinematics
            pin.forwardKinematics(self.model, self.data, self.q_real)
            pin.updateFramePlacements(self.model, self.data)
            
            pose = self.data.oMf[self.ee_frame_id]
            p_real = pose.translation
            R_real = pose.rotation

            # 2. Update Space Subplot (Red Dot)
            self.live_marker.set_data([p_real[0]], [p_real[1]])
            self.live_marker.set_3d_properties([p_real[2]])

            # 3. Update Orientation Subplot (Triad)
            self.ax_rot.cla()
            self.ax_rot.set_title("End-Effector Orientation (SO(3))")
            self.ax_rot.set_xlim([-1.5, 1.5]); self.ax_rot.set_ylim([-1.5, 1.5]); self.ax_rot.set_zlim([-1.5, 1.5])
            self.ax_rot.set_xlabel('X'); self.ax_rot.set_ylabel('Y'); self.ax_rot.set_zlabel('Z')
            
            # Plot fixed base reference frame (faint)
            self.ax_rot.plot([0, 1], [0, 0], [0, 0], color='gray', linestyle='--', alpha=0.5)
            self.ax_rot.plot([0, 0], [0, 1], [0, 0], color='gray', linestyle='--', alpha=0.5)
            self.ax_rot.plot([0, 0], [0, 0], [0, 1], color='gray', linestyle='--', alpha=0.5)

            # Extract column vectors from rotation matrix
            x_vec, y_vec, z_vec = R_real[:, 0], R_real[:, 1], R_real[:, 2]
            
            origin = np.zeros(3)
            self.ax_rot.quiver(*origin, *x_vec, color='r', length=1.0, normalize=True)
            self.ax_rot.quiver(*origin, *y_vec, color='g', length=1.0, normalize=True)
            self.ax_rot.quiver(*origin, *z_vec, color='b', length=1.0, normalize=True)
            
        except Exception as e:
            self.get_logger().error(f"Matplotlib Animation Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    mapper_node = WorkspaceMapper()

    # Spin ROS 2 in a background daemon thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(mapper_node,), daemon=True)
    spin_thread.start()

    # Block main thread with Matplotlib
    plt.show()

    # Cleanup
    if hasattr(mapper_node, 'urdf_path') and os.path.exists(mapper_node.urdf_path):
        os.remove(mapper_node.urdf_path)
        
    mapper_node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()

if __name__ == '__main__':
    main()