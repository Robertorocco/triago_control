#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String, Float64
from visualization_msgs.msg import Marker, MarkerArray # <--- NEW
import sys, select, termios, tty
import numpy as np
import time
import threading
import math

# --- CONFIGURATION ---
MAX_LIN_VEL = 0.5  # m/s

msg = """
--------------------------------------------------
DUAL ARM TELEOP (WITH DASHBOARD TELEMETRY)
--------------------------------------------------
    w           r (up)
 a  s  d        f (down)
    x

ORIENTATION CONTROL (Gripper):
  [Up Arrow]   : Pitch Down (Point at floor)
  [Down Arrow] : Pitch Up   (Point at ceiling)
  [Left Arrow] : Roll Left  (Twist Counter-Clockwise)
  [Right Arrow]: Roll Right (Twist Clockwise)
SPACE : STOP ALL

Selection Mode:
1 : Control LEFT Arm
2 : Control RIGHT Arm
3 : Control BOTH Arms

t/b : Increase/Decrease Max Speed
e   : END EXPERIMENT (Triggers Plotter 'R' Phase)
CTRL-C to quit
--------------------------------------------------
"""

# Key Mappings (X, Y, Z)
moveBindings = {
    'w': (1, 0, 0),   # +X (Forward)
    'x': (-1, 0, 0),  # -X (Backward)
    'a': (0, 1, 0),   # +Y (Left)
    'd': (0, -1, 0),  # -Y (Right)
    'r': (0, 0, 1),   # +Z (Up)
    'f': (0, 0, -1),  # -Z (Down)
}

# Orientation Mappings (Roll, Pitch, Yaw)
orientationBindings = {
    '\x1b[A': (0.0, -1.0, 0.0),  # Up Arrow: Negative Pitch
    '\x1b[B': (0.0, 1.0, 0.0),   # Down Arrow: Positive Pitch
    '\x1b[C': (1.0, 0.0, 0.0),   # Right Arrow: Positive Roll
    '\x1b[D': (-1.0, 0.0, 0.0),  # Left Arrow: Negative Roll
}

selectionBindings = {
    '1': 'LEFT',
    '2': 'RIGHT',
    '3': 'BOTH',
}

def euler_to_quaternion(roll, pitch, yaw):
    """Converts Fixed-Axis Roll/Pitch/Yaw into a Quaternion"""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
        # If the first byte is the Escape character, read the next two!
        if key == '\x1b':
            key += sys.stdin.read(2)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class DualArmTeleop(Node):
    def __init__(self):
        super().__init__('dual_arm_teleop')
        
        # Publishers for the CLF Controller
        self.pub_right = self.create_publisher(Float64MultiArray, '/arm_right/cartesian_reference', 10)
        self.pub_left  = self.create_publisher(Float64MultiArray, '/arm_left/cartesian_reference', 10)
        
        # --- NEW: Publishers for the Dashboard ---
        self.pub_dashboard = self.create_publisher(Float64MultiArray, '/trajectory/reference_state', 10)
        self.pub_phase = self.create_publisher(String, '/trajectory/phase', 10)
        self.pub_time_scale = self.create_publisher(Float64, '/trajectory/time_scale', 10)
        
        # --- NEW: Publisher for RViz ---
        self.pub_markers = self.create_publisher(MarkerArray, '/teleop/target_markers', 10)

        # Subscriber to anchor the initial state
        self.sub_ee = self.create_subscription(Float64MultiArray, '/qp_debug/ee_real', self.ee_cb, 10)
        
        # State
        self.active_selection = 'BOTH'
        self.speed = 0.05
        self.angular_speed = 0.05
        self.current_phase = 'S' # Start in 'Wait' phase
        
        # Integrator Memory
        self.anchored = False
        self.ref_p_r = np.zeros(3)
        self.ref_p_l = np.zeros(3)

        # --- Orientation Targets (Roll, Pitch, Yaw) ---
        self.ref_rpy_r = np.array([0.0, 0.0, 0.0])
        self.ref_rpy_l = np.array([0.0, 0.0, 0.0])

    def ee_cb(self, msg):
        """Anchors the virtual target to the physical robot exactly once at startup."""
        if not self.anchored and len(msg.data) >= 12:
            self.ref_p_r = np.array(msg.data[0:3])
            self.ref_p_l = np.array(msg.data[6:9])
            
            # --- NEW: Catch the true physical orientation ---
            if len(msg.data) >= 18:
                self.ref_rpy_r = np.array(msg.data[12:15])
                self.ref_rpy_l = np.array(msg.data[15:18])
            
            self.anchored = True
            
            # Instantly switch plotter to Tracking phase
            self.current_phase = 'T'
            self.pub_phase.publish(String(data='T'))
            print("\n[Teleop] Anchored to real physical positions & orientations. Dashboard Recording Started!")

    def create_target_marker(self, m_id, pos, rpy, r, g, b, ns="target"):
        markers = []
        
        # 1. The Cube (Hand approximation)
        cube = Marker()
        cube.header.frame_id = 'base_footprint' 
        cube.header.stamp = self.get_clock().now().to_msg()
        cube.ns = ns
        cube.id = m_id
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        
        cube.pose.position.x = float(pos[0])
        cube.pose.position.y = float(pos[1])
        cube.pose.position.z = float(pos[2])
        
        q = euler_to_quaternion(rpy[0], rpy[1], rpy[2])
        cube.pose.orientation.x = float(q[0])
        cube.pose.orientation.y = float(q[1])
        cube.pose.orientation.z = float(q[2])
        cube.pose.orientation.w = float(q[3])
        
        cube.scale.x = 0.03
        cube.scale.y = 0.03
        cube.scale.z = 0.03
        
        cube.color.r = float(r)
        cube.color.g = float(g)
        cube.color.b = float(b)
        cube.color.a = 0.5 
        markers.append(cube)

        # 2. The Direction Arrow
        arrow = Marker()
        arrow.header.frame_id = 'base_footprint'
        arrow.header.stamp = cube.header.stamp
        arrow.ns = ns
        arrow.id = m_id + 10  # Offset ID so it doesn't overwrite the cube
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = cube.pose  # Inherit the exact position and orientation!
        
        # RViz Arrow Scale using pose: x=length, y=shaft width, z=head width
        arrow.scale.x = 0.15  # 15cm long
        arrow.scale.y = 0.01
        arrow.scale.z = 0.01
        
        # Make the arrow solid yellow to stand out
        arrow.color.r = 1.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        markers.append(arrow)

        return markers
    
def ros_thread_entry(node):
    rclpy.spin(node)

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = DualArmTeleop()
    
    spinner = threading.Thread(target=ros_thread_entry, args=(node,), daemon=True)
    spinner.start()
    
    print(msg)
    print(f"Waiting for robot state to anchor... (Is the QP controller running?)")
    
    while not node.anchored and rclpy.ok():
        time.sleep(0.1)
        
    x, y, z = 0.0, 0.0, 0.0
    last_time = time.time()

    try:
        while rclpy.ok():
            key = getKey(settings)
            
            # 1. Check Movement Keys
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
            
            # --- NEW: Process Orientation Keys ---
            elif key in orientationBindings.keys():
                dr, dp, dy = orientationBindings[key]
                angular_step = node.angular_speed 
                
                if node.active_selection in ['RIGHT', 'BOTH']:
                    node.ref_rpy_r += np.array([dr, dp, dy]) * angular_step
                if node.active_selection in ['LEFT', 'BOTH']:
                    node.ref_rpy_l += np.array([dr, dp, dy]) * angular_step

            # 2. Check Selection Keys
            elif key in selectionBindings.keys():
                node.active_selection = selectionBindings[key]
                print(f"\rMode switched to: {node.active_selection}           ")
                
            # 3. Speed Adjustment
            elif key == 't':
                node.speed = min(node.speed + 0.01, MAX_LIN_VEL)
                print(f"\rSpeed increased: {node.speed:.2f} m/s")
            elif key == 'b':
                node.speed = max(node.speed - 0.01, 0.01)
                print(f"\rSpeed decreased: {node.speed:.2f} m/s")

            # 4. End Experiment
            elif key == 'e':
                print("\n[Teleop] Sending 'R' Phase to Dashboard. Plots will generate in 10s...")
                node.current_phase = 'R'
                node.pub_phase.publish(String(data='R'))
                x = 0; y = 0; z = 0

            # 5. Stop Logic
            elif key == ' ' or key == 's':
                x = 0; y = 0; z = 0
                dr = 0; dp = 0; dy = 0

            # 6. Quit
            elif key == '\x03': # Ctrl-C
                break
            
            # 7. Dead-Man Switch
            else:
                x = 0; y = 0; z = 0
                dr = 0; dp = 0; dy = 0

            # --- THE KINEMATIC INTEGRATOR ---
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            v_cmd = np.array([x, y, z]) * node.speed
            w_cmd = np.array([dr, dp, dy]) * (node.speed * 0.5) # Angular speed is half of linear speed
            
            v_r, w_r = np.zeros(3), np.zeros(3)
            v_l, w_l = np.zeros(3), np.zeros(3)
            
            # Evolve and Publish Right Arm (12 Floats)
            if node.active_selection in ['RIGHT', 'BOTH']:
                node.ref_p_r += v_cmd * dt
                node.ref_rpy_r += w_cmd * dt
                v_r, w_r = v_cmd, w_cmd
                
                msg_r = Float64MultiArray()
                msg_r.data = node.ref_p_r.tolist() + node.ref_rpy_r.tolist() + v_r.tolist() + w_r.tolist()
                node.pub_right.publish(msg_r)
            
            # Evolve and Publish Left Arm (12 Floats)
            if node.active_selection in ['LEFT', 'BOTH']:
                node.ref_p_l += v_cmd * dt
                node.ref_rpy_l += w_cmd * dt
                v_l, w_l = v_cmd, w_cmd
                
                msg_l = Float64MultiArray()
                msg_l.data = node.ref_p_l.tolist() + node.ref_rpy_l.tolist() + v_l.tolist() + w_l.tolist()
                node.pub_left.publish(msg_l)

            # --- PUBLISH TELEMETRY FOR PLOTTER ---
            # (Keeping dashboard as 12-element Position/Linear Vel to avoid breaking your Plotter!)
            dash_msg = Float64MultiArray()
            dash_msg.data = node.ref_p_r.tolist() + v_r.tolist() + node.ref_p_l.tolist() + v_l.tolist()
            node.pub_dashboard.publish(dash_msg)

            # 2. Publish Time Scale (Constant 1.0 during Teleop)
            ts_msg = Float64()
            ts_msg.data = 1.0
            node.pub_time_scale.publish(ts_msg)

            # 3. Keep publishing the current phase
            node.pub_phase.publish(String(data=node.current_phase))
            
            ts_msg = Float64()
            ts_msg.data = 1.0
            node.pub_time_scale.publish(ts_msg)

            # 3. Keep publishing the current phase
            node.pub_phase.publish(String(data=node.current_phase))

            # --- PUBLISH RVIZ MARKERS ---
            if node.anchored:
                marker_array = MarkerArray()
                
                # Right Target = Red Cube + Yellow Arrow
                m_right_list = node.create_target_marker(0, node.ref_p_r, node.ref_rpy_r, 1.0, 0.0, 0.0, "right_target")
                marker_array.markers.extend(m_right_list)
                
                # Left Target = Red Cube + Yellow Arrow
                m_left_list = node.create_target_marker(1, node.ref_p_l, node.ref_rpy_l, 1.0, 0.0, 0.0, "left_target")
                marker_array.markers.extend(m_left_list)
                
                node.pub_markers.publish(marker_array)

    except Exception as e:
        print(e)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()