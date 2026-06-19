import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import sys
import tempfile
import os

# --- 1. ROBUST COLLISION LIBRARY IMPORT ---
HPPFCL_FOUND = False
try:
    import hppfcl
    HPPFCL_FOUND = True
except ImportError:
    try:
        import pinocchio.hppfcl as hppfcl
        HPPFCL_FOUND = True
    except ImportError:
        print("\n[CRITICAL] 'hppfcl' library not found! No collision shapes will appear.\n")
        class MockFCL:
            def Capsule(self, r, l): return None
            def Cylinder(self, r, l): return None
            def Box(self, x, y, z): return None
        hppfcl = MockFCL()

# --- CONFIGURATION ---
REF_FRAME = 'torso_lift_link'

# 1. PALM (Attached to the wrist)
RIGHT_PALM_FRAME = 'arm_right_tool_link'
LEFT_PALM_FRAME  = 'arm_left_tool_link'

# 2. KNUCKLES (Proximal Phalanx - The base of the finger)
RIGHT_KNUCKLES = [
    'gripper_right_thumb_flexor_1_joint',
    'gripper_right_finger_1_flexor_1_joint',
    'gripper_right_finger_2_flexor_1_joint',
    'gripper_right_finger_3_flexor_1_joint'
]
LEFT_KNUCKLES = [
    'gripper_left_thumb_flexor_1_joint',
    'gripper_left_finger_1_flexor_1_joint',
    'gripper_left_finger_2_flexor_1_joint',
    'gripper_left_finger_3_flexor_1_joint'
]

# 3. TIPS (Distal Phalanx - The end of the finger)
RIGHT_TIPS = [
    'gripper_right_thumb_flexor_3_joint',
    'gripper_right_finger_1_flexor_3_joint',
    'gripper_right_finger_2_flexor_3_joint',
    'gripper_right_finger_3_flexor_3_joint'
]
LEFT_TIPS = [
    'gripper_left_thumb_flexor_3_joint',
    'gripper_left_finger_1_flexor_3_joint',
    'gripper_left_finger_2_flexor_3_joint',
    'gripper_left_finger_3_flexor_3_joint'
]

# Dimensions
PALM_BOX_SIZE = np.array([0.03, 0.11, 0.10]) # Depth(x), Width(y), Height(z)
KNUCKLE_LEN   = 0.04
TIP_LEN       = 0.03
FINGER_RAD    = 0.012 # Slightly thicker

RIGHT_CHAIN = ['arm_right_1_link', 'arm_right_2_link', 'arm_right_3_link', 'arm_right_4_link', 'arm_right_5_link', 'arm_right_6_link', 'arm_right_7_link']
LEFT_CHAIN  = ['arm_left_1_link', 'arm_left_2_link', 'arm_left_3_link', 'arm_left_4_link', 'arm_left_5_link', 'arm_left_6_link', 'arm_left_7_link']

# Helper
def get_skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

class MixedRealityVisualizer(Node):
    def __init__(self):
        super().__init__('mixed_reality_viz')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.current_q = None; self.model = None

    def get_urdf(self):
        client = self.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if not client.wait_for_service(timeout_sec=2.0): return None
        request = GetParameters.Request(); request.names = ['robot_description']
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().values[0].string_value

    def joint_callback(self, msg):
        if self.model is None: return
        q = pin.neutral(self.model)
        for i, name in enumerate(msg.name):
            if name in self.model.names:
                idx = self.model.joints[self.model.getJointId(name)].idx_q
                if idx >= 0: q[idx] = msg.position[i]
        self.current_q = q

    def calculate_arm_offsets(self, chain, tool_link_name):
        # ... (Same standard arm logic as before) ...
        offsets = {}
        full_chain = chain + [tool_link_name] 
        for i in range(len(chain)):
            link_name = chain[i]; next_link = full_chain[i+1]
            try:
                t_link = self.tf_buffer.lookup_transform(REF_FRAME, link_name, rclpy.time.Time())
                t_next = self.tf_buffer.lookup_transform(REF_FRAME, next_link, rclpy.time.Time())
                
                p_link = np.array([t_link.transform.translation.x, t_link.transform.translation.y, t_link.transform.translation.z])
                p_next = np.array([t_next.transform.translation.x, t_next.transform.translation.y, t_next.transform.translation.z])
                q_link = np.array([t_link.transform.rotation.x, t_link.transform.rotation.y, t_link.transform.rotation.z, t_link.transform.rotation.w])
                rot_link = pin.Quaternion(q_link[3], q_link[0], q_link[1], q_link[2]).matrix()
                
                vec_global = p_next - p_link
                length = np.linalg.norm(vec_global)
                if length < 0.001: length = 0.01 
                
                vec_local = rot_link.T @ vec_global
                midpoint = vec_local / 2.0
                
                z_axis = np.array([0,0,1]); target = vec_local / length
                rot_axis = np.cross(z_axis, target)
                if np.linalg.norm(rot_axis) < 0.001: R_cyl = np.eye(3)
                else:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis); angle = np.arccos(np.clip(np.dot(z_axis, target), -1.0, 1.0))
                    K = get_skew(rot_axis); R_cyl = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
                
                offsets[link_name] = (pin.SE3(R_cyl, midpoint), length)
            except Exception: pass
        return offsets

def main():
    rclpy.init(); node = MixedRealityVisualizer()
    urdf_str = node.get_urdf()
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.urdf') as f:
        f.write(urdf_str); urdf_path = f.name
        
    model = pin.buildModelFromUrdf(urdf_path)
    mesh_paths = ["/opt/pal/alum/share", "/opt/ros/humble/share", "/opt/pal/ferrum/share", "."]
    try: visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=mesh_paths)
    except: visual_model = pin.GeometryModel()

    # Green Skin
    for geom in visual_model.geometryObjects:
        geom.meshColor = np.array([0.0, 1.0, 0.0, 0.3]); geom.overrideMaterial = True

    print("Waiting for TF...")
    while not node.tf_buffer.can_transform(REF_FRAME, RIGHT_CHAIN[-1], rclpy.time.Time()):
        rclpy.spin_once(node, timeout_sec=0.1)

    # 1. Calc Arms
    right_arm = node.calculate_arm_offsets(RIGHT_CHAIN, 'arm_right_tool_link')
    left_arm  = node.calculate_arm_offsets(LEFT_CHAIN,  'arm_left_tool_link')

    # -------------------------------------------------------------------------
    # HAND COLLISION BUILDER
    # -------------------------------------------------------------------------
    def add_shape(parent_name, shape_type, dims, offset_pos, offset_rot=np.eye(3), color=[1,0,0,0.8], prefix=""):
        # parent_name: Can be a Link (Frame) OR a Joint
        
        # A. Try Finding Joint First (Best for Fingers)
        if model.existJointName(parent_name):
            joint_id = model.getJointId(parent_name)
            placement = pin.SE3(offset_rot, offset_pos)
            
        # B. Try Finding Frame (Best for Palm/Arm)
        elif model.existFrame(parent_name):
            frame_id = model.getFrameId(parent_name)
            joint_id = model.frames[frame_id].parentJoint
            placement = model.frames[frame_id].placement * pin.SE3(offset_rot, offset_pos)
        else:
            print(f"[Skip] {parent_name} not found")
            return

        # Create Shape
        if shape_type == "box": geometry = hppfcl.Box(*dims)
        elif shape_type == "capsule": geometry = hppfcl.Capsule(dims[0], dims[1]) # r, length
        else: geometry = hppfcl.Cylinder(dims[0], dims[1])

        obj = pin.GeometryObject(f"{prefix}_{parent_name}", joint_id, placement, geometry)
        obj.meshColor = np.array(color); obj.overrideMaterial = True
        visual_model.addGeometryObject(obj)

    # --- 1. ADD PALMS (BOX) ---
    # Offset: Shifted forward in Z (0.05) to cover the meat of the palm
    palm_pose = np.array([0.0, 0.0, 0.08]) 
    add_shape(RIGHT_PALM_FRAME, "box", PALM_BOX_SIZE, palm_pose, color=[1,0,0,0.6], prefix="r_palm")
    add_shape(LEFT_PALM_FRAME,  "box", PALM_BOX_SIZE, palm_pose, color=[0,0,1,0.6], prefix="l_palm")

    # --- 2. ADD KNUCKLES (Proximal) ---
    # Attached to flexor_1. Offset Z=0.03 covers the first phalanx.
    for joint in RIGHT_KNUCKLES:
        add_shape(joint, "capsule", (FINGER_RAD, KNUCKLE_LEN), np.array([0,0, KNUCKLE_LEN/2 + 0.01]), color=[1,0,0,0.8], prefix="r_knuckle")
    for joint in LEFT_KNUCKLES:
        add_shape(joint, "capsule", (FINGER_RAD, KNUCKLE_LEN), np.array([0,0, KNUCKLE_LEN/2 + 0.01]), color=[0,0,1,0.8], prefix="l_knuckle")

    # --- 3. ADD TIPS (Distal) ---
    # Attached to flexor_3. Offset Z=0.015 covers the tip.
    for joint in RIGHT_TIPS:
        add_shape(joint, "capsule", (FINGER_RAD, TIP_LEN), np.array([0,0, TIP_LEN/2]), color=[1,0,0,0.8], prefix="r_tip")
    for joint in LEFT_TIPS:
        add_shape(joint, "capsule", (FINGER_RAD, TIP_LEN), np.array([0,0, TIP_LEN/2]), color=[0,0,1,0.8], prefix="l_tip")

    # --- 4. ADD ARMS ---
    for name, (plc, length) in right_arm.items():
        add_shape(name, "cylinder", (0.05, length), plc.translation, plc.rotation, color=[1,0,0,0.6], prefix="r_arm")
    for name, (plc, length) in left_arm.items():
        add_shape(name, "cylinder", (0.05, length), plc.translation, plc.rotation, color=[0,0,1,0.6], prefix="l_arm")

    # --- 5. GROUND ---
    if HPPFCL_FOUND:
        g = pin.GeometryObject("ground", 0, pin.SE3(np.eye(3), np.array([0,0,-0.5])), hppfcl.Box(20,20,1))
        g.meshColor = np.array([0.5,0.5,0.5,0.5]); visual_model.addGeometryObject(g)

    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.initViewer(open=False); viz.clean(); viz.loadViewerModel()
    node.viz = viz; node.model = model
    
    print("------------------------------------------------")
    print("FULL HAND VISUALIZER RUNNING")
    print("http://127.0.0.1:7000/static/")
    print("------------------------------------------------")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.current_q is not None: node.viz.display(node.current_q)
    except KeyboardInterrupt: pass
    finally: os.remove(urdf_path); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()