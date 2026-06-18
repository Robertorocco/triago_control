#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

# Initializes ROS 2, executes the base stabilizer node, and safely tears down the context upon exit.
def main(args=None):
    rclpy.init(args=args)
    node = BaseStabilizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

class BaseStabilizerNode(Node):
    # Configures the velocity publisher and the direct Odometry subscriber for synchronous state feedback.
    def __init__(self):
        super().__init__('base_stabilizer')
        
        # Publish to the high-level multiplexer entry point
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe directly to the base odometry (Synchronous feedback)
        self.odom_sub = self.create_subscription(Odometry, '/mobile_base_controller/odom', self.odom_callback, 10)
        
        # Control gains (K > 0 for Lyapunov stability)
        self.K_x = 20.0
        self.K_y = 20.0
        self.K_theta = 30.0
        
        self.get_logger().info("Synchronous Base Stabilizer Active. Enforcing q -> 0.")

    # Extracts state from odometry, computes body-frame corrective Twist via Lyapunov law, and publishes.
    def odom_callback(self, msg: Odometry):
        # Extract global errors (e = q) directly from the odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.get_yaw(msg.pose.pose.orientation)

        # Global control law: \dot{q}_{cmd} = -K e
        v_x_global = -self.K_x * x
        v_y_global = -self.K_y * y
        omega_cmd  = -self.K_theta * theta

        # Coordinate transformation: Map global velocities to local body frame using R(-\theta)
        v_x_body = v_x_global * math.cos(-theta) - v_y_global * math.sin(-theta)
        v_y_body = v_x_global * math.sin(-theta) + v_y_global * math.cos(-theta)

        # DIAGNOSTIC LOGGING: Prove the feedback loop is closed and synchronously computing
        #self.get_logger().info(f"Odom [x:{x:.3f}, y:{y:.3f}, th:{theta:.3f}] -> Cmd [vx:{v_x_body:.3f}, vy:{v_y_body:.3f}, w:{omega_cmd:.3f}]")

        # Populate and publish
        cmd = Twist()
        cmd.linear.x = v_x_body
        cmd.linear.y = v_y_body
        cmd.angular.z = omega_cmd
        
        self.cmd_pub.publish(cmd)

    # Extracts and returns the scalar yaw angle (theta) from a geometry_msgs Quaternion.
    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

if __name__ == '__main__':
    main()