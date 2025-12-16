import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class PID:
    def __init__(self, Kp, Ki, Kd, output_limits=(-20, 20)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_out = output_limits[0]
        self.max_out = output_limits[1]
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error):
        p_out = self.Kp * error
        self.integral += error
        i_out = self.Ki * self.integral
        derivative = error - self.prev_error
        d_out = self.Kd * derivative
        self.prev_error = error
        output = p_out + i_out + d_out
        return max(self.min_out, min(self.max_out, output))

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.declare_parameter('max_speed_ms', 0.2)
        self.max_speed_ms = self.get_parameter('max_speed_ms').value
        self.lane_offset_value = 0.0
        self.lane_heading_value = 0.0

        self.lane_offset = self.create_subscription(Float32, 'lane_offset', self.offset_callback, 10)
        self.lane_heading = self.create_subscription(Float32, 'lane_heading', self.heading_callback, 10)
        
        # Publishes cmd_vel for the STM32 driver
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pid = PID(Kp=6.0, Ki=0.0, Kd=0.2, output_limits=(-25, 25))
        self.base_speed_ratio = 1.0 
        self.timer = self.create_timer(0.1, self.control_loop) 

    def offset_callback(self, msg): self.lane_offset_value = msg.data
    def heading_callback(self, msg): self.lane_heading_value = msg.data

    def control_loop(self):
        composite = float((20 * self.lane_offset_value * 0.8) + (20 * self.lane_heading_value * 0.2))
        steer_angle_deg = -float(self.pid.update(composite))
        
        # Slow down on turns
        reduction = (1 - min(abs(steer_angle_deg)/20, 1) * 0.5)
        target_speed_ms = (self.base_speed_ratio * self.max_speed_ms) * reduction

        msg = Twist()
        msg.linear.x = float(target_speed_ms)
        # Convert degrees to radians for the driver
        msg.angular.z = float(math.radians(steer_angle_deg))
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.pub_cmd.publish(Twist()) # Stop
    finally:
        node.destroy_node()
        rclpy.shutdown()
