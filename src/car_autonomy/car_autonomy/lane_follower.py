import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class PID:
    def __init__(self, Kp, Ki, Kd, limits=(-25, 25)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_out, self.max_out = limits
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )
        return max(self.min_out, min(self.max_out, output))

class ControllerNode(Node):
    def __init__(self):
        super().__init__('lane_controller')

        self.declare_parameter('max_speed', 0.25)
        self.max_speed = self.get_parameter('max_speed').value

        self.offset = 0.0
        self.heading = 0.0
        self.prev_steer = 0.0

        self.create_subscription(Float32, 'lane_offset', self.offset_cb, 10)
        self.create_subscription(Float32, 'lane_heading', self.heading_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)

        self.pid = PID(Kp=7.0, Ki=0.0, Kd=0.3)
        self.timer = self.create_timer(0.03, self.control_loop)

    def offset_cb(self, msg): self.offset = msg.data
    def heading_cb(self, msg): self.heading = msg.data

    def control_loop(self):
        msg = Twist()

        # --- T-SECTION OVERRIDE ---
        if abs(self.heading) > 0.6:
            steer = math.copysign(1.0, self.heading)
            msg.linear.x = 0.15
            msg.angular.z = steer
            self.pub_cmd.publish(msg)
            return

        # --- NORMAL LANE FOLLOWING ---
        error = (15 * self.offset) + (25 * self.heading)
        steer_deg = -self.pid.update(error)

        # Scale to teleop range
        steer_cmd = steer_deg * 0.03

        # Deadband compensation
        if abs(steer_cmd) < 0.15:
            steer_cmd = 0.15 * math.copysign(1, steer_cmd)

        # Smooth steering
        steer_cmd = -1* 0.7 * self.prev_steer + 0.3 * steer_cmd
        self.prev_steer = steer_cmd

        speed = self.max_speed * (1 - min(abs(steer_cmd), 1.0) * 0.3)

        msg.linear.x = float(speed)
        msg.angular.z = float(steer_cmd)

        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
