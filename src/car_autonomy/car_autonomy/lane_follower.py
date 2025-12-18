import rclpy
import math
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class PID:
    def __init__(self, Kp, Ki, Kd, limits=(-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_out, self.max_out = limits
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error
        # Anti-windup for integral
        self.integral = max(-10.0, min(10.0, self.integral))
        
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

        self.declare_parameter('max_speed', 0.3)
        self.max_speed = self.get_parameter('max_speed').value

        self.offset = 0.0
        self.heading = 0.0
        self.prev_steer = 0.0

        self.create_subscription(Float32, 'lane_offset', self.offset_cb, 10)
        self.create_subscription(Float32, 'lane_heading', self.heading_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, 'cmd_vel', 10)

        # UPDATED PID: Reduced gains to prevent twitching
        # We output directly to -1.0 to 1.0 range now
        self.pid = PID(Kp=6.0, Ki=0.0, Kd=0.1, limits=(-1.0, 1.0))
        
        self.timer = self.create_timer(0.03, self.control_loop)

    def offset_cb(self, msg): self.offset = msg.data
    def heading_cb(self, msg): self.heading = msg.data

    def control_loop(self):
        msg = Twist()

        # --- DISABLED T-SECTION OVERRIDE ---
        # This was causing the robot to lock up at 1.0 turn / 0.15 speed
        # if abs(self.heading) > 0.6:
        #     steer = math.copysign(1.0, self.heading)
        #     msg.linear.x = 0.15
        #     msg.angular.z = steer
        #     self.pub_cmd.publish(msg)
        #     return

        # --- NORMAL LANE FOLLOWING ---
        
        # 1. Calculate Weighted Error
        # Reduce these weights if the car oscillates too much
        # Assuming offset is roughly -1.0 to 1.0 range? 
        # If offset is pixels (e.g. 200), these weights are WAY too high.
        combined_error = (1.0 * self.offset) + (0.5 * self.heading)
        
        # 2. Compute PID Output (-1.0 to 1.0)
        # We invert the PID output because usually:
        # Positive Error (Line is Left) -> Requires Positive Turn (Left)
        # But previous code had -self.pid. Check this direction physically!
        steer_cmd = -self.pid.update(combined_error)

        # 3. Smooth steering (Low Pass Filter)
        # Helps prevent servo jitter
        steer_cmd = 0.7 * self.prev_steer + 0.3 * steer_cmd
        self.prev_steer = steer_cmd

        # 4. Dynamic Speed Adjustment
        # Slow down when turning sharply
        speed = self.max_speed * (1.0 - min(abs(steer_cmd), 0.8) * 0.5)
        speed = max(0.05, speed) # Ensure min speed isn't 0

        msg.linear.x = float(speed)
        msg.angular.z = float(steer_cmd)

        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.pub_cmd.publish(Twist()) # Stop on exit
        node.destroy_node()
        rclpy.shutdown()