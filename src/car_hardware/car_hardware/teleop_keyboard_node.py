#!/usr/bin/env python3
"""
Keyboard Teleop Node for RC Car
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

INSTRUCTIONS = """
RC Car Keyboard Control
-----------------------
   w
a  s  d

w/s : increase/decrease speed
a/d : turn left/right
space : stop
q : quit

CTRL-C to quit
"""

class TeleopKeyboardNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard_node')
        
        # Parameters
        self.declare_parameter('speed_step', 0.1)
        self.declare_parameter('turn_step', 0.3)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_turn', 1.0)
        
        self.speed_step = self.get_parameter('speed_step').value
        self.turn_step = self.get_parameter('turn_step').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_turn = self.get_parameter('max_turn').value
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Current velocities
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Terminal settings
        self.settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info('Teleop Keyboard Node initialized')
        print(INSTRUCTIONS)
    
    def get_key(self, timeout=0.1):
        """Get keyboard input"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def publish_twist(self):
        """Publish velocity"""
        twist = Twist()
        twist.linear.x = self.linear_vel
        twist.angular.z = self.angular_vel
        self.cmd_vel_pub.publish(twist)
        print(f'\rSpeed: {self.linear_vel:.2f} m/s | Turn: {self.angular_vel:.2f} rad/s', end='')
    
    def run(self):
        """Main loop"""
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == 'w':
                    self.linear_vel = min(self.linear_vel + self.speed_step, self.max_speed)
                elif key == 's':
                    self.linear_vel = max(self.linear_vel - self.speed_step, -self.max_speed)
                elif key == 'a':
                    self.angular_vel = min(self.angular_vel + self.turn_step, self.max_turn)
                elif key == 'd':
                    self.angular_vel = max(self.angular_vel - self.turn_step, -self.max_turn)
                elif key == ' ':
                    self.linear_vel = 0.0
                    self.angular_vel = 0.0
                elif key == 'q':
                    break
                
                self.publish_twist()
                
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
        finally:
            self.linear_vel = 0.0
            self.angular_vel = 0.0
            self.publish_twist()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboardNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
