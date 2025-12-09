#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, termios, tty
import time
import numpy as np

# --- CONFIGURATION ---
REAL_MAX_SPEED_M_S = 1.0    # 1.0 m/s
REAL_WHEELBASE_M = 0.25     # 0.25m
STEER_LIMIT_DEG = 40.0      # Max steering angle
CMD_FREQ = 20.0             # Hz

# Physics / Feel
ACCEL_STEP = 0.05           # Speed increase per loop while holding W
BRAKE_STEP = 0.10           # Speed decrease per loop while holding S
STEER_STEP = 4.0            # Degrees change per loop
FRICTION   = 0.02           # Deceleration when key released
STEER_CENTERING = 2.0       # Auto-center steering when key released

# Timeout to detect "key release"
KEY_TIMEOUT = 0.15          

class KeyboardController(Node):
    def __init__(self):
        super().__init__('game_control_cli')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.settings = termios.tcgetattr(sys.stdin)
        self.timer = self.create_timer(1.0/CMD_FREQ, self.control_loop)
        
        # State
        self.speed = 0.0      # -1.0 to 1.0
        self.steer = 0.0      # Degrees
        self.last_key_time = time.time()
        self.key_pressed = None

        print(f"""
        ---------------------------
          WASD GAME CONTROLLER
        ---------------------------
          W : Accelerate
          S : Brake / Reverse
          A : Steer Left
          D : Steer Right
        SPACE : E-Stop
        ---------------------------
        Running... Press Ctrl+C to quit.
        """)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01) # Non-blocking
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def control_loop(self):
        key = self.get_key()
        
        # Determine user input
        if key in ['w', 's', 'a', 'd', ' ']:
            self.last_key_time = time.time()
            self.key_pressed = key
        elif key == '\x03': # Ctrl+C
            self.shutdown()
            return

        # Check for key release (timeout)
        if time.time() - self.last_key_time > KEY_TIMEOUT:
            self.key_pressed = None

        # --- PHYSICS LOGIC ---
        
        # Throttle (W/S)
        if self.key_pressed == 'w':
            self.speed += ACCEL_STEP
        elif self.key_pressed == 's':
            self.speed -= BRAKE_STEP
        elif self.key_pressed == ' ':
            self.speed = 0.0
            self.steer = 0.0
        else:
            # Friction (Coast to stop)
            if self.speed > 0: self.speed -= FRICTION
            if self.speed < 0: self.speed += FRICTION
            if abs(self.speed) < FRICTION: self.speed = 0.0

        # Steering (A/D)
        if self.key_pressed == 'a':
            self.steer += STEER_STEP
        elif self.key_pressed == 'd':
            self.steer -= STEER_STEP
        else:
            # Auto-Center
            if self.steer > 0: self.steer -= STEER_CENTERING
            if self.steer < 0: self.steer += STEER_CENTERING
            if abs(self.steer) < STEER_CENTERING: self.steer = 0.0

        # Clamping
        self.speed = np.clip(self.speed, -1.0, 1.0)
        self.steer = np.clip(self.steer, -STEER_LIMIT_DEG, STEER_LIMIT_DEG)

        self.publish_cmd()

    def publish_cmd(self):
        msg = Twist()
        
        # Map percentage to real speed
        msg.linear.x = float(self.speed * REAL_MAX_SPEED_M_S)
        
        # Calculate Angular Velocity for Ackermann
        # w = (v * tan(theta)) / L
        steer_rad = np.radians(self.steer)
        
        # Small trick: Allow steering even if stopped (send tiny speed)
        # to prevent division by zero or logic errors in driver
        calc_speed = msg.linear.x if abs(msg.linear.x) > 0.01 else 0.01
        
        msg.angular.z = float((calc_speed * np.tan(steer_rad)) / REAL_WHEELBASE_M)

        self.publisher_.publish(msg)
        
        # Print status line (overwrite same line)
        sys.stdout.write(f"\rSpeed: {self.speed*100:3.0f}% | Steer: {self.steer:3.0f} deg   ")
        sys.stdout.flush()

    def shutdown(self):
        print("\nStopping...")
        msg = Twist()
        self.publisher_.publish(msg)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        self.destroy_node()
        rclpy.shutdown()
        sys.exit()

def main():
    rclpy.init()
    node = KeyboardController()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(e)
    finally:
        node.shutdown()

if __name__ == '__main__':
    main()