#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, termios, tty
import time
import numpy as np

# --- CONFIGURATION ---
REAL_MAX_SPEED_M_S = 1.0     # 1.0 m/s
REAL_WHEELBASE_M = 0.25      # 0.25m
STEER_LIMIT_DEG = 40.0       # Max steering angle
CMD_FREQ = 20.0              # Hz
LOOP_TIMEOUT = 1.0 / CMD_FREQ # New: Timeout used to control loop rate

# Physics / Feel
ACCEL_STEP = 0.05            # Speed increase per loop while holding W
BRAKE_STEP = 0.10            # Speed decrease per loop while holding S
STEER_STEP = 4.0             # Degrees change per loop
FRICTION   = 0.02            # Deceleration when key released
STEER_CENTERING = 2.0        # Auto-center steering when key released

# Timeout to detect "key release"
KEY_TIMEOUT = 0.15          

class KeyboardController(Node):
    def __init__(self):
        super().__init__('game_control_cli')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Save and set terminal settings once
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno()) # Set raw mode once for the duration of the node
        
        # State
        self.speed = 0.0     # -1.0 to 1.0
        self.steer = 0.0     # Degrees
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

    def get_key(self, timeout=LOOP_TIMEOUT):
        """Non-blocking keyboard read, assumes terminal is already in raw mode."""
        # Check for key press using select with a timeout
        rlist, _, _ = select.select([sys.stdin], [], [], timeout) 
        if rlist:
            # Read a character
            key = sys.stdin.read(1)
        else:
            key = ''
        return key

    def control_logic(self, key):
        """
        Calculates new speed/steer states based on key press 
        and publishes the Twist message.
        """
        
        # Determine user input
        if key in ['w', 's', 'a', 'd', ' ']:
            self.last_key_time = time.time()
            self.key_pressed = key
        elif key == '\x03': # Ctrl+C
            raise KeyboardInterrupt # Use an exception to cleanly exit the run loop
            
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

    def run(self):
        """Main execution loop that replaces rclpy.spin() for teleop."""
        while rclpy.ok():
            try:
                # 1. Get key input (blocks for LOOP_TIMEOUT)
                key = self.get_key(timeout=LOOP_TIMEOUT)
                
                # 2. Process logic and publish
                self.control_logic(key)
                
                # 3. Allow ROS to process any pending callbacks (like timer/subscriptions if added later)
                rclpy.spin_once(self, timeout_sec=0) 
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.get_logger().error(f'Runtime Error: {e}')
                break
        
        # Cleanup when the loop exits
        self.shutdown()


    def shutdown(self):
        print("\nStopping...")
        # Send zero velocity command
        msg = Twist()
        self.publisher_.publish(msg)
        
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0) # Exit cleanly

def main():
    rclpy.init()
    node = KeyboardController()
    node.run() # Start the main control loop

if __name__ == '__main__':
    main()