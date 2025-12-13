#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import sys
import termios
import tty
import time

class ServoCalibrationNode(Node):
    def __init__(self):
        super().__init__('servo_calibration_node')

        # --- Parameters ---
        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baud_rate', 115200)

        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud_rate').value

        # --- Serial Connection ---
        # We open this DIRECTLY (bypassing the main driver) to get raw access
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"Connected to STM32 on {self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            sys.exit(1)

        self.servo_val = 90  # Start at center
        self.print_instructions()

    def print_instructions(self):
        print("\n========================================")
        print("   STM32 SERVO CALIBRATION NODE")
        print("========================================")
        print("Controls:")
        print("  [a]   Left  (-1)")
        print("  [d]   Right (+1)")
        print("  [A]   Left  (-5)")
        print("  [D]   Right (+5)")
        print("  [s]   Reset Center (90)")
        print("  [q]   Quit & Calculate Results")
        print("----------------------------------------")

    def get_key(self):
        """Reads a single keypress from stdin"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def send_cmd(self, angle):
        """Sends raw #CMD packet. Speed is always 0 during calibration."""
        # PACKET FORMAT: #CMD,speed,angle* (No newline \n)
        packet = f"#CMD,0,{angle}*"
        try:
            self.serial.write(packet.encode())
        except Exception as e:
            self.get_logger().warn(f"Serial write failed: {e}")

    def run(self):
        # Interactive Loop
        try:
            # Initial center command
            self.send_cmd(self.servo_val)
            print(f"\rCurrent Servo PWM: {self.servo_val}", end="")

            while rclpy.ok():
                key = self.get_key()
                
                if key == 'a':
                    self.servo_val -= 1
                elif key == 'd':
                    self.servo_val += 1
                elif key == 'A':
                    self.servo_val -= 5
                elif key == 'D':
                    self.servo_val += 5
                elif key == 's':
                    self.servo_val = 90
                elif key == 'q':
                    self.calculate_and_exit()
                    break
                elif key == '\x03': # Ctrl+C
                    break

                # Safety Clamp
                self.servo_val = max(0, min(180, self.servo_val))
                
                self.send_cmd(self.servo_val)
                print(f"\rCurrent Servo PWM: {self.servo_val}   ", end="")
                
        except Exception as e:
            print(e)
        finally:
            self.send_cmd(90) # Reset to safe center on exit
            if self.serial:
                self.serial.close()

    def calculate_and_exit(self):
        # Restore terminal settings for input()
        # (termios/tty messed them up for get_key)
        # We simply use print/input normally here as the loop is done.
        
        print("\n\n========================================")
        print("       CALIBRATION DATA ENTRY")
        print("========================================")
        
        try:
            # 1. CENTER
            print(f"1. Center PWM (Last position: {self.servo_val})")
            c_input = input(f"   Enter Center PWM [default {self.servo_val}]: ")
            val_center = int(c_input) if c_input else self.servo_val

            # 2. LEFT LIMIT
            print("\n2. Left Limit")
            val_left = int(input("   Enter PWM used for MAX LEFT: "))
            angle_left = float(input("   Enter Measured INNER Wheel Angle (deg): "))

            # 3. RIGHT LIMIT
            print("\n3. Right Limit")
            val_right = int(input("   Enter PWM used for MAX RIGHT: "))
            angle_right = float(input("   Enter Measured INNER Wheel Angle (deg): "))

            # Calculations
            # Ratio = Delta PWM / Angle
            ratio_l = abs(val_left - val_center) / angle_left
            ratio_r = abs(val_right - val_center) / angle_right
            avg_ratio = (ratio_l + ratio_r) / 2.0
            
            offset = val_center - 90

            self.get_logger().info("CALIBRATION COMPLETE")
            print("\n****************************************")
            print("   COPY THESE VALUES TO PYTHON DRIVER")
            print("****************************************")
            print(f"STEERING_RATIO = {avg_ratio:.2f}")
            print(f"SERVO_OFFSET   = {offset}")
            print(f"PWM_RANGE      = ({min(val_left, val_right)}, {max(val_left, val_right)})")
            print("****************************************")

        except ValueError:
            self.get_logger().error("Invalid input. Please enter numbers only.")

def main(args=None):
    rclpy.init(args=args)
    node = ServoCalibrationNode()
    
    # We don't use rclpy.spin() because we have a blocking input loop
    # The node handles its own loop in node.run()
    node.run()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()