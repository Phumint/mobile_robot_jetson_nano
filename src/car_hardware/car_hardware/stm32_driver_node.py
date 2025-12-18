#!/usr/bin/env python3
"""
STM32 Driver Node (Optimized)
Improvements: 
1. Dynamic dt calculation for accurate Yaw integration.
2. Allows steering wheels to turn even when stopped.
3. Thread-safe serial writes.
4. Robust shutdown procedure.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32
import serial
import threading
import math
import time

class STM32DriverNode(Node):
    def __init__(self):
        super().__init__('stm32_driver_node')

        # --- Parameters ---
        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_base', 0.135)
        self.declare_parameter('max_speed_m_s', 1.0) # Renamed for clarity
        self.declare_parameter('max_steering_angle_deg', 25.0)

        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.max_speed = self.get_parameter('max_speed_m_s').value
        self.max_steer = self.get_parameter('max_steering_angle_deg').value

        # --- Serial Connection ---
        self.serial = None
        self.serial_lock = threading.Lock() # Prevent read/write collisions
        self._open_serial()

        # --- ROS2 Interfaces ---
        self.encoder_pub = self.create_publisher(Int32, 'encoder_count', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', 10)
        
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )

        # --- State Variables ---
        self.last_imu_time = self.get_clock().now()
        self.integrated_yaw = 0.0
        
        # --- Threading ---
        self.running = True
        self.read_thread = threading.Thread(target=self.serial_read_loop, daemon=True)
        self.read_thread.start()

        self.get_logger().info("STM32 Driver Node initialized.")

    def _open_serial(self):
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"Connected to STM32 on {self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.serial = None

    # ---------------------------
    #  CMD_VEL CALLBACK (Control)
    # ---------------------------
    def cmd_vel_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        # --- CALIBRATION VALUES ---
        # 90 is usually center. Adjust this if your car goes straight at 95 or 85.
        SERVO_CENTER = 90  
        SERVO_MIN    = 20
        SERVO_MAX    = 137
        
        # Max Steering Angle (in servo steps)
        # e.g. 30 means it can go from 60 (90-30) to 120 (90+30)
        STEERING_RANGE = 40 
        # --------------------------

        # 1. DIRECT MAPPING (Simplified)
        # We assume angular.z is normalized roughly between -1.0 and 1.0 by the lane follower.
        # -1.0 = Max Right, 0.0 = Straight, 1.0 = Max Left
        
        # Invert logic: If positive angular (Left) needs servo to go UP, use +
        # If positive angular (Left) needs servo to go DOWN, use -
        # Based on your previous code (90 - shift), it seems NEGATIVE is correct.
        servo_shift = int(angular * STEERING_RANGE)
        servo_cmd = SERVO_CENTER - servo_shift

        # 2. Safety Clamp
        servo_cmd = max(SERVO_MIN, min(SERVO_MAX, servo_cmd))

        # 3. Calculate Motor Speed
        # Map 0.0 - 1.0 m/s to 0 - 100 PWM
        speed_cmd = int((linear / self.max_speed) * 100)
        speed_cmd = max(-100, min(100, speed_cmd))

        self.send_packet(speed_cmd, servo_cmd)

    def send_packet(self, speed, angle):
        if not self.serial:
            return
        
        packet = f"#CMD,{speed},{angle}*\n" # Added \n for robustness
        try:
            with self.serial_lock:
                self.serial.write(packet.encode())
        except Exception as e:
            self.get_logger().warn(f"Serial write failed: {e}")
            self._open_serial()

    # ---------------------------
    #  SERIAL READ LOOP
    # ---------------------------
    def serial_read_loop(self):
        while self.running and rclpy.ok():
            if not self.serial:
                time.sleep(1)
                self._open_serial()
                continue

            try:
                # readline blocks for timeout (0.1s)
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    continue # Timeout or empty

                # Validate Packet Format
                if line.startswith('$') and line.endswith('*'):
                    self.parse_sensor_data(line)
            
            except serial.SerialException:
                self.get_logger().error("Serial connection lost.")
                self.serial.close()
                self.serial = None
            except Exception as e:
                self.get_logger().warn(f"Read error: {e}")

    # ---------------------------
    #  PARSING & PUBLISHING
    # ---------------------------
    def parse_sensor_data(self, line):
        # Format: $ENC,12345;IMU,ax,ay,az,gx,gy,gz,roll,pitch*
        try:
            content = line[1:-1] # Strip $ and *
            parts = content.split(';')
            
            for part in parts:
                if part.startswith("ENC"):
                    self.handle_encoder(part)
                elif part.startswith("IMU"):
                    self.handle_imu(part)
        except ValueError:
            pass # Malformed numbers

    def handle_encoder(self, part):
        # part = "ENC,12345"
        try:
            val = int(part.split(',')[1])
            msg = Int32()
            msg.data = val
            self.encoder_pub.publish(msg)
        except IndexError:
            pass

    def handle_imu(self, part):
        # part = "IMU,ax,ay,az,gx,gy,gz,roll,pitch" (integers scaled)
        # SCALING MUST MATCH STM32 CODE:
        # Accel / 1000.0, Gyro / 100.0, Euler / 100.0
        vals = part.split(',')[1:]
        if len(vals) < 8: return

        ax = float(vals[0]) / 1000.0
        ay = float(vals[1]) / 1000.0
        az = float(vals[2]) / 1000.0
        gx = float(vals[3]) / 100.0
        gy = float(vals[4]) / 100.0
        gz = float(vals[5]) / 100.0
        # roll/pitch ignored for orientation calculation 
        # because we calculate quaternion from gyro integration + accel usually, 
        # but here we just use what we have.

        # --- Dynamic Time Integration ---
        current_time = self.get_clock().now()
        dt = (current_time - self.last_imu_time).nanoseconds / 1e9
        self.last_imu_time = current_time

        # Avoid integration spikes if connection lagged
        if dt > 1.0: dt = 0.0 

        # Integrate Yaw (Gyro Z)
        self.integrated_yaw += math.radians(gz) * dt

        self.publish_imu_msg(ax, ay, az, gx, gy, gz)

    def publish_imu_msg(self, ax, ay, az, gx, gy, gz):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        # Linear Accel
        msg.linear_acceleration.x = ax * 9.81
        msg.linear_acceleration.y = ay * 9.81
        msg.linear_acceleration.z = az * 9.81

        # Angular Velocity
        msg.angular_velocity.x = math.radians(gx)
        msg.angular_velocity.y = math.radians(gy)
        msg.angular_velocity.z = math.radians(gz)

        # Orientation (Quaternion from Yaw only for planar robot)
        # Assuming the robot stays mostly flat, we simplify to Yaw rotation
        cy = math.cos(self.integrated_yaw * 0.5)
        sy = math.sin(self.integrated_yaw * 0.5)
        
        # Simple Yaw-only Quaternion (w, x, y, z)
        msg.orientation.w = cy
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = sy

        self.imu_pub.publish(msg)

    def destroy_node(self):
        self.running = False
        # Stop the robot on shutdown
        self.send_packet(0, 90) 
        if self.serial:
            self.serial.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = STM32DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
