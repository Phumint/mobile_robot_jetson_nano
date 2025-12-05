#!/usr/bin/env python3
"""
STM32 Driver Node (stable version)
Reads STM32 packets:
  $ENC,<count>;IMU,ax,ay,az,gx,gy,gz,roll,pitch*
Controls motor & steering.
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

        # Parameters
        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_base', 0.25)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_steering_angle', 40.0)

        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud_rate').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value

        # Try to open serial port
        self.serial = None
        self._open_serial()

        # Publishers
        self.encoder_pub = self.create_publisher(Int32, 'encoder_count', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', 10)

        # Subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )

        # Thread control
        self.running = True
        self.thread = threading.Thread(target=self.serial_thread, daemon=True)
        self.thread.start()

        self.integrated_yaw = 0.0

        self.get_logger().info("STM32 Driver Node initialized")

    def _open_serial(self):
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"Connected to STM32 on {self.port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port {self.port}: {e}")
            self.serial = None

    # ---------------------------
    # SEND MOTOR & STEERING CMD
    # ---------------------------
    def cmd_vel_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        # Motor speed (-100 to 100)
        speed = int((linear / self.max_speed) * 100)
        speed = max(-100, min(100, speed))

        # Ackermann steering
        if abs(linear) > 0.01:
            steering_rad = math.atan(angular * self.wheel_base / linear)
            steering_deg = math.degrees(steering_rad)
        else:
            steering_deg = 0

        steering_deg = max(-self.max_steering_angle, min(self.max_steering_angle, steering_deg))

        # Servo mapping
        servo = 90 - int(steering_deg)
        servo = max(50, min(130, servo))

        self.send_cmd(speed, servo)

    def send_cmd(self, speed, servo):
        if not self.serial:
            return
        try:
            packet = f"#CMD,{speed},{servo}*"
            self.serial.write(packet.encode())
        except Exception as e:
            self.get_logger().warn(f"Failed to send command: {e}")

    # ---------------------------
    # SERIAL READING THREAD
    # ---------------------------
    def serial_thread(self):
        while self.running:
            if not self.serial:
                time.sleep(0.5)
                self._open_serial()
                continue

            try:
                line = self.serial.readline().decode(errors='ignore').strip()

                if not line:
                    continue

                if not (line.startswith('$') and line.endswith('*')):
                    continue

                self.parse_packet(line)

            except Exception as e:
                self.get_logger().warn(f"Serial read warning: {e}")
                time.sleep(0.1)

    # ---------------------------
    # PACKET PARSER
    # ---------------------------
    def parse_packet(self, line):
        # Remove $ and *
        core = line[1:-1]
        parts = core.split(';')

        enc_part = parts[0]
        imu_part = parts[1]

        # ----- Encoder -----
        if enc_part.startswith("ENC,"):
            try:
                count = int(enc_part.split(',')[1])
                msg = Int32()
                msg.data = count
                self.encoder_pub.publish(msg)
            except:
                pass

        # ----- IMU -----
        if imu_part.startswith("IMU,"):
            vals = imu_part.split(',')[1:]
            if len(vals) >= 8:
                try:
                    ax = int(vals[0]) / 1000.0
                    ay = int(vals[1]) / 1000.0
                    az = int(vals[2]) / 1000.0
                    gx = int(vals[3]) / 100.0
                    gy = int(vals[4]) / 100.0
                    gz = int(vals[5]) / 100.0
                    roll = int(vals[6]) / 100.0
                    pitch = int(vals[7]) / 100.0

                    self.publish_imu(ax, ay, az, gx, gy, gz, roll, pitch)
                except:
                    pass

    # ---------------------------
    # IMU MESSAGE PUBLISHER
    # ---------------------------
    def publish_imu(self, ax, ay, az, gx, gy, gz, roll, pitch):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        msg.linear_acceleration.x = ax * 9.81
        msg.linear_acceleration.y = ay * 9.81
        msg.linear_acceleration.z = az * 9.81

        msg.angular_velocity.x = math.radians(gx)
        msg.angular_velocity.y = math.radians(gy)
        msg.angular_velocity.z = math.radians(gz)

        dt = 0.1
        self.integrated_yaw += math.radians(gz) * dt

        cy = math.cos(self.integrated_yaw * 0.5)
        sy = math.sin(self.integrated_yaw * 0.5)
        cp = math.cos(math.radians(pitch) * 0.5)
        sp = math.sin(math.radians(pitch) * 0.5)
        cr = math.cos(math.radians(roll) * 0.5)
        sr = math.sin(math.radians(roll) * 0.5)

        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy

        msg.orientation_covariance = [0.01,0,0, 0,0.01,0, 0,0,0.1]
        msg.angular_velocity_covariance = [0.01,0,0, 0,0.01,0, 0,0,0.01]
        msg.linear_acceleration_covariance = [0.01,0,0, 0,0.01,0, 0,0,0.01]

        self.imu_pub.publish(msg)

    # ---------------------------
    # CLEANUP
    # ---------------------------
    def destroy_node(self):
        self.running = False
        time.sleep(0.1)
        if self.serial:
            try:
                self.serial.close()
            except:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = STM32DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
