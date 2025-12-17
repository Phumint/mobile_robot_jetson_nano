import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
import threading
import os
import time

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_heading = self.create_publisher(Float32, 'lane_heading', 10)

        self.declare_parameter('record', True)
        self.declare_parameter('output_path', '/workspace/lane_output.avi')

        self.record = self.get_parameter('record').value
        self.output_path = self.get_parameter('output_path').value

        # --- Camera ---
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error("Camera not opened")

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        self.capture_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self.capture_thread.start()

        # --- Video writer ---
        self.out = None
        if self.record:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.out = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'MJPG'),
                20.0,
                (640, 480)
            )

        # --- Filtering ---
        self.prev_offset = 0.0
        self.prev_heading = 0.0
        self.alpha = 0.7

        self.timer = self.create_timer(0.05, self.timer_callback)

    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = cv2.resize(frame, (640, 480))
            else:
                time.sleep(0.01)

    def detect_lane_frame(self, frame):
        height, width = frame.shape[:2]

        # --- Color threshold ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([79, 0, 183])
        upper_white = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        lx, rx = self.sliding_window_lane(mask)

        debug = frame.copy()

        lane_width_px = 300  # approximate lane width (tune if needed)

        # --- Lane center estimation ---
        if len(lx) > 20 and len(rx) > 20:
            # Both lanes detected
            lane_center = (np.mean(lx) + np.mean(rx)) / 2

        elif len(lx) > 20:
            # Only left lane → search right
            lane_center = np.mean(lx) + lane_width_px / 2

        elif len(rx) > 20:
            # Only right lane → search left
            lane_center = np.mean(rx) - lane_width_px / 2

        else:
            # No lanes → go straight
            return 0.0, 0.0, debug

        # --- Offset ---
        pixel_offset = lane_center - width / 2
        offset_norm = pixel_offset / (width / 2)

        # --- Heading estimation ---
        heading_rad = 0.0
        if len(lx) > 10 and len(rx) > 10:
            left_fit = np.polyfit(np.arange(len(lx)), lx, 1)
            right_fit = np.polyfit(np.arange(len(rx)), rx, 1)
            slope = (left_fit[0] + right_fit[0]) / 2
            heading_rad = float(np.arctan(slope))

        # --- Low-pass filter ---
        offset_norm = self.alpha * self.prev_offset + (1 - self.alpha) * offset_norm
        heading_rad = self.alpha * self.prev_heading + (1 - self.alpha) * heading_rad
        self.prev_offset = offset_norm
        self.prev_heading = heading_rad

        # --- Visualization ---
        steer_x = int(width // 2 + np.tan(heading_rad) * 120)
        cv2.line(debug, (width//2, height), (steer_x, height-120), (0, 0, 255), 3)

        return float(offset_norm), float(heading_rad), debug

    def sliding_window_lane(self, mask):
        histogram = np.sum(mask[mask.shape[0]//2:], axis=0)
        midpoint = histogram.shape[0] // 2

        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        return np.array([left_base]), np.array([right_base])

    def timer_callback(self):
        with self.lock:
            frame = self.latest_frame

        if frame is None:
            return

        offset, heading, vis = self.detect_lane_frame(frame)

        self.pub_offset.publish(Float32(data=offset))
        self.pub_heading.publish(Float32(data=heading))

        if self.record and self.out:
            self.out.write(vis)

    def cleanup(self):
        self.running = False
        self.capture_thread.join(timeout=1.0)
        self.cap.release()
        if self.out:
            self.out.release()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
