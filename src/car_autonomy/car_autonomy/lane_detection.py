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

        # ---------------- ROS ----------------
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_heading = self.create_publisher(Float32, 'lane_heading', 10)

        self.declare_parameter('record', True)
        self.declare_parameter('output_path', '/workspace/lane_output.avi')

        self.record = self.get_parameter('record').value
        self.output_path = self.get_parameter('output_path').value

        # ---------------- Camera ----------------
        self.cap = cv2.VideoCapture(0)
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

        # ---------------- Video writer ----------------
        self.out = None
        if self.record:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.out = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*'MJPG'),
                20.0,
                (640, 480)
            )

        # ---------------- POLYGON ROI (YOUR ROI) ----------------
        self.roi_pts = np.array([
            [30, 477],   # bottom-left
            [115, 247],  # top-left
            [520, 255],  # top-right
            [638, 473]   # bottom-right
        ], dtype=np.int32)

        # ---------------- Perspective Transform ----------------
        self.dst_pts = np.float32([
            [0, 480],
            [0, 0],
            [640, 0],
            [640, 480]
        ])

        self.M = cv2.getPerspectiveTransform(
            self.roi_pts.astype(np.float32),
            self.dst_pts
        )

        # ---------------- Filtering ----------------
        self.prev_offset = 0.0
        self.prev_heading = 0.0
        self.alpha = 0.7

        self.timer = self.create_timer(0.05, self.timer_callback)

    # ======================================================
    # Frame capture thread
    # ======================================================
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = cv2.resize(frame, (640, 480))
            else:
                time.sleep(0.01)

    # ======================================================
    # Apply polygon ROI
    # ======================================================
    def apply_polygon_roi(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_pts], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        return roi, mask

    # ======================================================
    # Simple lane base detection
    # ======================================================
    def find_lane_centers(self, binary):
        histogram = np.sum(binary[binary.shape[0] // 2:], axis=0)
        midpoint = histogram.shape[0] // 2

        left_x = np.argmax(histogram[:midpoint])
        right_x = np.argmax(histogram[midpoint:]) + midpoint

        return left_x, right_x

    # ======================================================
    # Lane detection pipeline
    # ======================================================
    def detect_lane_frame(self, frame):
        h, w = frame.shape[:2]

        # --- ROI ---
        roi_frame, roi_mask = self.apply_polygon_roi(frame)

        debug = frame.copy()
        cv2.polylines(debug, [self.roi_pts], True, (0, 255, 255), 2)

        # --- Perspective transform ---
        bird = cv2.warpPerspective(roi_frame, self.M, (w, h))

        # --- Color threshold (white lanes) ---
        hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        lane_mask = cv2.inRange(hsv, lower_white, upper_white)

        # --- Overlay lane mask for visualization ---
        lane_overlay = bird.copy()
        lane_overlay[lane_mask > 0] = (0, 255, 0)

        # --- Lane center estimation ---
        left_x, right_x = self.find_lane_centers(lane_mask)
        lane_center = (left_x + right_x) / 2.0

        # --- Offset ---
        pixel_offset = lane_center - (w / 2)
        offset_norm = pixel_offset / (w / 2)

        # --- Heading (very rough) ---
        lane_width_px = right_x - left_x
        heading = np.arctan2((right_x - left_x), lane_width_px + 1e-6)

        # --- Low-pass filter ---
        offset_norm = self.alpha * self.prev_offset + (1 - self.alpha) * offset_norm
        heading = self.alpha * self.prev_heading + (1 - self.alpha) * heading
        self.prev_offset = offset_norm
        self.prev_heading = heading

        # --- Draw steering direction ---
        steer_x = int(w / 2 + offset_norm * (w / 2))
        cv2.line(debug, (w // 2, h), (steer_x, h - 120), (0, 0, 255), 3)

        # --- Compose debug view ---
        top = np.hstack((debug, lane_overlay))
        bottom = np.hstack((
            cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        ))
        vis = np.vstack((top, bottom))

        return offset_norm, heading, vis

    # ======================================================
    # ROS timer
    # ======================================================
    def timer_callback(self):
        with self.lock:
            frame = self.latest_frame

        if frame is None:
            return

        offset, heading, vis = self.detect_lane_frame(frame)

        self.pub_offset.publish(Float32(data=float(offset)))
        self.pub_heading.publish(Float32(data=float(heading)))

        if self.record and self.out:
            vis_resized = cv2.resize(vis, (640, 480))
            self.out.write(vis_resized)

    # ======================================================
    # Cleanup
    # ======================================================
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


if __name__ == '__main__':
    main()
