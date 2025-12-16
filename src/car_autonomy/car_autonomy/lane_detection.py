import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
import threading
import os

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_heading = self.create_publisher(Float32, 'lane_heading', 10)

        # --- Parameters ---
        self.declare_parameter('record', True)
        self.declare_parameter('output_path', '/workspace/lane_output.avi')
        self.declare_parameter('t_section_turn', 'left') 

        self.record = self.get_parameter('record').get_parameter_value().bool_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.t_section_turn = self.get_parameter('t_section_turn').get_parameter_value().string_value.lower()
        
        # --- T-section State ---
        self.t_section_active = False
        self.t_section_frames = 0
        self.T_SECTION_TURN_FRAMES = 50 
        
        # --- Camera setup ---
        # 1. Open Camera with V4L2 backend (More stable on Jetson)
        self.cap = cv2.VideoCapture(0)

        # 2. FORCE MJPG Format
        # Your v4l2-ctl output shows Index 0 is MJPG. We must request this.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 3. Set Resolution & FPS to match the supported mode
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 4. Verification
        if not self.cap.isOpened():
             self.get_logger().warning("⚠️ Camera not opened (index 0).")
        self.latest_frame = None
        self.lock = threading.Lock()

        # --- Video writer ---
        self.out = None
        if self.record:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (640, 480))

        # --- Threading ---
        threading.Thread(target=self._capture_frames, daemon=True).start()
        self.timer = self.create_timer(0.05, self.timer_callback)

    def _capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                with self.lock:
                    self.latest_frame = frame

    def detect_lane_frame(self, frame):
        height, width = frame.shape[:2]
        # roi_points = np.array([[(0, height - 30), (width, height - 30), (width, int(height * 0.8)), (0, int(height * 0.8))]], dtype=np.int32)
        roi_points = np.array([[(16, 475), (159, 241), (503, 237), (638, 444), (17, 477)]], dtype=np.int32)



        # Perspective Transform
        pts1 = np.float32([roi_points[0][3], roi_points[0][0], roi_points[0][2], roi_points[0][1]])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (width, height))

        # Color Mask (Blue)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([79, 0, 183])
        upper_blue = np.array([122, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        lx, rx = self.sliding_window_lane(mask)
        lanes_missing = (len(lx) < 200 and len(rx) < 200)

        # T-Section Logic
        if lanes_missing and not self.t_section_active:
            self.t_section_active = True
            self.t_section_frames = 0
            
        if self.t_section_active:
            if self.t_section_frames < self.T_SECTION_TURN_FRAMES:
                self.t_section_frames += 1
                heading_rad = np.pi/4 if self.t_section_turn == 'right' else -np.pi/4
                return 0.0, float(heading_rad), frame.copy()
            else:
                self.t_section_active = False

        # Normal Lane Following
        if len(lx) < 20 and len(rx) < 20:
             return 0.0, 0.0, frame.copy()

        left_fit = np.polyfit(np.arange(len(lx)), lx, 2) if len(lx) > 2 else None
        right_fit = np.polyfit(np.arange(len(rx)), rx, 2) if len(rx) > 2 else None

        y_eval = height * 0.9
        heading_rad = 0.0
        
        if left_fit is not None and right_fit is not None:
            left_slope = 2 * left_fit[0] * y_eval + left_fit[1]
            right_slope = 2 * right_fit[0] * y_eval + right_fit[1]
            lane_slope = (left_slope + right_slope) / 2.0
            heading_rad = float(np.arctan(lane_slope))

        lane_center = (np.mean(lx) + np.mean(rx)) / 2
        pixel_offset = lane_center - width / 2
        offset_norm = pixel_offset / (width / 2)

        return float(offset_norm), float(heading_rad), frame.copy()

    def sliding_window_lane(self, mask):
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        
        n_windows = 12
        window_height = mask.shape[0] // n_windows
        nonzero = mask.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        margin, minpix = 50, 50
        lx, rx = [], []
        l_current, r_current = left_base, right_base

        for window in range(n_windows):
            win_y_low = mask.shape[0] - (window + 1) * window_height
            win_y_high = mask.shape[0] - window * window_height
            win_xleft_low, win_xleft_high = l_current - margin, l_current + margin
            win_xright_low, win_xright_high = r_current - margin, r_current + margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            if len(good_left) > minpix: l_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: r_current = int(np.mean(nonzerox[good_right]))

            lx.extend(nonzerox[good_left])
            rx.extend(nonzerox[good_right])

        return np.array(lx), np.array(rx)

    def timer_callback(self):
        with self.lock:
            frame = self.latest_frame
        if frame is not None:
            offset, heading, visual = self.detect_lane_frame(frame)
            msg_offset = Float32()
            msg_offset.data = offset
            self.pub_offset.publish(msg_offset)
            msg_heading = Float32()
            msg_heading.data = heading
            self.pub_heading.publish(msg_heading)
            if self.record and self.out: self.out.write(visual)

    def cleanup(self):
        if self.cap: self.cap.release()
        if self.out: self.out.release()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
