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

        # --- Parameters ---
        self.declare_parameter('record', True)
        # We save to /workspace so it appears in your ~/phumint_ws folder on the Host
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
        # 1. Open Camera with V4L2 backend
        self.cap = cv2.VideoCapture(0)

        # 2. FORCE MJPG Format (Critical for Jetson/Docker stability)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 3. Set Resolution & FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # 4. Verification
        if not self.cap.isOpened():
             self.get_logger().error("⚠️ Camera not opened (index 0)!")
        
        self.latest_frame = None
        self.lock = threading.Lock()

        # --- Video writer ---
        self.out = None
        if self.record:
            try:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                # Use MJPG codec for Docker compatibility (XVID often fails)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (640, 480))
                
                if self.out.isOpened():
                    self.get_logger().info(f"✅ Recording started: {self.output_path}")
                else:
                    self.get_logger().error(f"❌ Failed to open VideoWriter! Check permissions for {self.output_path}")
            except Exception as e:
                self.get_logger().error(f"❌ VideoWriter error: {str(e)}")

        # --- Threading (with Safe Shutdown Flag) ---
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        self.timer = self.create_timer(0.05, self.timer_callback)

    def _capture_frames(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)

    def detect_lane_frame(self, frame):
        height, width = frame.shape[:2]

        # 1. Define ROI
        # Order for polylines: bl -> tl -> tr -> br -> bl
        bl = (199, 479)
        tl = (275, 359)
        tr = (404, 363)
        br = (495, 479)
        roi_points = np.array([[bl, tl, tr, br, bl]], dtype=np.int32)

        # 2. Perspective Transform (Bird's Eye View)
        # Standard Mapping: [Top-Left, Bottom-Left, Top-Right, Bottom-Right]
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (width, height))

        # 3. Color Mask (White Detection)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_white = np.array([79, 0, 183])
        upper_white = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 4. Find Lines (Sliding Window)
        lx, rx = self.sliding_window_lane(mask)
        
        # --- VISUALIZATION (Debug Overlay) ---
        debug_frame = frame.copy()

        # A. Draw the ROI Box (Yellow)
        cv2.polylines(debug_frame, [roi_points], True, (0, 255, 255), 1)

        # B. Draw the "Mask" (Picture-in-Picture)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(mask_bgr, (160, 120)) 
        debug_frame[0:120, 0:160] = small_mask
        cv2.rectangle(debug_frame, (0,0), (160,120), (255,255,255), 1)

        # --- LOGIC ---
        lanes_missing = (len(lx) < 100 and len(rx) < 100)

        # T-Section Logic
        if lanes_missing and not self.t_section_active:
            self.t_section_active = True
            self.t_section_frames = 0
            
        if self.t_section_active:
            cv2.putText(debug_frame, "T-SECTION", (width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if self.t_section_frames < self.T_SECTION_TURN_FRAMES:
                self.t_section_frames += 1
                heading_rad = np.pi/4 if self.t_section_turn == 'right' else -np.pi/4
                return 0.0, float(heading_rad), debug_frame
            else:
                self.t_section_active = False

        # Normal Lane Following
        if len(lx) < 20 and len(rx) < 20:
             cv2.putText(debug_frame, "NO LANE", (width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
             return 0.0, 0.0, debug_frame

        # Calculate Heading
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

        # C. Draw Steering Line (Red)
        steer_x = int(width//2 + np.tan(heading_rad) * 100)
        cv2.line(debug_frame, (width//2, height), (steer_x, height-100), (0, 0, 255), 3)

        return float(offset_norm), float(heading_rad), debug_frame

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
            
            # Write visual frame to video
            if self.record and self.out: 
                self.out.write(visual)

    def cleanup(self):
        self.get_logger().info("Stopping camera thread...")
        self.running = False  # 1. Signal thread to stop
        
        # 2. Wait briefly for thread to finish
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
            
        # 3. Release resources safely
        if self.cap: self.cap.release()
        if self.out: self.out.release()
        self.get_logger().info("Resources released.")

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