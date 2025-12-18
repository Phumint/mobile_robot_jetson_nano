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
    # Sliding Window Lane Detection (NEW ALGORITHM)
    # ======================================================
    def sliding_window_lane(self, binary_warped):
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves
        midpoint = int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        nwindows = 12
        margin = 50       # Width of the windows +/- margin
        minpix = 50       # Minimum number of pixels found to recenter window

        # Set height of windows
        window_height = int(binary_warped.shape[0]//nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # [Visual] Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), 
                          (win_xleft_high, win_y_high), (0, 255, 0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), 
                          (win_xright_high, win_y_high), (0, 255, 0), 2) 
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract line pixel positions
        leftx = nonzerox[left_lane_inds]
        # lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        # righty = nonzeroy[right_lane_inds] 

        # --- CALCULATE LANE CENTERS ---
        # To match your previous logic, we need a single X value for Left and Right.
        # We will take the average X position of all detected pixels.
        
        final_left_x = leftx_base # Default to histogram base if sliding window fails
        final_right_x = rightx_base

        if len(leftx) > 0:
            final_left_x = int(np.mean(leftx))
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0] # Color left red

        if len(rightx) > 0:
            final_right_x = int(np.mean(rightx))
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # Color right blue

        return final_left_x, final_right_x, out_img

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

        # --- Sliding Window Detection (NEW) ---
        left_x, right_x, sliding_vis = self.sliding_window_lane(lane_mask)
        
        lane_center = (left_x + right_x) / 2.0

        # --- Offset ---
        pixel_offset = lane_center - (w / 2)
        offset_norm = pixel_offset / (w / 2)

        # --- Heading ---
        lane_width_px = right_x - left_x
        # heading = np.arctan2((right_x - left_x), lane_width_px + 1e-6)
        heading = 0.0

        # --- Low-pass filter ---
        offset_norm = self.alpha * self.prev_offset + (1 - self.alpha) * offset_norm
        heading = self.alpha * self.prev_heading + (1 - self.alpha) * heading
        self.prev_offset = offset_norm
        self.prev_heading = heading

        # --- Draw steering direction ---
        steer_x = int(w / 2 + offset_norm * (w / 2))
        cv2.line(debug, (w // 2, h), (steer_x, h - 120), (0, 0, 255), 3)

        # --- Compose debug view ---
        # Resizing sliding_vis to fit alongside debug
        top = np.hstack((debug, sliding_vis))
        
        # Bottom row: ROI Mask + Raw Binary Mask (converted to color for stacking)
        mask_vis = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        bottom = np.hstack((
            cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR),
            mask_vis
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