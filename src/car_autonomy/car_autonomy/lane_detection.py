import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import cv2
import numpy as np
import threading

class LaneDetectionNode(Node):
    """
    Script 2 (Minimal):
    - Classical CV lane detection
    - Sliding window + polynomial curvature
    - Publishes lane offset and heading
    - NO FSM, NO cv_bridge, NO ROS image topics
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # Publishers (unchanged interface)
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_heading = self.create_publisher(Float32, 'lane_heading', 10)

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Camera not opened')

        self.latest_frame = None
        self.lock = threading.Lock()

        # Background capture thread
        threading.Thread(target=self._capture_frames, daemon=True).start()

        # Timer (20 Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)

    # --------------------------------------------------
    def _capture_frames(self):
        while True:
            if not self.cap.isOpened():
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue  # <-- VERY IMPORTANT on Jetson

            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.latest_frame = frame


    # --------------------------------------------------
    def detect_lane(self, frame):
        height, width = frame.shape[:2]

        # --- ROI ---
        roi = frame[int(height * 0.6):height, :]

        # --- Perspective transform ---
        pts1 = np.float32([
            [0, roi.shape[0]],
            [width, roi.shape[0]],
            [width, 0],
            [0, 0]
        ])
        pts2 = np.float32([
            [0, height],
            [width, height],
            [width, 0],
            [0, 0]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        Minv = cv2.getPerspectiveTransform(pts2, pts1)
        warped = cv2.warpPerspective(roi, M, (width, height))

        # --- Color mask (white + yellow) ---
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (106, 0, 219), (144, 255, 255))
        yellow_mask = cv2.inRange(hsv, (79, 43, 207), (72, 155, 233))
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        # --- Sliding window ---
        lx, rx = self.sliding_window(mask)

        if len(lx) < 10 or len(rx) < 10:
            return 0.0, 0.0, frame

        # --- Polynomial fit (curvature) ---
        # Get y coordinates from mask
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_inds  = np.isin(nonzerox, lx)
        right_inds = np.isin(nonzerox, rx)

        leftx  = nonzerox[left_inds]
        lefty  = nonzeroy[left_inds]
        rightx = nonzerox[right_inds]
        righty = nonzeroy[right_inds]

        if len(leftx) < 50 or len(rightx) < 50:
            return 0.0, 0.0, frame

        left_fit  = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


        center_fit = (left_fit + right_fit) / 2.0

        # Heading from curvature derivative
        y_eval = height * 0.9
        dx_dy = 2 * center_fit[0] * y_eval + center_fit[1]
        heading = float(np.arctan(dx_dy))

        # Offset
        lane_center = (np.mean(lx) + np.mean(rx)) / 2
        offset = (lane_center - width / 2) / (width / 2)

        # --- Visualization ---
        plot_y = np.linspace(0, height - 1, height)
        left_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        right_x = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

        lane_vis = warped.copy()
        for y, lx_p, rx_p in zip(plot_y.astype(int), left_x.astype(int), right_x.astype(int)):
            if 0 <= lx_p < width and 0 <= rx_p < width:
                cv2.circle(lane_vis, (lx_p, y), 1, (255, 0, 0), -1)
                cv2.circle(lane_vis, (rx_p, y), 1, (0, 0, 255), -1)

        lane_area = np.zeros_like(lane_vis)
        pts = np.vstack((np.transpose(np.vstack([left_x, plot_y])),
                          np.flipud(np.transpose(np.vstack([right_x, plot_y])))))
        cv2.fillPoly(lane_area, [pts.astype(np.int32)], (0, 255, 0))

        unwarped_lane = cv2.warpPerspective(lane_area, Minv, (width, roi.shape[0]))
        frame[int(height * 0.6):height, :] = cv2.addWeighted(
            frame[int(height * 0.6):height, :], 1.0, unwarped_lane, 0.3, 0)

        cv2.putText(frame, f'Offset: {offset:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Heading: {heading:.2f} rad', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return float(offset), float(heading), frame

    # --------------------------------------------------
    def sliding_window(self, mask):
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint

        n_windows = 12
        window_height = mask.shape[0] // n_windows
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 50
        minpix = 50
        lx, rx = [], []
        left_current, right_current = left_base, right_base

        for window in range(n_windows):
            win_y_low = mask.shape[0] - (window + 1) * window_height
            win_y_high = mask.shape[0] - window * window_height

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= left_current - margin) & (nonzerox < left_current + margin))
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= right_current - margin) & (nonzerox < right_current + margin))

            if np.sum(good_left) > minpix:
                left_current = int(np.mean(nonzerox[good_left]))
            if np.sum(good_right) > minpix:
                right_current = int(np.mean(nonzerox[good_right]))

            lx.extend(nonzerox[good_left])
            rx.extend(nonzerox[good_right])

        return np.array(lx), np.array(rx)

    # --------------------------------------------------
    def timer_callback(self):
        with self.lock:
            frame = self.latest_frame

        if frame is None:
            return

        offset, heading, visual = self.detect_lane(frame)

        msg_o = Float32()
        msg_o.data = offset
        self.pub_offset.publish(msg_o)

        msg_h = Float32()
        msg_h.data = heading
        self.pub_heading.publish(msg_h)

    # --------------------------------------------------
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
