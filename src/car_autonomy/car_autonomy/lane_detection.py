import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from std_msgs.msg import Float32
import threading

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Publishers
        self.pub_offset = self.create_publisher(Float32, 'lane_offset', 10)
        self.pub_heading = self.create_publisher(Float32, 'lane_heading', 10)

        # Subscriber: To see what the follower is actually doing
        self.steer_sub = self.create_subscription(Float32, 'control/steer_cmd', self.steer_cb, 10)
        self.current_steer = 0.0

        # Video Setup
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('lane_debug.avi', fourcc, 20.0, (640, 480))

        self.latest_frame = None
        self.lock = threading.Lock()

        # Background capture thread
        threading.Thread(target=self._capture_frames, daemon=True).start()

        # Timer (20 Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def steer_cb(self, msg):
        self.current_steer = msg.data

    def _capture_frames(self):
        while True:
            if not self.cap.isOpened(): continue
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.resize(frame, (640, 480))
            with self.lock:
                self.latest_frame = frame

    def detect_lane(self, frame):
        height, width = frame.shape[:2]

        # 1. Visualize ROI (Yellow Rectangle) - tackling the curve limit by raising to 0.5
        roi_top = int(height * 0.5) 
        cv2.rectangle(frame, (0, roi_top), (width, height), (0, 255, 255), 2)
        roi = frame[roi_top:height, :]

        # 2. Perspective Transform (Bird's Eye)
        pts1 = np.float32([[0, roi.shape[0]], [width, roi.shape[0]], [width, 0], [0, 0]])
        pts2 = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(roi, M, (width, height))

        # 3. Color Masking (Yellow/White)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (106, 0, 219), (144, 255, 255))
        yellow_mask = cv2.inRange(hsv, (79, 43, 207), (72, 155, 233))
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 4. Sliding Window (Increased margin=80, Lowered minpix=30 for curves)
        lx, rx, debug_view = self.sliding_window(mask)

        if len(lx) < 10 or len(rx) < 10:
            return 0.0, 0.0, frame

        # 5. Lane Fitting and Curvature
        left_fit = np.polyfit(np.arange(len(lx)), lx, 1) # Simple linear fit for heading
        right_fit = np.polyfit(np.arange(len(rx)), rx, 1)
        lane_center = (lx[-1] + rx[-1]) / 2
        
        offset = (lane_center - (width / 2)) / (width / 2) # Normalized
        heading = (left_fit[0] + right_fit[0]) / 2

        # 6. Overlays
        cv2.putText(frame, f'Offset: {offset:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Servo Cmd: {self.current_steer:.2f}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Picture-in-Picture: Show the warped sliding windows in the corner
        small_debug = cv2.resize(debug_view, (160, 120))
        frame[0:120, width-160:width] = small_debug

        return float(offset), float(heading), frame

    def sliding_window(self, mask):
        out_img = np.dstack((mask, mask, mask)) * 255
        histogram = np.sum(mask[mask.shape[0]//2:,:], axis=0)
        midpoint = histogram.shape[0] // 2
        left_c = np.argmax(histogram[:midpoint])
        right_c = np.argmax(histogram[midpoint:]) + midpoint
        
        n_windows, margin, minpix = 10, 80, 30
        window_height = mask.shape[0] // n_windows
        nonzero = mask.nonzero()
        nzy, nzx = np.array(nonzero[0]), np.array(nonzero[1])
        lx, rx = [], []

        for w in range(n_windows):
            y_low, y_high = mask.shape[0]-(w+1)*window_height, mask.shape[0]-w*window_height
            # Define window boundaries
            win_l_l, win_l_h = left_c-margin, left_c+margin
            win_r_l, win_r_h = right_c-margin, right_c+margin
            # Draw for video
            cv2.rectangle(out_img,(win_l_l,y_low),(win_l_h,y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_r_l,y_low),(win_r_h,y_high),(0,0,255), 2)
            # Find pixels
            good_l = ((nzy >= y_low) & (nzy < y_high) & (nzx >= win_l_l) & (nzx < win_l_h)).nonzero()[0]
            good_r = ((nzy >= y_low) & (nzy < y_high) & (nzx >= win_r_l) & (nzx < win_r_h)).nonzero()[0]
            if len(good_l) > minpix: left_c = int(np.mean(nzx[good_l]))
            if len(good_r) > minpix: right_c = int(np.mean(nzx[good_r]))
            lx.append(left_c); rx.append(right_c)

        return lx, rx, out_img

    def timer_callback(self):
        with self.lock:
            if self.latest_frame is None: return
            frame = self.latest_frame.copy()

        offset, heading, visual = self.detect_lane(frame)
        self.pub_offset.publish(Float32(data=offset))
        self.pub_heading.publish(Float32(data=heading))
        self.out.write(visual)

    def destroy_node(self):
        self.out.release()
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()