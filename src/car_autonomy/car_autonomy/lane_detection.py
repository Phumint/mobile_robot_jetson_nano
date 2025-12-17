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
        self.declare_parameter('output_path', '/workspace/lane_debug_output.avi')
        self.declare_parameter('t_section_turn', 'left')

        self.record = self.get_parameter('record').value
        self.output_path = self.get_parameter('output_path').value
        self.t_section_turn = self.get_parameter('t_section_turn').value.lower()

        # --- T-section state ---
        self.t_section_active = False
        self.t_section_frames = 0
        self.T_SECTION_TURN_FRAMES = 50

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

        self.timer = self.create_timer(0.05, self.timer_callback)

    # ===================== CAMERA THREAD =====================
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = cv2.resize(frame, (640, 480))
            else:
                time.sleep(0.01)

    # ===================== LANE DETECTION =====================
    def detect_lane_frame(self, frame):
        height, width = frame.shape[:2]
        debug = frame.copy()

        # ---------- ROI ----------
        bl = (199, 479)
        tl = (255, 359)
        tr = (425, 363)
        br = (495, 479)
        roi_pts = np.array([[bl, tl, tr, br]], dtype=np.int32)

        cv2.polylines(debug, roi_pts, True, (0, 255, 255), 2)

        # ---------- Perspective transform ----------
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([
            [0, 0],
            [0, height],
            [width, 0],
            [width, height]
        ])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, M, (width, height))

        # ---------- Mask ----------
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_white = np.array([79, 0, 183])
        upper_white = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # ---------- Sliding window ----------
        lx, rx, window_vis = self.sliding_window_lane(mask)

        lanes_missing = (len(lx) < 100 and len(rx) < 100)

        # ---------- State logic ----------
        state_text = "LANE FOLLOWING"

        if lanes_missing and not self.t_section_active:
            self.t_section_active = True
            self.t_section_frames = 0

        if self.t_section_active:
            self.t_section_frames += 1
            state_text = "T-SECTION"
            heading = np.pi / 4 if self.t_section_turn == 'right' else -np.pi / 4
            if self.t_section_frames > self.T_SECTION_TURN_FRAMES:
                self.t_section_active = False
            offset = 0.0
        else:
            if len(lx) < 20 or len(rx) < 20:
                state_text = "NO LANE"
                offset, heading = 0.0, 0.0
            else:
                lane_center = (np.mean(lx) + np.mean(rx)) / 2
                offset = (lane_center - width / 2) / (width / 2)
                heading = np.arctan((np.mean(lx) - np.mean(rx)) / width)

        # ---------- Draw steering arrow ----------
        arrow_x = int(width // 2 + np.tan(heading) * 120)
        cv2.arrowedLine(
            debug,
            (width // 2, height - 20),
            (arrow_x, height - 140),
            (0, 0, 255),
            3
        )

        # ---------- State text ----------
        cv2.putText(
            debug,
            f"STATE: {state_text}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # ---------- PiP overlays ----------
        def pip(img, x, y, w=160, h=120, label=""):
            img = cv2.resize(img, (w, h))
            debug[y:y+h, x:x+w] = img
            cv2.rectangle(debug, (x, y), (x+w, y+h), (255, 255, 255), 1)
            cv2.putText(debug, label, (x+5, y+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        pip(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0, 0, label="MASK")
        pip(warped, 160, 0, label="BIRD VIEW")
        pip(window_vis, 320, 0, label="SLIDING WINDOW")

        return float(offset), float(heading), debug

    # ===================== SLIDING WINDOW =====================
    def sliding_window_lane(self, mask):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        histogram = np.sum(mask[mask.shape[0]//2:], axis=0)
        midpoint = histogram.shape[0] // 2

        leftx = np.argmax(histogram[:midpoint])
        rightx = np.argmax(histogram[midpoint:]) + midpoint

        cv2.circle(vis, (leftx, mask.shape[0] - 50), 5, (0,255,0), -1)
        cv2.circle(vis, (rightx, mask.shape[0] - 50), 5, (255,0,0), -1)

        return np.array([leftx]), np.array([rightx]), vis

    # ===================== TIMER =====================
    def timer_callback(self):
        with self.lock:
            frame = self.latest_frame

        if frame is None:
            return

        offset, heading, visual = self.detect_lane_frame(frame)

        self.pub_offset.publish(Float32(data=offset))
        self.pub_heading.publish(Float32(data=heading))

        if self.record and self.out:
            self.out.write(visual)

    # ===================== CLEANUP =====================
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
