import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os


class CaveLineDetection(Node):
    def __init__(self):
        super().__init__('caveline_detection')

        model_path = self.declare_parameter(
            'model_path',
            '/home/xianyao/cavepi_ws/src/cavepi_detection/cavepi_detection/best.pt'
        ).value

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
            rclpy.shutdown()
            return

        self.model = YOLO(model_path)

        self.subscription = self.create_subscription(
            Image, 'downward_cam', self.image_callback, 10)
        self.pose_publisher = self.create_publisher(String, '/pose', 10)
        self.waypoints_publisher = self.create_publisher(Float32MultiArray, '/waypoints', 10)

        self.bridge = CvBridge()
        self.direction_history = []

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        frame_height, frame_width = frame.shape[:2]
        screen_center_x = frame_width // 2

        try:
            results = self.model.predict(frame, imgsz=[640, 480], conf=0.4)
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        rope_positions = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 10: 
                    x1, y1, x2, y2 = map(int, box[:4])
                    rope_positions.append((x1, y1, x2, y2))

        direction = "lost"
        offset_x = 0
        angle = None

        if not rope_positions:
            self.publish_data(direction, offset_x, angle)
            return

        x1, y1, x2, y2 = rope_positions[0]
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

        if lines is None or len(lines) < 2:
            self.publish_data(direction, offset_x, angle)
            return

        rope_center_x = 0
        points = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1_l = int(x0 + 1000 * (-b))
            y1_l = int(y0 + 1000 * (a))
            x2_l = int(x0 - 1000 * (-b))
            y2_l = int(y0 - 1000 * (a))
            points.append(((x1_l, y1_l), (x2_l, y2_l)))
            rope_center_x += (x1_l + x2_l) // 2

        rope_center_x = rope_center_x // len(lines)
        offset_x = rope_center_x - screen_center_x

        if len(points) >= 2:
            angle = self.calculate_angle(*points[0][0], *points[1][0])
            if self.is_turn(angle):
                direction = "turn"
            else:
                direction = "straight"
        else:
            direction = "straight"

        direction = self.smooth_direction(direction)

        self.publish_data(direction, offset_x, angle)

    def calculate_angle(self, x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    def is_turn(self, angle, angle_threshold=80):
        return (angle_threshold - 10) < abs(angle) < (angle_threshold + 10)

    def smooth_direction(self, new_direction, threshold=3):
        self.direction_history.append(new_direction)
        if len(self.direction_history) > threshold:
            self.direction_history.pop(0)
        return max(set(self.direction_history), key=self.direction_history.count)

    def publish_data(self, direction, offset_x, angle):
        pose_msg = String()
        pose_msg.data = direction
        self.pose_publisher.publish(pose_msg)

        waypoints_msg = Float32MultiArray()
        waypoints_msg.data = [float(offset_x), float(angle) if angle is not None else 0.0]
        self.waypoints_publisher.publish(waypoints_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CaveLineDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

