import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

class QRDetector(Node):
    def __init__(self):
        super().__init__('qr_detector')
        self.subscription = self.create_subscription(Image, 'front_cam', self.image_callback, 10)
        self.publisher = self.create_publisher(String, 'qr_info', 10)
        self.bridge = CvBridge()
        self.detector = cv2.QRCodeDetector()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        data, points, _ = self.detector.detectAndDecode(frame)
        if data:
            self.publisher.publish(String(data=data))

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

