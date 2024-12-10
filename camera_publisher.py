import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.front_camera = cv2.VideoCapture(0)  
        self.downward_camera = cv2.VideoCapture(2) 
        self.bridge = CvBridge()

        if not self.front_camera.isOpened() or not self.downward_camera.isOpened():
            self.get_logger().error("Can Not Open Camera！")
            rclpy.shutdown()

        self.front_publisher = self.create_publisher(Image, 'front_cam', 10)
        self.downward_publisher = self.create_publisher(Image, 'downward_cam', 10)

        self.timer = self.create_timer(0.1, self.publish_images)

    def publish_images(self):
        ret_front, frame_front = self.front_camera.read()
        ret_downward, frame_downward = self.downward_camera.read()

        if ret_front:
            front_msg = self.bridge.cv2_to_imgmsg(frame_front, encoding='bgr8')
            self.front_publisher.publish(front_msg)

        if ret_downward:
            downward_msg = self.bridge.cv2_to_imgmsg(frame_downward, encoding='bgr8')
            self.downward_publisher.publish(downward_msg)

    def destroy_node(self):
        self.front_camera.release()
        self.downward_camera.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

