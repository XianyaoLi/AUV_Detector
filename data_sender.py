import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import socket
import json


class DataSender(Node):
    def __init__(self):
        super().__init__('data_sender')

        self.udp_ip = "192.168.3.2"  
        self.udp_port = 5005       
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.qr_subscription = self.create_subscription(String, '/qr_info', self.qr_callback, 10)
        self.pose_subscription = self.create_subscription(String, '/pose', self.pose_callback, 10)
        self.waypoints_subscription = self.create_subscription(Float32MultiArray, '/waypoints', self.waypoints_callback, 10)

        self.qr_data = None
        self.pose_data = None
        self.waypoints_data = None

    def qr_callback(self, msg):
        self.qr_data = msg.data
        self.send_udp_data('qr_info', {'qr_data': self.qr_data})

    def pose_callback(self, msg):
        self.pose_data = msg.data
        self.send_udp_data('pose', {'direction': self.pose_data})

    def waypoints_callback(self, msg):
        self.waypoints_data = {'x': msg.data[0], 'y': msg.data[1]}
        self.send_udp_data('waypoints', self.waypoints_data)

    def send_udp_data(self, data_type, data):
        try:
            message = {
                'type': data_type,
                'data': data
            }
            serialized_message = json.dumps(message).encode('utf-8')
            self.sock.sendto(serialized_message, (self.udp_ip, self.udp_port))
            self.get_logger().info(f"Sent {data_type} data: {message}")
        except Exception as e:
            self.get_logger().error(f"Failed to send {data_type} data: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

