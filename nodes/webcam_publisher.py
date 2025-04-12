#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')

        # Declare ROS parameters with defaults
        self.declare_parameter('cam_id', 0)
        self.declare_parameter('resolution_width', 320)
        self.declare_parameter('resolution_height', 240)
        self.declare_parameter('frame_rate', 30.0)

        # Load parameters
        self.cam_id = self.get_parameter('cam_id').get_parameter_value().integer_value
        self.width = self.get_parameter('resolution_width').get_parameter_value().integer_value
        self.height = self.get_parameter('resolution_height').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)

        # Initialize camera
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera with ID {self.cam_id}")
            raise RuntimeError(f"Could not open camera with ID {self.cam_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Timer for publishing frames
        self.timer = self.create_timer(1.0 / self.frame_rate, self.publish_frame)

        self.get_logger().info(
            f"WebcamPublisher started with camera ID {self.cam_id} "
            f"({self.width}x{self.height} @ {self.frame_rate} FPS). Publishing to /camera/image_raw"
        )

    def publish_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.get_logger().warn("Camera is not opened.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame from webcam.")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down WebcamPublisher...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
