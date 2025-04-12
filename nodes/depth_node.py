#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from .depth_estimator import DepthEstimator
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import struct
import open3d as o3d

class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')

        self.publisher_ = self.create_publisher(PointCloud2, '/point_cloud', 10)
        
        # Parametreleri tanımla
        self.declare_parameter('use_gray_scale', False)
        self.declare_parameter('encoder_type', 'vitl')
        self.declare_parameter('input_source', 'webcam')
        self.use_gray_scale = self.get_parameter('use_gray_scale').value
        self.encoder_type = self.get_parameter('encoder_type').value
        self.input_source = self.get_parameter('input_source').value

        # Gerekli bileşenleri oluştur
        self.depth_estimator = DepthEstimator(encoder=self.encoder_type)
        self.bridge = CvBridge()
        self.get_logger().info(f"Depth Estimation Node started with encoder: {self.encoder_type}")
        if self.input_source == 'gazebo':
            self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
        elif self.input_source == 'webcam':
            self.webcam = cv2.VideoCapture(2)
            if not self.webcam.isOpened():
                self.get_logger().error("Webcam could not be opened!")
            else:
                self.create_timer(0.1, self.capture_webcam_frame)
    

    def convert_o3d_to_pointcloud2(self, pcd, frame_id="map"):
        """Open3D nokta bulutunu PointCloud2 mesajına çevirir."""
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)  # [0-255] arası uint8
        
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = points.shape[0]
        
        # Field'ları doğru hizalama ile tanımla (her alan 4 byte hizalanmalı)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16  # 12 (3xfloat) + 4 (RGB) = 16 byte
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = False  # NaN içermiyorsa True olabilir

        # Renkleri RGBA uint32 formatına paketle (8-bit per color, Alpha kullanma)
        packed_colors = [
            (r << 16) | (g << 8) | b
            for r, g, b in colors[:, :3]  # Alpha yoksa sadece RGB al
        ]
        
        # Nokta verisini oluştur
        buffer = bytearray()
        for pt, rgb in zip(points, packed_colors):
            buffer += struct.pack('3fI', pt[0], pt[1], pt[2], rgb)
        
        msg.data = bytes(buffer)
        return msg



    def capture_webcam_frame(self):
        """Webcam'den görüntü yakalar ve işler."""
        ret, frame = self.webcam.read()
        if ret:
            self.process_and_estimate_depth(frame)
        else:
            self.get_logger().error("Failed to capture frame from webcam")

    def image_callback(self, msg):
        """ROS 2 görüntü mesajını işler."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.process_and_estimate_depth(cv_image)
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
    
    def process_and_estimate_depth(self, image):
        """Derinlik tahmini yapar ve işlenmiş veriyi hazırlar."""
        pcd = self.depth_estimator.process_image(image)
        pointcloud_msg = self.convert_o3d_to_pointcloud2(pcd)
        self.publisher_.publish(pointcloud_msg)


    def convert_to_gray_scale(self, image):
        """Renkli görüntüyü gri tonlamaya çevirir."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def destroy_node(self):
        if self.input_source == 'webcam' and hasattr(self, 'webcam'):
            self.webcam.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
