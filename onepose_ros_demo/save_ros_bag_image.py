#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import os
import numpy as np

class ImageSaver(Node):
    def __init__(self, topic_name, save_dir="./saved_images"):
        super().__init__('image_saver_node')
        self.subscription = self.create_subscription(
            Image, 
            topic_name, 
            self.image_callback, 
            10
        )
        self.save_dir = save_dir
        self.image_count = 0

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def ros_image_to_cv2(self, msg: Image):
        """
        Convert sensor_msgs/Image to OpenCV image without cv_bridge.
        Supports: bgr8, rgb8, mono8.
        """
        if msg.encoding not in ("bgr8", "rgb8", "mono8"):
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")

        if msg.encoding == "mono8":
            expected_step = msg.width
            if msg.step < expected_step:
                raise ValueError(f"Invalid step for mono8: {msg.step} < {expected_step}")
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
            return image[:, :msg.width]

        # bgr8 / rgb8
        channels = 3
        expected_step = msg.width * channels
        if msg.step < expected_step:
            raise ValueError(f"Invalid step for {msg.encoding}: {msg.step} < {expected_step}")

        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        image = image[:, :expected_step].reshape((msg.height, msg.width, channels))

        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def image_callback(self, msg):
        try:
            # 将 ROS Image 消息转换为 OpenCV 格式（不依赖 cv_bridge）
            cv_image = self.ros_image_to_cv2(msg)

            # 生成文件名
            filename = os.path.join(self.save_dir, f"frame_{self.image_count:04d}.png")

            # 保存图片
            cv2.imwrite(filename, cv_image)
            self.get_logger().info(f"Saved: {filename}")

            # 增加计数
            self.image_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

def main(args=None):
    rclpy.init(args=args)

    # 指定要监听的图像主题
    topic_name = "/camera/color/image_raw"  # 修改为你的实际主题名

    image_saver = ImageSaver(topic_name, save_dir="./camera_images")

    rclpy.spin(image_saver)

    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()