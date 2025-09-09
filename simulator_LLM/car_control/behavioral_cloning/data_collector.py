#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
import cv2
import os
import json
from datetime import datetime
import threading

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        # Parameters
        self.declare_parameter('data_path', '~/f1tenth_data')
        self.declare_parameter('camera_topic', '/autodrive/f1tenth_1/front_camera')
        self.declare_parameter('steer_topic', '/autodrive/f1tenth_1/steering_command')
        self.declare_parameter('throttle_topic', '/autodrive/f1tenth_1/throttle_command')

        # State
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_steer = 0.0
        self.latest_throttle = 0.0
        self.is_recording = False
        self.lock = threading.Lock()

        # Data path
        self.data_path = os.path.expanduser(self.get_parameter('data_path').get_parameter_value().string_value)
        self.images_path = os.path.join(self.data_path, 'images')
        self.json_path = os.path.join(self.data_path, 'json')
        
        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)

        # Subscriptions
        self.create_subscription(Image, self.get_parameter('camera_topic').get_parameter_value().string_value, self.image_callback, 10)
        self.create_subscription(Float32, self.get_parameter('steer_topic').get_parameter_value().string_value, self.steer_callback, 10)
        self.create_subscription(Float32, self.get_parameter('throttle_topic').get_parameter_value().string_value, self.throttle_callback, 10)

        # Service to toggle recording
        self.toggle_recording_service = self.create_service(SetBool, 'toggle_recording', self.toggle_recording_callback)

        # Timer for saving data
        self.save_timer = self.create_timer(0.1, self.save_data) # 10 Hz

        self.get_logger().info(f"Data collector node started. Saving data to: {self.data_path}")
        self.get_logger().info("Use 'ros2 service call /toggle_recording std_srvs/srv/SetBool "{data: true}"' to start recording.")
        self.get_logger().info("Use 'ros2 service call /toggle_recording std_srvs/srv/SetBool "{data: false}"' to stop recording.")


    def image_callback(self, msg):
        with self.lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def steer_callback(self, msg):
        with self.lock:
            self.latest_steer = msg.data

    def throttle_callback(self, msg):
        with self.lock:
            self.latest_throttle = msg.data

    def toggle_recording_callback(self, request, response):
        self.is_recording = request.data
        response.success = True
        if self.is_recording:
            response.message = "Started recording."
            self.get_logger().info("Started recording.")
        else:
            response.message = "Stopped recording."
            self.get_logger().info("Stopped recording.")
        return response

    def save_data(self):
        if not self.is_recording:
            return

        with self.lock:
            if self.latest_image is None:
                return
            
            # Copy data to avoid race conditions
            image = self.latest_image.copy()
            steer = self.latest_steer
            throttle = self.latest_throttle

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Save image
        image_filename = os.path.join(self.images_path, f"{timestamp}.png")
        cv2.imwrite(image_filename, image)

        # Save control data
        json_filename = os.path.join(self.json_path, f"{timestamp}.json")
        data = {
            'steering': steer,
            'throttle': throttle,
            'image_filename': os.path.basename(image_filename)
        }
        with open(json_filename, 'w') as f:
            json.dump(data, f)

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
