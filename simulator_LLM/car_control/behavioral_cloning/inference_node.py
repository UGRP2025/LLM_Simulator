#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
from torchvision import transforms
from PIL import Image as PILImage
import os

from model import create_model
from dataset import ControlBins

class InferenceNode(Node):
    def __init__(self):
        super().__init__('behavioral_cloning_driver')

        # Parameters
        self.declare_parameter('model_path', 'behavioral_cloning_model.pth')
        self.declare_parameter('camera_topic', '/autodrive/f1tenth_1/front_camera')
        self.declare_parameter('steer_topic', '/autodrive/f1tenth_1/steering_command')
        self.declare_parameter('throttle_topic', '/autodrive/f1tenth_1/throttle_command')

        # Model and device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control_bins = ControlBins()
        self.model = create_model(num_outputs=self.control_bins.num_bins)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model path not found: {model_path}")
            self.get_logger().error("Please train the model first using train.py")
            rclpy.shutdown()
            return

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f"Model loaded from {model_path} and moved to {self.device}")

        # Image transformations (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ROS components
        self.bridge = CvBridge()
        self.create_subscription(Image, self.get_parameter('camera_topic').get_parameter_value().string_value, self.image_callback, 1)
        self.steer_pub = self.create_publisher(Float32, self.get_parameter('steer_topic').get_parameter_value().string_value, 10)
        self.throttle_pub = self.create_publisher(Float32, self.get_parameter('throttle_topic').get_parameter_value().string_value, 10)

        self.get_logger().info("Inference node started. Listening for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_image = PILImage.fromarray(cv_image[:, :, ::-1]) # Convert BGR to RGB

            # Preprocess image
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                output = self.model(input_batch)
                _, predicted_bin = torch.max(output, 1)
            
            # Convert bin to control values
            steer, throttle = self.control_bins.from_bin(predicted_bin.item())

            # Publish commands
            self.steer_pub.publish(Float32(data=float(steer)))
            self.throttle_pub.publish(Float32(data=float(throttle)))

            self.get_logger().info(f"Predicted bin: {predicted_bin.item()} -> Steer: {steer:.2f}, Throttle: {throttle:.2f}", throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
