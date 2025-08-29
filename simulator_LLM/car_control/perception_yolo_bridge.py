import time
import random
from typing import List, Dict, Tuple, NamedTuple
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# --- Type Definitions ---
# (x_min, y_min, x_max, y_max)
BoundingBox = Tuple[int, int, int, int]

class Detection(NamedTuple):
    bbox: BoundingBox
    class_name: str
    score: float
    distance: float  # Estimated distance in meters

RiskDict = Dict[str, float]

# --- Default Parameters (consider moving to params.yaml) ---
DEFAULT_PARAMS = {
    'image_width': 1280, # Default, will be updated from image message
    'sector_divisions': [0.33, 0.66],  # Proportions for left/front/right
    'weights': {
        'distance': 2.0,  # w_d
        'width': 1.0,     # w_w
        'class': 0.5,     # w_cls
    },
    'class_weights': {
        'car': 1.0,
        'cone': 0.6,
        'person': 1.2,
        'default': 0.5
    },
    'epsilon': 1e-6,
    'distance_estimation': {
        'heuristic_factor': 950.0 # Calibrate this factor: distance = factor / bbox_height_px
    }
}

class YOLOPerception:
    """
    Handles real-time object detection using YOLOv8, integrated into a ROS2 node.
    """
    def __init__(self, parent_node: Node, image_topic: str = "/autodrive/f1tenth_1/front_camera"):
        self.parent_node = parent_node
        self.logger = self.parent_node.get_logger()
        try:
            self.model = YOLO('yolov8n.pt')
            self.logger.info("YOLOv8n model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

        self.bridge = CvBridge()
        self.subscription = self.parent_node.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10)
        
        self.latest_detections: List[Detection] = []
        self.image_width = DEFAULT_PARAMS['image_width'] # Default, updated on first message
        self.lock = threading.Lock()
        self.distance_heuristic_factor = DEFAULT_PARAMS['distance_estimation']['heuristic_factor']

    def image_callback(self, msg: Image):
        """Processes incoming ROS Image messages."""
        if not self.model:
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_width = msg.width
        except Exception as e:
            self.logger.error(f"CvBridge Error: {e}")
            return

        # Run YOLO inference
        results = self.model(cv_image, verbose=False)
        
        new_detections = []
        if results and results[0]:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    # Extract data from YOLO result
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    score = float(box.conf[0].cpu().numpy())

                    # --- Distance Estimation Heuristic ---
                    # This is a simple heuristic and not a precise measurement.
                    # It assumes distance is inversely proportional to the bounding box height.
                    # The 'heuristic_factor' needs calibration for a specific camera setup.
                    bbox_height = xyxy[3] - xyxy[1]
                    if bbox_height > 0:
                        distance = self.distance_heuristic_factor / bbox_height
                    else:
                        distance = 100.0 # Assign a large distance if height is zero

                    detection = Detection(
                        bbox=tuple(xyxy),
                        class_name=class_name,
                        score=score,
                        distance=distance
                    )
                    new_detections.append(detection)
                except Exception as e:
                    self.logger.warn(f"Error processing a detection: {e}")

        with self.lock:
            self.latest_detections = new_detections

    def get_latest_detections(self) -> List[Detection]:
        """Thread-safe method to get the latest list of detections."""
        with self.lock:
            return self.latest_detections.copy()


def _get_sector(bbox_center_x: float, image_width: int, sector_divisions: List[float]) -> str:
    """Determines which sector the detection falls into based on its x-coordinate."""
    left_boundary = image_width * sector_divisions[0]
    right_boundary = image_width * sector_divisions[1]
    if bbox_center_x < left_boundary:
        return 'left'
    elif bbox_center_x > right_boundary:
        return 'right'
    else:
        return 'front'

def get_sector_risks(
    detections: List[Detection],
    params: Dict = DEFAULT_PARAMS
) -> RiskDict:
    """
    Calculates risk values for left, front, and right sectors based on YOLO detections.
    The risk for each sector is the maximum risk posed by any single object in it.
    """
    sector_risks = {'left': 0.0, 'front': 0.0, 'right': 0.0}

    # Unpack params
    w_d = params['weights']['distance']
    w_w = params['weights']['width']
    w_cls = params['weights']['class']
    class_weights = params['class_weights']
    epsilon = params['epsilon']
    # Use the actual image width from the perception module if available
    image_width = params.get('image_width', DEFAULT_PARAMS['image_width'])
    sector_divisions = params['sector_divisions']

    for det in detections:
        # 1. Calculate features of the detection
        bbox_width = det.bbox[2] - det.bbox[0]
        width_ratio = bbox_width / image_width
        distance = det.distance
        class_weight_val = class_weights.get(det.class_name, class_weights['default'])

        # 2. Calculate the individual risk score for this detection
        risk_score = (
            (w_d / (distance + epsilon)) +
            (w_w * width_ratio) +
            (w_cls * class_weight_val)
        )

        # 3. Assign risk to the correct sector
        bbox_center_x = (det.bbox[0] + det.bbox[2]) / 2
        sector = _get_sector(bbox_center_x, image_width, sector_divisions)

        # 4. Update sector risk (we care about the most dangerous object)
        sector_risks[sector] = max(sector_risks[sector], risk_score)

    sector_risks['timestamp'] = time.time()
    return sector_risks
