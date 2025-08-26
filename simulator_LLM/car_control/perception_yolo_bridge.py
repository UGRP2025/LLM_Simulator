import time
import random
from typing import List, Dict, Tuple, NamedTuple

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
    'image_width': 1280,
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
    'epsilon': 1e-6
}

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

    Args:
        detections: A list of Detection objects from the perception system.
        params: A dictionary of parameters for calculation.

    Returns:
        A dictionary with risk values for 'left', 'front', 'right' and a timestamp.
    """
    sector_risks = {'left': 0.0, 'front': 0.0, 'right': 0.0}

    # Unpack params
    w_d = params['weights']['distance']
    w_w = params['weights']['width']
    w_cls = params['weights']['class']
    class_weights = params['class_weights']
    epsilon = params['epsilon']
    image_width = params['image_width']
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

def get_latest_detections() -> List[Detection]:
    """
    Stub function to simulate getting dynamic and randomized detections from a YOLO topic.
    This provides more varied scenarios for testing.
    """
    detections = []
    # 80% chance of no detections
    if random.random() < 0.8:
        return []

    num_detections = random.randint(1, 3)
    possible_classes = ['car', 'cone', 'person']

    for _ in range(num_detections):
        class_name = random.choice(possible_classes)
        distance = random.uniform(5.0, 40.0)
        score = random.uniform(0.75, 0.98)
        
        # Randomize position and size
        image_width = DEFAULT_PARAMS['image_width']
        x_min = random.randint(0, image_width - 100)
        box_width = random.randint(80, 200)
        x_max = min(x_min + box_width, image_width)

        y_min = random.randint(200, 400)
        box_height = random.randint(100, 250)
        y_max = min(y_min + box_height, 600) # Assuming image height around 600

        detection = Detection(
            bbox=(x_min, y_min, x_max, y_max),
            class_name=class_name,
            score=score,
            distance=distance
        )
        detections.append(detection)
    
    return detections
