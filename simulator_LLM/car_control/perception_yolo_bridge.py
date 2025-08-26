import time
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
    Stub function to simulate getting the latest detections from a YOLO topic.
    In a real implementation, this would subscribe to a ROS topic.
    """
    # Mock data for testing and demonstration
    return [
        Detection(bbox=(100, 200, 250, 400), class_name='cone', score=0.9, distance=10.0),
        Detection(bbox=(500, 250, 700, 500), class_name='car', score=0.85, distance=8.0),
    ]