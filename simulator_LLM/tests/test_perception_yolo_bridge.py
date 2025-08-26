import pytest
from car_control.perception_yolo_bridge import get_sector_risks, Detection, DEFAULT_PARAMS

@pytest.fixture
def default_params():
    """Returns a copy of the default parameters for modification in tests."""
    return DEFAULT_PARAMS.copy()

def test_empty_detections(default_params):
    """Test that with no detections, all sector risks are zero."""
    risks = get_sector_risks([], params=default_params)
    assert risks['left'] == 0.0
    assert risks['front'] == 0.0
    assert risks['right'] == 0.0
    assert 'timestamp' in risks

@pytest.mark.parametrize("bbox, expected_sector", [
    # Bbox, expected sector
    ((100, 100, 200, 200), 'left'),   # Center at 150, should be left
    ((550, 100, 650, 200), 'front'),  # Center at 600, should be front
    ((1100, 100, 1200, 200), 'right'), # Center at 1150, should be right
])
def test_detection_sector_assignment(bbox, expected_sector, default_params):
    """Test that detections are correctly assigned to sectors."""
    det = Detection(bbox=bbox, class_name='cone', score=0.9, distance=10.0)
    risks = get_sector_risks([det], params=default_params)
    
    # Check that the risk is assigned to the correct sector
    for sector in ['left', 'front', 'right']:
        if sector == expected_sector:
            assert risks[sector] > 0.0
        else:
            assert risks[sector] == 0.0

def test_risk_increases_with_proximity(default_params):
    """Risk should increase as an object gets closer."""
    bbox = (550, 100, 650, 200) # Front sector
    det_far = Detection(bbox=bbox, class_name='car', score=0.9, distance=20.0)
    det_close = Detection(bbox=bbox, class_name='car', score=0.9, distance=5.0)

    risk_far = get_sector_risks([det_far], params=default_params)['front']
    risk_close = get_sector_risks([det_close], params=default_params)['front']

    assert risk_close > risk_far

def test_risk_increases_with_width(default_params):
    """Risk should increase as an object appears wider."""
    det_narrow = Detection(bbox=(600, 100, 650, 200), class_name='car', score=0.9, distance=10.0)
    det_wide = Detection(bbox=(550, 100, 700, 200), class_name='car', score=0.9, distance=10.0)

    risk_narrow = get_sector_risks([det_narrow], params=default_params)['front']
    risk_wide = get_sector_risks([det_wide], params=default_params)['front']

    assert risk_wide > risk_narrow

def test_class_weight_influence(default_params):
    """A higher-weight class (person) should yield higher risk than a lower-weight one (cone)."""
    bbox = (550, 100, 650, 200) # Front sector
    det_cone = Detection(bbox=bbox, class_name='cone', score=0.9, distance=10.0)
    det_person = Detection(bbox=bbox, class_name='person', score=0.9, distance=10.0)

    risk_cone = get_sector_risks([det_cone], params=default_params)['front']
    risk_person = get_sector_risks([det_person], params=default_params)['front']

    assert risk_person > risk_cone

def test_multiple_detections_max_risk(default_params):
    """The highest risk in a sector should be chosen, not the sum."""
    bbox = (550, 100, 650, 200) # Front sector
    det_low_risk = Detection(bbox=bbox, class_name='cone', score=0.9, distance=20.0)
    det_high_risk = Detection(bbox=bbox, class_name='car', score=0.9, distance=5.0)

    risk_low = get_sector_risks([det_low_risk], params=default_params)['front']
    risk_high = get_sector_risks([det_high_risk], params=default_params)['front']
    
    # Calculate risk with both detections present
    combined_risk = get_sector_risks([det_low_risk, det_high_risk], params=default_params)['front']

    assert risk_high > risk_low
    assert combined_risk == pytest.approx(risk_high)
