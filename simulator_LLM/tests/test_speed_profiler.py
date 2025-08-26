import pytest
from car_control.speed_profiler import target_speed, speed_to_throttle, V_MAX

# Fixtures for test data
@pytest.fixture
def base_metrics():
    """Provides default lane metrics with zero curvature and long free distance."""
    return {'curvature': 0.0, 'free_distance': 100.0}

@pytest.fixture
def base_risks():
    """Provides default risks with no obstacles nearby."""
    return {'front': 100.0, 'left': 100.0, 'right': 100.0}

@pytest.fixture
def hint_slow():
    """Provides a VLM hint for 'slow' speed."""
    return {'speed': 'slow', 'confidence': 0.9}

@pytest.fixture
def hint_fast():
    """Provides a VLM hint for 'fast' speed."""
    return {'speed': 'fast', 'confidence': 0.9}

# --- Test Cases ---

def test_speed_to_throttle_scaling():
    """Tests the speed to throttle conversion."""
    assert speed_to_throttle(0) == 0.0
    assert speed_to_throttle(V_MAX / 2) == 0.5
    assert speed_to_throttle(V_MAX) == 1.0
    assert speed_to_throttle(V_MAX * 1.5) == 1.0 # Should cap at 1.0
    assert speed_to_throttle(-1.0) == 0.0      # Should not be negative

def test_no_risk_scenario(base_metrics, base_risks):
    """Test that speed is at maximum when there are no risks."""
    speed = target_speed(base_metrics, base_risks, None)
    assert pytest.approx(speed) == V_MAX

def test_high_curvature_reduces_speed(base_metrics, base_risks):
    """Test that increasing curvature decreases the target speed."""
    base_speed = target_speed(base_metrics, base_risks, None)
    
    high_curvature_metrics = base_metrics.copy()
    high_curvature_metrics['curvature'] = 0.5 # High curvature
    
    slow_speed = target_speed(high_curvature_metrics, base_risks, None)
    assert slow_speed < base_speed

def test_frontal_obstacle_reduces_speed(base_metrics, base_risks):
    """Test that a close frontal obstacle reduces the target speed."""
    base_speed = target_speed(base_metrics, base_risks, None)
    
    risky_risks = base_risks.copy()
    risky_risks['front'] = 2.0 # Obstacle 2m ahead
    
    slow_speed = target_speed(base_metrics, risky_risks, None)
    assert slow_speed < base_speed

def test_lane_obstacle_reduces_speed(base_metrics, base_risks):
    """Test that a short free distance in the lane reduces speed."""
    base_speed = target_speed(base_metrics, base_risks, None)
    
    risky_metrics = base_metrics.copy()
    risky_metrics['free_distance'] = 2.0 # Obstacle in lane 2m ahead

    slow_speed = target_speed(risky_metrics, base_risks, None)
    assert slow_speed < base_speed

def test_vlm_hint_slow_reduces_speed(base_metrics, base_risks, hint_slow):
    """Test that a 'slow' hint from VLM reduces speed."""
    base_speed = target_speed(base_metrics, base_risks, None)
    hint_speed = target_speed(base_metrics, base_risks, hint_slow)
    assert hint_speed < base_speed
    assert pytest.approx(hint_speed) == V_MAX * 0.7

def test_vlm_hint_fast_increases_speed(base_risks, hint_fast):
    """Test that a 'fast' hint from VLM increases speed."""
    # First, create a scenario where speed is limited by curvature
    # A curvature of 0.3 should result in a speed < V_MAX
    metrics = {'curvature': 0.3, 'free_distance': 100.0}
    base_speed = target_speed(metrics, base_risks, None)
    assert base_speed < V_MAX  # Ensure we are not already at max speed

    # Now apply the fast hint
    hint_speed = target_speed(metrics, base_risks, hint_fast)
    assert hint_speed > base_speed

    # The speed should increase by the hint factor, but not exceed V_MAX
    expected_speed = min(base_speed * 1.2, V_MAX)
    assert hint_speed == pytest.approx(expected_speed)

def test_vlm_hint_fast_is_capped(base_metrics, base_risks, hint_fast):
    """Test that 'fast' hint doesn't exceed V_MAX."""
    # In a no-risk scenario, base speed is already V_MAX
    base_speed = target_speed(base_metrics, base_risks, None)
    hint_speed = target_speed(base_metrics, base_risks, hint_fast)
    assert hint_speed == base_speed
    assert hint_speed <= V_MAX
