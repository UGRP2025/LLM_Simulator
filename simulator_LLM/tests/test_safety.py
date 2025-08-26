import pytest
from car_control.safety import mask_unsafe_lanes, apply_lane_change_penalty

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_metrics():
    """Provides a sample metrics dictionary for three lanes."""
    return {
        'center': {'free_distance': 20.0, 'curvature': 0.1},
        'inner': {'free_distance': 5.0, 'curvature': 0.4},
        'outer': {'free_distance': 1.5, 'curvature': 0.3}
    }

@pytest.fixture
def sample_costs():
    """Provides a sample cost dictionary for three lanes."""
    return {'center': 1.0, 'inner': 1.2, 'outer': 1.5}


# --- Tests for mask_unsafe_lanes ---

def test_mask_unsafe_lane(sample_metrics):
    """Test that a lane with free_distance below the threshold is masked."""
    thresholds = {'min_free_distance': 2.0}
    unsafe = mask_unsafe_lanes(sample_metrics, thresholds)
    assert unsafe == {'outer'}

def test_mask_multiple_unsafe_lanes(sample_metrics):
    """Test that multiple unsafe lanes are correctly identified."""
    thresholds = {'min_free_distance': 6.0}
    unsafe = mask_unsafe_lanes(sample_metrics, thresholds)
    assert unsafe == {'inner', 'outer'}

def test_no_unsafe_lanes(sample_metrics):
    """Test that no lanes are masked if all are above the threshold."""
    thresholds = {'min_free_distance': 1.0}
    unsafe = mask_unsafe_lanes(sample_metrics, thresholds)
    assert not unsafe  # Expect an empty set

def test_masking_with_no_threshold(sample_metrics):
    """Test that no lanes are masked if the threshold is not provided."""
    unsafe = mask_unsafe_lanes(sample_metrics, {})
    assert not unsafe

def test_masking_with_missing_metric():
    """Test that a lane missing the 'free_distance' metric is ignored."""
    metrics = {
        'center': {'curvature': 0.1}, # No free_distance
        'inner': {'free_distance': 1.0}
    }
    thresholds = {'min_free_distance': 2.0}
    unsafe = mask_unsafe_lanes(metrics, thresholds)
    assert unsafe == {'inner'}

# --- Tests for apply_lane_change_penalty ---

def test_penalty_applied_on_change(sample_costs):
    """Test that penalty is added to all lanes except the previous one."""
    previous_lane = 'center'
    penalty = 0.5
    penalized_costs = apply_lane_change_penalty(sample_costs, previous_lane, penalty)

    assert penalized_costs['center'] == sample_costs['center']
    assert penalized_costs['inner'] == sample_costs['inner'] + penalty
    assert penalized_costs['outer'] == sample_costs['outer'] + penalty

def test_no_penalty_if_no_previous_lane(sample_costs):
    """Test that no penalty is applied if previous_lane is None."""
    penalized_costs = apply_lane_change_penalty(sample_costs, None, 0.5)
    assert penalized_costs == sample_costs

def test_no_penalty_if_zero_penalty(sample_costs):
    """Test that no penalty is applied if penalty is 0."""
    penalized_costs = apply_lane_change_penalty(sample_costs, 'center', 0)
    assert penalized_costs == sample_costs

def test_penalty_function_is_pure(sample_costs):
    """Test that the original cost dictionary is not modified."""
    original_costs_copy = sample_costs.copy()
    apply_lane_change_penalty(sample_costs, 'center', 0.5)
    assert sample_costs == original_costs_copy
