import pytest
from car_control.cost_fusion import select_lane, DEFAULT_PARAMS

@pytest.fixture
def default_params():
    """Returns a copy of the default parameters for modification in tests."""
    # Use deepcopy if params were nested dicts, but for this structure, copy is fine.
    return DEFAULT_PARAMS.copy()

@pytest.fixture
def base_metrics():
    """Provides a base set of metrics for three lanes."""
    return {
        'center': {'free_distance': 20.0, 'curvature': 0.1, 'progress': 10.0},
        'inner': {'free_distance': 20.0, 'curvature': 0.1, 'progress': 10.0},
        'outer': {'free_distance': 20.0, 'curvature': 0.1, 'progress': 10.0},
    }

def test_select_default_on_no_lanes(default_params):
    """Should return 'center' as a fallback if no lanes are available."""
    best_lane = select_lane({}, set(), None, None, default_params)
    assert best_lane == 'center'

def test_free_distance_influence(base_metrics, default_params):
    """The lane with more free distance should be preferred."""
    base_metrics['inner']['free_distance'] = 30.0 # More free distance
    best_lane = select_lane(base_metrics, {'center', 'inner'}, None, None, default_params)
    assert best_lane == 'inner'

def test_curvature_influence(base_metrics, default_params):
    """The lane with less curvature should be preferred."""
    base_metrics['inner']['curvature'] = 0.05 # Less curvy
    best_lane = select_lane(base_metrics, {'center', 'inner'}, None, None, default_params)
    assert best_lane == 'inner'

def test_progress_influence(base_metrics, default_params):
    """The lane with more progress should be preferred."""
    base_metrics['inner']['progress'] = 12.0 # More progress
    best_lane = select_lane(base_metrics, {'center', 'inner'}, None, None, default_params)
    assert best_lane == 'inner'

def test_vlm_hint_influence(base_metrics, default_params):
    """A VLM hint should be able to sway the decision."""
    # Make 'center' slightly worse than 'inner'
    base_metrics['center']['curvature'] = 0.11
    # Without hint, 'inner' would be chosen
    assert select_lane(base_metrics, {'center', 'inner'}, None, None, default_params) == 'inner'
    
    # With a hint for 'center', it should be chosen despite the slightly higher curvature
    hint = {'lane': 'center'}
    best_lane = select_lane(base_metrics, {'center', 'inner'}, hint, None, default_params)
    assert best_lane == 'center'

def test_lane_change_penalty_influence(base_metrics, default_params):
    """The lane change penalty should prevent flickering."""
    # Make 'inner' marginally better than 'center'
    base_metrics['inner']['curvature'] = 0.099
    # Without penalty, 'inner' would be chosen
    assert select_lane(base_metrics, {'center', 'inner'}, None, None, default_params) == 'inner'

    # With 'center' as the previous lane, the penalty on 'inner' should make 'center' win
    best_lane = select_lane(base_metrics, {'center', 'inner'}, None, 'center', default_params)
    assert best_lane == 'center'

def test_extreme_weights_nullify_effect(base_metrics, default_params):
    """Setting a weight to zero should nullify that component's effect."""
    # Make 'inner' have much less free distance, it should lose
    base_metrics['inner']['free_distance'] = 5.0
    assert select_lane(base_metrics, {'center', 'inner'}, None, None, default_params) == 'center'

    # Now, set the weight for free distance ('alpha') to 0
    default_params['weights']['alpha'] = 0.0
    # With distance ignored, the lanes are identical, so either can be chosen.
    # The min function will pick the first one in case of a tie, which is 'center'.
    # Let's make inner slightly better on another metric to be sure.
    base_metrics['inner']['curvature'] = 0.09
    best_lane = select_lane(base_metrics, {'center', 'inner'}, None, None, default_params)
    assert best_lane == 'inner'
