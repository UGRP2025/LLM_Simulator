import numpy as np
import pandas as pd
import pytest
import os
from car_control.lane_loader import load_three_lanes, Lane

@pytest.fixture
def dummy_csv_files(tmpdir):
    """Create dummy CSV files for testing."""
    center_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [0, 0, 0, 0]})
    inner_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [1, 1, 1, 1]})
    outer_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [-1, -1, -1, -1]})

    center_path = tmpdir.join("center.csv")
    inner_path = tmpdir.join("inner.csv")
    outer_path = tmpdir.join("outer.csv")

    center_df.to_csv(center_path, index=False)
    inner_df.to_csv(inner_path, index=False)
    outer_df.to_csv(outer_path, index=False)

    return str(center_path), str(inner_path), str(outer_path)

def test_load_three_lanes(dummy_csv_files):
    """Test loading of three lanes."""
    center_path, inner_path, outer_path = dummy_csv_files
    lanes = load_three_lanes(center_path, inner_path, outer_path)

    assert isinstance(lanes.center, Lane)
    assert isinstance(lanes.inner, Lane)
    assert isinstance(lanes.outer, Lane)

    # Test resampling
    for lane in [lanes.center, lanes.inner, lanes.outer]:
        distances = np.sqrt(np.sum(np.diff(lane.waypoints, axis=0)**2, axis=1))
        assert np.allclose(distances, lane.meta['ds'], atol=0.1)

    # Test curvature (straight line should have zero curvature)
    assert np.allclose(lanes.center.meta['curvature'], 0, atol=1e-9)

    # Test path length
    assert np.isclose(lanes.center.meta['s'][-1], 3, atol=0.1)
