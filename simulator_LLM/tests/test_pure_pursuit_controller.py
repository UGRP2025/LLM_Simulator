
import numpy as np
import pytest
from car_control.pure_pursuit_controller import PurePursuit, Pose2D

@pytest.fixture
def pp_controller():
    """Provides a PurePursuit controller instance with a simple straight path."""
    # A straight line along the x-axis from x=0 to x=100
    waypoints = np.array([[i, 0.0] for i in range(101)], dtype=float)
    # Basic parameters for testing
    params = {
        'wheelbase': 0.324,
        'lookahead_min': 1.0,
        'lookahead_max': 5.0,
        'lookahead_speed_gain': 0.5,
        'lookahead_curvature_gain': 0.1,
        'max_steering_angle_deg': 30.0
    }
    return PurePursuit(waypoints, params)

@pytest.fixture
def pp_controller_circle():
    """Provides a controller with a circular path."""
    radius = 10.0
    center = (0, radius)
    # Circle centered at (0, 10) with radius 10, starting from (0,0)
    angles = np.linspace(-np.pi / 2, 1.5 * np.pi, 200) # Full circle
    waypoints = np.array([
        [center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)] for a in angles
    ])
    params = {
        'wheelbase': 0.324,
        'lookahead_min': 2.0,
        'lookahead_max': 2.0, # Fixed lookahead for predictable test
        'lookahead_speed_gain': 0.0, # No speed influence
        'lookahead_curvature_gain': 0.0, # No curvature influence
        'max_steering_angle_deg': 30.0
    }
    return PurePursuit(waypoints, params)

# === Test Cases ===

def test_straight_path_on_track(pp_controller: PurePursuit):
    """Test steering when the vehicle is exactly on a straight path."""
    pose: Pose2D = (10.0, 0.0)
    yaw: float = 0.0  # Aligned with the path
    speed: float = 5.0
    
    steer = pp_controller.compute_steer(pose, yaw, speed)
    
    # On a perfect straight line, with zero yaw error, steering should be zero
    assert np.isclose(steer, 0.0)

def test_straight_path_parallel_offset(pp_controller: PurePursuit):
    """Test steering when the vehicle is parallel to a straight path but offset."""
    pose: Pose2D = (10.0, 1.0)  # 1 meter offset to the left
    yaw: float = 0.0  # Aligned with the path
    speed: float = 5.0
    
    steer = pp_controller.compute_steer(pose, yaw, speed)
    
    # Should steer to the right (negative steering) to correct the offset
    assert steer < 0.0

def test_straight_path_angle_offset(pp_controller: PurePursuit):
    """Test steering when the vehicle is on the path but at an angle."""
    pose: Pose2D = (10.0, 0.0)
    yaw: float = np.deg2rad(10)  # Pointing 10 degrees to the left
    speed: float = 5.0
    
    steer = pp_controller.compute_steer(pose, yaw, speed)
    
    # Should steer to the right (negative steering) to align with the path
    assert steer < 0.0

def test_circular_path_on_track(pp_controller_circle: PurePursuit):
    """Test steering for a circular path."""
    # Vehicle is at the start of the circle (0,0), heading along the x-axis
    pose: Pose2D = (0.0, 0.0)
    yaw: float = 0.0 # Heading tangent to the circle at the start point
    speed: float = 3.0
    
    steer = pp_controller_circle.compute_steer(pose, yaw, speed)
    
    # On a left-turning circle, steering should be positive (to the left)
    assert steer > 0.0

def test_end_of_path(pp_controller: PurePursuit):
    """Test behavior when the vehicle is near or past the end of the path."""
    # Position near the end of the 100m path, but slightly offset
    pose: Pose2D = (99.5, 0.1)
    yaw: float = 0.0
    speed: float = 5.0
    
    # The controller should still find a target and steer to correct the offset
    steer = pp_controller.compute_steer(pose, yaw, speed)
    assert steer < 0.0 # Steer right to correct left offset

    # Position past the end of the path
    pose_after_end: Pose2D = (102.0, 0.0)
    # Manually set last_best_idx to simulate being at the end
    pp_controller._last_best_idx = len(pp_controller.waypoints) - 1
    steer_after_end = pp_controller.compute_steer(pose_after_end, yaw, speed)
    
    # No target point should be found, so steering should be zero
    assert np.isclose(steer_after_end, 0.0)
