import pytest
import json
import io
from contextlib import redirect_stdout
import rclpy

# We import the class directly, not as a ROS node
from car_control.behavior_planner_node import BehaviorPlanner

@pytest.fixture
def planner_offline():
    """Provides an instance of the BehaviorPlanner in offline mode."""
    # rclpy.init() must be called before a Node can be created
    if not rclpy.ok():
        rclpy.init()
    
    planner = BehaviorPlanner(offline_mode=True)
    # Yield the planner to the test
    yield planner
    # Teardown: ensure background threads are stopped after the test
    planner.stop_all()
    if rclpy.ok():
        rclpy.shutdown()

def test_offline_control_loop_produces_output(planner_offline):
    """
    Tests that a single run of the control loop in offline mode
    produces a valid JSON output to stdout.
    """
    # Set a valid initial state, otherwise the loop will just return
    planner_offline.pose = (10.0, 5.0)
    planner_offline.yaw = 0.5

    # The offline loop prints JSON to stdout. We capture it here.
    f = io.StringIO()
    with redirect_stdout(f):
        planner_offline.control_loop()
    
    output = f.getvalue()

    # 1. Check that we got some output
    assert output.strip(), "The control loop should produce a log line to stdout"

    # 2. Check that the output is valid JSON
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail(f"Output was not valid JSON: {output}")

    # 3. Check that the JSON log contains the expected keys
    expected_keys = ['timestamp', 'pose', 'yaw', 'chosen_lane', 'steer', 'throttle']
    for key in expected_keys:
        assert key in data, f"Output JSON is missing expected key: '{key}'"
    
    # 4. Check data types for some important fields
    assert isinstance(data['chosen_lane'], str)
    assert isinstance(data['steer'], float)
    assert isinstance(data['throttle'], float)
    assert data['chosen_lane'] in ['center', 'inner', 'outer']
