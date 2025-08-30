from typing import Dict, Set, Optional

# --- Type Aliases ---
LaneName = str
# e.g., {'center': {'free_distance': 20.0, 'curvature': 0.1}, 'inner': ...}
MetricsByLane = Dict[LaneName, Dict[str, float]]
# e.g., {'min_free_distance': 2.5}
SafetyThresholds = Dict[str, float]
# e.g., {'center': 1.2, 'inner': 2.5, 'outer': 2.5}
CostDict = Dict[LaneName, float]


def mask_unsafe_lanes(
    metrics_by_lane: MetricsByLane,
    thresholds: SafetyThresholds
) -> Set[LaneName]:
    """
    Identifies and returns a set of lanes that are considered unsafe.

    Currently, safety is determined by checking if the free distance ahead
    in a lane is below a minimum threshold.

    TODO: Implement more sophisticated checks, such as:
      - Off-track detection (checking if the car's position is already
        outside the drivable area of a lane).
      - Collision with lane boundaries.

    Args:
        metrics_by_lane: A dictionary where keys are lane names and values are
                         dictionaries of their calculated metrics.
                         Expected metric: 'free_distance'.
        thresholds: A dictionary of safety thresholds.
                    Expected threshold: 'min_free_distance'.

    Returns:
        A set of lane names that are deemed unsafe.
    """
    unsafe_lanes: Set[LaneName] = set()
    min_dist = thresholds.get('min_free_distance')

    if min_dist is None:
        return unsafe_lanes  # No threshold, no-op

    for lane_name, metrics in metrics_by_lane.items():
        free_distance = metrics.get('free_distance')
        if free_distance is not None and free_distance < min_dist:
            unsafe_lanes.add(lane_name)

    return unsafe_lanes


def apply_lane_change_penalty(
    costs: CostDict,
    previous_lane: Optional[LaneName],
    penalty: float
) -> CostDict:
    """
    Applies a penalty to the cost of lanes that are not the previously
    selected lane. This helps prevent rapid, unnecessary lane changes (flickering).

    Args:
        costs: A dictionary of calculated costs for each candidate lane.
        previous_lane: The name of the lane that was selected in the
                       previous control cycle.
        penalty: The cost penalty to add for changing lanes.

    Returns:
        The updated dictionary of costs with the hysteresis penalty applied.
    """
    if previous_lane is None or penalty == 0:
        return costs

    # Create a new dictionary to avoid modifying the original in place
    penalized_costs = costs.copy()
    for lane_name in penalized_costs:
        if lane_name != previous_lane:
            penalized_costs[lane_name] += penalty

    return penalized_costs
