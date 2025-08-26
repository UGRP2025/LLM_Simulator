from typing import Dict, Optional, Set
import numpy as np

from car_control.safety import apply_lane_change_penalty

# --- Type Aliases ---
LaneName = str
MetricsByLane = Dict[LaneName, Dict[str, float]]
HintDict = Dict[str, any]
ParamsDict = Dict[str, float]
CostDict = Dict[LaneName, float]

DEFAULT_PARAMS = {
    'weights': {
        'alpha': 2.0,   # free_distance
        'beta': 1.0,    # curvature
        'gamma': 1.0,   # progress
        'delta': 0.5,   # lane_change_penalty
        'w_lane': 0.5,  # vlm_lane_hint
    },
    'epsilon': 1e-6
}

def _calculate_vlm_hint_cost(hint: Optional[HintDict], lane_name: LaneName, w_lane: float) -> float:
    """
    Calculates the cost reduction (bonus) based on the VLM lane hint.
    A hint for a specific lane reduces its cost.
    """
    if hint and hint.get('lane') == lane_name:
        # This is a cost reduction, so we return a negative value
        return -w_lane
    return 0.0

def select_lane(
    metrics_by_lane: MetricsByLane,
    available_lanes: Set[LaneName],
    hint: Optional[HintDict],
    previous_lane: Optional[LaneName],
    params: Dict = DEFAULT_PARAMS
) -> LaneName:
    """
    Selects the best lane by calculating a normalized cost for each component.
    This prevents any single metric from dominating the decision due to scale.
    """
    if not available_lanes:
        return 'center'

    weights = params['weights']
    epsilon = params['epsilon']
    raw_costs = {r: {'dist': 0.0, 'curve': 0.0, 'prog': 0.0} for r in available_lanes}

    # 1. Calculate raw values for each cost component
    for r in available_lanes:
        metrics = metrics_by_lane.get(r, {})
        raw_costs[r]['dist'] = 1 / (metrics.get('free_distance', epsilon) + epsilon)
        raw_costs[r]['curve'] = abs(metrics.get('curvature', 0.0))
        # Progress is a reward, so we use its negative value in cost calculation
        raw_costs[r]['prog'] = -metrics.get('progress', 0.0)

    # 2. Normalize each cost component and apply weights
    normalized_costs: CostDict = {r: 0.0 for r in available_lanes}
    for component in ['dist', 'curve', 'prog']:
        values = {r: raw_costs[r][component] for r in available_lanes}
        min_val = min(values.values())
        max_val = max(values.values())
        range_val = max_val - min_val

        if range_val < epsilon:
            continue # All lanes have the same cost for this component

        weight = weights[{ 'dist': 'alpha', 'curve': 'beta', 'prog': 'gamma' }[component]]
        for r in available_lanes:
            normalized_value = (values[r] - min_val) / range_val
            normalized_costs[r] += weight * normalized_value

    # 3. Add VLM hint cost (already a small, weighted value)
    for r in available_lanes:
        normalized_costs[r] += _calculate_vlm_hint_cost(hint, r, weights['w_lane'])

    # 4. Apply hysteresis penalty for lane changes
    final_costs = apply_lane_change_penalty(
        normalized_costs,
        previous_lane,
        penalty=weights['delta']
    )

    # 5. Select the lane with the minimum final cost
    best_lane = min(final_costs, key=final_costs.get)
    
    return best_lane