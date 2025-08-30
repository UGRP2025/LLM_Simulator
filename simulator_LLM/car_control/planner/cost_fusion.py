from typing import Dict, Optional, Set
import numpy as np
import time

from car_control.planner.safety import apply_lane_change_penalty

# --- Type Aliases ---
LaneName = str
MetricsByLane = Dict[LaneName, Dict[str, float]]
HintDict = Dict[str, any]
ParamsDict = Dict[str, float]
CostDict = Dict[LaneName, float]

def _calculate_vlm_hint_cost(hint: Optional[HintDict], lane_name: LaneName, params: Dict) -> float:
    """
    Calculates the cost reduction (bonus) based on the VLM lane hint.
    The bonus decays linearly with the age of the hint.
    """
    if not hint or hint.get('lane') != lane_name or 'timestamp' not in hint:
        return 0.0

    weights = params['weights']
    w_lane = weights.get('w_lane', 0.5)
    max_age_s = weights.get('max_hint_age_s', 2.0)

    hint_age = time.time() - hint['timestamp']

    if hint_age > max_age_s:
        return 0.0 # Hint is too old

    # Linear decay of the hint's influence
    age_scaling_factor = 1.0 - (hint_age / max_age_s)
    
    # This is a cost reduction (bonus), so we return a negative value
    return -w_lane * age_scaling_factor

def select_lane(
    metrics_by_lane: MetricsByLane,
    available_lanes: Set[LaneName],
    hint: Optional[HintDict],
    previous_lane: Optional[LaneName],
    params: Dict
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

        weight_key = {'dist': 'alpha', 'curve': 'beta', 'prog': 'gamma'}[component]
        weight = weights.get(weight_key, 1.0)
        for r in available_lanes:
            normalized_value = (values[r] - min_val) / range_val
            normalized_costs[r] += weight * normalized_value

    # 3. Add VLM hint cost, which is now dynamically weighted by age
    for r in available_lanes:
        normalized_costs[r] += _calculate_vlm_hint_cost(hint, r, params)

    # 4. Apply center lane preference bonus
    for r in available_lanes:
        if r != 'center':
            normalized_costs[r] += weights.get('zeta', 0.2) # Add penalty to non-center lanes

    # 5. Apply hysteresis penalty for lane changes
    final_costs = apply_lane_change_penalty(
        normalized_costs,
        previous_lane,
        penalty=weights.get('delta', 0.3)
    )

    # 6. Select the lane with the minimum final cost
    best_lane = min(final_costs, key=final_costs.get)
    
    return best_lane