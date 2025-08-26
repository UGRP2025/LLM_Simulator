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
    Selects the best lane from a set of available lanes based on a cost function.

    The cost function J(r) for a lane r is:
    J(r) = α*(1/free_dist) + β*|κ| - γ*progress + VLM_hint + lane_change_penalty

    Args:
        metrics_by_lane: Metrics calculated for each potential lane.
        available_lanes: A set of lane names that are considered safe to drive in.
        hint: The suggestion from the VLM advisor.
        previous_lane: The lane selected in the previous cycle for hysteresis.
        params: A dictionary containing weights and other parameters.

    Returns:
        The name of the lane with the minimum cost.
    """
    if not available_lanes:
        # Default to center lane if no lanes are available (safety fallback)
        return 'center'

    costs: CostDict = {}
    weights = params['weights']
    epsilon = params['epsilon']

    # TODO: Normalize cost components before applying weights,
    # as they have very different scales (e.g., 1/dist vs curvature).

    for r in available_lanes:
        metrics = metrics_by_lane.get(r, {})
        
        # 1. Free distance cost (higher cost for less free space)
        free_dist = metrics.get('free_distance', epsilon)
        cost_dist = weights['alpha'] * (1 / (free_dist + epsilon))

        # 2. Curvature cost (higher cost for sharper turns)
        curvature = metrics.get('curvature', 0.0)
        cost_curve = weights['beta'] * abs(curvature)

        # 3. Progress reward (negative cost for making more progress)
        progress = metrics.get('progress', 0.0)
        cost_prog = -weights['gamma'] * progress

        # 4. VLM hint bonus (negative cost if hinted by VLM)
        cost_vlm = _calculate_vlm_hint_cost(hint, r, weights['w_lane'])

        # 5. Sum up the costs for lane r
        costs[r] = cost_dist + cost_curve + cost_prog + cost_vlm

    # 6. Apply hysteresis penalty for lane changes
    final_costs = apply_lane_change_penalty(
        costs,
        previous_lane,
        penalty=weights['delta']
    )

    # 7. Select the lane with the minimum cost
    best_lane = min(final_costs, key=final_costs.get)
    
    return best_lane