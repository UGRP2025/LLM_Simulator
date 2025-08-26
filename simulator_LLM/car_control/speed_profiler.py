import numpy as np
from typing import Optional, Dict

# --- Constants (Consider moving to params.yaml) ---
G = 9.81  # Gravity
MU = 0.9  # Assumed friction coefficient
EPSILON = 1e-6 # Small number to avoid division by zero
V_MAX = 6.0 # m/s, maximum speed limit
OBS_KD = 0.8 # Proportional gain for obstacle-based speed reduction

# Type Aliases for clarity
RiskDict = Dict[str, float]
HintDict = Dict[str, any]
MetricsDict = Dict[str, float]

def _get_speed_hint_factor(hint: Optional[HintDict]) -> float:
    """
    Determines the speed adjustment factor based on the VLM hint.
    """
    if not hint or 'speed' not in hint:
        return 1.0

    speed_token = hint.get('speed')
    if speed_token == 'slow':
        return 0.7
    elif speed_token == 'fast':
        return 1.2
    else: # 'normal' or other
        return 1.0

def speed_to_throttle(speed: float, max_speed: float = V_MAX) -> float:
    """
    Scales a target speed (m/s) to a normalized throttle command [0, 1].
    
    NOTE: This is a simple linear scaler. A more sophisticated mapping
    (e.g., polynomial, lookup table) might be needed depending on the simulator's
    vehicle dynamics.
    """
    if max_speed <= 0:
        return 0.0
    throttle = speed / max_speed
    return np.clip(throttle, 0.0, 1.0)

def target_speed(
    metrics_for_lane: MetricsDict,
    risks: RiskDict,
    hint: Optional[HintDict]
) -> float:
    """
    Calculates the target speed based on curvature, obstacles, and VLM hint.

    Args:
        metrics_for_lane: Dictionary with metrics for the selected lane.
                          Expected keys: 'curvature', 'free_distance'.
        risks: Dictionary with obstacle risk assessment.
               Expected keys: 'front'.
        hint: Optional dictionary with VLM advisor's suggestion.
              Expected keys: 'speed'.

    Returns:
        The calculated target speed in m/s.
    """
    # 1. Curvature-based speed limit
    curvature = abs(metrics_for_lane.get('curvature', 0.0))
    v_curve = np.sqrt(max(EPSILON, MU * G / (curvature + EPSILON)))

    # 2. Obstacle-based speed limit
    front_dist = risks.get('front', float('inf'))
    lane_free_dist = metrics_for_lane.get('free_distance', float('inf'))
    
    # Use the closest obstacle distance between frontal and in-lane
    closest_dist = min(front_dist, lane_free_dist)
    v_obs = OBS_KD * closest_dist

    # 3. Apply VLM hint
    speed_factor = _get_speed_hint_factor(hint)

    # 4. Combine and constrain
    v_target = min(v_curve, v_obs, V_MAX) * speed_factor
    
    # Final clamp to ensure it never exceeds absolute max speed
    return np.clip(v_target, 0.0, V_MAX)