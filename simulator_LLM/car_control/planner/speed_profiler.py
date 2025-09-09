import numpy as np
from typing import Optional, Dict

# Type Aliases for clarity
RiskDict = Dict[str, float]
HintDict = Dict[str, any]
MetricsDict = Dict[str, float]

class SpeedProfiler:
    """Calculates target speed based on vehicle dynamics, obstacles, and hints."""

    def __init__(self, params: Dict):
        """
        Initializes the SpeedProfiler with parameters.
        Args:
            params: A dictionary containing speed-related parameters.
        """
        self.V_MAX = params.get('vmax', 1.0)
        self.DEFAULT_SPEED = params.get('default_speed', 0.1)
        self.V_MIN = params.get('vmin', 0.05)
        self.MU = params.get('mu', 0.9)
        self.OBS_KD = params.get('obs_kd', 0.8)
        self.CURVATURE_START_SLOWING = params.get('curvature_start_slowing', 0.2)
        self.CURVATURE_MAX_SLOWING = params.get('curvature_max_slowing', 1.0)
        self.G = 9.81
        self.EPSILON = 1e-6

    def _get_speed_hint_factor(self, hint: Optional[HintDict]) -> float:
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
        else:  # 'normal' or other
            return 1.0

    def speed_to_throttle(self, speed: float) -> float:
        """
        Scales a target speed value to a normalized throttle command [0, 1].
        """
        if self.V_MAX <= 0:
            return 0.0
        throttle = speed / self.V_MAX
        return np.clip(throttle, 0.0, 1.0)

    def target_speed(
        self,
        metrics_for_lane: MetricsDict,
        risks: RiskDict,
        hint: Optional[HintDict]
    ) -> float:
        """
        Calculates the target speed based on curvature, obstacles, and VLM hint.
        """
        # 1. Curvature-based speed limit (New Logic)
        curvature = abs(metrics_for_lane.get('curvature', 0.0))
        if curvature < self.CURVATURE_START_SLOWING:
            v_curve = self.DEFAULT_SPEED  # No slowing needed
        else:
            # Linearly interpolate between default_speed and vmin based on curvature
            slowing_range = max(self.EPSILON, self.CURVATURE_MAX_SLOWING - self.CURVATURE_START_SLOWING)
            interp_fraction = (curvature - self.CURVATURE_START_SLOWING) / slowing_range
            interp_fraction = np.clip(interp_fraction, 0.0, 1.0)
            
            speed_range = self.DEFAULT_SPEED - self.V_MIN
            v_curve = self.DEFAULT_SPEED - interp_fraction * speed_range

        # 2. Obstacle-based speed limit
        lane_free_dist = metrics_for_lane.get('free_distance', float('inf'))
        closest_dist = lane_free_dist
        v_obs = self.OBS_KD * closest_dist

        # 3. Apply VLM hint
        speed_factor = self._get_speed_hint_factor(hint)

        # 4. Combine and constrain
        # Aim for the curvature-adjusted speed, but also respect obstacles
        v_target = min(v_curve, v_obs) * speed_factor
        
        # If target speed is positive but below minimum, raise to minimum
        if v_target > self.EPSILON:
            v_target = max(v_target, self.V_MIN)

        # Final clamp to ensure it never exceeds absolute max speed
        return np.clip(v_target, 0.0, self.V_MAX)
