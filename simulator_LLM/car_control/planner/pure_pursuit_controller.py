import numpy as np
from typing import Tuple, Optional, Dict, Any

class PurePursuit:
    """
    A Pure Pursuit controller using the robust logic from the reference implementation.
    This version directly ports the known-good algorithm from Pure_pursuit_opt.py
    into a class structure compatible with the behavior_planner.
    """

    def __init__(self, waypoints: np.ndarray, curvatures: np.ndarray, params: Dict[str, Any]):
        """
        Initializes the controller.

        Args:
            waypoints: An array of waypoints (N, 2).
            curvatures: An array of curvatures (N,). Not used in this implementation
                        to match the reference, but kept for API compatibility.
            params: A dictionary of controller parameters.
        """
        self.waypoints = waypoints
        self.params = params
        self.wheelbase = self.params['wheelbase']
        self.L_min = self.params['lookahead_min']
        self.L_max = self.params['lookahead_max']
        self.k_speed = self.params['lookahead_speed_gain']
        self.max_steer_rad = np.deg2rad(self.params['max_steering_angle_deg'])

    def compute_steer(self, pose: Tuple[float, float], yaw: float, speed: float) -> float:
        """
        Computes the required steering angle using the reference logic.
        """
        # 1. Find the nearest waypoint on the path (full search, as in reference)
        distances = np.linalg.norm(self.waypoints[:, :2] - np.array(pose), axis=1)
        nearest_idx = int(np.argmin(distances))

        # 2. Calculate dynamic lookahead distance (speed-dependent only)
        lookahead_dist = np.clip(self.L_min + self.k_speed * speed, self.L_min, self.L_max)

        # 3. Find the goal point by accumulating distance along the path
        goal_idx = nearest_idx
        accumulated_dist = 0.0
        while goal_idx < len(self.waypoints) - 1 and accumulated_dist < lookahead_dist:
            dist_step = np.linalg.norm(self.waypoints[goal_idx + 1, :2] - self.waypoints[goal_idx, :2])
            accumulated_dist += dist_step
            goal_idx += 1
        
        goal_point = self.waypoints[goal_idx, :2]

        # 4. Transform the goal point to the vehicle's coordinate frame
        dx = goal_point[0] - pose[0]
        dy = goal_point[1] - pose[1]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        local_x =  cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        # 5. Calculate curvature and steering angle
        # This uses the direct curvature formula from the reference implementation
        Ld_sq = local_x**2 + local_y**2
        if Ld_sq < 1e-6:
            Ld_sq = 1e-6 # Avoid division by zero

        curvature = 2.0 * local_y / Ld_sq
        steer_rad = np.arctan(self.wheelbase * curvature)

        # 6. Clamp and normalize the steering command
        steer_rad_clamped = np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad)
        
        return steer_rad_clamped / self.max_steer_rad
