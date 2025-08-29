import numpy as np
import yaml
from typing import Tuple, Optional, Dict, Any

# Type alias for clarity
Pose2D = Tuple[float, float]

class PurePursuit:
    """
    A Pure Pursuit controller for path tracking.

    This controller computes steering commands to follow a given set of waypoints.
    It features an adaptive lookahead distance based on vehicle speed.
    """

    def __init__(self, waypoints: np.ndarray, curvatures: np.ndarray, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Pure Pursuit controller.

        Args:
            waypoints: A numpy array of shape (N, M) where N is the number of points
                       and M >= 2 (x, y, ...).
            curvatures: A numpy array of shape (N,) containing the curvature at each waypoint.
            params: A dictionary containing controller parameters. If None, default
                    values will be used.
        """
        if waypoints.ndim != 2 or waypoints.shape[1] < 2:
            raise ValueError("Waypoints must be a 2D array with at least 2 columns (x, y).")
        if waypoints.shape[0] != curvatures.shape[0]:
            raise ValueError("Waypoints and curvatures must have the same number of elements.")

        self.waypoints = waypoints
        self.curvatures = curvatures
        self._last_best_idx = 0

        # TODO: Load parameters from params.yaml instead of using a stub
        if params is None:
            # Default parameters if none are provided
            self.params = {
                'wheelbase': 0.324,
                'lookahead_min': 0.3,
                'lookahead_max': 1.5,
                'lookahead_speed_gain': 0.5,
                'lookahead_curvature_gain': 0.1, # c in the user formula
                'max_steering_angle_deg': 30.0
            }
        else:
            self.params = params
            
        self.wheelbase = self.params['wheelbase']
        self.L_min = self.params['lookahead_min']
        self.L_max = self.params['lookahead_max']
        self.k_speed = self.params['lookahead_speed_gain']
        self.c_curvature = self.params['lookahead_curvature_gain']
        self.max_steer_rad = np.deg2rad(self.params['max_steering_angle_deg'])


    def _get_adaptive_lookahead(self, speed: float, curvature_ahead: float = 0.0) -> float:
        """
        Calculates the adaptive lookahead distance (Ld) based on speed.
        The curvature term has been removed to match the reference implementation and improve stability.
        Ld = clamp(Lmin + k * v, Lmin, Lmax)
        """
        ld = self.L_min + self.k_speed * speed
        return np.clip(ld, self.L_min, self.L_max)

    def _find_target_point(self, pose: Pose2D, speed: float) -> Optional[Tuple[np.ndarray, float]]:
        """
        Finds the target waypoint on the path for the pure pursuit algorithm.

        Args:
            pose: The current 2D position of the vehicle (x, y).
            speed: The current speed of the vehicle.

        Returns:
            A tuple containing the target point (np.ndarray) and the lookahead distance (float)
            used to find it. Returns None if no suitable point is found.
        """
        # Find the closest point on the path to the vehicle (using a cached search window)
        search_slice = slice(self._last_best_idx, self._last_best_idx + 100)
        relative_positions = self.waypoints[search_slice, :2] - np.array(pose)
        distances_sq = np.sum(relative_positions**2, axis=1)
        
        if len(distances_sq) == 0: # Reached end of path in previous slice
             return None

        best_idx_local = np.argmin(distances_sq)
        self._last_best_idx += best_idx_local
        
        # Get adaptive lookahead distance
        # Use the curvature at the vehicle's current closest point on the path
        current_curvature = self.curvatures[self._last_best_idx]
        lookahead_dist = self._get_adaptive_lookahead(speed, curvature_ahead=current_curvature)

        # Search forward from the closest index by accumulating path distance (more robust)
        goal_idx = self._last_best_idx
        accumulated_dist = 0.0
        while goal_idx < len(self.waypoints) - 1 and accumulated_dist < lookahead_dist:
            dist_step = np.linalg.norm(self.waypoints[goal_idx + 1, :2] - self.waypoints[goal_idx, :2])
            accumulated_dist += dist_step
            goal_idx += 1

        # Return the found goal point
        return self.waypoints[goal_idx], lookahead_dist

    def compute_steer(self, pose: Pose2D, yaw: float, speed: float) -> float:
        """
        Computes the required steering angle to follow the path.

        Args:
            pose: The current 2D position of the vehicle (x, y).
            yaw: The current heading of the vehicle in radians.
            speed: The current speed of the vehicle.

        Returns:
            The normalized steering command in the range [-1, 1].
        """
        target_data = self._find_target_point(pose, speed)
        if target_data is None:
            return 0.0  # No target found, maintain current steering

        target_point, lookahead_dist = target_data
        
        # === Coordinate Transformation (World to Vehicle Frame) ===
        dx = target_point[0] - pose[0]
        dy = target_point[1] - pose[1]
        
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        target_x_vehicle = cos_yaw * dx + sin_yaw * dy
        target_y_vehicle = -sin_yaw * dx + cos_yaw * dy
        
        # === Pure Pursuit Formula ===
        # Calculate the steering angle alpha needed to point towards the target
        alpha = np.arctan2(target_y_vehicle, target_x_vehicle)
        
        # Calculate the final steering angle using the pure pursuit geometric model
        # Formula: delta = atan2(2 * L * sin(alpha), Ld)
        steer_rad = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), lookahead_dist)
        
        # Clamp the steering angle to physical limits
        steer_rad_clamped = np.clip(steer_rad, -self.max_steer_rad, self.max_steer_rad)
        
        # Normalize the steering command to [-1, 1]
        return steer_rad_clamped / self.max_steer_rad

# Example of loading parameters from a YAML file (stub)
def load_params_from_yaml(file_path: str) -> Dict[str, Any]:
    """
    Loads parameters from a YAML file.
    NOTE: This is a stub. In a real ROS2 node, this would be handled by the
          parameter server.
    """
    try:
        with open(file_path, 'r') as f:
            all_params = yaml.safe_load(f)
            # Assuming a structure like the one in params.yaml
            pp_params = {
                'wheelbase': all_params.get('pure_pursuit', {}).get('wheelbase', 0.324),
                'lookahead_min': all_params.get('pp', {}).get('lookahead_min', 1.0),
                'lookahead_max': all_params.get('pp', {}).get('lookahead_max', 4.5),
                'lookahead_speed_gain': all_params.get('pp', {}).get('lookahead_gain_vs_speed', 0.35),
                'max_steering_angle_deg': all_params.get('pure_pursuit', {}).get('max_steering_angle', 30.0),
                'lookahead_curvature_gain': 0.1 # Not in sample YAML, adding default
            }
            return pp_params
    except (IOError, yaml.YAMLError) as e:
        print(f"Error loading params from {file_path}: {e}")
        return {}