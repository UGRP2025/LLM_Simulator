import numpy as np
from typing import Tuple

def find_closest_waypoint(pose: Tuple[float, float], waypoints: np.ndarray) -> Tuple[int, float]:
    """
    Finds the index of the closest waypoint to a given pose and the distance to it.

    Args:
        pose: The (x, y) coordinate of the vehicle.
        waypoints: A numpy array of waypoints, shape (N, 2).

    Returns:
        A tuple containing:
        - The index of the closest waypoint.
        - The distance from the pose to the closest waypoint.
    """
    if waypoints.shape[0] == 0:
        return -1, float('inf')

    vehicle_pos = np.array(pose).reshape(1, 2)
    distances = np.linalg.norm(waypoints - vehicle_pos, axis=1)
    
    closest_idx = np.argmin(distances)
    min_dist = distances[closest_idx]
    
    return int(closest_idx), float(min_dist)
