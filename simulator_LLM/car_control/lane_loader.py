import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.interpolate import splprep, splev

@dataclass
class Lane:
    waypoints: np.ndarray
    meta: dict

@dataclass
class Lanes:
    center: Lane
    inner: Lane
    outer: Lane

def _process_lane(file_path: str, ds: float = 0.2) -> Lane:
    """Loads waypoints from a CSV, resamples them to a constant distance, and computes metrics."""
    print(f"Processing lane from {file_path}...")
    # TODO: Add assert for map units and coordinate frame
    df = pd.read_csv(file_path)
    waypoints = df[['x', 'y']].to_numpy()
    print(f"  Loaded {len(waypoints)} waypoints.")

    # Resample points
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, per=0)
    path_length = np.sum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
    num_points = int(np.ceil(path_length / ds))
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)
    resampled_waypoints = np.vstack((x_new, y_new)).T
    print(f"  Resampled to {len(resampled_waypoints)} waypoints (ds={ds}m).")

    # Calculate path length (s)
    distances = np.sqrt(np.sum(np.diff(resampled_waypoints, axis=0)**2, axis=1))
    s = np.concatenate(([0], np.cumsum(distances)))

    # Calculate curvature
    dx_dt = np.gradient(resampled_waypoints[:, 0], s)
    dy_dt = np.gradient(resampled_waypoints[:, 1], s)
    d2x_dt2 = np.gradient(dx_dt, s)
    d2y_dt2 = np.gradient(dy_dt, s)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    # Calculate tangents and normals
    tangents = np.vstack((dx_dt, dy_dt)).T
    normals = np.vstack((-dy_dt, dx_dt)).T

    meta = {
        "ds": ds,
        "s": s,
        "curvature": curvature,
        "tangents": tangents,
        "normals": normals
    }

    return Lane(waypoints=resampled_waypoints, meta=meta)

def load_three_lanes(center_csv: str, inner_csv: str, outer_csv: str, ds: float = 0.2) -> Lanes:
    """Loads and processes three lanes from CSV files."""
    print("--- Loading and processing lanes ---")
    lanes = Lanes(
        center=_process_lane(center_csv, ds),
        inner=_process_lane(inner_csv, ds),
        outer=_process_lane(outer_csv, ds),
    )
    print("--- All lanes loaded successfully ---")
    return lanes

def compute_lane_metrics(pose, yaw, waypoints, risks):
    """
    Computes metrics for each lane.

    TODO: Implement the calculation of free distance, curvature, and progress.
    """
    metrics = {}
    for lane_name, lane_wp in waypoints.items():
        metrics[lane_name] = {
            'free_distance': 100.0,  # Placeholder
            'curvature': 0.0,  # Placeholder
            'progress': 0.0  # Placeholder
        }
    return metrics