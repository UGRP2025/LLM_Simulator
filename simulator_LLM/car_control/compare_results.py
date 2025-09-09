import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os

# --- Core Functions (from speedProf.py) ---

def compute_curvature(trajectory):
    """Computes the curvature of a given trajectory."""
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    curvature[np.isnan(curvature)] = 0  # Handle division by zero
    return curvature

def compute_ideal_speed(trajectory, a_max=7.0, a_lat_max=8.6, v_max=22.88):
    """Computes the ideal speed profile for a trajectory based on physical limits."""
    curvature = compute_curvature(trajectory)
    v_curvature = np.sqrt(a_lat_max / (curvature + 1e-6))  # Avoid division by zero
    v_curvature = np.clip(v_curvature, 0, v_max)

    N = len(trajectory)
    speed = np.zeros(N)
    speed[0] = 0

    # Forward pass: Respect acceleration limits
    for i in range(1, N):
        d = np.linalg.norm(trajectory[i] - trajectory[i - 1])
        speed[i] = min(v_curvature[i], np.sqrt(speed[i - 1]**2 + 2 * a_max * d))

    # Backward pass: Respect deceleration limits
    for i in range(N - 2, -1, -1):
        d = np.linalg.norm(trajectory[i + 1] - trajectory[i])
        speed[i] = min(speed[i], np.sqrt(speed[i + 1]**2 + 2 * a_max * d))

    return speed

# --- Data Loading ---

def load_trajectory_data(filepath):
    """Loads trajectory data from a CSV, handling headers and different column structures."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}. Skipping.")
        return None, None, None

    x, y, speed = [], [], []
    with open(filepath, 'r', newline='') as f:
        try:
            has_header = csv.Sniffer().has_header(f.read(2048))
            f.seek(0)
        except csv.Error:
            has_header = False
            f.seek(0)

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x.append(float(row['x']))
                    y.append(float(row['y']))
                    if 'speed' in row:
                        speed.append(float(row['speed']))
                except (ValueError, KeyError) as e:
                    print(f"Skipping malformed row in {filepath}: {row} ({e})")
                    continue
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x.append(float(row[0]))
                        y.append(float(row[1]))
                        if len(row) >= 3:
                            speed.append(float(row[2]))
                    except ValueError:
                        continue
    
    path_xy = np.array(list(zip(x, y)))
    speed_array = np.array(speed) if speed else None
    
    # Calculate cumulative distance for plotting against a common axis
    distances = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)

    return path_xy, speed_array, cumulative_dist

# --- Plotting ---

def plot_comparison(data_dict):
    """Plots the comparison of trajectories and speed profiles."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Performance Comparison of Driving Algorithms', fontsize=18, fontweight='bold')

    # Plot 1: Trajectories
    ax1.set_title('Trajectory Comparison', fontsize=14)
    colors = {'Ideal': 'k', 'Pure Pursuit': 'b', 'VLM Planner': 'g'}
    linestyles = {'Ideal': '--', 'Pure Pursuit': '-', 'VLM Planner': '-'}

    for name, data in data_dict.items():
        if data['path'] is not None and len(data['path']) > 0:
            ax1.plot(data['path'][:, 0], data['path'][:, 1], 
                     label=name, color=colors[name], linestyle=linestyles[name])
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.axis('equal')
    ax1.legend()

    # Plot 2: Speed Profiles
    ax2.set_title('Speed Profile Comparison', fontsize=14)
    for name, data in data_dict.items():
        if data['speed'] is not None and data['dist'] is not None and len(data['speed']) > 0:
            ax2.plot(data['dist'], data['speed'], 
                     label=name, color=colors[name], linestyle=linestyles[name])

    ax2.set_xlabel('Distance Along Path (meters)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("research_result_comparison.png")
    print("Saved comparison plot to research_result_comparison.png")
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # Define absolute paths to default CSV files
    base_path = "/home/ugrp/simulator/simulator_LLM/car_control/CSVs"
    
    parser = argparse.ArgumentParser(description="Compare the performance of different driving algorithms.")
    parser.add_argument("--centerline", default=os.path.join(base_path, 'Centerline_points.csv'), help="Path to the centerline CSV for ideal performance.")
    parser.add_argument("--pp_actual", default=os.path.join(base_path, 'PP_actual.csv'), help="Path to the Pure Pursuit actual trajectory CSV.")
    parser.add_argument("--vlm_actual", default=os.path.join(base_path, 'planner_actual.csv'), help="Path to the VLM Planner actual trajectory CSV.")
    args = parser.parse_args()

    # Load data for all three cases
    ideal_path, _, ideal_dist = load_trajectory_data(args.centerline)
    pp_path, pp_speed, pp_dist = load_trajectory_data(args.pp_actual)
    vlm_path, vlm_speed, vlm_dist = load_trajectory_data(args.vlm_actual)

    # Compute ideal speed for the centerline path
    ideal_speed = None
    if ideal_path is not None and len(ideal_path) > 0:
        ideal_speed = compute_ideal_speed(ideal_path)

    # Organize data for plotting
    data_to_plot = {
        "Ideal": {'path': ideal_path, 'speed': ideal_speed, 'dist': ideal_dist},
        "Pure Pursuit": {'path': pp_path, 'speed': pp_speed, 'dist': pp_dist},
        "VLM Planner": {'path': vlm_path, 'speed': vlm_speed, 'dist': vlm_dist}
    }

    plot_comparison(data_to_plot)
