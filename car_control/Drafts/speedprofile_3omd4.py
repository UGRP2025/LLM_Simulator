import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Vehicle and trajectory parameters
a_max = 5.2  # Maximum acceleration (m/s^2)
a_lat_max = 3.6  # Maximum lateral acceleration (m/s^2) try
v_max = 22.88  # Maximum velocity (m/s)

def compute_curvature(trajectory):#will be removed
    """
    Compute the curvature of the trajectory.
    Args:
        trajectory: Nx2 array of [x, y] waypoints.
    Returns:
        curvatures: N array of curvature values.
    Example
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
    curvature[np.isnan(curvature)] = 0  # Handle division by zero
    return curvature
    

def compute_speed_profile(trajectory, a_max, a_lat_max, v_max):
    """
    Generate the optimal speed profile for the trajectory.
    Args:
        trajectory: Nx2 array of [x, y] waypoints.
        a_max: Maximum forward/backward acceleration (m/s^2).
        a_lat_max: Maximum lateral acceleration (m/s^2).
        v_max: Maximum velocity (m/s).
    Returns:
        speed_profile: N array of speeds (m/s).

    Example  
    """  
        # Compute curvature
    curvature = compute_curvature(trajectory)
    
    # Compute maximum speeds based on lateral acceleration
    v_curvature = np.sqrt(a_lat_max / (curvature + 1e-6))  # Avoid div by zero
    v_curvature = np.clip(v_curvature, 0, v_max)
    
    # Forward pass: Respect acceleration limits
    N = len(trajectory)
    #speed = np.zeros(N)
    initial_speed = 0
    speed=[]
    speed[0] = initial_speed  # Start from rest
    for i in range(1, N):
        d = np.linalg.norm(trajectory[i] - trajectory[i - 1])  # Distance
        speed[i] = min(v_curvature[i], np.sqrt(speed[i - 1]**2 + 2 * a_max * d))
    
    # Backward pass: Respect deceleration limits
    for i in range(N - 2, -1, -1):
        d = np.linalg.norm(trajectory[i + 1] - trajectory[i])  # Distance
        speed[i] = min(speed[i], np.sqrt(speed[i + 1]**2 + 2 * a_max * d))
        initial_speed = speed[i]
    
    return speed


        


def plot_speed_profile(trajectory, speed_profile):
    """
    Plot the trajectory and speed profile.
    Args:
        trajectory: Nx2 array of [x, y] waypoints.
        speed_profile: N array of speeds (m/s).

        

        Example
    """
    plt.figure(figsize=(12, 6))
    
    # Trajectory plot
    plt.subplot(1, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")
    plt.axis("equal")
    
    # Speed profile plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(speed_profile)), speed_profile, label="Speed Profile", color='red')
    plt.xlabel("Waypoint Index")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Profile")
    
    plt.tight_layout()
    plt.savefig("speed_profile.png")



# Example usage
if __name__ == "__main__":
    # Example trajectory (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    t = np.linspace(0, 2 * np.pi, 100)
    x = 2 * np.cos(t)
    y = np.sin(2 * t)
    trajectory = np.column_stack((x, y))
    
    # Compute speed profile
    speed_profile = compute_speed_profile(trajectory, a_max, a_lat_max, v_max)
    
    # Plot results
    plot_speed_profile(trajectory, speed_profile)
