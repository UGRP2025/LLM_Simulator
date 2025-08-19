import numpy as np
import matplotlib.pyplot as plt
from csv_to_arr import CSVConverter

class SpeedProfileGenerator:
    def __init__(self, a_max, a_lat_max, v_max):
        
        self.a_max = a_max
        self.a_lat_max = a_lat_max
        self.v_max = v_max

    @staticmethod
    def compute_curvature(trajectory):

        x = trajectory[:, 0]
        y = trajectory[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(ddx * dy - ddy * dx) / (dx**2 + dy**2)**1.5
        curvature[np.isnan(curvature)] = 0  # Handle division by zero
        return curvature

    def compute_speed_profile(self, trajectory):

        curvature = self.compute_curvature(trajectory)
        v_curvature = np.sqrt(self.a_lat_max / (curvature + 1e-6))  # Avoid division by zero
        v_curvature = np.clip(v_curvature, 0, self.v_max)

        N = len(trajectory)
        speed = np.zeros(N)  # Initialize speed array
        speed[0] = 0  # Start from rest

        # Forward pass: Respect acceleration limits
        for i in range(1, N):
            d = np.linalg.norm(trajectory[i] - trajectory[i - 1])  # Distance
            speed[i] = min(v_curvature[i], np.sqrt(speed[i - 1]**2 + 2 * self.a_max * d))

        # Backward pass: Respect deceleration limits
        for i in range(N - 2, -1, -1):
            d = np.linalg.norm(trajectory[i + 1] - trajectory[i])  # Distance
            speed[i] = min(speed[i], np.sqrt(speed[i + 1]**2 + 2 * self.a_max * d))

        return speed
    
    @staticmethod
    def plot_speed_profile(trajectory, speed_profile):
        
        plt.figure(figsize=(12, 6))

        # Trajectory plot
        plt.subplot(1, 2, 1)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory")
        plt.axis("equal")
        plt.legend()
        
        # Speed profile plot
        plt.subplot(1, 2, 2)
        plt.plot(range(len(speed_profile)), speed_profile, label="Speed Profile", color='red')
        plt.xlabel("Waypoint Index")
        plt.ylabel("Speed (m/s)")
        plt.title("Speed Profile")
        plt.legend()

        plt.tight_layout()
        plt.savefig("speed_profile.png")
        plt.show()

if __name__ == "__main__":
    # Vehicle parameters
    a_max = 7  # Maximum acceleration (m/s^2)
    a_lat_max = 8.6  # Maximum lateral acceleration (m/s^2)
    v_max = 22.88  # Maximum velocity (m/s)

    path_obj = CSVConverter('CSVs/Centerline_points.csv')
    trajectory_data = path_obj.to_array()

    if not trajectory_data or len(trajectory_data) < 2:
        print("Error: Invalid or empty trajectory data.")
    else:
        try:
            # Convert trajectory to NumPy array
            trajectory = np.array(trajectory_data, dtype=float)
            # Generate and plot speed profile
            generator = SpeedProfileGenerator(a_max, a_lat_max, v_max)
            speed_profile = generator.compute_speed_profile(trajectory)
            generator.plot_speed_profile(trajectory, speed_profile)
        except ValueError as e:
            print(f"Error converting trajectory data to numeric values: {e}")