import numpy as np
import cvxpy as cp
from scipy.ndimage import binary_dilation 
import csv 
from scipy.spatial import KDTree
from skimage.io import imsave
from PIL import Image
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import distance
import cv2


# Configuration dictionary
config = {
    'HORIZON': 50,        # Number of trajectory points
    'MAX_VELOCITY': 2.0,  # Maximum velocity (m/s)
    'MAX_ACCELERATION': 1.5,  # Maximum acceleration (m/s²)
    'TRACK_WIDTH': 1.0,   # Track width (m)
    'VEHICLE_LENGTH': 0.5,  # Vehicle length (m)
    'VEHICLE_WIDTH': 0.3    # Vehicle width (m)
}


def compute_inner_outer_distances(occupancy_grid, skeleton_points):
    """
    Compute the inner and outer boundary distances for each point on the centerline.

    Parameters:
    - occupancy_grid: 2D numpy array representing the map (1 for occupied, 0 for free space)
    - skeleton: 2D numpy array representing the centerline

    Returns:
    - inner_distances: Array of distances to the inner boundary for each centerline point
    - outer_distances: Array of distances to the outer boundary for each centerline point
    - centerline_points: Array of (x, y) coordinates for the centerline points
    """
    # Ensure the grid is binary
    occupancy_grid = (occupancy_grid > 0).astype(np.bool_)

    # Identify inner and outer boundaries using dilation/erosion
    outer_boundary = binary_dilation(occupancy_grid) & ~occupancy_grid
    inner_boundary = binary_erosion(~occupancy_grid) & ~occupancy_grid

    # Extract coordinates of the boundaries and skeleton points
    inner_coords = np.argwhere(inner_boundary)
    outer_coords = np.argwhere(outer_boundary)

    # Compute distances from each skeleton point to inner and outer boundaries
    inner_distances = []
    outer_distances = []
    for point in skeleton_points:
        inner_dist = distance.cdist([point], inner_coords).min()
        outer_dist = distance.cdist([point], outer_coords).min()
        inner_distances.append(inner_dist)
        outer_distances.append(outer_dist)

    return np.array(inner_distances), np.array(outer_distances)

def load_occupancy():
    with open("/home/autodrive_devkit/src/car_control/car_control/occupancy_grid.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        occupancy_grid = np.array([list(map(float, row)) for row in reader])
    return occupancy_grid



def preprocess_occupancy_grid(occupancy_grid):
    """
    Preprocess occupancy grid for trajectory optimization
    
    Args:
    - occupancy_grid: 2D NumPy array (0 = free space, 1 = occupied)
    
    Returns:
    - Processed safe space grid
    """    
    safe_space = binary_dilation(occupancy_grid, iterations=4)
    safe_space = safe_space.astype(float)
    return safe_space


def occupancy_to_png(skeleton ,output_file="skeleton_path.png"):
    # Save the pruned skeleton as a PNG file
    imsave(output_file, skeleton.astype(np.uint8) * 255)

def png_to_skeleton(png_file):
    """
    Convert a PNG file to a skeletonized binary array.
    
    Parameters:
    - png_file: Path to the PNG file.
    
    Returns:
    - skeleton: 2D binary NumPy array representing the skeleton.
    """
    # Load the image
    image = Image.open(png_file)
    
    # Convert to grayscale
    grayscale_image = image.convert("L")
    
    # Convert to a binary image (thresholding)
    binary_image = np.array(grayscale_image) > 128  # Threshold at 128 (mid-gray)
    
    # Skeletonize the binary image
    skeleton = skeletonize(binary_image)
    
    return skeleton



def extract_centerline(occupancy_grid):


    # Invert the occupancy grid to get free space
    free_space = 1 - occupancy_grid

    # Skeletonize the free space
    skeleton = skeletonize(free_space)


    return skeleton


def skeleton_coordinates(skeleton):
    y_coords, x_coords = np.nonzero(skeleton)
    skeleton_points = np.column_stack((x_coords, y_coords))

    # Create a KDTree to efficiently find nearest neighbors for ordering points
    tree = KDTree(skeleton_points)
    ordered_points = [skeleton_points[0]]
    visited = {0}

    while len(ordered_points) < len(skeleton_points):
        last_point = ordered_points[-1]
        distances, indices = tree.query(last_point, k=len(skeleton_points))  # Find all neighbors
        for idx in indices:
            if idx not in visited:
                visited.add(idx)
                ordered_points.append(skeleton_points[idx])
                break
        else:
            # If no new point is found, break to avoid an infinite loop
            break

    ordered_points = np.array(ordered_points)

    return  ordered_points




def optimize_trajectory(occupancy_grid, start_point, end_point, config):
    """
    Perform minimum time trajectory optimization
    
    Args:
    - occupancy_grid: 2D NumPy array of track
    - start_point: [x, y] starting coordinates
    - end_point: [x, y] ending coordinates
    - config: Dictionary of configuration parameters
    
    Returns:
    - Optimized trajectory
    - Velocity profile
    - Total time
    """
    safe_space = preprocess_occupancy_grid(occupancy_grid)
    
    # Configuration shortcuts
    N = config['HORIZON']
    v_max = config['MAX_VELOCITY']
    a_max = config['MAX_ACCELERATION']
    
    # Optimization Variables
    x = cp.Variable((N, 2))    # Position trajectory
    v = cp.Variable(N)          # Velocity profile
    t = cp.Variable()            # Total time
    
    # Initial trajectory guess
    initial_trajectory = compute_initial_trajectory_guess(start_point, end_point, N)# center line
    
    # Objective: Minimize total time
    objective = cp.Minimize(t)
    
    # Constraints
    constraints = [
        t >= 0,
        v >= 0,
        v <= v_max,
        x[0] == start_point,
        x[-1] == end_point,
        v[0] == 0,
        v[-1] == 0
    ]
    
    for k in range(N - 1):
        constraints += [
            x[k + 1, 0] == x[k, 0] + v[k] * (t / N),
            x[k + 1, 1] == x[k, 1] + v[k] * (t / N),
            cp.abs(v[k + 1] - v[k]) / (t / N) <= a_max
        ]
        
        grid_x = int(x[k, 0])
        grid_y = int(x[k, 1])
        
        if 0 <= grid_x < safe_space.shape[0] and 0 <= grid_y < safe_space.shape[1]:
            constraints.append(safe_space[grid_x, grid_y] == 1)
    
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS)
        return {
            'trajectory': x.value,
            'velocity_profile': v.value,
            'total_time': t.value
        }
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None





def visualization(visual):

    visual = visual.astype(float)
    # Visualize the occupancy grid
    plt.imshow(visual, cmap="viridis", origin="upper")
    plt.colorbar(label="Occupancy Probability")
    plt.title("Occupancy Grid Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def visualize_occupancy_grid(occupancy_grid, center_line=None, boundary_distance=None):
    """
    Visualize the occupancy grid with optional center line and boundary distances.
    
    Parameters:
    -----------
    occupancy_grid : numpy.ndarray
        2D binary grid where 1 represents occupied space and 0 represents free space.
    center_line : numpy.ndarray, optional
        Coordinates of the center line path
    boundary_distance : numpy.ndarray, optional
        Distances from each center line point to the nearest boundary
    """
    plt.figure(figsize=(12, 6))
    
    # Original Occupancy Grid
    plt.subplot(121)
    plt.imshow(occupancy_grid, cmap='binary', interpolation='nearest')
    plt.title('Occupancy Grid')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Occupancy Grid with Center Line
    plt.subplot(122)
    plt.imshow(occupancy_grid, cmap='binary', interpolation='nearest')
    

    plt.legend()
    plt.title('Center Line and Boundary Distances')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt

def visualize_points(points, title="Point Visualization", color='b', marker='o'):
    """
    Visualize a set of 2D points.

    Parameters:
    - points: A list or numpy array of points [(x1, y1), (x2, y2), ...].
    - title: Title of the plot.
    - color: Color of the points.
    - marker: Marker style for the points.
    """
    # Convert points to x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c=color, marker=marker, label="Points")
    
    # Invert the Y-axis
    plt.gca().invert_yaxis()
    
    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Show plot
    plt.show()


def plot_points_on_occupancy_grid(grid, points, title="Occupancy Grid with Points"):
    """
    Plot a set of points on top of an occupancy grid.

    Parameters:
    - grid: 2D numpy array representing the occupancy grid.
            (e.g., 0 for free, 1 for occupied, -1 for unknown)
    - points: List or numpy array of points [(x1, y1), (x2, y2), ...].
    - title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))

    # Display the occupancy grid
    plt.imshow(grid, cmap="gray", origin="upper")  # Use "gray" colormap and set the origin at the top

    # Extract X and Y coordinates of the points
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    # Overlay points on the grid
    plt.scatter(x_coords, y_coords, c="red", marker="o", label="Points")

    # Add labels, legend, and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def visualize_distances(occupancy_grid, distances, skeleton_coords):
    """
    Visualize the distances between inner and outer boundaries.

    Parameters:
    - occupancy_grid: 2D numpy array of the map.
    - distances: Array of distances between boundaries.
    - skeleton_coords: Coordinates of the centerline.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(occupancy_grid, cmap='gray')
    plt.scatter(skeleton_coords[:, 1], skeleton_coords[:, 0], c=distances, cmap='viridis', s=5)
    plt.colorbar(label='Distance Between Boundaries')
    plt.title("Distance Between Inner and Outer Boundaries")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def ploting(line1_points, line2_points, line3_points, optimized_path):
    # Define a helper function to plot a single line
    def plot_line(points, color, label):
        x_coords, y_coords = zip(*points)  # Extract x and y coordinates
        plt.plot(x_coords, y_coords, color=color, label=label, marker='o')  # Plot line with markers
    
    # Plot all three lines
    plt.figure(figsize=(8, 6))
    plot_line(line1_points, color='red', label='center Line')
    plot_line(line2_points, color='blue', label='out_bound')
    plot_line(line3_points, color='green', label='inn_bound')
    plot_line(optimized_path, color='pink', label='oppa')
    
    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Visualization')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()



def main():
    #  Example usage
    skeleton_path = "/home/autodrive_devkit/src/car_control/car_control/skeleton_path_masked.png"
    outer_bound_path = "/home/autodrive_devkit/src/car_control/car_control/outer_boundry.png"
    inner_bound_path = "/home/autodrive_devkit/src/car_control/car_control/IneerBounds.png"


    occupancy = load_occupancy()
    heat_map = preprocess_occupancy_grid(occupancy)

    skeleton = png_to_skeleton(skeleton_path)

    keleton_points = skeleton_coordinates(skeleton)

    inner_bound = png_to_skeleton(inner_bound_path)
    inner_bound_points = skeleton_coordinates(inner_bound)

    outer_bound = png_to_skeleton(outer_bound_path)
    outer_bound_points = skeleton_coordinates(outer_bound)

    ploting(keleton_points,inner_bound_points,outer_bound_points)


    
if __name__ == "__main__":
    main()