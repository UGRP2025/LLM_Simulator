import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import splprep, splev
import cv2

# Step 1: Load the image
image_path = 'car_control/Plots/Figure_1.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Threshold the image to isolate the spline
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Step 3: Detect edges of the spline
edges = cv2.Canny(binary_image, 100, 200)

# Step 4: Get the coordinates of the spline pixels
y_coords, x_coords = np.nonzero(edges)

# Step 5: Fit a spline to the detected points
points = np.array([x_coords, y_coords]).T
points = points[np.argsort(points[:, 0])]  # Sort points by x-coordinate

x_sorted = points[:, 0]
y_sorted = points[:, 1]

# Fit a spline using splprep
tck, u = splprep([x_sorted, y_sorted], s=0)

# Generate interpolated points
u_fine = np.linspace(0, 1, 500)
x_spline, y_spline = splev(u_fine, tck)

# Step 6: Plot the original and extracted spline
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray', origin='upper', extent=(0, image.shape[1], image.shape[0], 0))
plt.plot(x_coords, y_coords, 'ro', markersize=2, label='Detected Points')
plt.plot(x_spline, y_spline, 'b-', linewidth=2, label='Fitted Spline')
plt.legend()
plt.title('Spline Extraction')
plt.show()

# Step 7: Extracted spline points
spline_points = np.column_stack((x_spline, y_spline))
print("Spline Points:")
print(spline_points)