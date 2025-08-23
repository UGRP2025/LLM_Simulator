import csv
import rclpy
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
import time
import os
from ament_index_python.packages import get_package_share_directory


yaw = 0.0
flag = 0






def CSV_SAVE():
    global flag
    with open("/home/autodrive_devkit/src/simulator_LLM/car_control/CSVs/actual.csv", mode="a") as csvfile:
        fieldnames = ["positions_X", "positions_y"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        if flag == 0:
            writer.writeheader()
            flag = 1
        writer.writerow(Dict)


def csv_reading(file_path, column_name):
    column_data = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if column_name in row:
                column_data.append(float(row[column_name]))
    return column_data

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def euler_from_quaternion(quaternion):
    """
    Convert a ROS2 Quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        quaternion (geometry_msgs.msg.Quaternion): Input quaternion
    
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Extract quaternion components directly from the Quaternion object
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw



def convert_to_steer_Command(steering):
    """
    Convert steering angle to a normalized command
    
    Args:
        steering (float): Steering angle in degrees
    
    Returns:
        Float32: Normalized steering command
    """
    # Create a new Float32 message
    command = Float32()
    
    # Clip the steering angle to the valid range
    steering_clipped = np.clip(steering, -30, 30)
    
    # Normalize to range [-1, 1]
    command.data = steering_clipped / 30.0
    
    return command

def get_yaw(Imu):
    """
    Extract yaw angle from IMU orientation quaternion
    
    Args:
        Imu (sensor_msgs.msg.Imu): IMU message containing orientation quaternion
    """
    global yaw
    
    # Extract quaternion from IMU message
    orientation_q = Imu.orientation
    
    # Convert quaternion to Euler angles
    # Unpack all three angles, but we only care about yaw
    _, _, yaw = euler_from_quaternion(orientation_q)

def get_point(Point):
    global C_pose, yaw, path, Dict

    C_pose = [Point.x, Point.y]

    # Log car's position
    Dict = {"positions_X": C_pose[0], "positions_y": C_pose[1]}
    CSV_SAVE()

    # Find the nearest point on the path to the car
    distances = np.linalg.norm(path - np.array(C_pose), axis=1)
    nearest_index = np.argmin(distances)

    # Set a lookahead distance (in terms of number of points on the path)
    lookahead_points = 10  # This is a tunable parameter

    # Find the goal point on the path
    goal_index = min(nearest_index + lookahead_points, len(path) - 1)
    goal_point = path[goal_index]
    
    # Calculate curvature and send commands
    calculate_curv(goal_point)



def calculate_curv(point, wheel_base =0.3240):
    global cmd_pub, steering_pub, yaw
   # Calculate relative position 
    dx = point[0] - C_pose[0]
    dy = point[1] - C_pose[1]

    # Transform to local coordinates
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_x = cos_yaw * dx +sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    
    # Avoid division by zero or very small numbers
    if abs(local_x) < 1e-6:
        local_x = 1e-6 if local_x >= 0 else -1e-6
    
    # Calculate curvature
    curvature = 2.0 * local_y  / (local_x**2 + local_y**2)
    
    steering_angle = np.arctan2(wheel_base * curvature, 1.0)

    # Convert to degrees and limit the steering angle
    steering_angle_deg = np.degrees(steering_angle)

    steering_command = convert_to_steer_Command(steering_angle_deg)

    steering_pub.publish(steering_command)
    
    vel_command = Float32()
    vel_command.data = 0.08

    cmd_pub.publish(vel_command) 


def main(arg = None):
    global path, wheel_base, node, cmd_pub, steering_pub

    # Initialize ROS2 first to use its features like logging and package path finding
    rclpy.init(args=arg)
    node = rclpy.create_node('pure_pursuit_controller')

    # --- Get path to the CSV file in a ROS2-idiomatic way ---
    try:
        package_share_dir = get_package_share_directory('car_control')
        file_path = os.path.join(package_share_dir, 'CSVs', 'Centerline_points.csv')
        node.get_logger().info(f"Loading waypoints from: {file_path}")
    except Exception as e:
        node.get_logger().error(f"Error getting package share directory: {e}")
        rclpy.shutdown()
        return
    # ---

    # Paramaeters
    wheel_base = 0.3240
    column_x = 'positions_X'
    column_y = 'positions_y'
    x_values = csv_reading(file_path, column_x)
    y_values = csv_reading(file_path, column_y)
    path = list(zip(x_values, y_values))
    path = np.array(path)

    # Check if the path was loaded correctly
    if path.shape[0] == 0:
        node.get_logger().error(f"Failed to load waypoints from {file_path}. Is the path correct, file not empty, and columns '{column_x}', '{column_y}' present?")
        rclpy.shutdown()
        return  # Exit if path is not loaded

    node.get_logger().info(f"Successfully loaded {path.shape[0]} waypoints.")

    cmd_pub = node.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
    steering_pub = node.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)

    node.create_subscription(Point, "/autodrive/f1tenth_1/ips", get_point, 10)
    node.create_subscription(Imu, "/autodrive/f1tenth_1/imu", get_yaw, 10)

    rclpy.spin(node)



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass