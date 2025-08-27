import csv
import rclpy
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
import time
import os
from ament_index_python.packages import get_package_share_directory
import math

# ================== Globals & Params ==================
yaw = 0.0
flag = 0

# Vehicle
WHEEL_BASE = 0.3240  # m

# Dynamic Lookahead (only mechanism kept)
LD_MIN = 0.6   # m
LD_MAX = 2.5   # m
K_LD_V = 0.5   # s (lookahead growth per m/s)

# Simple speed switch (as given)
MAX_VELOCITY = 0.10
MIN_VELOCITY = 0.05
STEER_SLOW_THRESH = 0.3  # rad

# Speed estimate (from IPS)
v_est = 0.0
POS_PREV = None
T_PREV = None
V_ALPHA = 0.3  # low-pass gain for speed estimate

# CSV logging
Dict = {}
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
    if column_name == 'positions_X':
        column_index = 0
    elif column_name == 'positions_y':
        column_index = 1
    else:
        return []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:
                try:
                    column_data.append(float(row[column_index]))
                except ValueError:
                    pass
    return column_data

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def euler_from_quaternion(quaternion):
    x = quaternion.x; y = quaternion.y; z = quaternion.z; w = quaternion.w
    # roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def convert_to_steer_Command(steering_deg):
    """Convert steering angle [deg] → normalized [-1,1] (±30° clip)."""
    command = Float32()
    steering_clipped = np.clip(steering_deg, -30.0, 30.0)
    command.data = steering_clipped / 30.0
    return command

# ================== PP Core ==================
def get_yaw(ImuMsg):
    global yaw
    _, _, yaw = euler_from_quaternion(ImuMsg.orientation)

def get_point(PointMsg):
    """Main PP callback using ONLY dynamic lookahead."""
    global C_pose, yaw, path, Dict
    global v_est, POS_PREV, T_PREV

    # --- Pose & CSV log ---
    C_pose = [PointMsg.x, PointMsg.y]
    Dict = {"positions_X": C_pose[0], "positions_y": C_pose[1]}
    CSV_SAVE()

    # --- Speed estimate from IPS (low-pass) ---
    t_now = time.time()
    if POS_PREV is not None and T_PREV is not None:
        dt = max(1e-3, t_now - T_PREV)
        v_raw = np.linalg.norm(np.array(C_pose) - np.array(POS_PREV)) / dt
        v_est = (1.0 - V_ALPHA) * v_est + V_ALPHA * v_raw
    POS_PREV = C_pose[:]
    T_PREV = t_now

    # --- Find nearest waypoint ---
    distances = np.linalg.norm(path - np.array(C_pose), axis=1)
    nearest_index = int(np.argmin(distances))

    # --- Dynamic Lookahead in meters ---
    Ld = np.clip(LD_MIN + K_LD_V * v_est, LD_MIN, LD_MAX)

    # --- Goal index by accumulated path distance from nearest ---
    goal_index = nearest_index
    acc = 0.0
    while goal_index < len(path) - 1 and acc < Ld:
        step = np.linalg.norm(path[goal_index + 1] - path[goal_index])
        acc += step
        goal_index += 1
    goal_point = path[goal_index]

    # --- Pure Pursuit geometry in vehicle frame ---
    dx = goal_point[0] - C_pose[0]
    dy = goal_point[1] - C_pose[1]
    cos_y = np.cos(yaw); sin_y = np.sin(yaw)
    local_x =  cos_y * dx + sin_y * dy
    local_y = -sin_y * dx + cos_y * dy

    # Avoid singular
    Ld_sq = max(1e-6, local_x**2 + local_y**2)

    # Curvature & steering
    curvature = 2.0 * local_y / Ld_sq
    steering_angle = np.arctan2(WHEEL_BASE * curvature, 1.0)  # [rad]

    # === Publish steering ===
    # Option A) normalized [-1,1] (as in your original PP)
    steering_deg = np.degrees(steering_angle)
    steering_command = convert_to_steer_Command(steering_deg)
    steering_pub.publish(steering_command)

    # Option B) if your simulator expects radians directly:
    # rad_msg = Float32(); rad_msg.data = float(steering_angle)
    # steering_pub.publish(rad_msg)

    # === Very simple speed logic (unchanged) ===
    vel_command = Float32()
    vel_command.data = MAX_VELOCITY if abs(steering_angle) < STEER_SLOW_THRESH else MIN_VELOCITY
    cmd_pub.publish(vel_command)

def main(arg=None):
    global path, node, cmd_pub, steering_pub

    rclpy.init(args=arg)
    node = rclpy.create_node('pure_pursuit_controller')

    # Load waypoints (ROS2-idiomatic)
    try:
        package_share_dir = get_package_share_directory('car_control')
        file_path = os.path.join(package_share_dir, 'CSVs', 'Centerline_points.csv')
        node.get_logger().info(f"Loading waypoints from: {file_path}")
    except Exception as e:
        node.get_logger().error(f"Error getting package share directory: {e}")
        rclpy.shutdown()
        return

    # Path
    column_x = 'positions_X'
    column_y = 'positions_y'
    x_values = csv_reading(file_path, column_x)
    y_values = csv_reading(file_path, column_y)
    path_np = np.array(list(zip(x_values, y_values)), dtype=float)
    if path_np.shape[0] == 0:
        node.get_logger().error("Failed to load waypoints")
        rclpy.shutdown()
        return
    node.get_logger().info(f"Successfully loaded {path_np.shape[0]} waypoints.")
    # Make it global for simplicity (keeps your original structure)
    globals()['path'] = path_np

    # Publishers & Subscribers
    cmd_pub = node.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
    steering_pub = node.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
    node.create_subscription(Point, "/autodrive/f1tenth_1/ips", get_point, 10)
    node.create_subscription(Imu,   "/autodrive/f1tenth_1/imu", get_yaw,   10)

    rclpy.spin(node)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
