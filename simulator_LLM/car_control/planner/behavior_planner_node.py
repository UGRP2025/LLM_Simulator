import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

import numpy as np
import argparse
import time
import json
import pandas as pd
import yaml
import os

# Import all the implemented modules
from car_control.planner.lane_loader import load_three_lanes
from car_control.planner.pure_pursuit_controller import PurePursuit
from car_control.planner.speed_profiler import target_speed, speed_to_throttle
from car_control.planner.perception_yolo_bridge import get_sector_risks, YOLOPerception, DEFAULT_PARAMS as PERCEPTION_PARAMS
from car_control.planner.vlm_advisor import VLMAdvisor
from car_control.planner.cost_fusion import select_lane
from car_control.planner.safety import mask_unsafe_lanes
from car_control.planner.planner_utils import find_closest_waypoint

# Placeholder for a new module to compute metrics
# In a real scenario, this would be a proper module
def compute_lane_metrics(lanes, pose, yaw, risks):
    """
    Computes real-time metrics for each lane based on the vehicle's pose.
    """
    metrics = {}
    base_free_distance = 50.0  # Default lookahead distance if no obstacles

    lanes_dict = {
        'center': lanes.center,
        'inner': lanes.inner,
        'outer': lanes.outer
    }

    for name, lane_obj in lanes_dict.items():
        # Find the closest point on the lane's path
        closest_idx, dist_to_lane = find_closest_waypoint(pose, lane_obj.waypoints)

        if closest_idx == -1:
            continue # Skip if lane has no waypoints

        # Get pre-computed metrics from the closest point on the path
        # This assumes the lane data is rich with metadata
        progress = lane_obj.meta['s'][closest_idx]
        curvature = lane_obj.meta['curvature'][closest_idx]

        # Start with a high default free distance
        free_dist = base_free_distance

        # Reduce free distance based on sector risks from perception
        if name == 'center' and risks.get('front', 0.0) > 0.1:
            free_dist /= (1 + risks['front'] * 2)
        if name == 'inner' and risks.get('left', 0.0) > 0.1:
            free_dist /= (1 + risks['left'])
        if name == 'outer' and risks.get('right', 0.0) > 0.1:
            free_dist /= (1 + risks['right'])

        metrics[name] = {
            'free_distance': free_dist,
            'curvature': curvature,
            'progress': progress,
            'dist_to_lane': dist_to_lane # Bonus metric: how far are we from the lane center
        }

    return metrics

def extract_yaw_from_quaternion(q: Quaternion) -> float:
    """Extracts the yaw angle from a quaternion."""
    # Simple conversion assuming yaw is around the z-axis
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(t3, t4)

class BehaviorPlanner(Node):
    def __init__(self, offline_mode=False):
        super().__init__('behavior_planner')
        self.offline_mode = offline_mode

        # Load parameters from the YAML file
        params_path = "/home/autodrive_devkit/src/simulator_LLM/car_control/planner/params.yaml"
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        # Load lanes and instantiate controllers
        lane_paths = self.params['lane_csv_paths']
        self.lanes = load_three_lanes(
            lane_paths['center'],
            lane_paths['inner'],
            lane_paths['outer']
        )
        lanes_dict = {
            'center': self.lanes.center,
            'inner': self.lanes.inner,
            'outer': self.lanes.outer
        }
        self.pp_controllers = {name: PurePursuit(lane.waypoints, lane.meta['curvature'], self.params['pp_params']) for name, lane in lanes_dict.items()}

        # VLM Advisor
        self.vlm = VLMAdvisor(self, self.params['vlm_params'])
        self.vlm.start()

        # Perception Module
        # Note: The image topic might need to be adjusted based on the simulator's configuration
        self.perception = YOLOPerception(self, image_topic="/autodrive/f1tenth_1/front_camera")

        # State variables
        self.pose = None
        self.yaw = None
        self.current_speed = 0.0 # Start at 0, will be estimated
        self.last_pose_time = None
        self.current_lane = 'center'
        # Watchdog timers for stale data
        self.last_ips_time = None
        self.last_imu_time = None
        self.data_timeout_sec = 1.0 # Seconds to wait before considering data stale

        if not self.offline_mode:
            # Publishers
            self.throttle_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
            self.steer_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
            # Subscribers
            self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)
            self.create_subscription(Imu, "/autodrive/f1tenth_1/imu", self.cb_imu, 10)
            # Control loop timer
            self.timer = self.create_timer(0.04, self.control_loop) # 25 Hz

    def cb_pose(self, msg: Point):
        self.last_ips_time = self.get_clock().now()
        current_time_sec = self.last_ips_time.nanoseconds / 1e9
        new_pose = (msg.x, msg.y)

        if self.pose is not None and self.last_pose_time is not None:
            dt = current_time_sec - self.last_pose_time
            if dt > 1e-4: # Avoid division by zero and ensure meaningful timestep
                dist = np.linalg.norm(np.array(new_pose) - np.array(self.pose))
                speed_raw = dist / dt
                
                # Low-pass filter for speed estimation (exponential moving average)
                alpha = 0.2 
                self.current_speed = alpha * speed_raw + (1.0 - alpha) * self.current_speed

        self.pose = new_pose
        self.last_pose_time = current_time_sec

    def cb_imu(self, msg: Imu):
        self.last_imu_time = self.get_clock().now()
        self.yaw = extract_yaw_from_quaternion(msg.orientation)

    def control_loop(self):
        # --- Safety: Check for stale sensor data ---
        current_time = self.get_clock().now()
        if self.last_ips_time is None or (current_time - self.last_ips_time).nanoseconds / 1e9 > self.data_timeout_sec:
            self.get_logger().error(f"IPS data is stale (older than {self.data_timeout_sec}s). Halting control.")
            self.throttle_pub.publish(Float32(data=0.0))
            self.steer_pub.publish(Float32(data=0.0))
            return

        if self.last_imu_time is None or (current_time - self.last_imu_time).nanoseconds / 1e9 > self.data_timeout_sec:
            self.get_logger().error(f"IMU data is stale (older than {self.data_timeout_sec}s). Halting control.")
            self.throttle_pub.publish(Float32(data=0.0))
            self.steer_pub.publish(Float32(data=0.0))
            return

        # 1. Perception
        detections = self.perception.get_latest_detections()
        
        # Pass the perception module's actual image width to the risk calculation
        # This ensures the sectors are calculated correctly based on the real image size.
        perception_params = PERCEPTION_PARAMS.copy()
        perception_params['image_width'] = self.perception.image_width
        risks = get_sector_risks(detections, params=perception_params)

        # 2. Lane Metrics
        metrics_by_lane = compute_lane_metrics(self.lanes, self.pose, self.yaw, risks)

        # 3. VLM Hint
        hint = self.vlm.latest_hint()

        # 4. Safety Masking
        unsafe_lanes = mask_unsafe_lanes(metrics_by_lane, self.params['safety_thresholds'])
        available_lanes = {'center', 'inner', 'outer'} - unsafe_lanes

        # 5. Lane Selection
        self.current_lane = select_lane(metrics_by_lane, available_lanes, hint, self.current_lane, self.params['cost_fusion_params'])

        # 6. Low-level Control
        pp_controller = self.pp_controllers[self.current_lane]
        steer = pp_controller.compute_steer(self.pose, self.yaw, self.current_speed)
        v_cmd = target_speed(metrics_by_lane[self.current_lane], risks, hint)

        # TODO: Proper scaling of outputs to simulator requirements
        throttle = speed_to_throttle(v_cmd, max_speed=self.params['vmax'])

        # Log the state for debugging in live mode
        if not self.offline_mode:
            vlm_summary = "N/A"
            if hint:
                vlm_summary = f"{hint.get('lane')}/{hint.get('speed')} (conf: {hint.get('confidence', 0):.2f})"
            
            risks_summary = f"F:{risks['front']:.1f} L:{risks['left']:.1f} R:{risks['right']:.1f}"

            log_msg = (
                f"Lane: {self.current_lane: <6} | "
                f"Steer: {steer: .2f}, Thr: {throttle:.2f} | "
                f"Risks: {risks_summary} | "
                f"VLM: {vlm_summary}"
            )
            self.get_logger().info(log_msg)

        # 7. Publish or Log
        if not self.offline_mode:
            self.steer_pub.publish(Float32(data=float(steer)))
            self.throttle_pub.publish(Float32(data=float(throttle)))
        else:
            # In offline mode, we return the log data for the replay loop to handle
            return {
                'timestamp': time.time(), 'pose': self.pose, 'yaw': self.yaw,
                'chosen_lane': self.current_lane, 'steer': steer, 'throttle': throttle,
                'vlm_hint': hint, 'risks': risks
            }

    def stop_all(self):
        self.vlm.stop()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline_replay', type=str, help='Path to CSV for offline replay')
    # In a ROS2 launch file, args can be passed differently. For standalone, this works.
    cli_args, _ = parser.parse_known_args()

    if cli_args.offline_replay:
        print("Running in OFFLINE REPLAY mode.")
        try:
            # Assuming columns are 'x', 'y', 'yaw_rad'
            replay_df = pd.read_csv(cli_args.offline_replay)
        except FileNotFoundError:
            print(f"Error: Replay file not found at {cli_args.offline_replay}")
            return
        except KeyError:
            print(f"Error: Replay CSV must contain 'x', 'y', and 'yaw_rad' columns.")
            return

        if not rclpy.ok():
            rclpy.init()
        planner_node = BehaviorPlanner(offline_mode=True)
        output_log_path = "replay_output.jsonl"

        print(f"Processing {len(replay_df)} data points from {cli_args.offline_replay}...")
        with open(output_log_path, "w") as log_file:
            for index, row in replay_df.iterrows():
                planner_node.pose = (row['x'], row['y'])
                planner_node.yaw = row['yaw_rad']
                
                log_data = planner_node.control_loop()
                
                if log_data:
                    log_file.write(json.dumps(log_data) + '\n')
        
        planner_node.stop_all()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"Offline replay complete. Output saved to {output_log_path}")
    else:
        print("Running in LIVE ROS2 mode.")
        rclpy.init(args=args)
        planner_node = BehaviorPlanner()
        try:
            rclpy.spin(planner_node)
        except KeyboardInterrupt:
            pass
        finally:
            planner_node.stop_all()
            planner_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
