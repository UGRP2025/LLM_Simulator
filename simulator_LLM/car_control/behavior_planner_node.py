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
import json

# Import all the implemented modules
from car_control.lane_loader import load_three_lanes
from car_control.pure_pursuit_controller import PurePursuit
from car_control.speed_profiler import target_speed, speed_to_throttle
from car_control.perception_yolo_bridge import get_sector_risks, get_latest_detections
from car_control.vlm_advisor import VLMAdvisor
from car_control.cost_fusion import select_lane, DEFAULT_PARAMS
from car_control.safety import mask_unsafe_lanes

# Placeholder for a new module to compute metrics
# In a real scenario, this would be a proper module
def compute_lane_metrics(lanes, pose, yaw, risks):
    """
    Placeholder function to compute metrics for each lane.
    This version provides more sensible defaults and reacts to risks.
    """
    metrics = {}
    base_free_distance = 50.0  # A long, clear road ahead

    # Base metrics for each lane - center is usually straightest and preferred
    base_metrics_map = {
        'center': {'curvature': 0.05, 'progress': 10.0},
        'inner':  {'curvature': 0.2, 'progress': 9.5},
        'outer':  {'curvature': 0.2, 'progress': 9.5}
    }

    for name, base_vals in base_metrics_map.items():
        metrics[name] = base_vals.copy()
        # Start with a high default free distance
        free_dist = base_free_distance

        # Reduce free distance based on sector risks from perception
        # A risk score of 0.5 is a moderate risk
        if name == 'center' and risks.get('front', 0.0) > 0.1:
            # Penalize front risk more heavily
            free_dist /= (1 + risks['front'] * 2)
        if name == 'inner' and risks.get('left', 0.0) > 0.1:
            free_dist /= (1 + risks['left'])
        if name == 'outer' and risks.get('right', 0.0) > 0.1:
            free_dist /= (1 + risks['right'])

        metrics[name]['free_distance'] = free_dist

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

        # TODO: Load parameters from a YAML file
        self.params = {
            'pp_params': { 'wheelbase': 0.33, 'lookahead_min': 1.0, 'lookahead_max': 4.5, 'max_steering_angle_deg': 30.0, 'lookahead_speed_gain': 0.35, 'lookahead_curvature_gain': 0.1 },
            'vmax': 6.0,
            'safety_thresholds': {'min_free_distance': 2.0},
            'cost_fusion_params': DEFAULT_PARAMS, # from cost_fusion
        }

        # Load lanes and instantiate controllers
        # TODO: Make file paths configurable
        self.lanes = load_three_lanes(
            "car_control/CSVs/Centerline_points.csv",
            "car_control/CSVs/inner_bound_points.csv",
            "car_control/CSVs/outer_bound_points.csv"
        )
        lanes_dict = {
            'center': self.lanes.center,
            'inner': self.lanes.inner,
            'outer': self.lanes.outer
        }
        self.pp_controllers = {name: PurePursuit(lane.waypoints, self.params['pp_params']) for name, lane in lanes_dict.items()}

        # VLM Advisor
        self.vlm = VLMAdvisor(hz=8, timeout_s=0.08, conf_thresh=0.6)
        self.vlm.start()

        # State variables
        self.pose = None
        self.yaw = None
        self.current_speed = 3.0 # TODO: Estimate from pose or get from another topic
        self.current_lane = 'center'

        if not self.offline_mode:
            # Publishers
            self.throttle_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
            self.steer_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
            # Subscribers
            self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)
            self.create_subscription(Imu, "/autodrive/f1tenth_1/imu", self.cb_imu, 10)
            # Control loop timer
            self.timer = self.create_timer(0.04, self.control_loop) # 25 Hz

    def cb_pose(self, msg: Point): self.pose = (msg.x, msg.y)
    def cb_imu(self, msg: Imu): self.yaw = extract_yaw_from_quaternion(msg.orientation)

    def control_loop(self):
        if self.pose is None or self.yaw is None:
            self.get_logger().warn("Waiting for pose and yaw data...")
            return

        # 1. Perception
        detections = get_latest_detections() # Mocked
        risks = get_sector_risks(detections)

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
