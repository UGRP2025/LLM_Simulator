import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

import numpy as np
import argparse
import time
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
    """Placeholder function to compute metrics for each lane."""
    metrics = {}
    lanes_dict = {
        'center': lanes.center,
        'inner': lanes.inner,
        'outer': lanes.outer
    }
    for name, lane in lanes_dict.items():
        # Mocked metrics for demonstration
        metrics[name] = {
            'free_distance': 30.0 - len(name) * 5, # e.g., center=20, inner=25, outer=15
            'curvature': 0.1 + len(name) * 0.05,
            'progress': 10.0
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

        # 7. Publish
        if not self.offline_mode:
            self.steer_pub.publish(Float32(data=float(steer)))
            self.throttle_pub.publish(Float32(data=float(throttle)))
        else:
            # In offline mode, we can log the output
            log_data = {
                'timestamp': time.time(), 'pose': self.pose, 'yaw': self.yaw,
                'chosen_lane': self.current_lane, 'steer': steer, 'throttle': throttle,
                'vlm_hint': hint, 'risks': risks
            }
            print(json.dumps(log_data))

    def stop_all(self):
        self.vlm.stop()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline_replay', type=str, help='Path to CSV for offline replay')
    # In a ROS2 launch file, args can be passed differently. For standalone, this works.
    cli_args, _ = parser.parse_known_args()

    if cli_args.offline_replay:
        print("Running in OFFLINE REPLAY mode.")
        # TODO: Implement offline replay logic using pandas to read the CSV
        # and call the control_loop manually.
        print(f"Offline replay from {cli_args.offline_replay} is not fully implemented.")
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
