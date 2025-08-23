import rclpy
import json
import threading
import time
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from lane_loader import load_three_lanes, compute_lane_metrics
from pure_pursuit_controller import PurePursuit
from speed_profiler import target_speed
from perception_yolo_bridge import get_sector_risks
from vlm_advisor import VLMAdvisor
from cost_fusion import select_lane

# TODO: Implement yaw extraction from IMU data
def extract_yaw(msg):
    return 0.0

class BehaviorPlanner(Node):
    def __init__(self):
        super().__init__('behavior_planner')
        # pubs
        self.throttle_pub = self.create_publisher(Float32, "/autodrive/f1tenth_1/throttle_command", 10)
        self.steer_pub    = self.create_publisher(Float32, "/autodrive/f1tenth_1/steering_command", 10)
        # subs
        self.create_subscription(Point, "/autodrive/f1tenth_1/ips", self.cb_pose, 10)
        self.create_subscription(Imu,   "/autodrive/f1tenth_1/imu", self.cb_imu, 10)

        # load lanes & controllers
        # TODO: Load from params.yaml
        lanes = load_three_lanes("CSVs/Centerline_points.csv",
                                 "CSVs/inner_bound_points.csv",
                                 "CSVs/outer_bound_points.csv")
        self.wp = {"center": lanes['center'], "inner": lanes['inner'], "outer": lanes['outer']}
        self.pp = {k: PurePursuit(self.wp[k]) for k in self.wp}  # lookahead per-lane 가능

        # perception + vlm
        self.vlm = VLMAdvisor(hz=8, timeout_s=0.08)  # 비동기 스레드 내부 구동
        self.perception_state = None

        # control loop
        self.timer = self.create_timer(0.04, self.control_loop)  # 25 Hz

        # state
        self.pose = None; self.yaw = None

    def cb_pose(self, msg: Point): self.pose = (msg.x, msg.y)
    def cb_imu(self, msg: Imu): self.yaw = extract_yaw(msg)

    def control_loop(self):
        if self.pose is None or self.yaw is None:
            return

        # 1) Perception
        risks = get_sector_risks()  # dict(left, front, right, timestamp)

        # 2) Lane metrics (free distance, curvature, progress, etc.)
        metrics = compute_lane_metrics(self.pose, self.yaw, self.wp, risks)

        # 3) VLM hint (thread-safe fetch). 실패/타임아웃/낮은 신뢰도면 내부적으로 None 반환
        hint = self.vlm.get_latest_hint()  # {"lane","speed","confidence"} or None

        # 4) Lane selection with hard constraints
        lane = select_lane(metrics, hint)  # default="center" on failure

        # 5) Low-level control
        steer = self.pp[lane].compute_steer(self.pose, self.yaw)  # Pure Pursuit
        v_cmd = target_speed(metrics[lane], risks, hint)          # speed profile

        # 6) Publish
        self.steer_pub.publish(Float32(data=float(steer)))   # [-1,1] 정규화 가정
        self.throttle_pub.publish(Float32(data=float(v_cmd)))# [0,1] or m/s -> sim 스케일에 맞춤

def main():
    rclpy.init()
    rclpy.spin(BehaviorPlanner())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
