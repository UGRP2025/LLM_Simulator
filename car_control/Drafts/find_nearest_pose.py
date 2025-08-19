import rclpy
import sys
import os
import math
import csv
from nav_msgs.msg import Odometry
from std_msgs.msg import Int64
from geometry_msgs.msg import PoseStamped

class NearestPoseIsolator(rclpy.node.Node):
    def __init__(self):
        super().__init__('nearest_pose_isolator')
        self.car_name = sys.argv[1]
        self.trajectory_name = sys.argv[2]
        self.plan = []
        self.min_index_pub = self.create_publisher(Int64, f'/{self.car_name}/purepursuit_control/index_nearest_point', 1)
        self.min_pose_pub = self.create_publisher(PoseStamped, f'/{self.car_name}/purepursuit_control/visualize_nearest_point', 1)
        self.construct_path()
        self.create_subscription(Odometry, f'/{self.car_name}/base/odom', self.odom_callback, 1)

    def construct_path(self):
        file_path = os.path.expanduser(f'~/catkin_ws/src/f1tenth_purepursuit/path/{self.trajectory_name}.csv')
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for waypoint in csv_reader:
                self.plan.append([float(x) for x in waypoint])

    def odom_callback(self, data):
        min_index = Int64()
        curr_x = data.pose.pose.position.x
        curr_y = data.pose.pose.position.y
        min_index.data = self.find_nearest_point(curr_x, curr_y)
        self.min_index_pub.publish(min_index)
        pose = PoseStamped()
        pose.pose.position.x = self.plan[min_index.data][0]
        pose.pose.position.y = self.plan[min_index.data][1]
        self.min_pose_pub.publish(pose)

    def find_nearest_point(self, curr_x, curr_y):
        ranges = [math.sqrt(math.pow(curr_x - point[0], 2) + math.pow(curr_y - point[1], 2)) for point in self.plan]
        return ranges.index(min(ranges))

def main():
    rclpy.init()
    node = NearestPoseIsolator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()