import rclpy
import math
import tf
import sys
import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PoseStamped, Pose
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64

class PurePursuitController(rclpy.node.Node):
    def __init__(self):
        super().__init__('vehicle_control_node')
        self.car_name = sys.argv[1]
        self.use_ackermann_model = sys.argv[2] == 'true'
        self.adaptive_lookahead = sys.argv[3] == 'true'
        self.scale_vel_no_adaptive_lookahead = float(sys.argv[4])

        self.ang_goal_x = 0.0
        self.ang_goal_y = 0.0
        self.vel_goal_x = 0.0
        self.vel_goal_y = 0.0
        self.lookahead_state = 'caution'

        self.command_pub = self.create_publisher(AckermannDrive, f'/{self.car_name}/command', 1)
        self.deviation_pub = self.create_publisher(Float64, f'/{self.car_name}/deviation', 1)

        self.create_subscription(Odometry, f'/{self.car_name}/base/odom', self.vehicle_control_node, 1)
        if self.adaptive_lookahead:
            self.create_subscription(String, f'/{self.car_name}/purepursuit_control/adaptive_lookahead', self.dist_callback, 1)
        self.create_subscription(PoseStamped, f'/{self.car_name}/purepursuit_control/ang_goal', self.ang_pose_callback, 1)
        self.create_subscription(PoseStamped, f'/{self.car_name}/purepursuit_control/vel_goal', self.vel_pose_callback, 1)

    def vehicle_control_node(self, data):
        command = AckermannDrive()
        log_dev = Float64()

        curr_x = data.pose.pose.position.x
        curr_y = data.pose.pose.position.y
        heading = tf.transformations.euler_from_quaternion((data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w))[2]

        if self.use_ackermann_model:
            front_axle_x = (WHEELBASE_LEN * math.cos(heading)) + curr_x
            front_axle_y = (WHEELBASE_LEN * math.sin(heading)) + curr_y
            curr_x = front_axle_x
            curr_y = front_axle_y

        # Rest of the vehicle_control_node logic

        self.command_pub.publish(command)
        self.deviation_pub.publish(log_dev)

    def ang_pose_callback(self, data):
        self.ang_goal_x = data.pose.position.x
        self.ang_goal_y = data.pose.position.y

    def vel_pose_callback(self, data):
        self.vel_goal_x = data.pose.position.x
        self.vel_goal_y = data.pose.position.y

    def dist_callback(self, data):
        self.lookahead_state = data.data

def main():
    rclpy.init()
    node = PurePursuitController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()