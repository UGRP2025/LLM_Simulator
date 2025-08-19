import rclpy 
import math 
import numpy as np 
from nav_msgs.msg import  OccupancyGrid, Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan

grid_range = 3
grid_resolution = 0.1
grid_size = 12
grid_center = 6
occupancy_grid = np.full((grid_size, grid_size), 0)
clear_obstacles_count = 0
one_D = 0

def Occ_pub():
    global node, occupancy_grid, OC_pub
    occ_grid_msg = OccupancyGrid()
    occ_grid_msg.header.stamp = node.get_clock().now().to_msg()
    occ_grid_msg.header.frame_id = "3mk_Moataz"
    occ_grid_msg.info.resolution = grid_resolution
    occ_grid_msg.info.height = 100
    occ_grid_msg.info.width = 100
    occ_grid_msg.info.origin.position.x = 0.0
    occ_grid_msg.info.origin.position.y = 0.0
    occ_grid_msg.info.origin.position.z = 0.0
    occ_grid_msg.info.origin.orientation.x = 0.0
    occ_grid_msg.info.origin.orientation.y = 0.0
    occ_grid_msg.info.origin.orientation.z = 0.0

    # occ_grid_msg.data = occupancy_grid
    # OC_pub.publish(occ_grid_msg)

def lidar_cb(data):
    global node, clear_obstacles_count, one_D
    x_off = 0.01
    y_off = 0.01
    for i in range(0,271):  
        angle = data.angle_min + i * data.angle_increment 
        x_obstacle =  (data.ranges[i] - x_off) * np.cos(angle)
        y_obstacle =  (data.ranges[i] - y_off) * np.sin(angle)

        col = int((x_obstacle + grid_range) / grid_resolution)
        # row = int((y_obstacle + grid_range) / grid_resolution)

        x_index = int(x_obstacle /grid_resolution)
        y_index = int(y_obstacle /grid_resolution)
        one_D_arr = []
        inflation_radius = 5
        for i in range (-inflation_radius + x_index, inflation_radius + 1 + x_index ):
    
            for j in range (-inflation_radius + y_index, inflation_radius + 1 + y_index):
                if 0 <= i < col and 0 <= j < col:
                    one_D=  j * col + i
                    one_D_arr.append(one_D)

        for index in one_D_arr:
            if occupancy_grid[int(index)] != 1:
                occupancy_grid[int(index)] = 1
                new_obstacles.append(int(index))
    new_obstacles =[]    
    clear_obstacles_count += 1
    if clear_obstacles_count > 6:
        for index in new_obstacles:
            occupancy_grid[index] = 0  

        new_obstacles.clear()
        clear_obstacles_count = 0
    print( "col:",col)
    print( "x_index:",x_index)
    print( "y_index:",y_index)
    print( "occ:",occupancy_grid)
    print( "1_D:",one_D)      
    Occ_pub()





def main(args = None):
    global OC_pub, node
    rclpy.init(args = args )
    node  = rclpy.create_node('grid_map_plotter')
    OC_pub = node.create_publisher(OccupancyGrid, 'Occupancybruh', 1)
    # node.create_subscription(Odometry,'/topic',Odom_read,1)
    node.create_subscription(LaserScan, '/autodrive/f1tenth_1/lidar',lidar_cb, 1)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()