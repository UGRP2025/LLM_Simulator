#!/usr/bin/env python3



import rclpy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
import matplotlib.pyplot as plt 
import time 


node=None
thrtl=0.05
close_threshold=1
margin=100
gap_counter=0
max_gap_width=0
steer_value=0
gap_end_disparity_index=0


def plot_variable(values):
    """
    Plots the behavior of a variable.
    
    Args:
        values (list or array): The values of the variable to be plotted.
    """
    plt.figure()
    plt.plot(values, marker='o')
    plt.title("Variable Behavior")
    plt.xlabel("disparity_Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


def lidar_callback (msg):
    global node,gap_counter,max_gap_width,steer_value,margin,gap_end_disparity_index,close_threshold
#    start_time = time.time()

    angle_min=msg.angle_min
    angle_max=msg.angle_max
    angle_increment=msg.angle_increment
    lidar_ranges=msg.ranges


    control_ranges = np.array(lidar_ranges[180:901]) 
    start_index = 180
    control_ranges = np.nan_to_num(control_ranges, nan=10.0, posinf=10.0)
    control_ranges = np.clip(control_ranges, 0.0, 7.0)   
    right_car = np.array(lidar_ranges[:181])
    left_car = np.array(lidar_ranges[900:])


    if abs(np.min(right_car)) < 0.2:
        return
        steer_value = 0.2  # Turn left if too close to right wall
        node.get_logger().info("getting away RIGHT WALL")
    elif abs(np.min(left_car)) < 0.4:
        return
        steer_value = -0.2  # Turn right if too close to left wall
        node.get_logger().info("etting away LEFT WALL")

    else:
        car_width = 0.5
        chord_length = car_width *.75  #for tolerance


        differences = (np.diff(control_ranges))
        positive_differences = np.abs(differences)
        disparity_index = np.argmax(positive_differences)  

        ratio = chord_length / (2.0 * np.min([control_ranges[disparity_index], control_ranges[disparity_index+1]]))
        ratio = np.clip(ratio, -1, 1)  # Ensure the ratio is between -1 and 1
        arc_degrees = 2* np.arcsin(ratio)
        no_disparity_index = int(arc_degrees/angle_increment)+1
        node.get_logger().info(f'control range = {control_ranges[disparity_index]} , {control_ranges[disparity_index+1]}')
        no_disparity_index = int(arc_degrees/angle_increment)+1

        
        if control_ranges[disparity_index] - control_ranges[disparity_index+1] > 0:
        # Fill backward
            node.get_logger().info(f'gap is right')    
            start_idx = max(0, disparity_index - no_disparity_index)
            control_ranges[start_idx:disparity_index] = control_ranges[disparity_index+1]

        elif control_ranges[disparity_index] - control_ranges[disparity_index+1] < 0:
        # Fill Forward
            node.get_logger().info(f'gap is Left')    
            end_idx = min(len(control_ranges), disparity_index + no_disparity_index + 1)
            control_ranges[(disparity_index+1):end_idx] = control_ranges[disparity_index]


        differences = np.abs(np.diff(control_ranges))
        gap_index = np.argmax(differences)+1


        # Normal steering calculation if not too close to walls
        #if 355 <= disparity_index <= 365: 
        #    return 
        #    steer_value = 0  # Go straight if disparity is in middle
        #    node.get_logger().info("NO Need STEERING")
        if True:
            control_angle_min = -90  # in degrees
            gap_angle = control_angle_min + (gap_index) * np.degrees(angle_increment)
            gap_angle = np.radians(gap_angle)
            max_steer_angle = np.radians(100)
            steer_value = np.clip(gap_angle / max_steer_angle, -1, 1)
            if abs(gap_angle) < 0.1:  # Tolerance for centered gaps
                steer_value = 0
            node.get_logger().info(f'disparity_Index value is: {disparity_index} but Gap indes is {gap_index}')
            node.get_logger().info(f'Gap disparity_Index: {gap_index}, Gap Angle: {gap_angle*180/np.pi:.2f} degrees, Steer Value: {steer_value*30:.2f} ')



#    end_time = time.time()
#    processing_time = end_time - start_time
#   node.get_logger().info(f'Lidar callback processing time: {processing_time:.6f} seconds') takes almost .0015 sec


    
def main(args=None):
    global node
    rclpy.init(args=args)

    node=rclpy.create_node('wall_follower')
    node.create_subscription(LaserScan, '/autodrive/f1tenth_1/lidar', lidar_callback , 10)
    #node.create_subscription(Imu, '/autodrive/f1tenth_1/imu', imu_callback , 10)
    # node.create_subscription(Float32, '/autodrive/f1tenth_1/lap_time', time_callback , 10)
    
    throttling = node.create_publisher(Float32,'/autodrive/f1tenth_1/throttle_command', 10)
    steering= node.create_publisher(Float32,'/autodrive/f1tenth_1/steering_command', 10)
    throttle=Float32()
    steer=Float32()

    def timer_callback ():
        global thrtl,steer_value
        if abs(steer_value) <= 10/30 :
            throttle.data = (1.5/30)
        elif abs(steer_value) >10/30 and abs(steer_value) <= 20/30:
            throttle.data = (1/30)
        else:
            throttle.data = (.5/30) 
        
        throttle.data = .05 #testing 
        steer.data= float(steer_value)
        throttling.publish(throttle)
        steering.publish(steer)

    timer = node.create_timer(0.02, timer_callback)    # 50 Hz

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        throttling.publish(Float32(data = 0.0))  # Publish 0 throttle to stop motion
        steering.publish(Float32(data = 0.0))    # Publish 0 steering to stop steering
    finally:
        node.destroy_timer(timer)
        node.destroy_node()
        rclpy.shutdown()

    
   
    
if __name__=='__main__':
    main()

