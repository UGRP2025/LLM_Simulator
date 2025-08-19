import rclpy
import math
from rclpy import time
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

kp = 0.6 
ki = 0.00001
kd = 0.3
desired = 0.5
previous_time = 0
previous_error = 0
integral = 0
indicator = 1
rclpy.init()

#creating node
wall_follow = rclpy.create_node("wall_following")

#publish
speed = wall_follow.create_publisher (Float32, '/autodrive/f1tenth_1/throttle_command', 1 )
steer = wall_follow.create_publisher (Float32, '/autodrive/f1tenth_1/steering_command', 1 )

def getRange(data, angle):
    angle_r = angle * (math.pi / 180)
    index = int((angle_r - data.angle_min)/data.angle_increment )
    return data.ranges[index]

def callback(data):
    global msg, previous_time, previous_error, integral

    #calculcating Delta t
    # current_time = wall_follow.get_clock().now().nanoseconds * 1e-9
    # delta_time = current_time - previous_time
    
    # print('delta time :', delta_time)
    #finding a and b
    theta = 45
    thetaL = 45
    a = getRange(data,theta)
    b = getRange(data,90)
    theta_in_radians = math.radians(theta)
    thetaL_in_radians = math.radians(thetaL)
    aL = getRange(data,-thetaL)
    bL = getRange(data,-90)
    
    print ("a is :", a)
    print ("b is :", b)
    print ("al is :", aL)
    print ("bl is :", bL)


    if a > 3 :
        a = 3
    if b > 3 :
        b = 3
    if aL > 3 :
        aL = 3
    if bL > 3 :
        bL = 3

    


    #calcualting AB, CD, and alpha
    alpha = math.atan((a * math.cos(theta_in_radians) - b) / a * math.cos(theta_in_radians))
    AB = b * math.cos(alpha)
    CD = AB + 1.1 * math.sin(alpha)

    alphaL = math.atan((aL * math.cos(thetaL_in_radians) - bL) / aL * math.cos(thetaL_in_radians))
    ABL = bL * math.cos(alpha)
    CDL = ABL + 1.1 * math.sin(alphaL)



    print('AB is !:', AB)
    print('CD is :', CD)
    print('alpha is :', alpha)
    print('AB_left is :', ABL)
    print('CD_Left is :', CDL)
    print('alpha is :', alphaL)


    #PID
    error= CD -  CDL 
    integral += error 

    # if integral > 20:
    #     integral = 20
    # elif integral < -20:
    #     integral = -20 

    delta_error = error - previous_error
    PID_output = kp * error + kd * (delta_error) + ki * integral
    previous_error = error 
    steering = float(PID_output)

    print('proportional is :', error)
    print('integral is :', integral)
    print('delta error is :', delta_error)

    if steering > 1.0:
        steering = 1.0
    elif steering < -1.0:
        steering = -1.0


    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = wall_follow.get_clock().now().to_msg()
    drive_msg.header.frame_id = "laser"
    drive_msg.drive.steering_angle = steering 
    drive_msg.drive.speed = 0.03
    
    seer =  Float32()
    seer.data = steering
    
    velo = Float32()
    velo.data = 0.03

    #publishing
    speed.publish()
    steer.publish()


#subscribe
wall_follow.create_subscription(LaserScan,'/autodrive/f1tenth_1/lidar', callback, 1)


rclpy.spin(wall_follow)
node.destroy_node()
rclpy.shutdown()