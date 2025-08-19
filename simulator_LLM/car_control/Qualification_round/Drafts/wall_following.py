import rclpy
import math
import numpy as np
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import Float32

global lap_count, throttle, steering_angle
throttle = Float32()
steering_angle = Float32()


def get_angle_index(scan, angle):#to get the index of the angle in the scanning range
    index=angle*len(scan.ranges)/((-1 *scan.angle_min + scan.angle_max)*180/np.pi )
    return(int(index))

def lidar_callback(scan):
    global steering_pub, throttle_pub, throttle, steering_angle
    prev_rd=0.0
    prev_ld=0.0
    prev_steering=0.0
    theta= 30

    ang_Neg90_distance=scan.ranges[get_angle_index(scan, -90)]
    ang_Neg60_distance=scan.ranges[get_angle_index(scan,-90+theta)]

    ang_90_distance=scan.ranges[get_angle_index(scan, 90)]
    ang_60_distance=scan.ranges[get_angle_index(scan,90-theta)]
        
    angle_to_Rwall= -1* math.atan((ang_90_distance*math.cos(theta)-ang_60_distance)/(ang_90_distance*math.sin(theta)))+0.5291789535531515
    angle_to_Lwall= -1* math.atan((ang_Neg90_distance*math.cos(theta)-ang_Neg60_distance)/(ang_Neg90_distance*math.sin(theta)))+0.5291789535531515
    D_L=ang_Neg60_distance*math.cos(math.radians(angle_to_Lwall))
    D_R=ang_60_distance*math.cos(math.radians(angle_to_Rwall))

###     
    if math.isinf(ang_Neg90_distance) or math.isinf(ang_Neg60_distance):
        if(prev_ld==0):
            prev_ld=0.6
        steering_angle.data=prev_ld-D_R
        if abs(angle_to_Rwall)>0.1:
            steering_angle.data-=5*angle_to_Rwall
    elif math.isinf(ang_90_distance) or math.isinf(ang_60_distance):
        if(prev_rd==0):
            prev_rd=0.6
        steering_angle.data=D_L-prev_rd
        if abs(angle_to_Lwall)>0.1:
            steering_angle.data+=5*angle_to_Lwall      
    else:
        steering_angle.data=(D_L-D_R)/2
        if abs(angle_to_Lwall)>0.1:
            steering_angle.data+=2.5*angle_to_Lwall
        if abs(angle_to_Rwall)>0.1:
            steering_angle.data-=2.5*angle_to_Rwall
    
    steering_angle.data=max(-1.0, min(1.0, steering_angle.data))
    steering_pub.publish(steering_angle)


    prev_ld=D_L
    prev_rd=D_R
    prev_steering=steering_angle.data

    throttle.data=0.11
    if min(scan.ranges)>=0.6 and abs(angle_to_Rwall)<=0.05 and abs(angle_to_Lwall)<=0.05:
        throttle.data=0.5#1.3

    

    throttle_pub.publish(throttle)





def main(arg = None):
    global node, steering_pub, throttle_pub, throttle, steering_angle

    rclpy.init(args = arg)
    node=rclpy.create_node('PID_wall_following')
    steering_pub= node.create_publisher(Float32, 'autodrive/f1tenth_1/steering_command', 0)
    throttle_pub= node.create_publisher(Float32, 'autodrive/f1tenth_1/throttle_command', 0)
    lidar_sub=node.create_subscription(LaserScan, '/autodrive/f1tenth_1/lidar', lidar_callback, 0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        throttle_pub.publish(Float32(data = 0.0))  # Publish 0 throttle to stop motion
        steering_pub.publish(Float32(data = 0.0))    # Publish 0 steering to stop steering
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
