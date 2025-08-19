import rclpy
import csv
from geometry_msgs.msg import Point
import std_msgs.msg 
import time


flag = 0
start = time.time()

def position(POSE):
    global X , Y, Dict, Timer
    msg = POSE
    X = msg.x
    Y = msg.y
    Dict = {"positions_X": X, "positions_y": Y, "Time_Sec": (time.time()- start)}
    CSV_SAVE()

def main(args = None):
    rclpy.init(args = args)
    node = rclpy.create_node("CSV_maker")
    Subscriber = node.create_subscription(Point, '/autodrive/f1tenth_1/ips', position ,10)
    while rclpy.ok ():
        rclpy .spin(node)
    node.destroy_timer(Timer)
    node.destroy_node()
    rclpy.shutdown()



def CSV_SAVE():
    global flag
    with open("/home/autodrive_devkit/src/car_control/car_control/Test.csv", mode="a") as csvfile:
        fieldnames = ["positions_X", "positions_y", "Time_Sec"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        if flag == 0:
            writer.writeheader()
            flag = 1
        writer.writerow(Dict)


if __name__ == '__main__':
    main()
    while rclpy.ok ():
        rclpy .spin(node)
    node.destroy_timer(Timer)
    node.destroy_node()
    rclpy.shutdown()