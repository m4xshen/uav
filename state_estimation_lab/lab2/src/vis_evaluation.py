#!/usr/bin/env python
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
import csv

def create_pose(x, y, z, yaw):
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose

def read_csv(csv_file_path):
    data_list = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data_list.append(list(map(float, row[:3] + row[5:6])))
    return data_list

if __name__ == "__main__":

    # Path to the CSV files    
    BENCHMARK_CSV_PATH = "../benchmark/benchmark_track2.csv"
    KF_CSV_PATH = "../result/Track2.csv"

    benchmark_data = read_csv(BENCHMARK_CSV_PATH)
    kf_data = read_csv(KF_CSV_PATH)

    rospy.init_node('path_publisher')

    # Publisher for the Path
    benchmark_path_pub = rospy.Publisher('/uav_benchmark_path', Path, queue_size=10)
    kf_path_pub = rospy.Publisher('/uav_kf_path', Path, queue_size=10)

    # Create the Path message
    benchmark_path = Path()
    benchmark_path.header.frame_id = "map"
    kf_path = Path()
    kf_path.header.frame_id = "map"

    rate = rospy.Rate(1)  # 1 Hz

    for data in benchmark_data:
        x = data[0]
        y = data[1]
        z = data[2]
        yaw = data[3]

        # Create the new pose
        new_pose = create_pose(x, y, z, yaw)

        # Add the new pose to the path
        benchmark_path.poses.append(new_pose)

    benchmark_path.header.stamp = rospy.Time.now()

    rospy.sleep(1)
    benchmark_path_pub.publish(benchmark_path)
    rospy.sleep(1)

    benchmark_path.poses = []
    for data in kf_data:
        x = data[0]
        y = data[1]
        z = data[2]
        yaw = data[3]

        # Create the new pose
        new_pose = create_pose(x, y, z, yaw)

        # Add the new pose to the path
        benchmark_path.poses.append(new_pose)

        benchmark_path.header.stamp = rospy.Time.now()
        kf_path_pub.publish(benchmark_path)
        rospy.sleep(0.05)
        if rospy.is_shutdown():
            break
