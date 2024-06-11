import os
import csv
import cv2
import rospy
import numpy as np

import tf
import tf2_ros
import geometry_msgs.msg

import pupil_apriltags as apriltag

def rotation_matrix_to_euler_angles(rot_matrix):
    
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

    sin_pitch = -rot_matrix[2, 0]
    cos_pitch = np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    sin_roll = rot_matrix[2, 1] / cos_pitch
    cos_roll = rot_matrix[2, 2] / cos_pitch
    roll = np.arctan2(sin_roll, cos_roll)

    return roll, pitch, yaw

def get_rotation_matrix(roll, pitch, yaw):
    
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx

def rad2deg(rad):
    return rad * 180 / np.pi

def deg2rad(deg):
    return deg * np.pi / 180

def get_camera_pose(img_path, tag_detector, intrinsic, dist, apriltag_id_info):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)
    
    exists_id = apriltag_id_info.keys()
    find_tag_count = 0

    pose_t_sum = np.zeros((3, 1), dtype = np.float32) 
    roll_sum = 0
    pitch_sum = 0
    yaw_sum = 0

    for tag in tags:
        id = tag.tag_id
        if str(id) in exists_id:
            
            find_tag_count += 1
            _, rvec, tvec = cv2.solvePnP(apriltag_id_info[str(id)]['corner_list'], tag.corners, intrinsic, dist, None, None, False, 0)
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            pose_t = rotation_matrix.T @ (-tvec)
            pose_R = rotation_matrix.T

            pose_t_sum += pose_t
            roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_R)
            
            roll_sum += (roll + 2 * np.pi)
            pitch_sum += (pitch + 2 * np.pi)
            yaw_sum += (yaw + 2 * np.pi)
             
    
    pose_t = pose_t_sum / find_tag_count
    roll = roll_sum / find_tag_count
    pitch = pitch_sum / find_tag_count
    yaw = yaw_sum / find_tag_count
            
    pose_R = get_rotation_matrix(roll, pitch, yaw)        

    return pose_t, pose_R

def rotation_matrix_to_euler_angles(rot_matrix):
    
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

    sin_pitch = -rot_matrix[2, 0]
    cos_pitch = np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    sin_roll = rot_matrix[2, 1] / cos_pitch
    cos_roll = rot_matrix[2, 2] / cos_pitch
    roll = np.arctan2(sin_roll, cos_roll)

    return roll, pitch, yaw

def put_apriltag_origin(world_x, world_y, world_z, name, broadcaster):
    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "map"
    static_transform.child_frame_id  = name
    static_transform.transform.translation.x = float(world_x)
    static_transform.transform.translation.y = float(world_y)
    static_transform.transform.translation.z = float(world_z)
    roll, pitch, yaw = [deg2rad(-90), deg2rad(0), deg2rad(90)]
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transform.transform.rotation.x = quat[0]
    static_transform.transform.rotation.y = quat[1]
    static_transform.transform.rotation.z = quat[2]
    static_transform.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transform)
    rospy.loginfo("Publish " + name + " coordinate")
    rospy.sleep(1)

def put_groundtruth(x, y, z, roll, pitch, yaw, name, broadcaster):
    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "map"
    static_transform.child_frame_id  = name
    static_transform.transform.translation.x = float(x)
    static_transform.transform.translation.y = float(y)
    static_transform.transform.translation.z = float(z)
    roll, pitch, yaw = [deg2rad(-90) + roll, deg2rad(0) + pitch, deg2rad(90) + yaw]
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transform.transform.rotation.x = quat[0]
    static_transform.transform.rotation.y = quat[1]
    static_transform.transform.rotation.z = quat[2]
    static_transform.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transform)
    rospy.loginfo("Publish " + name)
    rospy.sleep(1)

def put_camera_pose(pose_t, pose_R, name, broadcaster):
    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "map"
    static_transform.child_frame_id  = name
    static_transform.transform.translation.x = pose_t[0]
    static_transform.transform.translation.y = pose_t[1]
    static_transform.transform.translation.z = pose_t[2]
    roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_R)
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transform.transform.rotation.x = quat[0]
    static_transform.transform.rotation.y = quat[1]
    static_transform.transform.rotation.z = quat[2]
    static_transform.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transform)
    rospy.loginfo("Publish " + name)
    rospy.sleep(1)

def read_csv(csv_file_path):
    data_list = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            data_list.append(list(map(float, row)))
    return data_list

def get_apriltag_corners(world_x, world_y, world_z, tag_size):

    return np.array([[world_x, world_y - tag_size / 2, world_z - tag_size / 2],  # First point
                     [world_x, world_y + tag_size / 2, world_z - tag_size / 2],  # Second point
                     [world_x, world_y + tag_size / 2, world_z + tag_size / 2],  # Third point
                     [world_x, world_y - tag_size / 2, world_z + tag_size / 2]], # Fourth point
                     dtype = np.float32)

if __name__ == "__main__":

    # Apriltag parameter 
    APRILTAG_SIZE = 0.13
    APRILTAG_LOCATION_PATH = "../apriltag_id_3_4.csv"
    
    apriltag_id_info = {}
    apriltag_location_list = read_csv(APRILTAG_LOCATION_PATH)
    for location in apriltag_location_list:
        id, x, y, z = location
        apriltag_id_info[str(int(id))] = {}
        apriltag_id_info[str(int(id))]['location'] = (x, y, z)
        apriltag_id_info[str(int(id))]['corner_list'] = get_apriltag_corners(world_x = x, world_y = y, world_z = z, tag_size = APRILTAG_SIZE)

    # Camera parameters
    FX = 1791 
    FY = 1788
    CX = 882
    CY = 1164

    DIST_COEFF = np.array([0.28147, -1.66096, -0.00401858, -0.00098643, 3.327])
    
    INTRINSIC = np.array([[ FX,  0, CX],
                          [  0, FY, CY],
                          [  0,  0,  1]])
    

    # Get Benchmark
    bench_mark_path = "../benchmark.csv"
    bench_mark_pose_list = read_csv(bench_mark_path)

    # Ros init
    rospy.init_node('lab1')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    at_detector = apriltag.Detector(families = 'tag36h11')

    # Publish Apriltag location
    for tag_id in apriltag_id_info.keys():
        x, y, z = apriltag_id_info[tag_id]['location']
        put_apriltag_origin(world_x = x, world_y = y, world_z = z, name = "tag" + tag_id, broadcaster = broadcaster)
    

    # Publish Benchmark pose
    for i, bench_mark_pose in enumerate(bench_mark_pose_list):
        x, y, z, _, _, yaw = bench_mark_pose
        put_groundtruth(x = x, y = y, z = z, roll = 0, pitch = 0, yaw = yaw, name = 'benchmark_' + str(i + 1), broadcaster = broadcaster)
    

    # Get camera pose from image
    Image_folder = "./image1"
    image_name_list = os.listdir(Image_folder)

    at_detector = apriltag.Detector(families = 'tag36h11')
    content = []
    for  i, image_name in enumerate(image_name_list):
        img_path = os.path.join(Image_folder, image_name)
        pose_t, pose_R = get_camera_pose(img_path, at_detector, INTRINSIC, DIST_COEFF, apriltag_id_info)
        put_camera_pose(pose_t, pose_R, image_name, broadcaster)
        
        x, y, z = pose_t.squeeze()
        roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_R)
        content.append((x, y, z, roll, pitch, yaw))


    ## Write the result to csv
    Output_csv_file = "object3.csv"   
    header = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    with open(Output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  
        writer.writerows(content) 

    rospy.loginfo(f'CSV file saved to {Output_csv_file}')
    rospy.sleep(10)
