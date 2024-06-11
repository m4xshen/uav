# Python
import csv
import cv2
import numpy as np

import rospy
import tf
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

def create_pose_stamped(tvec, roll, pitch, yaw):

    pose_stamped = PoseStamped()

    pose_stamped.header.stamp = rospy.Time.now()
    pose_stamped.header.frame_id = "camera"

    pose_stamped.pose.position.x = tvec[0]
    pose_stamped.pose.position.y = tvec[1]
    pose_stamped.pose.position.z = tvec[2]

    quaternion = quaternion_from_euler(roll, pitch, yaw)

    pose_stamped.pose.orientation.x = quaternion[0]
    pose_stamped.pose.orientation.y = quaternion[1]
    pose_stamped.pose.orientation.z = quaternion[2]
    pose_stamped.pose.orientation.w = quaternion[3]

    return pose_stamped

def sorted_corner(points):

    average_y = np.average(points[:, 0])
    average_x = np.average(points[:, 1])
    
    first_idx = 0
    second_idx = 0
    third_idx = 0
    fourth_idx = 0

    for i, point in enumerate(points):
        
        y, x = point
        if  y < average_y and x > average_x:
            first_idx = i
        elif y > average_y and x > average_x:
            second_idx = i
        elif y > average_y and x < average_x:
            third_idx = i
        else:
            fourth_idx = i

    rearranged_points = np.array([
        points[first_idx],
        points[second_idx],
        points[third_idx],
        points[fourth_idx]
    ])

    return rearranged_points


def get_apriltag_corners(tag_size):
    tag_half_length = tag_size / 2
    return np.array([[-tag_half_length,  tag_half_length, 0],
                    [  tag_half_length,  tag_half_length, 0],
                    [  tag_half_length, -tag_half_length, 0],
                    [ -tag_half_length, -tag_half_length, 0]], dtype = np.float32)

def rotation_matrix_to_euler_angles(rot_matrix):
    
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

    sin_pitch = -rot_matrix[2, 0]
    cos_pitch = np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    sin_roll = rot_matrix[2, 1] / cos_pitch
    cos_roll = rot_matrix[2, 2] / cos_pitch
    roll = np.arctan2(sin_roll, cos_roll)

    return roll, pitch, yaw

def put_camera_origin():
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "map"
    static_transform.child_frame_id  = "camera"
    static_transform.transform.translation.x = 0
    static_transform.transform.translation.y = 0
    static_transform.transform.translation.z = 2
    roll, pitch, yaw = [np.deg2rad(0), np.deg2rad(90), np.deg2rad(0)]
    quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    static_transform.transform.rotation.x = quat[0]
    static_transform.transform.rotation.y = quat[1]
    static_transform.transform.rotation.z = quat[2]
    static_transform.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transform)
    rospy.loginfo("Publish camera coordinate")
    rospy.sleep(1)

def read_csv_to_list(file_path):

    data_list = []
    frame_information = []

    with open(file_path, mode = 'r') as csv_file:

        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        for row in csv_reader:
            frame_id, _, track_id, x1, y1, x2, y2, x3, y3, x4, y4 = row
            data_list.append((frame_id, track_id, x1, y1, x2, y2, x3, y3, x4, y4))
            frame_information.append(int(frame_id))

    frame_information = sorted(list(set(frame_information)))
    return frame_information, data_list

def output_frame_information(target_frame_id, data_list):

    FX = 1729.5132050527152
    FY = 1748.0612914520668
    CX = 962.9786159834048
    CY = 591.4451729256456
    
    intrinsic = np.array([[ FX,  0, CX],
                          [  0, FY, CY],
                          [  0,  0,  1]])
    
    dist = np.array([ 0.40408531, -1.78957082,  0.02626639,  0.01158871,  4.15482281])

    object_list = []
    pose_list = []

    for data in data_list:

        frame_id, track_id, x1, y1, x2, y2, x3, y3, x4, y4 = data

        if str(frame_id) == str(target_frame_id):

            object_list.append(track_id)
            corners = np.array([[x1, y1],
                                [x2, y2],
                                [x3, y3],
                                [x4, y4]], dtype = np.float32)
            
            corners = sorted_corner(corners)

            ## Calculate PnP
            world_corners = get_apriltag_corners(0.19)

            _, rvec, tvec = cv2.solvePnP(world_corners, corners, intrinsic, dist, None, None, False, 0)
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

            pose = create_pose_stamped(tvec, roll, pitch, yaw)

            pose_list.append(pose)

    return object_list, pose_list

if __name__ == "__main__":

    # Ros init
    rospy.init_node('lab3')
    put_camera_origin()
    ID_PATH_MAPPING = {}

    # TODO: CSV setting
    csv_path = 
    frame_info, data_list = read_csv_to_list(csv_path)
    rate = rospy.Rate(3)
    
    for frame_id in frame_info:

        object_list, pose_list = output_frame_information(frame_id, data_list)

        if len(object_list) != 0 and len(pose_list) != 0:
            
            for id, pose in zip(object_list, pose_list):
            
                if str(id) not in ID_PATH_MAPPING.keys():
                    rospy.loginfo("Create new id.")
                    ID_PATH_MAPPING[str(id)] = {'Path': Path(), 'Path_Publisher': rospy.Publisher('/id_' + str(id) + 'path', Path, queue_size=10), 'Pose_Publisher': rospy.Publisher('/id_' + str(id) + 'pose', PoseStamped, queue_size=10), 'exists': True}
                    ID_PATH_MAPPING[str(id)]['Path'].header.frame_id = "camera"
                    ID_PATH_MAPPING[str(id)]['Path'].poses.append(pose)
                else:
                    ID_PATH_MAPPING[str(id)]['Path'].poses.append(pose)
                    ID_PATH_MAPPING[str(id)]['exists'] = True


            # Make sure the pose contains only 10 steps
            for id in ID_PATH_MAPPING.keys():

                path = ID_PATH_MAPPING[id]['Path']
                if len(path.poses) > 30 or ID_PATH_MAPPING[id]['exists'] == False:
                    #rospy.loginfo("Pop path")
                    path.poses.pop(0)

            # Publish Path and pose
            for id in ID_PATH_MAPPING.keys():

                path = ID_PATH_MAPPING[id]['Path']
                path_publisher = ID_PATH_MAPPING[id]['Path_Publisher']
                pose_publisher = ID_PATH_MAPPING[id]['Pose_Publisher']
                
                if ID_PATH_MAPPING[id]['exists'] == True:

                    # Publish pose
                    pose = path.poses[-1]
                    pose.header.stamp = rospy.Time.now()
                    rospy.sleep(0.1)
                    pose_publisher.publish(pose)
                    #rospy.loginfo("Publish pose")

                    # Publish path
                    path.header.stamp = rospy.Time.now()
                    rospy.sleep(0.1)
                    path_publisher.publish(path)
                    #rospy.loginfo("Publish path")

                else:
                    
                    pose = create_pose_stamped([-1000, -1000, -1000], 0, 0, 0)
                    pose.header.stamp = rospy.Time.now()
                    rospy.sleep(0.1)
                    pose_publisher.publish(pose)

                ID_PATH_MAPPING[id]['exists'] = False

            rospy.loginfo("Publish data")
            rate.sleep()
        
        if rospy.is_shutdown():
            break

    # Reset
    for id in ID_PATH_MAPPING.keys():

        path = ID_PATH_MAPPING[id]['Path']
        path_publisher = ID_PATH_MAPPING[id]['Path_Publisher']
        pose_publisher = ID_PATH_MAPPING[id]['Pose_Publisher']
        path.poses = []
            
        path.header.stamp = rospy.Time.now()
        rospy.sleep(0.1)
        path_publisher.publish(path)

        pose = create_pose_stamped([-1000, -1000, -1000], 0, 0, 0)
        pose.header.stamp = rospy.Time.now()
        rospy.sleep(0.1)
        pose_publisher.publish(pose)


        