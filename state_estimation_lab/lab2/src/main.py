# Python
import os
import csv
import cv2
import numpy as np

# ROS
import rospy
import tf
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

# Other
from UAV import *
from BuildMap import *
from KalmanFilter import *
import pupil_apriltags as apriltag

def read_data(csv_file_path):
    data_list = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
        for row in csv_reader:
            time, control, camera = row
            data_list.append([float(time), control, camera])
    return data_list

def normalize_angle(angle):

    normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return normalized_angle

def rotation_matrix_to_euler_angles(rot_matrix):
    
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

    sin_pitch = -rot_matrix[2, 0]
    cos_pitch = np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    sin_roll = rot_matrix[2, 1] / cos_pitch
    cos_roll = rot_matrix[2, 2] / cos_pitch
    roll = np.arctan2(sin_roll, cos_roll)

    return roll, pitch, yaw

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

def get_uav_pose(img_path, tag_detector, intrinsic, dist, apriltag_id_info):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = tag_detector.detect(gray)
    
    exists_id = apriltag_id_info.keys()

    find_tag_count = 0

    pose_t_sum = np.zeros((3, 1), dtype = np.float32) 
    roll_sum = 0
    pitch_sum = 0
    yaw_sum = 0

    uav_cam_R = np.array([[ 0, 1, 0],
                          [ 0, 0,-1],
                          [ 1, 0, 0]], dtype = np.float32)
    
    uav_state = None

    if len(tags) != 0:
        
        for tag in tags:

            if tag.decision_margin > 40 and str(int(tag.tag_id)) in exists_id:

                find_tag_count += 1
                corners = sorted_corner(tag.corners)
                
                cv2.circle(img, tuple(corners[0].astype(int)), 4, (0, 0, 255, 0), 2)
                cv2.circle(img, tuple(corners[1].astype(int)), 4, (0, 255, 0, 0), 2)
                cv2.circle(img, tuple(corners[2].astype(int)), 4, (255, 0,   0, 0), 2)
                cv2.circle(img, tuple(corners[3].astype(int)), 4, (0, 255, 255, 0), 2)
                
                _, rvec, tvec = cv2.solvePnP(apriltag_id_info[str(int(tag.tag_id))]['corner_list'], corners, intrinsic, dist, None, None, False, 0)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                pose_t = rotation_matrix.T @ (-tvec)
                pose_R = rotation_matrix.T @ uav_cam_R

                pose_t_sum += pose_t
                roll, pitch, yaw = rotation_matrix_to_euler_angles(pose_R)
                
                roll_sum += ((roll + 2 * np.pi) % (2 * np.pi))
                pitch_sum += ((pitch + 2 * np.pi) % (2 * np.pi))
                yaw_sum += ((yaw + 2 * np.pi) % (2 * np.pi))
        
        if find_tag_count != 0:
            pose_t = pose_t_sum / find_tag_count
            roll = roll_sum / find_tag_count
            pitch = pitch_sum / find_tag_count
            yaw = yaw_sum / find_tag_count

            roll = normalize_angle(roll)
            pitch = normalize_angle(pitch)
            yaw = normalize_angle(yaw)

            uav_state = np.array([pose_t[0,0], pose_t[1,0], pose_t[2,0], yaw], dtype = np.float32)

    return uav_state, img

def publish_pose(publisher, uav_state, uav_state_variance):

    pose_with_cov = PoseWithCovarianceStamped()
    pose_with_cov.header.frame_id = "map"
    pose_with_cov.header.stamp = rospy.Time.now()

    pose_with_cov.pose.pose.position.x = uav_state[0]
    pose_with_cov.pose.pose.position.y = uav_state[1]
    pose_with_cov.pose.pose.position.z = uav_state[2]

    ori = pose_with_cov.pose.pose.orientation
    ori.x, ori.y, ori.z, ori.w = tf.transformations.quaternion_from_euler(0, 0, uav_state[3])

    covariance = np.array([uav_state_variance[0, 0], 0, 0, 0, 0, 0,
                           0, uav_state_variance[1, 1], 0, 0, 0, 0, 
                           0, 0, uav_state_variance[2, 2], 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, uav_state_variance[3, 3]], dtype = np.float32)

    pose_with_cov.pose.covariance = covariance
    rospy.sleep(0.1)
    publisher.publish(pose_with_cov)

    return  pose_with_cov.pose.pose

def motion_model(control, uav_state, velocity, delta_t):
    # Initialize changes in state
    dx, dy, dz, d_yaw = 0, 0, 0, 0

    if control == "up":
        dz = velocity * delta_t

    elif control == "down":
        dz = -velocity * delta_t

    elif control == "stop":
        velocity = 0

    elif control == "left":
        dy = -velocity * delta_t
        velocity = velocity / 2

    elif control == "right":
        dy = velocity * delta_t
        velocity = velocity / 2

    else:
        raise ValueError("Invalid control input")

    # Calculate the changes in x, y based on the current yaw angle
    # current_yaw = uav_state[3]
    # dx = velocity * delta_t * np.cos(current_yaw)
    # dy = velocity * delta_t * np.sin(current_yaw)

    return np.array([dx, dy, dz, d_yaw], dtype=np.float32)

if __name__ == "__main__":

    TRACK = "Track2"

    # Path setting
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    OUTPUT_PATH =  os.path.join(ROOT_DIR, "result", (TRACK + ".csv"))
    
    OUTPUT_CSV_HEADER = ["x", "y", "z", "row", "pitch", "yaw"]
    with open(OUTPUT_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(OUTPUT_CSV_HEADER)

    # ROS Node
    rospy.init_node('lab2')

    # Build Map
    MAP_PATH = os.path.join(ROOT_DIR, "map_information.csv")
    LAB2_MAP = BuildMap(MAP_PATH)
    APRILTAG_INFO = LAB2_MAP.get_apriltag_info()

    # ROS Camera Publisher
    CV_BRIDGE = CvBridge()
    CAM_PUB = rospy.Publisher('uav_camera', Image, queue_size = 10)
    rospy.loginfo("Create Image Publisher")
    rospy.sleep(1)

    # ROS UAV Pose Publisher
    UAV_POSE_PUB = rospy.Publisher('uav_pose', PoseWithCovarianceStamped, queue_size = 10)
    rospy.loginfo("UAV Pose Publisher")
    rospy.sleep(1)

    # ROS UAV Path Publisher
    PATH_PUB = rospy.Publisher('uav_path', Path, queue_size=10)
    UAV_PATH = Path()
    UAV_PATH.header.frame_id = "map"
    rospy.loginfo("UAV Path Publisher")
    rospy.sleep(1)

    # Data Path
    DATA_PATH     = os.path.join(DATA_DIR, TRACK, "Motion_Observation.csv")
    CAMERA_FOLDER = os.path.join(DATA_DIR, TRACK, "camera_image")
    DATA_LIST = read_data(DATA_PATH)

    # UAV setting parameters
    UAV_CFG_FILE = os.path.join(DATA_DIR, TRACK, "uav_config.yml")
    LAB2_UAV = UAV(UAV_CFG_FILE)
    CAMERA_INTRINSIC, CAMERA_DISS = LAB2_UAV.get_camera_parameter()
    UAV_INIT_STATE, UAV_INIT_VARIANCE, UAV_VELOCITY = LAB2_UAV.get_init_value()

    # Apriltag detector
    AT_DETECTOR = apriltag.Detector(families = 'tag16h5')

    # Kalman Filter init
    KF = KalmanFilter(x0 = UAV_INIT_STATE, s0 = UAV_INIT_VARIANCE)

    # Program start
    while not rospy.is_shutdown():

        for i in range(len(DATA_LIST)):

            # Get state and variance of uav
            uav_state, uav_variance = KF.get_current_state()
            rospy.loginfo(f"Current state: {uav_state}")

            # Write the data
            with open(OUTPUT_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([uav_state[0], uav_state[1], uav_state[2], 0, 0, uav_state[3]])

            # Publish state and variance to rviz
            uav_pose = publish_pose(publisher = UAV_POSE_PUB, 
                                    uav_state = uav_state, 
                                    uav_state_variance = uav_variance)
            rospy.sleep(0.2)

            # Publish uav path
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose = uav_pose

            UAV_PATH.poses.append(pose)

            if len(UAV_PATH.poses) > 20:
                UAV_PATH.poses.pop(0)

            UAV_PATH.header.stamp = rospy.Time.now()
            PATH_PUB.publish(UAV_PATH)
            rospy.sleep(0.2)

            # Get delta_t, control and observation
            delta_t = 0
            control_data = "None"
            observation = "None"

            if i == len(DATA_LIST) - 1: 
                delta_t = 0.1
                _, control_data, observation_data  = DATA_LIST[i]
            
            else:
                
                time_start, control_data, observation_data  = DATA_LIST[i]
                time_end, _ , _= DATA_LIST[i + 1]
                delta_t = time_end - time_start


            # If observe data
            if observation_data != "None":

                # Get camera data
                image_path = os.path.join(CAMERA_FOLDER, observation_data)

                # Use camera to get uav state
                uav_state_observe, img = get_uav_pose(img_path = image_path, 
                                                      tag_detector = AT_DETECTOR, 
                                                      intrinsic = CAMERA_INTRINSIC, 
                                                      dist = CAMERA_DISS, 
                                                      apriltag_id_info = APRILTAG_INFO)
                
                # Publish camera image
                CAM_PUB.publish(CV_BRIDGE.cv2_to_imgmsg(img))

                # Kalman Filter update
                if uav_state_observe is not None:
                    
                    KF.update(z = uav_state_observe)
                    #rospy.loginfo(f"Observe state: {uav_state_observe}")
            
            # Predicted next state
            uav_state, _ = KF.get_current_state()
            d_motion = motion_model(control = control_data, uav_state =uav_state, velocity = UAV_VELOCITY, delta_t = delta_t)
            KF.predict(u = d_motion)
            #rospy.loginfo(f"Control motion: {control_data}, dx = {d_motion[0]}, dy = {d_motion[1]}, dz = {d_motion[2]}, d_yaw = {d_motion[3]}")
        
        break
                    






