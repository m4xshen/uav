import tf
import csv
import rospy
import tf2_ros
import numpy as np
import geometry_msgs.msg


class BuildMap:

    def __init__ (self, csv_path):

        self.csv_path = csv_path
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.apriltag_id_info = {}
        with open(self.csv_path, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  
            for row in csv_reader:
                id, x, y, z, tag_size = list(map(float, row))
                self.apriltag_id_info[str(int(id))] = {}
                self.apriltag_id_info[str(int(id))]['location'] = (x, y, z)
                self.apriltag_id_info[str(int(id))]['tag_size'] = tag_size
                self.apriltag_id_info[str(int(id))]['corner_list'] = self.get_apriltag_corners(world_x = x, world_y = y, world_z = z, tag_size = tag_size)

        for tag_id in self.apriltag_id_info.keys():
            x, y, z = self.apriltag_id_info[tag_id]['location']
            self.put_apriltag_origin(world_x = x, world_y = y, world_z = z, name = "tag" + tag_id)
    
    def map_information(self):

        for tag_id in self.apriltag_id_info.keys():
            
            print(f"Tag id: {tag_id}")
            print(f"Tag size: {self.apriltag_id_info[tag_id]['tag_size']} [m]")
            print(f"Tag location: {self.apriltag_id_info[tag_id]['location']} [m]")
            print("-" * 50)
    
                
    def get_apriltag_corners(self, world_x, world_y, world_z, tag_size):
        tag_half_length = tag_size / 2
        return np.array([[world_x, world_y - tag_half_length, world_z - tag_half_length],
                        [world_x, world_y + tag_half_length, world_z - tag_half_length],
                        [world_x, world_y + tag_half_length, world_z + tag_half_length],
                        [world_x, world_y - tag_half_length, world_z + tag_half_length]], dtype = np.float32)


    def put_apriltag_origin(self, world_x, world_y, world_z, name):
        static_transform = geometry_msgs.msg.TransformStamped()
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "map"
        static_transform.child_frame_id  = name
        static_transform.transform.translation.x = float(world_x)
        static_transform.transform.translation.y = float(world_y)
        static_transform.transform.translation.z = float(world_z)
        roll, pitch, yaw = [np.deg2rad(-90), np.deg2rad(0), np.deg2rad(90)]
        quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        static_transform.transform.rotation.x = quat[0]
        static_transform.transform.rotation.y = quat[1]
        static_transform.transform.rotation.z = quat[2]
        static_transform.transform.rotation.w = quat[3]
        self.broadcaster.sendTransform(static_transform)
        rospy.loginfo("Publish " + name + " coordinate")
        rospy.sleep(1)
    
    def get_apriltag_info(self):
        return self.apriltag_id_info


if __name__ == "__main__":

    rospy.init_node('map_publisher')

    csv_path = "apriltag_coordinate.csv"

    lab2_map = BuildMap(csv_path = csv_path)
    lab2_map.map_information()