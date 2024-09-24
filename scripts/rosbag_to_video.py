import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import cv2

class ROSBagToDataset:
    def __init__(self, bag_file, topic_name, output_dir):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.output_dir = output_dir
        self.bridge = CvBridge()

        # Create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_frames_as_images(self):
        # Open rosbag
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
        except Exception as e:
            raise RuntimeError(f"Failed to open ROS bag file: {e}")

        frame_count = 0
        for topic, msg, t in bag.read_messages(topics=[self.topic_name]):
            try:
                # Convert ROS Image message to OpenCV image
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # Generate image filename based on frame count
                img_filename = os.path.join(self.output_dir, f"frame_{frame_count:06d}.png")

                # Save image
                cv2.imwrite(img_filename, frame)

                print(f"Saved: {img_filename}")
                frame_count += 1
            except Exception as e:
                print(f"Error converting ROS Image message: {e}")

        bag.close()
        print(f"Total frames saved: {frame_count}")

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('rosbag_to_dataset_node')

    # Parameters (provide default values if not set in ROS parameter server)
    bag_file = rospy.get_param('~bag_file', '/home/kwon/lane_ws/src/lane_dataset_rosbag/2024-08-12-17-47-07.bag')
    topic_name = rospy.get_param('~topic_name', '/zed2/zed_node/right/image_rect_color')
    output_dir = rospy.get_param('~output_dir', '/home/kwon/lane_ws/src/lane_detection_240830/dataset/images')

    # Create dataset generator and run
    dataset_creator = ROSBagToDataset(bag_file, topic_name, output_dir)
    dataset_creator.save_frames_as_images()
