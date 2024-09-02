import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

class ROSBagLaneDetection:
    def __init__(self, bag_file, topic_name, model, frame_processor, output_path):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.model = model
        self.frame_processor = frame_processor
        self.output_path = output_path

        # Initialize the CvBridge instance
        self.bridge = CvBridge()

        # Add these lines to initialize frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        # Initialize the bag and video properties
        self.bag = None
        self.out = None

        self._initialize_video_properties()
        self._open_bag()

    def _open_bag(self):
        try:
            self.bag = rosbag.Bag(self.bag_file, 'r')
        except Exception as e:
            raise RuntimeError(f"Failed to open ROS bag file: {e}")

    def _initialize_video_properties(self):
        first_frame = self._get_first_frame()
        if first_frame is None:
            raise RuntimeError("Failed to retrieve the first frame. Ensure the bag file and topic are correct.")
        
        self.frame_width = first_frame.shape[1]  # Width of the frame
        self.frame_height = first_frame.shape[0]  # Height of the frame
        # self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        # self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.frame_width, self.frame_height))
    
    def _get_first_frame(self):
        with rosbag.Bag(self.bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=[self.topic_name]):
                try:
                    frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    if frame is not None:
                        print("First frame retrieved successfully.")
                        return frame
                except Exception as e:
                    print(f"Error converting ROS Image message to OpenCV frame: {e}")
        print("No frames were retrieved from the specified topic.")
        return None

    def run(self):
        if self.bag is None:
            raise RuntimeError("ROS bag not opened. Cannot run the processing.")
        
        total_messages = self.bag.get_message_count(self.topic_name)
        with tqdm(total=total_messages, desc="Processing Frames") as pbar:
            for topic, msg, t in self.bag.read_messages(topics=[self.topic_name]):
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                results = self.model(frame)
                processed_frame = self.frame_processor.process_frame(frame, results)

                self.out.write(processed_frame)
                pbar.update(1)

                cv2.imshow("Lane Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                    break

        self._clean_up()

    def _clean_up(self):
        if self.out is not None:
            self.out.release()
        if self.bag is not None:
            self.bag.close()
        cv2.destroyAllWindows()

class FrameProcessor:
    def __init__(self, colors=None):
        self.colors = colors if colors else {
            0: (255, 0, 255),  # Pink for Dashed
            1: (255, 0, 0)     # Blue for Solid
        }
        self.class_names = {
            0: "Dashed",
            1: "Solid"
        }

    def process_frame(self, frame, results):
        for result in results:
            lane_type = result["class"]
            confidence = result["confidence"] * 100
            points = result["points"]

            x_coords = np.array([point['x'] for point in points])
            y_coords = np.array([point['y'] for point in points])
            
            for j in range(len(x_coords) - 1):
                start_point = (int(x_coords[j]), int(y_coords[j]))
                end_point = (int(x_coords[j+1]), int(y_coords[j+1]))
                cv2.line(frame, start_point, end_point, self.colors.get(lane_type, (255, 255, 255)), 2)
            
            mid_idx = len(points) // 2
            mid_point = (int(x_coords[mid_idx]), int(y_coords[mid_idx]))
            cv2.putText(frame, f"{lane_type} {confidence:.0f}%", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors.get(lane_type, (255, 255, 255)), 2, cv2.LINE_AA)

        return frame

if __name__ == "__main__":
    rospy.init_node('lane_detection_node')

    bag_file = rospy.get_param('~bag_file', '/home/kwon/Downloads/lane_dataset_rosbag/2024-08-12-17-47-07.bag')
    topic_name = rospy.get_param('~topic_name', '/camera/image_raw')
    output_path = rospy.get_param('~output_path', '/home/kwon/lane_ws/src/lane_detection_240830/videos/output_video.mp4')

    model = YOLO('/home/kwon/lane_ws/yolov8_lane_dataset/runs/detect/train6/weights/best.pt')
    frame_processor = FrameProcessor()

    lane_detection = ROSBagLaneDetection(bag_file, topic_name, model, frame_processor, output_path)
    lane_detection.run()
