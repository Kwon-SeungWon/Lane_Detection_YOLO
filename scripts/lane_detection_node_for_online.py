#!/usr/bin/env python3

import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from tqdm import tqdm
import base64
import json
from inference_sdk import InferenceHTTPClient

class ROSBagLaneDetection:
    def __init__(self, bag_file, topic_name, client, frame_processor, output_path):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.client = client
        self.frame_processor = frame_processor
        self.output_path = output_path

        # Initialize the CvBridge instance
        self.bridge = CvBridge()

        # Initialize video properties
        self.frame_width = None
        self.frame_height = None
        self.bag = None
        self.out = None

        self._open_bag()
        self._initialize_video_properties()

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
        self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (self.frame_width, self.frame_height))
    
    def _get_first_frame(self):
        for topic, msg, _ in self.bag.read_messages(topics=[self.topic_name]):
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
            for topic, msg, _ in self.bag.read_messages(topics=[self.topic_name]):
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # Encode the frame to JPEG format and base64 encode it
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Create the payload to send to the inference API
                payload = {
                    'image': image_base64,
                    'model_id': 'lane-detection-segmentation-edyqp/5'
                }

                # Send the frame to the inference client and get predictions
                response = self.client.infer(payload)
                results = self._parse_response(response)

                # Projection to frame for prediction result
                processed_frame = self.frame_processor.process_frame(frame, results)

                # Save the video file
                self.out.write(processed_frame)
                pbar.update(1)

                # Visualize the result frame
                cv2.imshow("Lane Detection", processed_frame)
                key = cv2.waitKey(1)

                # Press ESC to exit
                if key & 0xFF == 27:
                    break

                # Adjust threshold with up/down arrow keys
                elif key == 82:  # Up arrow
                    self.frame_processor.threshold += 5
                    print(f"Threshold increased to {self.frame_processor.threshold}%")
                elif key == 84:  # Down arrow
                    self.frame_processor.threshold -= 5
                    print(f"Threshold decreased to {self.frame_processor.threshold}%")

        self._clean_up()

    def _parse_response(self, response):
        """
        Parse the API response and extract the necessary information.
        Assuming the API returns a dictionary with 'boxes' and 'masks'.
        """
        try:
            return json.loads(response)  # Parse the JSON response
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {e}")
            return {}

    def _clean_up(self):
        if self.out is not None:
            self.out.release()
        if self.bag is not None:
            self.bag.close()
        cv2.destroyAllWindows()


class FrameProcessor:
    def __init__(self, colors=None, threshold=50):
        self.colors = colors if colors else {
            0: (255, 0, 255),  # Pink for Dashed
            1: (255, 0, 0)     # Blue for Solid
        }
        self.class_names = {
            0: "Dashed",
            1: "Solid"
        }
        self.threshold = threshold  # Set the threshold value

    def process_frame(self, frame, results):
        overlay = frame.copy()  # Create a copy of the frame for overlay

        for result in results.get('predictions', []):
            boxes = result.get('boxes', [])  # Bounding boxes object
            masks = result.get('masks', [])  # Masks object, if available

            # If masks are available, process them
            if masks:
                for i, mask in enumerate(masks):
                    lane_type = int(boxes[i]['class'])  # Get the class (0 for Dashed, 1 for Solid)
                    confidence = boxes[i]['confidence'] * 100  # Get the confidence

                    # Apply the threshold to filter out low-confidence predictions
                    if confidence < self.threshold:
                        continue

                    # Convert mask to numpy array and then to uint8
                    mask_np = np.array(mask, dtype=np.uint8)

                    # Resize the mask to match the original image size if necessary
                    mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

                    # Find contours from the mask
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw the contours on the overlay with transparency
                    for contour in contours:
                        # Draw a filled contour with transparency
                        cv2.drawContours(overlay, [contour], -1, self.colors.get(lane_type, (255, 255, 255)), thickness=cv2.FILLED)

                        # Optional: Draw a thicker contour to act as a border
                        cv2.drawContours(frame, [contour], -1, (0, 0, 0), thickness=2)

                        # Draw the contours on the overlay
                        cv2.polylines(frame, [contour], isClosed=False, color=self.colors.get(lane_type, (255, 255, 255)), thickness=3)

                    # Draw label and confidence at the center of the lane
                    if len(contours) > 0:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            label = f"{self.class_names[lane_type]} {confidence:.0f}%"
                            cv2.putText(overlay, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors.get(lane_type, (255, 255, 255)), 2, cv2.LINE_AA)

        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        return frame


if __name__ == "__main__":
    rospy.init_node('lane_detection_node')

    bag_file = rospy.get_param('~bag_file', '/home/kwon/lane_ws/src/lane_dataset_rosbag/2024-08-12-17-47-07.bag')
    topic_name = rospy.get_param('~topic_name', '/zed2/zed_node/right/image_rect_color')
    output_path = rospy.get_param('~output_path', '/home/kwon/lane_ws/src/lane_detection_240830/videos/rosbag_output_online.mp4')

    client = InferenceHTTPClient(
        api_url="https://outline.roboflow.com",
        api_key="7fLFcWQOy3nrBlMEZBNC"
    )
    frame_processor = FrameProcessor()

    lane_detection = ROSBagLaneDetection(bag_file, topic_name, client, frame_processor, output_path)
    lane_detection.run()
