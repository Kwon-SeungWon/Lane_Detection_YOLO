import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

class ROSBagLaneDetection:
    def __init__(self, bag_file, topic_name, model, frame_processor, output_path, conf_threshold=0.6, iou_threshold=0.05):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.model = model
        self.frame_processor = frame_processor
        self.output_path = output_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

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

        start_time = time.time()
        frame_count = 0

        total_messages = self.bag.get_message_count(self.topic_name)
        executor = ThreadPoolExecutor(max_workers=2)  # 병렬 처리용 스레드풀
        with tqdm(total=total_messages, desc="Processing Frames") as pbar:
            for topic, msg, _ in self.bag.read_messages(topics=[self.topic_name]):
                try:
                    frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                except Exception as e:
                    print(f"Error converting ROS Image message: {e}")
                    continue

                # 비동기 YOLO 모델 예측
                future = executor.submit(self._predict_and_process, frame)
                try:
                    processed_frame = future.result()  # 예측 결과 받기
                    self.out.write(processed_frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

                frame_count += 1
                pbar.update(1)

                # Visualize the result frame
                cv2.imshow("Lane Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                    break

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"Total processing time: {total_time:.2f} seconds, Average FPS: {avg_fps:.2f}")

        self._clean_up()

    def _predict_and_process(self, frame):
        start_frame_time = time.time()

        # Sharpening kernel
        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        # Apply the sharpening filter to the frame
        sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)

        # YOLO model prediction with confidence and IOU thresholds on the sharpened frame
        results = self.model.predict(sharpened_frame, conf=self.conf_threshold, iou=self.iou_threshold)

        # Process the sharpened frame with the detection results
        processed_frame = self.frame_processor.process_frame(sharpened_frame, results)

        elapsed_time = time.time() - start_frame_time
        print(f"Frame processed in {elapsed_time:.2f} seconds")

        return processed_frame

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
        overlay = frame.copy()  # Create a copy of the frame for overlay

        for result in results:
            boxes = result.boxes  # Bounding boxes object
            masks = result.masks  # Masks object

            # If masks are available, process them
            if masks:
                for i, mask in enumerate(masks.data):
                    lane_type = int(boxes.cls[i])  # Get the class (0 for Dashed, 1 for Solid)
                    confidence = boxes.conf[i].item() * 100  # Get the confidence

                    # Convert mask tensor to numpy array and then to uint8
                    mask_np = mask.cpu().numpy().astype(np.uint8)

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
                        cv2.polylines(frame, [contour], isClosed=True, color=self.colors.get(lane_type, (255, 255, 255)), thickness=2)

                    # Draw label and confidence at the center of the lane
                    if len(contours) > 0:
                        M = cv2.moments(contours[0])
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            label = f"{self.class_names[lane_type]} {confidence:.0f}%"
                            cv2.putText(overlay, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors.get(lane_type, (255, 255, 255)), 2, cv2.LINE_AA)

        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.35, frame, 0.5, 0, frame)

        return frame


if __name__ == "__main__":
    rospy.init_node('lane_detection_node')

    bag_file = rospy.get_param('~bag_file', '/home/kwon/lane_ws/src/lane_dataset_rosbag/2024-08-12-17-47-07.bag')
    topic_name = rospy.get_param('~topic_name', '/zed2/zed_node/right/image_rect_color')
    output_path = rospy.get_param('~output_path', '/home/kwon/lane_ws/src/lane_detection_240830/videos/output_video_240910_yolov8_thresold60_iouthreshold5_with_sharpening.mp4')

    model = YOLO('/home/kwon/lane_ws/yolov8_lane_dataset/runs/segment/train9/weights/best.pt')

    frame_processor = FrameProcessor()

    lane_detection = ROSBagLaneDetection(bag_file, topic_name, model, frame_processor, output_path)
    lane_detection.run()
