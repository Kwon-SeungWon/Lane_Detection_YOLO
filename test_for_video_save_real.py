import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from queue import Queue
import threading

class FrameProcessor:
    def __init__(self, colors=None):
        # Set default colors if none are provided
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

class VideoLaneDetection:
    def __init__(self, video_path, output_path, model, frame_processor, conf_threshold=0.55, iou_threshold=0.001, queue_size=5):
        self.video_path = video_path
        self.output_path = output_path
        self.model = model
        self.frame_processor = frame_processor
        self.conf_threshold = conf_threshold  # Confidence threshold
        self.iou_threshold = iou_threshold  # IoU threshold
        self.image_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue()
        self._initialize_video_properties()

    def _initialize_video_properties(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def _preprocess_image(self, frame):
        # Convert image to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the Y channel
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(12, 12))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        
        # Convert back to BGR color space
        equalized_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        sharpening_kernel = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])

        # Apply the sharpening filter to the frame
        sharpened_frame = cv2.filter2D(equalized_frame, -1, sharpening_kernel)

        return sharpened_frame

    def _perform_inference(self):
        while True:
            frame = self.image_queue.get()
            if frame is None:
                break

            # Perform preprocessing
            preprocessed_frame = self._preprocess_image(frame)

            # Perform inference using the YOLO model with specified thresholds
            results = self.model.predict(preprocessed_frame, conf=self.conf_threshold, iou=self.iou_threshold)

            # Put the results into the result queue for processing
            self.result_queue.put((results, preprocessed_frame))
            self.image_queue.task_done()

    def run(self):
        inference_thread = threading.Thread(target=self._perform_inference)
        inference_thread.start()

        frame_processed = 0
        with tqdm(total=self.total_frames, desc="Processing Frames") as pbar:
            while self.cap.isOpened() or not self.result_queue.empty():
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    self.image_queue.put(frame)

                while not self.result_queue.empty():
                    results, original_frame = self.result_queue.get()
                    processed_frame = self.frame_processor.process_frame(original_frame, results)
                    self.out.write(processed_frame)
                    pbar.update(1)
                    frame_processed += 1

                    cv2.imshow("Lane Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                        self.cap.release()
                        self.out.release()
                        cv2.destroyAllWindows()
                        return

        self._clean_up()

    def _clean_up(self):
        self.image_queue.put(None)
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# Instantiate the required classes
model = YOLO('/home/kwon/lane_ws/yolov8_lane_dataset/runs/segment/train9/weights/best.pt')

frame_processor = FrameProcessor()

video_lane_detection = VideoLaneDetection(
    video_path="/home/kwon/lane_ws/src/lane_detection_240830/videos/rosbag_video.mp4",
    output_path="/home/kwon/lane_ws/src/lane_detection_240830/videos/rosbag_video_240911_with_sharpening.mp4",
    model=model,
    frame_processor=frame_processor,
    conf_threshold=0.5,  # Set your custom confidence threshold
    iou_threshold=0.05    # Set your custom IoU threshold
)

# Run the lane detection
video_lane_detection.run()
