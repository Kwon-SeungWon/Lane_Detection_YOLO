import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from tqdm import tqdm
from torchvision import transforms
totensor = transforms.ToTensor()

import sys
sys.path.insert(0, '/home/kwon/lane_ws/src/lane_detection_240830/scripts')
from Networks.LSQ_layer import Net
from Networks.utils import define_args

class ROSBagLaneDetection:
    def __init__(self, bag_file, topic_name, model, frame_processor, output_path, use_gpu=True):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.model = model
        self.frame_processor = frame_processor
        self.output_path = output_path
        self.use_gpu = use_gpu

        # Initialize CvBridge instance
        self.bridge = CvBridge()

        # Initialize video properties
        self.frame_width = None
        self.frame_height = None
        self.bag = None
        self.out = None

        self._initialize_video_properties()
        self._open_bag()

        # Move model to GPU if available and use_gpu is True
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.to('cuda')

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
                try:
                    frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                    # Forward pass through the model
                    input_tensor = self.frame_processor.prepare_input(frame, self.use_gpu)
                    with torch.no_grad():  # No gradient computation for inference
                        if self.use_gpu and torch.cuda.is_available():
                            input_tensor = input_tensor.to('cuda')
                        results = self.model(input_tensor)

                    # Process and annotate frame
                    processed_frame = self.frame_processor.process_frame(frame, results)

                    self.out.write(processed_frame)
                    pbar.update(1)

                    cv2.imshow("Lane Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                        break

                except Exception as e:
                    print(f"Error processing frame: {e}")

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

    def prepare_input(self, frame, use_gpu):
        # Convert the frame to a tensor
        input_tensor = torch.from_numpy(frame).float()
        input_tensor = input_tensor.permute(2, 0, 1)  # Convert from HWC to CHW format
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Normalize the tensor if needed (example: if using pretrained models)
        input_tensor = input_tensor / 255.0  # Normalize to [0, 1] range
        input_tensor = torch.nn.functional.interpolate(input_tensor, size=(224, 224))  # Resize to match model input size

        return input_tensor

    def process_frame(self, frame, results):
        # Assuming results is a list of dictionaries containing boxes, labels, and scores
        for result in results:
            boxes = result['boxes']
            labels = result['labels']
            scores = result['scores']
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = scores[i] * 100
                cls = labels[i]

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors.get(cls, (255, 255, 255)), 2)
                label = f"{self.class_names.get(cls, 'Unknown')} {confidence:.0f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                label_origin = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), self.colors.get(cls, (255, 255, 255)), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

if __name__ == "__main__":
    rospy.init_node('lane_detection_node')

    bag_file = rospy.get_param('~bag_file', '/home/kwon/Downloads/lane_dataset_rosbag/2024-08-12-17-47-07.bag')
    topic_name = rospy.get_param('~topic_name', '/camera/image_raw')
    output_path = rospy.get_param('~output_path', '/home/kwon/lane_ws/src/lane_detection_240830/videos/output_video.mp4')

    global args
    parser = define_args()
    args = parser.parse_known_args()[0]
    print(args.resize)

    # Load the custom model
    model = Net(args)  # Instantiate your custom model
    checkpoint = torch.load('/home/kwon/lane_ws/src/lane_detection_240830/models/model_best_epoch_340.pth.tar', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # Load model weights
    model = model.cuda

    frame_processor = FrameProcessor()

    lane_detection = ROSBagLaneDetection(bag_file, topic_name, model, frame_processor, output_path, use_gpu=torch.cuda.is_available())
    lane_detection.run()
