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

        # 출력 폴더 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_frames_as_images(self):
        # rosbag 열기
        try:
            bag = rosbag.Bag(self.bag_file, 'r')
        except Exception as e:
            raise RuntimeError(f"Failed to open ROS bag file: {e}")

        frame_count = 0
        for topic, msg, t in bag.read_messages(topics=[self.topic_name]):
            try:
                # ROS Image 메시지를 OpenCV 이미지로 변환
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # 이미지 파일명 생성 (프레임 번호를 기반으로)
                img_filename = os.path.join(self.output_dir, f"frame_{frame_count:06d}.png")

                # 이미지 저장
                cv2.imwrite(img_filename, frame)

                print(f"Saved: {img_filename}")
                frame_count += 1
            except Exception as e:
                print(f"Error converting ROS Image message: {e}")

        bag.close()
        print(f"Total frames saved: {frame_count}")

if __name__ == "__main__":
    # ROS 노드 초기화
    rospy.init_node('rosbag_to_dataset_node')

    # 파라미터 설정 (필요시 ROS 파라미터 서버에서 가져옴)
    bag_file = '~bag_file', '/home/kwon/lane_ws/src/lane_dataset_rosbag/2024-08-12-17-47-07.bag'
    topic_name = '~topic_name', '/zed2/zed_node/right/image_rect_color'
    output_dir = '/home/kwon/lane_ws/src/lane_detection_240830/dataset/images'

    # 데이터셋 생성기 실행
    dataset_creator = ROSBagToDataset(bag_file, topic_name, output_dir)
    dataset_creator.save_frames_as_images()
