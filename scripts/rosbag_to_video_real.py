#!/usr/bin/env python3

import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge

# ROS bag file path
bag_file = '/home/kwon/lane_ws/src/lane_dataset_rosbag/2024-08-12-17-47-07.bag'

# Video file path and settings
video_file = '/home/kwon/lane_ws/src/lane_detection_240830/videos/rosbag_video_sharpening.mp4'
fps = 5  # Frames per second
frame_width = 640  # Video width resolution
frame_height = 480  # Video height resolution

# Create CvBridge instance
bridge = CvBridge()

# Create VideoWriter object for saving the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

# Sharpening kernel
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# Open the ROS bag file
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/zed2/zed_node/right/image_rect_color']):
        # Convert ROS image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize to match the desired resolution (if necessary)
        resized_image = cv2.resize(cv_image, (frame_width, frame_height))

        # Apply sharpening filter
        sharpened_image = cv2.filter2D(resized_image, -1, sharpening_kernel)

        # Write the sharpened frame to the video file
        out.write(sharpened_image)

# Release the VideoWriter object after all frames are processed
out.release()
