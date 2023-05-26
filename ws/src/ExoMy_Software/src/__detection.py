#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('exomy')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse
import sys
import time

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

base_options = core.BaseOptions(
      file_name='/home/pi/efficientdet_lite0.tflite', use_coral=False, num_threads=2)
detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
counter, fps = 0, 0

def detect(image):

    start_time = time.time()

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    counter += 1

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return rgb_image
 
class image_converter:
 
  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=1)
 
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/pi_cam/image_raw",Image,self.callback)
 
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    image = detect(image)
    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
    #  cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(27)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
     main(sys.argv)
