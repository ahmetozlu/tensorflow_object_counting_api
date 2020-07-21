
import os

# Imports
import tensorflow as tf

from api import object_counting_api
# Object detection imports
from utils import backbone

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def hello():
    print('hello')

def run(input_video, roi, axis, is_color_recognition_enabled = 0, deviation = 1):
  # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
  detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

  # main counting all the objects
  if (axis == 'y'):
    object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation)
  else:
    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation)
