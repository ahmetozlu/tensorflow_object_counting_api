
import os

# Imports
import tensorflow as tf

from api import object_counting_api
# Object detection imports
from utils import backbone

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(tf.version)


input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model(
    'ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

# set it to 1 for enabling the color prediction for the detected objects
is_color_recognition_enabled = 0

# the constant that represents the object counting area
deviation = 1

# roi line position
roi = 250

# axis for te object detection
axis = 'x'

# main counting all the objects
if (axis == 'y'):
    object_counting_api.cumulative_object_counting_y_axis(
        input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation)
else:
    object_counting_api.cumulative_object_counting_x_axis(
        input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation)
