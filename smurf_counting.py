#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th July 2019
#----------------------------------------------

# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

input_video = "./input_images_and_videos/smurf_input.avi"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('custom_frozen_inference_graph', 'detection.pbtxt')

is_color_recognition_enabled = 0

object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled)
