#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import os
import tensorflow as tf
import cv2

# Object detection imports
from utils import backbone

from api import object_counting_api

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "New Office TOUR!  Karlie Kloss.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

# Detection
object_counting_api.object_detection_function(input_video, detection_graph, category_index, 0)
