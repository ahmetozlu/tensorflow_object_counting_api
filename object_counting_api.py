#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import backbone

#initialize .csv
with open('traffic_measurement.csv', 'w') as f:
        writer = csv.writer(f)  
        csv_line = "Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)"                 
        writer.writerows([csv_line.split(',')])

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# input video
cap = cv2.VideoCapture('sub-1504614469486.mp4')

# Variables
total_passed_vehicle = 0 # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
PATH_TO_LABELS, NUM_CLASSES, detection_graph = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
def object_detection_function(mode, y_min=None, y_max=None):
        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                if(mode == 0):
                    counter, csv_line = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
		    input_frame,
                    mode,
		    np.squeeze(boxes),
		    np.squeeze(classes).astype(np.int32),
		    np.squeeze(scores),
		    category_index,
		    use_normalized_coordinates=True,
		    line_thickness=4)
                
                    total_passed_vehicle = total_passed_vehicle + counter          

                    cv2.putText(input_frame,"Detected Vehicles: " + str(total_passed_vehicle), (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)

                    # when the vehicle passed over line and counted, make the color of ROI line green
                    if(counter == 1):
                            cv2.line(input_frame,(0,200),(640,200),(0,255,0),5)
                    else:
                            cv2.line(input_frame,(0,200),(640,200),(0,0,255),5)

                    # insert information text to video frame
                    cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                    cv2.putText(input_frame,"ROI Line", (545, 190), font, 0.6,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(input_frame,"LAST PASSED VEHICLE INFO", (11, 290), font, 0.5, (255,255, 255), 1,cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.putText(input_frame,"-Movement Direction: " + direction, (14, 302), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    cv2.putText(input_frame,"-Speed(km/h): " + speed, (14, 312), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    cv2.putText(input_frame,"-Color: " + color, (14, 322), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    cv2.putText(input_frame,"-Vehicle Size/Type: " + size, (14, 332), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)

                elif (mode == 1):
                    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                          input_frame,
			                                                                                  mode,
                                                                                                          np.squeeze(boxes),
                                                                                                          np.squeeze(classes).astype(np.int32),
                                                                                                          np.squeeze(scores),
                                                                                                          category_index,
                                                                                                          use_normalized_coordinates=True,
                                                                                                          line_thickness=4)
                    if(len(counting_mode) == 0):
                        cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                    else:
                        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                cv2.imshow('vehicle detection',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         
            cap.release()
            cv2.destroyAllWindows()
            
object_detection_function(0)
