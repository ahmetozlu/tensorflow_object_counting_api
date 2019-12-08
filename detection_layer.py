#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 14th August 2019
#----------------------------------------------

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))

from utils import visualization_utils

class ObjectDetector(object):
    def __init__(self):

        self.object_boxes = []
        
        os.chdir(cwd)
         
        detect_model_name = 'custom_frozen_inference_graph'
        
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'       
        
        self.detection_graph = tf.Graph()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')    

    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)       

    def box_normal_to_pixel(self, box, dim):    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):         
        category_index={1: {'id': 1, 'name': u'player'}}  
        
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})          
              if visual == True:
                  visualization_utils.visualize_boxes_and_labels_on_image_array_tracker(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)   
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()               
              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)  
              cls = classes.tolist()
              idx_vec = [i for i, v in enumerate(cls) if ((scores[i]>0.6))]              
              if len(idx_vec) ==0:
                  print('there are not any detections, passing to the next frame...')
              else:
                  tmp_object_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)
                      
                      #if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                      tmp_object_boxes.append(box)
                      #print(box, ', confidence: ', scores[idx], 'ratio:', ratio)                                                   
                  
                  self.object_boxes = tmp_object_boxes             
        return self.object_boxes                              
