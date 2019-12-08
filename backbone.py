#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 14th August 2019
#----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import detection_layer
import cv2

from utils.object_tracking_module import tracking_layer
from utils.object_tracking_module import tracking_utils

max_detection = 15
min_detection =1

tracker_list =[]
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '72', '73', '74', '75', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100'])

det = detection_layer.ObjectDetector()

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):           
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = tracking_utils.box_iou2(trk,det) 
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    
def processor(img):
    global tracker_list
    global max_detection
    global min_detection
    global track_id_list 
    
    img_dim = (img.shape[1], img.shape[0])
    z_box = det.get_localization(img)       
    x_box =[]       
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
       
    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)
         
    # matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
    
    # unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracking_layer.Tracker() # new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
    
    # unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                         
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_detection) and (trk.no_losses <=max_detection)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box             
             img= tracking_utils.draw_box_label(trk.id,img, x_cv2)

    deleted_tracks = filter(lambda x: x.no_losses >max_detection, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_detection]

    return img
