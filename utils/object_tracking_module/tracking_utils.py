#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 14th August 2019
#----------------------------------------------

import numpy as np
import cv2

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def box_iou2(a, b):    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)

def convert_to_pixel(box_yolo, img, crop_range):    
    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape
    
    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)
    
    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))
    
    # Deal with corner cases
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0
    
    # Return the coordinates (in the unit of the pixels)  
    box_pixel = np.array([left, top, width, height])
    return box_pixel

def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])
    
    return (left, top, right, bottom)
        
def draw_box_label(id,img, bbox_cv2, box_color=(0, 255, 0), show_label=True):
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 255)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)   
    
    if show_label:       
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        
        object_id = 'object_ID:'+str(id)
        cv2.putText(img,object_id,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img    
