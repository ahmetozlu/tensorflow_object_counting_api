#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 14th August 2019
#----------------------------------------------

import numpy as np
import cv2
import backbone

from utils.object_tracking_module import tracking_layer
                     
cap = cv2.VideoCapture("./input_images_and_videos/vehicle_survaillance.mp4")
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('the_output.avi',fourcc, fps, (width, height))

while(True):            
    ret, img = cap.read()           
    np.asarray(img)
    processed_img = backbone.processor(img)
    out.write(processed_img)
    print("writing frame...")
        
print("end of the video!")
