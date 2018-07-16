#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

import tensorflow as tf
import csv
import cv2
import numpy as np
from utils import visualization_utils as vis_util

def object_detection_function2(input_video, output_video, detection_graph, category_index, mode, color_recognition_status, y_reference, deviation):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
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
  
                if (width_heigh_taken == True and mode == 0):
                  width_heigh_taken = False
                  height, width = frame.shape[:2]

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
                    color_recognition_status,                    
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    y_reference=y_reference,
                    deviation=deviation,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                
                    total_passed_object = total_passed_object + counter          

                    cv2.putText(input_frame,"Detected Objects: " + str(total_passed_object), (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)

                    # when the object passed over line and counted, make the color of ROI line green
                    if(counter == 1):
                            cv2.line(input_frame,(0,y_reference),(width,y_reference),(0,255,0),5)
                    else:
                            cv2.line(input_frame,(0,y_reference),(width,y_reference),(0,0,255),5)

                    # insert information text to video frame
                    #cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                    cv2.putText(input_frame,"ROI Line", (width-200, y_reference-10), font, 0.6,(0,0,255),2,cv2.LINE_AA)
                    #cv2.putText(input_frame,"LAST PASSED OBJECT INFO", (11, 290), font, 0.5, (255,255, 255), 1,cv2.FONT_HERSHEY_SIMPLEX)
                    #cv2.putText(input_frame,"-Movement Direction: " + direction, (14, 302), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Speed(km/h): " + speed, (14, 312), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Color: " + color, (14, 322), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Object Size/Type: " + size, (14, 332), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)

                elif (mode == 1):
                    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                          input_frame,
                                                                                                          mode,
                                                                                                          color_recognition_status,
                                                                                                          np.squeeze(boxes),
                                                                                                          np.squeeze(classes).astype(np.int32),
                                                                                                          np.squeeze(scores),
                                                                                                          category_index,
                                                                                                          y_reference=y_reference,
                                                                                                          deviation=deviation,
                                                                                                          use_normalized_coordinates=True,
                                                                                                          line_thickness=4)
                    if(len(counting_mode) == 0):
                        cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                    else:
                        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                cv2.imshow('object counting',input_frame)
                #output_video.write(input_frame)
                #print ("writeing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         
            cap.release()
            cv2.destroyAllWindows()

def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
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
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
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
                
                output_movie.write(input_frame)
                print ("writing frame")
                #cv2.imshow('object counting',input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()

def object_detection_function2(input_video, output_video, detection_graph, category_index, mode, color_recognition_status, y_reference, deviation):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        counting_mode = "..."
        width_heigh_taken = True
        height = 0
        width = 0
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
  
                if (width_heigh_taken == True and mode == 0):
                  width_heigh_taken = False
                  height, width = frame.shape[:2]

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
                    color_recognition_status,                    
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    y_reference=y_reference,
                    deviation=deviation,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                
                    total_passed_object = total_passed_object + counter          

                    cv2.putText(input_frame,"Detected Objects: " + str(total_passed_object), (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)

                    # when the object passed over line and counted, make the color of ROI line green
                    if(counter == 1):
                            cv2.line(input_frame,(0,y_reference),(width,y_reference),(0,255,0),5)
                    else:
                            cv2.line(input_frame,(0,y_reference),(width,y_reference),(0,0,255),5)

                    # insert information text to video frame
                    #cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                    cv2.putText(input_frame,"ROI Line", (width-200, y_reference-10), font, 0.6,(0,0,255),2,cv2.LINE_AA)
                    #cv2.putText(input_frame,"LAST PASSED OBJECT INFO", (11, 290), font, 0.5, (255,255, 255), 1,cv2.FONT_HERSHEY_SIMPLEX)
                    #cv2.putText(input_frame,"-Movement Direction: " + direction, (14, 302), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Speed(km/h): " + speed, (14, 312), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Color: " + color, (14, 322), font, 0.4, (255,255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)
                    #cv2.putText(input_frame,"-Object Size/Type: " + size, (14, 332), font, 0.4, (255, 255, 255), 1,cv2.FONT_HERSHEY_COMPLEX_SMALL)

                elif (mode == 1):
                    counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                          input_frame,
                                                                                                          mode,
                                                                                                          color_recognition_status,
                                                                                                          np.squeeze(boxes),
                                                                                                          np.squeeze(classes).astype(np.int32),
                                                                                                          np.squeeze(scores),
                                                                                                          category_index,
                                                                                                          y_reference=y_reference,
                                                                                                          deviation=deviation,
                                                                                                          use_normalized_coordinates=True,
                                                                                                          line_thickness=4)
                    if(len(counting_mode) == 0):
                        cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                    else:
                        cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                cv2.imshow('object counting',input_frame)
                #output_video.write(input_frame)
                #print ("writeing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         
            cap.release()
            cv2.destroyAllWindows()

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object, fps, width, height):
        #initialize .csv
        with open('object_counting_report.csv', 'w') as f:
                writer = csv.writer(f)  
                csv_line = "Object Type, Object Color, Object Movement Direction, Object Speed (km/h)"                 
                writer.writerows([csv_line.split(',')])

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        # input video
        cap = cv2.VideoCapture(input_video)

        total_passed_vehicle = 0
        speed = "waiting..."
        direction = "waiting..."
        size = "waiting..."
        color = "waiting..."
        the_result = "..."
        width_heigh_taken = True
        height = 0
        width = 0
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
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                
                #cv2.imshow('object counting',input_frame)

                output_movie.write(input_frame)
                print ("writing frame")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if(csv_line != "not_available"):
                        with open('traffic_measurement.csv', 'a') as f:
                                writer = csv.writer(f)                          
                                size, direction = csv_line.split(',')                                             
                                writer.writerows([csv_line.split(',')])         

            cap.release()
            cv2.destroyAllWindows()
