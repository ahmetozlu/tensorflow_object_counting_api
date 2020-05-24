import cv2
import os

vehicle_count = [0]

current_path = os.getcwd()

def save_image(source_image):
	cv2.imwrite(current_path + "/detected_objects/object" + str(len(vehicle_count)) + ".png", source_image)
	vehicle_count.insert(0,1)
