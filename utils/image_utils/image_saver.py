import cv2
import os

object_count = [0]

current_path = os.getcwd()
the_path = current_path + "/detected_objects/"
def save_image(source_image):
	cv2.imwrite(the_path + "object" + str(len(object_count)) + ".png", source_image)
	object_count.insert(0,1)
	print("*detected object image saved: "+ the_path)
