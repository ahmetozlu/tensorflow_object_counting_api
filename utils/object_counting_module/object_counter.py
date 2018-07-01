from utils.image_utils import image_saver

is_vehicle_detected = [0]
bottom_position_of_detected_vehicle = [0]

def count_objects(top, bottom, right, left, crop_img, roi_position, y_min, y_max):	
	direction = "n.a." # means not available, it is just initialization
	isInROI = True # is the object that is inside Region Of Interest
	update_csv = False

	if (y_min<bottom and bottom<y_max and roi_position<bottom):
		is_vehicle_detected.insert(0,1)
		update_csv = True
		image_saver.save_image(crop_img) # save detected vehicle image

	if(bottom > bottom_position_of_detected_vehicle[0]):
		direction = "down"
	else:
		direction = "up"

	bottom_position_of_detected_vehicle.insert(0,(bottom))

	return direction, is_vehicle_detected, update_csv
