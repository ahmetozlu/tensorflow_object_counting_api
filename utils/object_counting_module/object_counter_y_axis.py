from utils.image_utils import image_saver

is_object_detected = [0]
bottom_position_of_previous_detected_object = [0]

def count_objects(top, bottom, right, left, crop_img, roi_position, y_min, y_max, deviation):   
        direction = "n.a." # means not available, it is just initialization
        isInROI = True # is the object that is inside Region Of Interest
        update_csv = False

        if (abs(((bottom+top)/2)-roi_position) < deviation):
            is_object_detected.insert(0,1)
            update_csv = True
            image_saver.save_image(crop_img) # save detected object image

        if(bottom > bottom_position_of_previous_detected_object[0]):
            direction = "down"
        else:
            direction = "up"

        bottom_position_of_previous_detected_object.insert(0,(bottom))

        return direction, is_object_detected, update_csv

