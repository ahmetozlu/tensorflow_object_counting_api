#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------
# author      : Ahmet Ozlu
# mail        : ahmetozlu93@gmail.com
# date        : 05.05.2019
# -----------------------------------------

import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob

# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import custom 

def predict (src_image):
    
	# Root directory of the project
	ROOT_DIR = os.getcwd()
	sys.path.append(ROOT_DIR)  # To find local version of the library

	# Root directory of the project
	ROOT_DIR = os.getcwd()


	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	custom_WEIGHTS_PATH = "mask_rcnn_barilla-spaghetti_0040.h5"

	# Configurations
	config = custom.CustomConfig()
	custom_DIR = os.path.join(ROOT_DIR, "customImages")

	class InferenceConfig(config.__class__):
		# Run detection on one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()

	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
	TEST_MODE = "inference"

	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights
	print("Loading weights ", custom_WEIGHTS_PATH)
	model.load_weights(custom_WEIGHTS_PATH, by_name=True)

	try:

		image = src_image
		# Run object detection
		results = model.detect([image], verbose=1)

		# results
		r = results[0]
		#print(str(r))
		class_names = ['BG', 'book']
		masked_image_result = visualize.finalize(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Predictions")
		cv2.imwrite("result.png", masked_image_result.astype(np.uint8))
		print("result image saved")
            
	except Exception as e:
	    print (e)
