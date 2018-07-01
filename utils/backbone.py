import glob, os
import tensorflow as tf

def set_model(model_name):
	model_found = 0

	for file in glob.glob("*"):
		if (file == model_name):
			model_found = 1

	# What model to download.
	model_name = model_name
	model_file = model_name + '.tar.gz'
	download_base = 'http://download.tensorflow.org/models/object_detection/'

	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	path_to_ckpt = model_name + '/frozen_inference_graph.pb'

	# List of the strings that is used to add correct label for each box.
	path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')

	num_classes = 90

	# Download Model if it has not been downloaded yet
	if (model_found == 0):		
		opener = urllib.request.URLopener()
		opener.retrieve(download_base + model_file, model_file)
		tar_file = tarfile.open(model_file)
		for file in tar_file.getmembers():
		  file_name = os.path.basename(file.name)
		  if 'frozen_inference_graph.pb' in file_name:
		    tar_file.extract(file, os.getcwd())

	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
	  od_graph_def = tf.GraphDef()
	  with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
	    serialized_graph = fid.read()
	    od_graph_def.ParseFromString(serialized_graph)
	    tf.import_graph_def(od_graph_def, name='')

	return path_to_labels, num_classes, detection_graph
