# TensorFlow Object Counting API
The TensorFlow Object Counting API is an open source framework built on top of TensorFlow that makes it easy to develop object counting systems.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42237325-1f964e82-7f06-11e8-966b-dfde98701c66.gif" | width=430> <img src="https://user-images.githubusercontent.com/22610163/42238435-77ac0d34-7f09-11e8-9609-e7c3c2c5af74.gif" | width=430>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42241094-14163cc8-7f12-11e8-83ed-68021b5e3b33.gif" | width=430><img src="https://user-images.githubusercontent.com/22610163/42237904-d6a3ac22-7f07-11e8-88f8-5f21430d9503.gif" | width=430>
</p>

**DEVELOPMENT IS ON PROGRESS! DETAILED API DOCUMENTATION AND SAMPLE JUPYTER NOTEBOOKS THAT EXPLAIN BASIC USAGES OF API WILL BE SHARED SOON!**

Here are some cool capabilities of TensorFlow Object Counting API:

- Detect just targeted object/s
- Count just targeted object/s
- Predict color of the targeted object/s
- Predict speed of the targeted object/s
- Print out the detection-counting result in a .csv file as an analysis report
- Save and store detected objects as new images under [detected object folder](www)
- Select, download  and use state of the art [models that are trained by Google Brain Team](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- Save detection and counting results as a new video or show detection and counting results in real time
- Process images or videos depending your requirements

Here are some cool architectural design features of TensorFlow Object Counting API:

- Lightweigth, runs in real-time
- Scalable and well-designed framework, easy usage
- Gets "Pythonic Approach" advantages
- It supports REST Architecture and RESTful Web Services

***You can find a sample project - case study that uses TensorFlow Object Counting API in [this repo](https://github.com/ahmetozlu/vehicle_counting_tensorflow).***

## Theory

### Architecture

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42291975-db58d884-7fd7-11e8-848d-52cb79d36f7a.png" | width=700>
</p>

- Object detection and classification have been developed on top of TensorFlow Object Detection API, [see](https://github.com/tensorflow/models/tree/master/research/object_detection) for more info.

- Object color prediction has been developed using OpenCV via K-Nearest Neighbors Machine Learning Classification Algorithm is Trained Color Histogram Features, [see](https://github.com/ahmetozlu/tensorflow_object_counting_api/tree/master/utils/color_recognition_module) for more info.

[TensorFlow™](https://www.tensorflow.org/) is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

[OpenCV (Open Source Computer Vision Library)](https://opencv.org/about.html) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products.

### Tracker

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/41812993-a4b5a172-7735-11e8-89f6-083ec0625f21.png" | width=700>
</p>

Source video is read frame by frame with OpenCV. Each frames is processed by ["SSD with Mobilenet" model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17) is developed on TensorFlow. This is a loop that continue working till reaching end of the video. The main pipeline of the tracker is given at the above Figure.

### Model

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/41813283-79528968-773b-11e8-8069-0494cd59a813.png" | width=700>
</p>

By default I use an ["SSD with Mobilenet" model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17) in this project. You can find more information about SSD in [here](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab). See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies. You can easily select, download and use the models that is suitable for your requeirements using TensorFlow Object Detection API.

## Project Demo

Demo video of the project is available on [My YouTube Channel](https://www.youtube.com/watch?v=bas6c8d1JyU).

## Installation

### Dependencies

Tensorflow Object Detection API depends on the following libraries:

- Protobuf 3.0.0
- Python-tk
- Pillow 1.0
- lxml
- tf Slim (which is included in the "tensorflow/models/research/" checkout)
- Jupyter notebook
- Matplotlib
- Tensorflow
- Cython
- contextlib2
- cocoapi

For detailed steps to install Tensorflow, follow the [Tensorflow installation instructions](https://www.tensorflow.org/install/). For more information, please see [this](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{tfocapi,
        author = "Ahmet Özlü",
        title  = "TensorFlow Object Counting API
        year   = "2018",
        url    = "https://github.com/ahmetozlu/tensorflow_object_counting_api"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.
