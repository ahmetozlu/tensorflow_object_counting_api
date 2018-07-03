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
- Print out the detection-counting result in a .csv file as analysis report
- Save and store detected objects as new images under [detected object folder](www)
- Select, download  and use desired [models that are trained by Google Brain Team](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- Save detection and counting results as a new video or show detection and counting results in real time

Here are some cool architectural design features of TensorFlow Object Counting API:

- Lightweigth, runs in real-time
- Scalable and well-designed framework, easy usage
- Get "Pythonic Approach" advantages
- It supports REST Architecture and RESTful Web Services

## Theory

### Tracker
Source video is read frame by frame with OpenCV. Each frames is processed by ["SSD with Mobilenet" model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17) is developed on TensorFlow. This is a loop that continue working till reaching end of the video. The main pipeline of the tracker is given at the above Figure.


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
