# TensorFlow Object Counting API
The TensorFlow Object Counting API is an open source framework built on top of TensorFlow that makes it easy to develop object counting systems!

---
### Cumulative Counting Mode:
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166455-45964aac-8f9f-11e8-9ddf-f71d05f0c7f5.gif" | width=430> <img src="https://user-images.githubusercontent.com/22610163/43166945-c0744de0-8fa0-11e8-8985-9f863c59e859.gif" | width=411>
</p>

---
### Real-Time Counting Mode:
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42237325-1f964e82-7f06-11e8-966b-dfde98701c66.gif" | width=430> <img src="https://user-images.githubusercontent.com/22610163/42238435-77ac0d34-7f09-11e8-9609-e7c3c2c5af74.gif" | width=430>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42241094-14163cc8-7f12-11e8-83ed-68021b5e3b33.gif" | width=430><img src="https://user-images.githubusercontent.com/22610163/42237904-d6a3ac22-7f07-11e8-88f8-5f21430d9503.gif" | width=430>
</p>

---

***The development is on progress! The API will be updated soon, the more talented and light-weight API will be available in this repo!***

- ***Detailed API documentation and sample jupyter notebooks that explain basic usages of API will be added!***

**You can find a sample project - case study that uses TensorFlow Object Counting API in [this repo](https://github.com/ahmetozlu/vehicle_counting_tensorflow).**

---

## USAGE

### 1.) Usage of "Cumulative Counting Mode"

#### 1.1) For detecting, tracking and counting *the pedestrians* with disabled color prediction

*Usage of "Cumulative Counting Mode" for the "pedestrian counting" case:*

    fps = 30 # change it with your input video fps
    width = 626 # change it with your input video width
    height = 360 # change it with your input vide height
    is_color_recognition_enabled = 0

    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, 400) # 400 is the x location of the ROI Line
    
*Result of the "pedestrian counting" case:*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166945-c0744de0-8fa0-11e8-8985-9f863c59e859.gif" | width=700>
</p>

---

**Source code of "pedestrian counting case-study": [pedestrian_counting.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/pedestrian_counting.py)**

---

**1.2)** For detecting, tracking and counting *the vehicles* with enabled color prediction

*Usage of "Cumulative Counting Mode" for the "vehicle counting" case:*

    fps = 24 # change it with your input video fps
    width = 640 # change it with your input video width
    height = 352 # change it with your input vide height
    is_color_recognition_enabled = 0

    object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, 200) # 200 is the y location of the ROI Line
    
*Result of the "vehicle counting" case:*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166455-45964aac-8f9f-11e8-9ddf-f71d05f0c7f5.gif" | width=700>
</p>

---

**Source code of "vehicle counting case-study": [vehicle_counting.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/vehicle_counting.py)**

---

### 2.) Usage of "Real-Time Counting Mode"

#### 2.1) For detecting, tracking and counting the *targeted object/s* with disabled color prediction
 
 *Usage of "the targeted object is bicycle":*
 
    targeted_objects = "bicycle"
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height
    is_color_recognition_enabled = 0

    object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
    
 *Result of "the targeted object is bicycle":*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411751-1ae1d3f0-820a-11e8-8465-9ec9b44d4fe7.gif" | width=700>
</p>

*Usage of "the targeted object is person":*

    targeted_objects = "person"
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height
    is_color_recognition_enabled = 0

    object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
 
 *Result of "the targeted object is person":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411749-1a80362c-820a-11e8-864e-acdeed85b1f2.gif" | width=700>
</p>

*Usage of "detecting, counting and tracking all the objects":*

    is_color_prediction_enabled = 0
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
 
 *Result of "detecting, counting and tracking all the objects":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411750-1aae0d72-820a-11e8-8726-4b57480f4cb8.gif" | width=700>
</p>
 
 
#### 1.2) For detecting, tracking and counting "all the objects with disabled color prediction"

*Usage of detecting, counting and tracking "all the objects with disabled color prediction":*
    
    is_color_prediction_enabled = 0
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
    
 *Result of detecting, counting and tracking "all the objects with disabled color prediction":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411748-1a5ab49c-820a-11e8-8648-d78ffa08c28c.gif" | width=700>
</p>


*Usage of detecting, counting and tracking "all the objects with enabled color prediction":*

    is_color_prediction_enabled = 1
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
    
 *Result of detecting, counting and tracking "all the objects with enabled color prediction":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411747-1a215e4a-820a-11e8-8aef-faa500df6836.gif" | width=700>
</p>

---

**For sample usages of "Real-Time Counting Mode": [test.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/test.py)**

---

## General Capabilities of The TensorFlow Object Counting API

Here are some cool capabilities of TensorFlow Object Counting API:

- Detect just the targeted object/s
- Detect all the objects
- Count just the targeted object/s
- Count all the objects
- Predict color of the targeted object/s
- Predict color of all the objects
- Predict speed of the targeted object/s
- Predict speed of all the objects
- Print out the detection-counting result in a .csv file as an analysis report
- Save and store detected objects as new images under [detected_object folder](www)
- Select, download and use state of the art [models that are trained by Google Brain Team](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- Use [your own trained models](https://www.tensorflow.org/guide/keras) or [a fine-tuned model](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb) to detect spesific object/s
- Save detection and counting results as a new video or show detection and counting results in real time
- Process images or videos depending on your requirements

Here are some cool architectural design features of TensorFlow Object Counting API:

- Lightweigth, runs in real-time
- Scalable and well-designed framework, easy usage
- Gets "Pythonic Approach" advantages
- It supports REST Architecture and RESTful Web Services

## Theory

### Architecture

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42447990-5ef59ace-8384-11e8-8b0f-4a7522d47fc8.jpg" | width=700>
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

### Models

By default I use an ["SSD with Mobilenet" model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17) in this project. You can find more information about SSD in [here](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab). 

Please, See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies. You can easily select, download and use state-of-the-art models that are suitable for your requeirements using TensorFlow Object Detection API.

## Project Demo

Demo video of the project is available on [My YouTube Channel](https://www.youtube.com/watch?v=bas6c8d1JyU).

## Installation

### Dependencies

Tensorflow Object Counting API depends on the following libraries:

- TensorFlow Object Detection API
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

For detailed steps to install Tensorflow, follow the [Tensorflow installation instructions](https://www.tensorflow.org/install/). 

TensorFlow Object Detection API have to be installed to run TensorFlow Object Counting API, for more information, please see [this](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

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
