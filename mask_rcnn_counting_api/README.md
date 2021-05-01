# Mask R-CNN Object Counting API

This is an Object Counting API, which is an open source framework built on top of the implementation of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow, makes it easy to develop object counting systems! You can check the [Mask R-CNN](https://arxiv.org/abs/1703.06870) paper to get more information about its theory!

## QUICK DEMO

### Detect and Count:
<p align="center">
<img src="https://user-images.githubusercontent.com/22610163/57814531-97539c80-777c-11e9-9bf1-6c44d5304248.png" | width=750></p>

### Detect, Segment and Count:
<p align="center">
<img src="https://user-images.githubusercontent.com/22610163/57814689-2bbdff00-777d-11e9-9229-e2c5749d26e8.png" | width=750></p>

## USAGE

- Image Processing:

Here are [a sample usage](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/mask_rcnn_counting_api_keras_tensorflow/single_image_object_counting.py#L636):

    masked_image = get_masked_fixed_color(im, r['rois'], r['masks'], r['class_ids'], class_names, colors, r['scores'], show=False)
    
Make "show=True" to enable masking operation!

- Video Processing:

You can find [a sample python program](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/mask_rcnn_counting_api_keras_tensorflow/real_time_object_counting.py) for video processing to count custom objects via Mask R-CNN.

## Theory

You can check the [Mask R-CNN](https://arxiv.org/abs/1703.06870) paper to get more information about its theory!

### Model

I use the pre-trained [COCO weights](https://github.com/matterport/Mask_RCNN/releases). You can find more information about Mask R-CNN in [here](https://github.com/matterport/Mask_RCNN). 

* Architecture of Mask RCNN structure

<p align="center">
<img src="https://user-images.githubusercontent.com/22610163/57968672-195ae580-7976-11e9-96f7-33ac0f99e232.png" | width=600></p>

* Illustration of Mask RCNN structure

<p align="center">
<img src="https://user-images.githubusercontent.com/22610163/57968609-8752dd00-7975-11e9-9221-455c61c0b332.jpeg" | width=600></p>

## Installation

1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

### Dependencies

Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in requirements.txt.

MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{tfocapi,
        author = "Ahmet Özlü",
        title  = "TensorFlow Object Counting API",
        year   = "2019",
        url    = "https://github.com/ahmetozlu/tensorflow_object_counting_api/tree/master/mask_rcnn_counter"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.
