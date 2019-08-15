# Smurf Counting

In this project readers will learn how to train TensorFlow models with their own training data to built their own custom object counter system! It covers some of the theory of transfer learning and show how to apply it in useful projects.

## QUICK DEMO

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/62861574-9d6e0080-bd0c-11e9-9e38-b63226df8aa1.gif" | width=750>
</p>

## THEORY

### Converting XML to TFRecord

To convert XML files to TFRecord, first, they are needed to be converted to CSV. [**Here**](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/smurf_counter_training/xml_to_csv.py) is the code that converts your XML files to CSV files.

Once, the XML files have been converted to CSV files, TFRecords can be generated using a python which can be found in [**here**](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/smurf_counter_training/generate_tfrecord.py) script from the same repository with some changes. Using Tensorflow TFRecords is a convenient way to get your data into your machine learning pipeline.

Type this in your terminal to generate the tfrecord for the training data:

    python generate_tfrecord.py — csv_input=data/train_labels.csv — output_path=data/train.record

Ahh finally!! we have our train.record. Similarly do this for the test data:

    python generate_tfrecord.py — csv_input=data/test_labels.csv — output_path=data/test.record

### Training Smurf Detector

To get our Smurf detector we can either use a pre-trained model and then use transfer learning to learn a new object, or we could learn new objects entirely from scratch. The benefit of transfer learning is that training can be much quicker, and the required data that you might need is much less. For this reason, we’re going to be doing transfer learning here. TensorFlow has quite a few pre-trained models with checkpoint files available, along with configuration files.

For this task I have used Faster R-CNN Inception. You can always use some other model. You can get a list of models and their download links from here. To get the configuration file of your corresponding model [**click here**](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). After you download your model and your configuration file, you will need to edit the configuration file according to your dataset.

In your configuration file search for “PATH_TO_BE_CONFIGURED” and change it to something similar to what has been shown in [**here**](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/smurf_counter_training/legacy/training/faster_rcnn_inception_v2_coco.config). Also, change the number of classes in the config file.

One last thing that still remains before we can start training is creating the label map. Label map is basically a dictionary which contains the id and name of the classes that we want to detect. You can find it in [**here**](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/smurf_counter_training/legacy/training/detection.pbtxt).

That’s it. That’s it. We can now start training. Ahh finally!
Copy all your data to the cloned tensorflow repository on your system (Clone it if you haven’t earlier). And from within ‘models/object_detection’ type this command in your terminal.

    python3 train.py --logtostderr --train_dir=training/--pipeline_config_path=training/faster_rcnn_inception_v2_coco_2018_01_28.config

You can wait until the total loss reaches around 1.

### Testing Smurf Detector

To test how good our model is doing, we need to export the inference graph. In the ‘models/object_detection’ directory, there is a script that does this for us: ‘export_inference_graph.py’

To run this, you just need to pass in your checkpoint and your pipeline config. Your checkpoint files should be in the ‘training’ directory. Just look for the one with the largest step (the largest number after the dash), and that's the one you want to use. Next, make sure the pipeline_config_path is set to whatever config file you chose, and then finally choose the name for the output directory. For example:

    python3 export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path training/ssd_inception_v2_coco_2017_11_17.config \
        --trained_checkpoint_prefix training/model.ckpt-7051 \
        --output_directory logos_inference_graph

Once this runs successfully you should have a new directory with the name ‘frozen_inference_graph’. That's all! You performed transfer learning and got your own custom frozen inference graph! Congrat! :) You can start to count your own cutom objects using [**tensorflow_object_counting_api**](https://github.com/ahmetozlu/tensorflow_object_counting_api)!

## Citation
If you use this code for your publications, please cite it as:

    @ONLINE{tfocapi,
        author = "Ahmet Özlü",
        title  = "TensorFlow Object Counting API",
        year   = "2018",
        url    = "https://github.com/ahmetozlu/tensorflow_object_counting_api"
    }

## Author
Ahmet Özlü

## License
This system is available under the MIT license. See the LICENSE file for more info.

