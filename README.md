# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission

### Project overview
The goal is to develop the most accurate approach for recognizing three classes of objects in an image dataset. Our dataset is the Waymo Open Dataset, and the three classes are cars, pedestrians, and cyclists. The desired level of accuracy is to achieve or surpass human-level performance, where a human can recognize the three classes of objects in less than one second per image. We will apply exploratory data analysis, use pre-trained object detection models to detect the classes, fine-tune hyperparameters, and augment our dataset to improve accuracy. Achieving higher accuracy will improve our ability to make better decisions and ensure the safety of self-driving cars. 
As a final step, we will use the evaluation dataset to predict a sequence of images and accurately classify the three classes in a way that is representative of real-world scenarios.

### Set up
At the top, you can find the steps to follow to run the code. I used the virtual machine provided by Udacity, which has a GPU that is important for training neural networks. To perform exploratory data analysis, open the `Exploratory Data Analysis.ipynb` notebook in Jupyter Notebook. To train the model, use `experiment-2` in the `experiment` directory. If you prefer to run the code locally, you can either use the Docker file or the requirements.txt file to install the necessary libraries.

### Dataset
#### Dataset analysis
My first step was to check the dataset. However, it wasn't as simple as just opening a folder and checking the images. I had to use the display_images function to view some examples of the classes. During this process, I discovered that the classification of the classes used label encoding, where 1 represented cars, 2 represented pedestrians, and 4 represented bicycles. However, I noticed that there was no class 3.
<img width="1243" alt="" src="https://user-images.githubusercontent.com/8989089/231610738-9fbc4046-1928-4c7d-807a-72cf7507688b.jpg">

For my analysis, I decided to shuffle the dataset and select 10,000 images. I believe that this subset of the dataset should provide a representative sample of the entire dataset.


<img width="512" alt="" src="https://user-images.githubusercontent.com/8989089/231611369-cd5b26a5-fe4d-4e0b-b4e8-b121a8a0cac8.png">


My recommendation is to add more pedestrians and bicycles to the dataset to create a more balanced dataset, which can help improve the prediction accuracy of these classes. Additionally, data augmentation techniques can be applied specifically to these classes to further improve their representation in the dataset.

#### Cross validation
In the project, a script was provided to split up the dataset. As a general guideline, the recommended split is 60%-70% for training, 20% for testing, and 10%-20% for evaluation. The evaluation dataset contains a continuous image sequence that can be used to create animations and simulate real-world situations. This suggests that using a recurrent neural network (RNN) may help improve the accuracy of the predictions. To avoid overfitting, the training data is shuffled.

Although the provided script splits the dataset into training, testing, and evaluation sets, there are many other methods for cross-validation that can be used, including: Leave p out cross-validation, Leave one out cross-validation, Holdout cross-validation, Repeated random subsampling validation, k-fold cross-validation, and others.

### Training
#### Reference experiment
I was surprised by the poor results of the first experiment. The model I used was a pre-trained Single Shot Detection ResNet50, but the test results showed a very high loss value and poor performance. The model was trained on images of size 640x640 with a batch size of 2 and around 2k steps.
<img width="1243" alt="Screenshot 2023-04-12 at 7 19 26 PM" src="https://user-images.githubusercontent.com/8989089/231613717-f7e8ada4-3c77-4525-8d1d-e833afab16b0.png">

After analyzing the loss and learning rate graphs, I realized that I needed to use a smaller learning rate for my model."
<img width="472" alt="Screenshot 2023-04-12 at 7 22 39 PM" src="https://user-images.githubusercontent.com/8989089/231614075-43657c90-cf56-42d1-82d4-aeb339c9c9c7.png">

#### Improve on the reference
##### Experiment 1
I started experimenting with the `Explore augmentations.ipynb` Notebook. The random_crop_image function worked well for me, but I set the random_horizontal_flip probability to 0.3 to reduce the likelihood of flipping the image. 
I also added several other augmentations, including: 
- random_image_scale
- random_adjust_hue
- random_adjust_saturation
- random_adjust_brightness
- random_adjust_contrast
- random_distort_color. 
 
These augmentations were meant to simulate different lighting and camera conditions, which can impact the model accuracy.

After that, I started testing with a learning rate of 0.004 and a batch size of 8 because 2 was too low, but a high number should have more resources. However, I realized that the learning rate should be very small because we are using a pre-trained model. If the optimizer overshoots the optimal weights, it can lead to poor convergence and high loss values.
After making these changes, the model was learning and producing better results.

##### Experiment 2
My model was ready for submission, but I wanted to improve its performance. Upon checking, I found that the tools available for preventing overfitting were limited to pre-processing, augmentation, and shuffling the dataset. To address this, I added some dropout configurations to the model, and further reduced the learning rate to 0.001. I used a dropout keep probability of 0.8 to prevent overfitting.

I would like to have used a bigger batch size and more epochs, but the resources of the workspace were limited. However, in the future, we can improve the resources available to us.

<img width="1118" alt="Screenshot 2023-04-12 at 8 08 25 PM" src="https://user-images.githubusercontent.com/8989089/231619553-560a6f55-9ce5-4826-a695-540f0f9afc54.png">

I saw that we were able to continue training without any overfitting problems and that we could have kept improving the model, but unfortunately, the resources available were limited. Therefore, I had to stop the project, but we achieved good results nonetheless.

<img width="1429" alt="Screenshot 2023-04-12 at 8 11 05 PM" src="https://user-images.githubusercontent.com/8989089/231619848-77ff8c45-9f27-4fbd-95f5-4f69eadb615d.png">

## Submission
<img width="1429" alt="Screenshot 2023-04-12 at 8 11 05 PM" src="https://github.com/wgcv/nd013-c1-vision-starter/raw/main/animation_compress.gif">

