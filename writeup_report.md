#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/model.png "Model Visualization"
[center]: ./images/center.png "Center camera"
[left]: ./images/left.png "Left camera"
[right]: ./images/right.png "Right camera"
[postflip]: ./images/post-flipped.png "Flipped Image"
[postbright]: ./images/post-brightened.png "Brightened Image"
[postwarp]: ./images/post-warp.png "Original Post-Warp image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My final model (model.py lines 103-132) consists of convolution layers of 1x1, 3x3 and 5x5 filter sizes of depths between 3 and 64.

The model includes RELU layers to introduce nonlinearity (code line 101), and the data is normalized in the model using a Keras lambda layer (code line 105). 


####2. Attempts to reduce overfitting in the model

The model contains BatchNormalization layers in order to reduce overfitting (model.py lines 111). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 140). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 135).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used provided center lane driving data and a reverse route. I did not add any additional recovery data. The next track was too hard for even me to drive it, so I could not make use of that.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start simple, add more layers if underfitting, add more augmentation if overfitting. Then finally BatchNormalization after last overfit and train for a long time.

I first started with LeNet because I understood it from class. It did surpringly well for starters but it confusing patch of dirt after bridge with the road and would sometimes pull left.

I switched to a convolution neural network model similar to the NVIDIA model[https://arxiv.org/abs/1604.07316] I thought this model might be appropriate because it was trained on the task. It was a little peculiar because everything was done through convolutions. Instead of pooling, it seems like they are using bigger strides. This is probably for speed but I cannot say for certain.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added augmented images from left and right cameras as well as random image flips. Then the model underfit and would drive off sometimes.

I added 1x1 convolutions to learn the color space on top.

I then added some random image contrasts and warps after reading this [https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9] and the model worked well.

I added a reverse lap just for fun. I did not gather data on the more difficult track, because it was hard even for me manually to keep the car on the road.

Then I finally added Batch Normalization and tried to run it for 5 epochs, but saved checkpoints on every one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

Here is a visualization of the architecture

![alt text][model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from off-center positions. These images show what a recovery looks like starting from left and right respectively:

![alt text][left]
![alt text][right]

I hard-coded correction of 0.2 from either side for now.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would simulate driving the other way on the track. For example, here is an image that has then been flipped:

![alt text][center]
![alt text][postflip]

This stopped the car from veering left.

I added random brightness to the image. Here is an example

![alt text][center]
![alt text][postbright]

This seems to have helped with the dirt road.

I also added a random warps to the image to help with turns

![alt text][center]
![alt text][postwarp]

After the collection process, I had 10,559 data points, times 3 as I used left and right images. The augmentation was done with independant probabilities and it is possible for something to be both brightened and warped. I only cropped the top (sky) and bottom (the hood). I left that as part of the first model layer so it does the same thing in predict mode.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. I saved on every epoch and evaluated which checkpoint to use after via learning curves. Oddly, it worked fine with not just the lowest validation error but the first one that got us in that loss range, third epoch. We got to val_error of 0.0262 in second third epoch already. It could already drive around the lap without any issues. Fourth epoch brought it down further to 0.0235 but at this point, weare probably overfitting. So I used the third epoch one as the final model.

I also had to change samples_per_batch parameter to 3 * len(train_samples)/BATCH_SIZE because we are generating 3x examples per batch with left and right camera use. 

The car drove around the lap nicely.

