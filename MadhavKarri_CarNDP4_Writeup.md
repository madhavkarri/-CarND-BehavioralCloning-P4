# **Self-Driving Car Engineer Nanodegree**
# **Project4: Behavioral Cloning**

## MK

Overview
---
Develop a convolutional neural network model using keras to clone driving behavior. Train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

The Project
---
The goals/steps for this project:
* Use simulator to collect data for good driving behavior
* Build convolution neural network in Keras, that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./Writeup_IV/CNNModelSummary.png "CNNModelSummary"
[image2]: ./Writeup_IV/CNNArch.png "CNNArch"


## [Rubric](https://review.udacity.com/#!/rubrics/432/view) Points

### Consider the rubric points individually and describe how each point has been addressed.

---
### Files Submitted & Code Quality

#### 1. Files included with submission to run the simulator in autonomous mode

Project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* MadhavKarri_CarNDP4_Writeup.md summarizes the results

#### 2. Functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Project code (model.py)

The model.py file contains the code for the following set of tasks.
* Load all images and steering angles from memory
* Data augumentation using vertical flip
* Data normilization and centering
* Image cropping to retain, only pixels that contain useful information
* Define and build convolution neural network model
* Training, validation, and saving the model

### Model Architecture and Training Strategy

#### 1. Model architecture 

Model consists of a convolution neural network with the following set of features 
![][image1]
![][image2]

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 3. Appropriate training data

Training data was collected to keep the vehicle driving on the road. Used only center lane driving. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
