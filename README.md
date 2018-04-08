
# Behavioral Cloning**

The steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_cam.jpg "Grayscaling"
[image3]: ./examples/left_cam.jpg "Recovery Image"
[image4]: ./examples/right_cam.jpg "Recovery Image"
[image5]: ./examples/original.jpg "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[image7]: ./examples/loss.png "Loss Graph"


---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 3 convolution neural network with 5x5 filter sizes, strides=(2,2) and depths between 24 and 48 (model.py lines 78-86), 2 convolution neural network with 3x3 filter sizes, no strides and depths 64 (model.py lines 190-194).

The model includes RELU layers to introduce nonlinearity (code line 178-194), and the data is normalized in the model using a Keras lambda layer (code line 175). 

The model consists of 4 fully conection layers with no activation and Neurons from 100, 50, 10 to 1(model.py lines 200-210).

#### 2. Attempts to reduce overfitting in the model 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 49,166 and 167). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 213).

#### 4. Appropriate training data
In order to make the network work faster and reduce the noise of data, I cropped all the images from shape (160,320,3) into shape(90,320,3)

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, to avoid overfitting, I flipped the images to double the dataset

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to transfer successful network to fit this project.

My first step was to use a convolution neural network model similar to the network published in the paper 'End to End Learning for Self-driving Cars' by NVIDIA, I thought this model might be appropriate because it is initially designed for end to end problem and  it has proved successful.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I added a maxpooling layer after layer conv4, so I could get data with shape=(1,15,64) after layer conv5, which is more like the how the original network works.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
	1.Cropping(input:160x320x3, ouput:90x320x3)
	2.Normalization(lambda x: x/255 - 0.5)
	3.Convolution2D_1(krenel:5x5, strides:2x2, output:43x158x24)
	4.Convolution2D_2(kernel:5x5, strides:2x2, output:20x76x36)
	5.Convolution2D_3(kernel:5x5, strides:2x2, output:8x36x48)
	6.Convolution2D_4(kernel:3x3, output:6x34x64)
	7.MaxPooling2D(kernel:2x2, output:3x17x64)
	8.Convolution2D_5(kernel:3x3, output:1x15x64)
	9.Flatten
	10.Dense_1(output neuronss:100)
	11.Dense_2(output neuronss:50)
	12.Dense_3(output neuronss:10)	
	13.output(steering angle)	

Here is a visualization of the architecture (update later...)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust steering angle to drive back to the center of the road when it is about to fell off the track. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]


To augment the data set, I also flipped images and angles thinking that this would avoid underfitting, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had 36162 number of data points. I then preprocessed this data by cropping and normalization.


I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5(not sure yet, beacause I got a strage loss graph though the model ran good on the simulator test when epoches=5)  as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary. Here is the loss graph:

![alt text][image7]
