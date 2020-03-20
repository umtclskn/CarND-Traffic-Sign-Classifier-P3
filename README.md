
## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image1_2]: ./examples/bar_chart.png "Bar Chart"
[image2]: ./examples/gray_scale.png  "Grayscaling"
[image2_1]: ./examples/rgb_scale.png  "RGB Color"
[image3]: ./examples/augmented_images.png "Augmented Images"
[image4]: ./examples/test_datas.png "Samples"

You're reading it! and here is a link to my [project code]()

### Data Set Summary & Exploration

#### 1. In the code, the analysis should be done using python, numpy  methods rather than hardcoding results manually.

* I used the numpy library to calculate summary statistics of the traffic signs.


signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
*  I used the matplotlib library to visualization summary statistics of the traffic signs.

Here is an exploratory histogram of the data set. 

![alt text][image1]

It is a bar chart showing how the train, test, validation datasets' distribustion bar chart.

![alt text][image1_2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because we try to find a specific pattern by taking samples from the images we have, so the color layer can be overlooked. For example, it is necessary to determine the image features that cause the letter A to be a yellow, red or pink and beyond A. That's why we processed our image by turning it into a gray color space.

Another advantage is that our matrix operations are performed in one dimension instead of 3 dimensions, and advantage is obtained from the space occupied in performance and RAM.

Firstt sample from RGB Color space:
![alt text][image2_1]

Here is an example grayscaling image samples:

![alt text][image2]

As a last step, I normalized the image data because we use the tensorflow and find weights and biases. Tensorflow use statistical functions when finding weight and biases so we need valid probability distrubition  to determine contrasts the features of the image.

I decided to create additional data because, as we see in the histogram, there is quite a difference between some classes. For example, class 0 has 180 images in total, while class 1 has nearly 2000 images. We will try to close this distribution by creating additional images. We will use the following functions for this. Each function adheres to a sample from that class, creating a different image. For example, by rotating 45 degrees, we get a new image example.
>def affine_transoform(img):
>def rotate_r_45(img):
>def rotate_l_45(img):
>def noise(img):
>def flip(img):

Here is an example of an original image and an augmented image. The difference between the original data set and the augmented data set is the following :

![alt text][image3]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 		|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 						|
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16 	|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 						|
| Flatten	        	| outputs 400 										|
| Fully connected		| outputs 120  										|
| RELU					|													|
| Dropout				| keep probability = 0.75 							|
| Fully connected		| outputs 84  										|
| RELU					|													|
| Dropout				| keep probability = 0.75 							|
| Fully connected		| outputs 43 logits  								|
|						|													|
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Number of epochs = 10. Experimental way: increasing of this parameter doesn't give significant improvements.
* Batch size = 128
* Learning rate = 0.001
* Optimizer - Adam algorithm (alternative of stochastic gradient descent). Optimizer uses backpropagation to update the network and minimize training loss.
* Dropout = 0.75 (for training set only)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.940 
* test set accuracy of 0.919

If an iterative approach was chosen:
* Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
* Number of EPOCHs = 25
* Batch Size = 128
* Learning Rate = 0.001
* Use the Adam Optimizer for backprop to minimize the training loss.
* Use the dropuot for regularization of parameters (features map) 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The second image might be difficult to classify because it has a dark texture.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)sign   									| 
| No passing     			| No passing										|
| Priority road					| Priority road											|
| Speed limit (80km/h)	      		| Speed limit (80km/h)					 				|
| Dangerous curve to the left			| Dangerous curve to the left       							|


The model was able to correctly guess %100. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0 ), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit (30km/h)   									| 
| 0.     				| Speed limit (80km/h) 										|
| 0.					| Speed limit (50km/h) 											|
| 0.	      			| Speed limit (20km/h)  					 				|
| 0.				    | End of speed limit (80km/h)        							|


For the second image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| No passing (30km/h)   									| 
| 0.     				| End of all speed and passing limits  										|
| 0.					| No passing for vehicles over 3.5 metric tons 											|
| 0.	      			| No entry  					 				|
| 0.				    | Ahead only        							|

For the thirdimage 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Priority road    									| 
| 0.     				| Roundabout mandatory 										|
| 0.					| No vehicles 											|
| 0.	      			|   	End of no passing by vehicles over 3.5 metric tons				 				|
| 0.				    | Speed limit (20km/h)        							|

For the fourth image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Speed limit (80km/h)    									| 
| 0.     				| Speed limit (100km/h)  										|
| 0.					| Vehicles over 3.5 metric tons prohibited 											|
| 0.	      			| Speed limit (60km/h)  					 				|
| 0.				    | Speed limit (120km/h)        							|

For the fourth image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Dangerous curve to the left    									| 
| 0.     				| Slippery road									|
| 0.					| Dangerous curve to the right 											|
| 0.	      			| Double curve 					 				|
| 0.				    | Children crossing       	

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



