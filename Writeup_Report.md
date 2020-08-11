# **Traffic Sign Recognition** 

## Report


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./histogram.png "Visualization"
[image2]: ./grayScaled2.jpg "Grayscaling"
[image3]: ./grayScaled.jpg "Grayscaling"
[image4]: ./extra-data/Sample_1.jpg "Label 25"
[image5]: ./extra-data/Sample_2.jpg "Label 23"
[image6]: ./extra-data/Sample_3.jpg "Label 13"
[image7]: ./extra-data/Sample_4.jpg "Label 30"
[image8]: ./extra-data/Sample_6.jpg "Label 37"
[image9]: ./extra-data/Sample_7.jpg "Label 20"
[image10]: ./extra-data/Sample_10.jpg "Label 22"
[image11]: ./img_b4_warp.jpg "Image before warp"
[image12]: ./warped.png "Example of added image to training"
[image13]: ./histogram_added_images.png "Histogram after adding data"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to extract the statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the distribution of the dataset that is getting trained on:

![alt text][image1]

It shows that some images are trained by many pictures while the rest have notably lower traning sets. The disparity of the training sets can impact the results of the final estimated W.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale after comparing the results of colored data and gray scaled, eventhough the test results where higher by 1% but on the extra images the colored data didn't pefrom better than using the gray scale. The results of using colored image availible on another notebook called copy_Traffic_Sign_Classifier_colored.ipynb 

As a last step, I normalized the image data since it doesnt change the content of the image as explained in the lectures but it makes it easier for optimization. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]


I decided to generate additional data because of the imbalance in the training dataset. To add more data to the the data set, a warping and rotating the availible images used, only for the labels that had lower number of traing dataset.   

Here is an example of an original image and an augmented image:

![alt text][image11]
![alt text][image12]

The histogram of the augmented dataset:

![alt text][image13]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 30x30x30 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 26x26x40 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 24x24x40 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 23x23x40 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 19x19x50 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 17x17x50 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 16x16x50 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 12x12x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 9x9x64 					|
| Dropout	            | 0.3											|
| Convolution 3x3	    | 1x1 stride, Same padding, outputs 9x9x84		|
| Fully connected		| Input 4096, Output 120						|
| Dropout	            | 0.5											|
| Fully connected		| Input 120, Output 84  						|
| Fully connected		| Input 84, Output n-classes					|
| Softmax				| etc.        									|
|						|												|
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model: given the logits, used softmax_cross_enthropy to compare the estimated labels with ground truth to estimate the cross_entropy. To optimize the loss, AdamOptimizer is used finding the min error and updating weights and Biases with learning rate of 0.001.
BATCH_SIZE = 128, Epoch=20

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.981
* Validation Accuracy = 0.978 
* Test Accuracy = 0.960

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 

The model had fewer layers and thus could extract/learn less number of features. The first model was inputed coloured images and once compared to the gray scaled results, it was obvious that the gray scale performed much better on the extra data and slightly better on the Validation and test dataset. 

* What were some problems with the initial architecture?

It was saturated at validation rate of 0.93 but on the test dataset it performed around 0.8, possibly the network was overfited and meorized patterns.  

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Using LeNet model reached the Validation Accuracy of 0.911 after 10 epoch. So in order to improve the model implemented a network called Net_Improved which doesn't have dropouts, and reach Validation Accuracy of 0.975 after 10 epoch and after 20 epoch 0.981. Then added a dropout to the improved model and prevent the model to overfilt; called Net_Improved_Dropout which reached Validation Accuracy of 0.960 after 10 epoch and after 20 epoch reached 0.978; which was slightly lower than the one without the dropouts. The Third model experimented was similar to VGG, after 10 epoch Validation Accuracy reached 0.960 and after 20 epoch 0.973. It seems this model had a slower learning and possibly less overfitting. However, on the extra images the VGG didn't perform as the other two, the other networks performed about 10% better on the extra images collected from internet.



* Which parameters were tuned? How were they adjusted and why?

The combination of changes experimented such as filter and layer sizes such as depth at each layer, padding and strides. Once these changes appeared to have not impacted the results, the increase in network depth meaning adding additional layers implemented. For that reason, to be able to have many layers, improving the extraction of the higher number of features, same padding with a unit stride implemented. Thus at each convolution layer the network preserves spatial dimension and the output can be fed to a higher number of convolution layers.



* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The design change that had a significant improvament was the depth of the newtwork; once the number of layers increased the accuracy of validation and test datasets improved significantly. Surprizingly for the same network adding dropouts didn't change the validation and test results significanly. 


If a well known architecture was chosen:

* What architecture was chosen?

Also implemented a network similar to VGG, after 10 epoch Validation Accuracy reached 0.960 and after 20 epoch 0.973. However, on the extra images the VGG didn't perform as the other two, the other networks performed about 10% better. 

* Why did you believe it would be relevant to the traffic sign application?

It is similar to LeNet but with deeper layers, as if there are few LeNet networks are connected in series, excluding the last fully connected layers, and that three fully connected layers moved to the very end of the network. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The model seem to work well within the dataset, since the accuracy of Validation and test dataset is very close. It means that the algorithm still can generalize and not memorized the 
However, when introduced with images that are somewhat tricky, it can't properly identify it at the highest probability, and will be rather a second or 4th choice. Those images that are detected incorrectly mostly are the ones that have lower number of training sets.  

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

The image 9 which has label 20 is difficult to classify because the number of trained data with this label is quiet low compared to the rest of the training dataset. However, the images with labels 12, 13 the prediction is most accurate since there are high number of training dataset for these labels. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                | Prediction			 						| 
|:-----------------------------:|:---------------------------------------------:| 
| Bumpy road (22) 				| Bumpy road   									| 
| Road work (25)    			| Road work 									|
| Yield (13)					| Yield											|
| Beware of ice/snow (30)		| Slippery road					 				|
| Slippery Road (23)			| Slippery Road      							|


The model was able to correctly guess 7 out of the 11 traffic signs, which gives an accuracy of 63.64%. One of the images, label 30, is partially occluded by snow. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located where its called "Predict the Sign Type for Each Image" in the notebook.

For most of the image, the model is relatively sure that this is a correct (probability of 0.75-0.99) rightfully. However, for the image of "Beaware of ice/snow" it is also certain that its slipper road where the prediction is incorrect. For two of the samples that incorrectly identified; the sample 8 and sample 4; the correct label is in the second choice with of course lower probability.  

| Probability         	| Prediction											| 
|:---------------------:|:-----------------------------------------------------:| 
| .99         			| Bumpy road (22)   									| 
| 1     				| Road work (25) 										|
| 1						| Yield (13)											|
| 1						| Beware of ice/snow (30)				 				|
|.75					| Slippery Road (23) 		   							|

 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


