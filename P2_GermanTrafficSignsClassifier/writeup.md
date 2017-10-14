#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[example_data]: ./images_for_writeup/example_data.jpg "Example Data"
[counts_per_class_train_data]: ./images_for_writeup/counts_per_class_train_data.jpg "Counts per class in training data set"
[counts_per_class_validation_data]: ./images_for_writeup/counts_per_class_validation_data.jpg "Counts per class in validation data set"
[counts_per_class_test_data]: ./images_for_writeup/counts_per_class_test_data.jpg "Counts per class in testing data set"
[example_augmented_data]: ./images_for_writeup/example_augmented_data.jpg "Example augmented data"
[lenet_architecture]: ./images_for_writeup/lenet_architecture.jpg "lenet architecture"
[NN_architecture]: ./images_for_writeup/NN_architecture.jpg "NN architecture"
[test_image1]: ./examples/test_image1.jpg "Traffic Sign 1"
[test_image2]: ./examples/test_image2.jpg "Traffic Sign 2"
[test_image3]: ./examples/test_image3.jpg "Traffic Sign 3"
[test_image4]: ./examples/test_image4.jpg "Traffic Sign 4"
[test_image5]: ./examples/test_image5.jpg "Traffic Sign 5"

## Rubric Points
### Writeup / README

[project code](https://github.com/ShanmukhaManoj11/CarND_Term1/blob/master/P2_GermanTrafficSignsClassifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

numpy is used to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### Exploratory visualization of the dataset.

Below are few example images from the data set

![alt text][example_data]

Distribution of data, i.e. counts per class in training, validation and test data sets are shown below,

![alt text][counts_per_class_train_data]

![alt text][counts_per_class_validation_data]

![alt text][counts_per_class_test_data]

### Design and Test a Model Architecture

#### Image preprocessing

**Data Augmentation:**

Inorder to provide the network with lots of data, additional images can be generated from the available images by applying random translations, rotations, affine or perspective transformations along with additional random brightness to them. This helps the network to learn more.

For every image in the training data set, 3 random image transformations are applied and a final data set with 4 times the size of given data set is generated

![alt text][example_augmented_data]

**Data Normalization:**

Every image is normalized by applying the following transformation to each pixel,

pixel = (pixel - 128.0)/128.0

### The definitions of model architecture are included in the "utils.py" script along with definition of class "Model" that provides functions "train" and "evaluate" data set

#### Model/ network architecture 1: 'lenet'

![alt text][lenet_architecture]

**Training procedure for lenet architecture:**

For training the model from scratch keep the "RESUME_TRAINING" parameter (in the Model.train(...) function) "False" and inorder to use pretrained model change the parameter to "True"

For the 'lenet' architecture, batch size and keep prob are fixed at 128 and 0.5 (respectively) and 6 runs with varying learning rate and epochs are performed as follows,

Run 1: EPOCHS = 5, LEARNING_RATE = 0.001, RESUME_TRAINING = False - resulted in a validation accuracy of 95.46%
Run 2: EPOCHS = 10, LEARNING_RATE = 0.00085, RESUME_TRAINING = True - resulted in a validation accuracy of 98.28%
Run 3: EPOCHS = 5, LEARNING_RATE = 0.00075, RESUME_TRAINING = True - resulted in a validation accuracy of 98.25%
Run 4: EPOCHS = 5, LEARNING_RATE = 0.0005, RESUME_TRAINING = True - resulted in a validation accuracy of 98.57%
Run 5: EPOCHS = 5, LEARNING_RATE = 0.00025, RESUME_TRAINING = True - resulted in a validation accuracy of 98.66%
Run 6: EPOCHS = 5, LEARNING_RATE = 0.0001, RESUME_TRAINING = True - resulted in a validation accuracy of 98.71%

#### Model/ network architecture 2: 'NN' - based on Sermanet and Lecunn's paper

![alt text][NN_architecture]

For training the model from scratch keep the "RESUME_TRAINING" parameter (in the Model.train(...) function) "False" and inorder to use pretrained model change the parameter to "True"

For the 'NN' architecture, batch size and keep prob are fixed at 128 and 0.5 (respectively) and 4 runs with varying learning rate and epochs are performed as follows,

Run 1: EPOCHS = 5, LEARNING_RATE = 0.00085, RESUME_TRAINING = False - resulted in a validation accuracy of 95.53%
Run 2: EPOCHS = 5, LEARNING_RATE = 0.00075, RESUME_TRAINING = True - resulted in a validation accuracy of 96.67%
Run 3: EPOCHS = 5, LEARNING_RATE = 0.0005, RESUME_TRAINING = True - resulted in a validation accuracy of 97.01%
Run 4: EPOCHS = 5, LEARNING_RATE = 0.0001, RESUME_TRAINING = True - resulted in a validation accuracy of 97.37%

However by the end of run 4, training accuracy reached 100% and validation accuracy leveled at approximately 97% (compared to "lenet" that resulted in a final 98.71% validation accuracy). This shows some kind of overfitting on the train data. Since the number of parameters in the fully connected layers are too high in the above "NN" architecture implementation, probably decreasing the keep probablity and adding L2 regularization to loss calculation might help in reducing the overfitting behavior.

#### Final model results - 

**for lenet architecture:**

* training accuracy = 0.9973
* validation accuracy = 0.9871
* test accuracy = 0.9748

**for NN architecture:**

* training accuracy = 1.0000
* validation accuracy = 0.9737
* test accuracy = 0.9788

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


