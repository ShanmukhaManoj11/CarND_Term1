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

[example_data]: ./images_for_writeup/example_data.jpg "Example Data"
[counts_per_class_train_data]: ./images_for_writeup/counts_per_class_train_data.jpg "Counts per class in training data set"
[counts_per_class_validation_data]: ./images_for_writeup/counts_per_class_validation_data.JPG "Counts per class in validation data set"
[counts_per_class_test_data]: ./images_for_writeup/counts_per_class_test_data.jpg "Counts per class in testing data set"
[example_augmented_data]: ./images_for_writeup/example_augmented_data.jpg "Example augmented data"
[lenet_architecture]: ./images_for_writeup/lenet_architecture.jpg "lenet architecture"
[NN_architecture]: ./images_for_writeup/NN_architecture1.jpg "NN architecture"
[test_image1]: ./examples/test_image1.jpg "Traffic Sign 1"
[test_image2]: ./examples/test_image2.jpg "Traffic Sign 2"
[test_image3]: ./examples/test_image3.jpg "Traffic Sign 3"
[test_image4]: ./examples/test_image4.jpg "Traffic Sign 4"
[test_image5]: ./examples/test_image5.jpg "Traffic Sign 5"
[resized_test_data]: ./images_for_writeup/resized_test_data.jpg "Resized test images" 
[test_predictions_lenet_architecure]: ./images_for_writeup/test_predictions_lenet_architecure.jpg "Top 5 predictions for each test image - lenet" 
[test_predictions_NN_architecure]: ./images_for_writeup/test_predictions_NN_architecure.JPG "Top 5 predictions for each test image - NN" 

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

**Probable issues with my data augmentation procedure:**

In the above histogram of counts per class in training data, the distribution of data is unequal - certain classes have more examples and other classes have fewer. New images can also be generated (by performing random transformations as explained above) for the classes with fewer examples and make the data distribution equal.
This helps the network to learn equally on each class type thus increasing precision in prediction. But the data augmentation procedure I haved used in the project, is that for every training image (irrespective of which class it belongs to) I have generated 3 new example performing random transformations. By doing this the resutling new data distribution is the same as the original distribution.
So, probably I should have used data augmentation to make the distribution more even rather than simply generating new images for every image

![alt text][example_augmented_data]

**Data Normalization:**

Every image is normalized by applying the following transformation to each pixel,

pixel = (pixel - 128.0)/128.0

By applying this, mean is shifted to 0 and thus makes the data dsitributed around the 0 mean which help the optimizer to converge easily. Without data normalization, optimizer generally takes lot of steps to converge.

**Didn't convert images to gray scale:**

In this experiment I have used RGB images for triaining rather than converting them to gray scale in preprocessing stage. Though converting to gray scale reduces input image dimension and might reduce the training complexity, I assumed that using RGB images might help the network to extract features pertaining to each color channel while learning to classify the input. Probably I should perform experiments with gray scale images to compare the results.

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

I have used the color images for training, as I feel providing color images instead of gray scale might allow network to learn better by extracting features pertaining to color channels.
I have stated with simple lenet architecture which resulted in lower accuracies, and hence made the network deep by adding more kernels for convolution operation and decreasing learning rate after certain no. of epochs.
With this approach validation accuracy has reached 98.71%

Then to compare, I have tried implementing architecture described in Sermanet and Lecunn's paper. This architecture contains skip connections and I felt this would allow the fully connected classifier layer to receive both high level and low level features as input.
My implementation of that architecture ('NN') has started overfitting the training data, so probably I should have reduced the no. of nodes by applying further maxpooling to the branched skip connections (see the NN architecture image above for details)

### Test on New Images from web

Here are five German traffic signs that I found on the web:

![alt text][test_image1] ![alt text][test_image2] ![alt text][test_image3] 
![alt text][test_image4] ![alt text][test_image5]

All the above images are resized to 32x32x3 and the normalized as explained above.

Following are the resized test images with their original labels,

![alt text][resized_test_data]

**Comments on the new test images:**

The images with the speed limit 30 kmph sign and stop sign from the new test set, contains more than 50% background information in them. If we closely observe the training data set on which the network has been trained, the background details in them are less than 20%. So probably aforementioned test images might be difficult to classify - by which I mean that the network might miss classify these images.

#### Perfromance of lenet architecture on the new test images

Following figure shows the top 5 predictions lenet architecture provides for each image among the new test images

![alt text][test_predictions_lenet_architecure]

#### Perfromance of NN architecture on the new test images

Following figure shows the top 5 predictions NN architecture provides for each image among the new test images

![alt text][test_predictions_NN_architecure]

#### Comments on the accuracies achieved on the new test images

Out of the 5 images downloaded from the web, NN architecture has wrongly predicted 2 of them and lenet architecture has wrongly predicted 1 of them. The reason for the wrong prediction might be the presence of backgound scene in the image. If notices carefully all the images used for triaining have above 80% of the pixels filled only with the sign related intensities and only less than 20% is background, but that's not the case in those images downloaded from web - more than 50% in the image contains backgound details. This might be a reason for the wrong predictions