**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NVIDIA_architecture.JPG "NVIDIA architecture"
[image2]: ./examples/preprocessed_image_samples.png "Preprocessed samples"
[image3]: ./examples/data_distribution.png "Data distribution"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

My project includes the following files:
* data_loading.py - loads data from all the 3 cameras along with measured steering angle adjustments (increment by +0.25 for images from left camera and -0.25 for images from right camera)
* model.py - containing the script to create and train the model 
* drive.py - for driving the car in autonomous mode
* model_NVIDIA.h5 - containing a trained convolution neural network (NVIDIA architecture)
* writeup_report.md - summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_NVIDIA.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. NVIDIA architecture

I have used the NVIDIA model (https://arxiv.org/pdf/1604.07316.pdf) in this project. Following image describes the architecture in detail.

![alt text][image1]

I have used 'ELU' activation at each layer (except the last layer with one output node). The model contains dropout layers (with dropout probability of 0.5) at every layer in order to reduce overfitting. 


#### 2. Training data

I have used only **sample data** (provided by the Udacity) for training the model.
Though I have collected data, I have found issues with the results when I used that.
So, as advised by mentor, I went forward with only the provided sample data.

#### 3. Training Procedure

Data augmentation is necessary for training the network. Apart from using images from all the 3 cameras with measured steering angle adjustement, every image that is fed into the network during training is preprocessed as follows,
* random brightness addition
* random shadow addition
* cropped bottom 25 pixels (removing the hood) and top (approx.) 50 pixels (removing the trees and sky)
* random translation in both x and y directions - for every pixel translated in x direction measured angle is adjusted by adding or subtracting 0.004 depending on direction of translation in x direction.
* randomly flip the image
* resized to 66x200 (input for NVIDIA architecture)

The above data augmentation startegies are inspired from the following work - https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9

Following are sample preprocessed images
![alt text][image2]

Instead of having all the images loaded in memory, **data_generator** is used to create batches of images. Also, the data distribution is not even as shown below
![alt text][image3]

I have then assumed that the angles whose absolute value < 0.1 are considered low and since the measurements are more in the 'low' zone, for the model to learn unifromly follwoing strategy (inspired from the work referenced above for the data augmentation) is implemented
* At the start of every epoch a threshold is defined as 1/(epoch+1), where epoch starts from 0
* if the absolute value of measured steering angle is < 0.1 and a random number in range [0,1) is < threshold value - the corresponing data set is not added to the batch, otherwise added to the bacth 

This makes the intial batches with more of the angles in the 'high' zone (whose absolute value >= 0.1), than in 'low' zone (whose absolute value < 0.1) and with increasing epoch number, this distribution changes.
This allows the trained model to not be biased more with 'low' zone angles.

The model used an adam optimizer with initial learning rate = 0.0001.

#### 4. Performance

The perfroms good on the track 1. It can drive for hours with out going out of road markings. run1.mp4 is a sample video depicting the model performance on track 1.

The car in autonomous mode, indeed takes smooth turns, but could perform better while driving straight. This is probably the result of the 'low' zone threshold I assumed (probably instead of 0.1 I could have tried 0.2 or played around with that parameter). 
Also, note that the model has been trained only on the sample data. I should probably collect more data and also the recovery data as advised by the Udacity lectures.

I have not collected any data on track 2. And current model doesn't generalize well on the track 2. Track 2 has very sharp turns and ramps and slopes for the roads, which track 1 doesn't include. So probably some data should be collected on track 2 and the network should be trained on those.

