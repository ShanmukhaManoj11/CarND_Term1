**Vehicle Detection Project**

#### Files included

1. writeup.md - discussion about the project implementation
2. utils.py - python code with all the functions required for the project
3. project5.ipynb - ipython notebook for the step-by-step implementation
4. sample.ipynb - ipython notebook generating images for illustration
5. test_videos_output/ - folder containing output video
6. output_images/ - folder containing output images

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg

[image3]: ./output_images/test1_sliding_window.jpg
[image4]: ./output_images/test2_sliding_window.jpg
[image5]: ./output_images/test3_sliding_window.jpg
[image6]: ./output_images/test4_sliding_window.jpg
[image7]: ./output_images/test5_sliding_window.jpg
[image8]: ./output_images/test6_sliding_window.jpg

[image9]: ./output_images/test1.jpg
[image10]: ./output_images/test2.jpg
[image11]: ./output_images/test3.jpg
[image12]: ./output_images/test4.jpg
[image13]: ./output_images/test5.jpg
[image14]: ./output_images/test6.jpg

[image15]: ./examples/heatmap_det_thresh.JPG

[video1]: ./project_video.mp4

### Training Data

The training data comprises of 64x64 images of vehicles and non vehicles.

![alt text][image1]

I have used spatial binning, color histograms and HOG features combined as feature vectors to train a Linear SVM classifier. I have also converted the images into 'YCrCb' color space - experimented with 'RGB', 'HSV' and 'YCrCb' color spaces and found decent enough perfromance with 'YCrCb' color space.

**Note:** Most of the functions implemented as referred from the Vehicle detections and tracking lesson from the course.

#### Spatial binning

Refer to bin_spatial() function in utils.py. This function resizes the images into specified spatial size and ravels the input into a 1D vector. After experimenting with spatial sizes (16,16) and (32,32), I have decided to use (16,16) as I didn't notice much variation in the final perfromance on test images based only on spatial bins.

#### Color histograms

Refer to color_hist() function in utils.py. Creates histograms in all three channels and concatenate the forming a 1D vector of features. I have used 16 bins to create histograms for these experiments.

#### Histogram of Oriented Gradients (HOG)

I have used the 'hog' function provided by skimage.feature to compute hog features. Refer to function **get_hog_features()** in 'utils.py'. After experiementing with color spaces, orientations, pixels_per_cell, cells_per_block and other parameters, I have decided on using 'YcrCb' color space, 9 orientations, 8 pix_per_cell, 2 cell_per_block - as these parameters were resulting in satisfactory outputs on test images.

![alt text][image2]

#### Combining feature vectors

Refer to functions single_img_features() and extract_features() functions in utils.py for extracting combined feature vector from single images and list of image files respectively.


### Training SVM classifier

I computed features as explained above for each of the image in provided vehicle and non-vehicle images data. Then applied normalization using StandardScaler() from sklearn.preprocessing. Then I split the data randomly into 80% training set and 20% test set. (Refer to cell 5 in project5.ipynb)

Then trained a linear SVM classifier that resulted in a 99.13% accuracy on the test set. (refer to cell 6 in project5.ipynb)

### Finding cars by sliding windows

#### Basic sliding window approach

Based on the information provided in the course, I have used 3 different window sizes and slided them over corresponding image portions. Refer to slide_window_find_cars() in utils.py.

Used heatmap and thresholding technique as explained in the lesson, to generate bounding boxes over vehicles detected. Using multiple window sizes and thresholding, I removed false positives in the detections. Refer to draw_boxes_thresholded() function in utils.py. Following are the resulting detections in test images, (refer to cell 10 in project5.ipynb)

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

#### Faster approach by computing the hog features for entire image only one time.

Though the previous search technique perfromed well, it is slow as every time hog features are to be computed over a small window of image and this is expensive. So to overcome that, it was explained in the lesson to compute hog features for the entire image in the desired region only one time. Refer to find_cars() function in utils.py.

Above function has a paramter 'scale' that convers that serves the similar purpose of sliding multiple sized windows in the previous approach. I ahve used 2 scales - 1.5 and 2 for multiple detections before thershodling the detections. Following figures show raw detection windows, corresponding heatmaps before and after threshold (here threshold value is 2)

![alt text][image15]

Following are the results on test images, (refer to cell 11 in project5.ipynb)

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

### Video Implementation

Here's a [link to my video result](./test_videos_output/project_video.mp4)

For each frame of the video I have used find_cars() function previously described function to retrieve possible locations (bounding boxes) for the cars detetcted. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap that corresponded to a vehicle.

To make the tracking relatively stable, I haved used history of bounding boxes. I maintained a global variable with 20 previously deteted bounding boxes, and applied heat for all the 20 previously detected boxes and current detections before thresholding. This helped in reducing the false positives detected and also made the tracking a little smoother.

### Discussion

It can be seen in my [video result](./test_videos_output/project_video.mp4) that the tracking is not extremely smooth, and there is an instance with false positive detection. 

There is paramter 'threshold' in my function draw_boxes_thresholded() in utlis.py, that decides the threshold for the heatmap. With my current implementation, increasing the threshold value reduces the false positives but at the same time reduces smoothness in tracking. 

Also I would like to experiment with neural net classifier and compare the perfromance with the linear SVM classifier.

