# **Finding Lane Lines on the Road** 

---


[//]: # (Image References)

[image1]: ./writeup_images/pipeline.png "Pipeline"

---

### Reflection

### 1. Pipeline

This pipeline includes 7 steps. 

a. Read frame/ image - either from file or from video clip

b. convert the image into grayscale

c. apply gaussian kernel and blur the grayscale image

d. apply canny edge detection on the blurred image and find edge image (thresholded image with edges)

e. with predetermined polygon constituting a region of interest, create a mask and apply the mask to the edge image (canny output)

f. detect lines in the masked image using hough transforms

g. overlay detected lines on the initial image

![alt text][image1]

Function "draw_lines_2" draws extended lanes on the image by averaging the lines detected from the above pipeline. 
Following is a high level description of the function draw_lines_2,

	a. A threshold of 0.5 is applied to the slopes, i.e. 
        all lines with slope < -0.5 are considered to be part of the left lane and 
        lines with slope > 0.5 are considered to be part of right lane 
		
    b. Lines gathered as part of left lane are then sorted based on their lengths and in the hough space (i.e., m-b space)
    average 'm' and 'b' values are calucalted from the top 5 lines with higher lengths. 
    Similarly average slope and intercept (m and b) for the right lane are found   
	
    c. Then with the m,b values in hough space, corresponding points on the image plane are calculated by substituing y values
    in the line equation y = mx+b such that the lanes, that would be drawn on to the image,
    are fully drawn in the region of interest


### 2. Shortcomings with current pipeline

1. Several parameters - (thresholds for canny and hough transforms, vertices for region of interest) are to be fine tuned and these values change from scenario to scenario. 
The optional challenge alone required different parameter values when compared to the other two test cases. 
With changing brightness/ weather conditions or road conditions, parameters vary.
This can be a big shortcoming as different scenarios need different parameters

2. Generally lanes on roads are not just straight line, so using hough transforms and detecting only straight line can be an important shortcoming

### 3. Possible improvements

1. Instead of straight lines, lanes can be fitted with polynomials

2. With changing road/ weather (brightness) conditions, and noise in the images, detected lanes misalign or at few time steps disappear.
To avoid this, a memory can be included which stores previous lane slopes and can be used to filter the noisy output (can be a simple averaging filter)

3. May be a deep learning model can learn to detect lanes under all conditions thus unifying the code for road lane detection
