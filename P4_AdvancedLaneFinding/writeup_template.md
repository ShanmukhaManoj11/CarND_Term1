**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./writeup_images/undistorted_image.jpg "undistorted image example"
[image3]: ./writeup_images/thresholded_image.JPG "thresholded image example"
[image4]: ./writeup_images/roi_mask_applied.JPG "roi mask applied example"
[image5]: ./writeup_images/warped_binary.JPG "warped binary example"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

The code for this step is provided as an example.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Undistort images

From the camera matrix and distortion coefficients computed in the above step, `cv2.undistort()` function is used to undistort the images. Following image shows the undistorted image of a test image,

![alt text][image2]


#### 2. Pipeline to detect lanes

First converted the BGR image to HLS using `cv2.cvtColor()` and then thresholded the S channel and gradient of L channel in x and y directions of L channel. Refer to `pipeline()` function in the **project4_2.ipynb**

![alt text][image3]

Then a region of interest mask is applied on the thresholded image to remove pixels that most porbably don't include in a lane. Following image shows a sample after applying the mask.

![alt text][image4]


#### 3. Perspective transform of the resulting binary image

Using openCV's functions for computing perspective transform matrix - `cv2.getPerspectiveTransform()` and warping the image using the computed matrix - `cv2.warpPerspective()`, resulting binary image after applying region of interest mask in the above step is transformed. I have used following source and destination points for computing the transform matrix.

```python
roi_vertices_test=np.array([[(0,imshape[0]),(imshape[1]/2-100,imshape[0]/2+100),(imshape[1]/2+100,imshape[0]/2+100),
                        (imshape[1],imshape[0])]],dtype=np.int32)
src_pts_test=np.float32(roi_vertices_test)
dst_pts_test=np.float32([[(0,imshape[0]),(0,0),(imshape[1],0),(imshape[1],imshape[0])]])
```

FOllowing image shows a transformed sample image. And observe that the left and right lanes are almost parallel to each other.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
