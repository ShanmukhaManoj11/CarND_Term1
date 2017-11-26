## Advanced Lane Finding Project

### Files inculded

1. writeup.md - the writeup file
2. project4_1.ipynb - code for calibrating camera and computing camera matrix and distortion coefficients
3. project4_2.ipynb - pipeline and code for finding lanes in images and videos. Also inculdes sample images
4. ./test_videos_output - results of finding lanes in videos

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./writeup_images/undistorted_image.jpg "undistorted image example"
[image3]: ./writeup_images/thresholded_image.JPG "thresholded image example"
[image4]: ./writeup_images/roi_mask_applied.JPG "roi mask applied example"
[image5]: ./writeup_images/warped_binary.JPG "warped binary example"
[image6]: ./writeup_images/lanes_drawn_warped.JPG "lanes on warped binary example"
[image7]: ./writeup_images/sample_result.JPG "sample result"
[video1]: ./test_videos_output/project_video.mp4 "Video"
[video2]: ./test_videos_output/challenge_video.mp4 "challenge Video"

### Camera Calibration

The code for this step is provided as an example.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline

#### 1. Undistort images

From the camera matrix and distortion coefficients computed in the above step, `cv2.undistort()` function is used to undistort the images. Following image shows the undistorted image of a test image,

![alt text][image2]

After undistorting, I have smoothed the resulted image with gaussian blur.

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

Following image shows a transformed sample image. And observe that the left and right lanes are almost parallel to each other.

![alt text][image5]

#### 4. Find lane pixels and fit 2nd degree polynomial to the left and right lanes

For single image I have used the histogram technique explained in the lecture to locate the lanes starting positions and applied sliding window approach to find the pixels that correpsond to left and right lanes.

For video input, to the initial frames histogram technique is applied. For the subsequent frames, since the previous lane positions are known, I have used the previous fit to locate the lanes and slided the windows to the top to compute new fits. Also, I have averaged the new fits using first order filter with a coefficient of 0.1

After pixels correspoinding to the lanes are located, 2nd degree polynomial fits are computed. Image below shows a sample warped image with lanes drawn.

![alt text][image6]

Then using `cv2.fillPoly()` the lane region is computed on a mask and the mask is overlayed on an empty image. This image is unwarped using the inverse transformation matrix, and a weighted average of the unwarped image and the undistorted image is the result.

#### 5. Calculating radius of curvature and offset from the lane center.

As described in the lecture video, I have applied the formula for calcualting radius of curvature. Following function is does the radius calculation, where the 2nd order polynomial fit is given as x=Ay^2+By+C

```python
def calculate_ROC(A,B,y): 
    return ((1+((2*A*y+B)**2))**1.5)/(2*A)
```

With the position of left and right lane, average between the two values gives the location of the camera and the difference between position of the x-center in the image and the camera location gives the offset from the lane center in pixels.

#### Sample result

![alt text][image7]

---

### Result on video

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

Though the implementation works quite good on the project video, it fails to perfrom well on the challenge video. Here is the [link](./test_videos_output/challenge_video.mp4)  to my challenge video output. Probably playing with the pipeline thresholding color and gradients might improve the performance. Also currently I am just using one previous fit data, if a history of fits (may be 3 or 5 previous fits) could be included, this might omprove the perfromace by resutling in a more smoother lane detection.
