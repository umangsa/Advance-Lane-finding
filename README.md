## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Calibration checkerboard"
[image2]: ./output_images/warped_calibration1.jpg "Calibrated checkerboard"
[image3]: ./test_images/test11.jpg "Original lane image"
[image4]: ./output_images/warped_test11.jpg "Pipeline"
[image6]: ./output_images/warped_straight_lines1.jpg "Pipeline"
[image5]: ./output_images/undistort_test11.jpg "Un-distorted lane image"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

All the code for this project is implemented in the advance_lanes.py
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


The code for this step is implemented in the calibrate_camera function, lines 43-61.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

I used all the checkerboard images provided in the camera_cal directory for calibration calculation. This ensures that the calibration errors are reduced, thus getting more accurate calibration.

The original checkerboard used for calibration 

![alt text][image1]

and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

The image looks like this after un-distort
![alt text][image5]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used the following transforms to isolate the lane markers (code in implemented in the lane_pipeline function, line 261:
Color threshold of the R & G channels of the RGB image. This helps to identify the yellow and white lines

L channel to filter the shadows from HLS

Y channel to help with contrast and yellows, from YUV

Sobel of X of the S Channel from HSV (Channel name in code incorrectly marked as L Channel)

Threshold of the V Channel from HSV (Channel name in code incorrectly marked as S Channel)

Combination of the above was used to generate a binary image that did a reasonable job of isolating the desired pixels.

My complete pipeline is plotted in stages in the following image

![alt text][image4]

The 1st plot on the left is the original image. The 3rd image in 1st row shows the output of the thresholding step. This step has been performed on the top view as it gives a better result

One more sample image of the pipeline

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `fix_perspective()`, which appears in lines 73 through 78   The `fix_perspective()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# Perspective transform to get top view
	bottom_left = [282 , 666]
	bottom_right = [1025, 666]
	top_left = [597, 450]
	top_right = [684, 450]
	corners = np.float32([top_left, top_right, bottom_right, bottom_left])
	
	bottom_left_dst = [300,720]
	bottom_right_dst = [1000, 720]
	top_left_dst = [300, 0]
	top_right_dst = [1000, 0]
	dest = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 282, 666      | 300, 720      | 
| 1025, 666     | 1000, 720     |
| 684, 450      | 1000, 0       |
| 597, 450      | 300, 0        |

I perform the perspective transform immediately after undistort. 

The image after the perspective transform to get the top view is seen in the 2nd image of the 1st row in the above matrix

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I initially (submission 1) used the convolution method to find the centroids of each lane. Each image was split into 9 horizontal strips to get the centroids. A convolution filter of all ones for the window_width was used to identify where the lane pixels were.I then did a polyfit of order 2 for all the centroids found for each lane to plot the lanes. To smoothen the lanes across the images,I did an averaging across the last 15 images

I faced a number of problems using the convolution method. For the right lanes, which are mostly dashed lines. Several strips in the images did not contain any pixels of the image. Convolution window would return very low or no activations. In order to detect or predict where the lane marker would be, I placed the centroid at the average road width that I was calculating whenever I detected good centroids. To get a good lane width, I started with 600 pixels and then performed moving average across last 5 centroids

I found that working with the convolution method was very unstable. I modified my image processing pipeline to detect the pixels of the lanes better. However, I could not make the convolution method to become stable. I then switched to the lane finding using histogram peak. This is implemented in the search_lanes function, line 235.

For the first image, I do a full image search horizontally to find the histogram peak (full_search() function, line 107). Once I have a good peak detection done, for the next image onwards, I do a window search. This reduces the computation time and lanes are detected more quickly. (margin_search function, line 184)

To make the detection more robust, I do a sanity check on the coefficients found in the margin_search step. If the coeffiecients of the left and right lane cross a certain threshold, it means the left and right lanes are not parallel to each other. I chose a threshold that points to various problems in detection. I then ignore the detections and do a full image search for the lanes. THe code for sanity is implemented in the sanity_check function, line 226

The output of the lane  detection is seen in the row 2, 1st image of the above image matrix. 

Output after polyfit is seen in the row 2, 2nd image


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did the radius calculation in calculate_radius function lines 434 through #449 in my code.

To find the car offset from the center, I assumed the camera was placed in the center of the image. It should ideally be the center of the lanes. The difference between the detected center of the lanes and the camera center is the car offset from center. This is calculated in line 427-443

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 317 through 333 in the function `project_lanes()`.  The output can be seen in the row 2, 3rd image in the above matrix

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest challenge I faced was finding or estimating the centroid when the lane markers were not visible or detected. To solve this, I used an estimated road width to estimate location of the centroid. However, the road width appeared to vary image to image. This was causing the polyfit to fail under several conditions and would lead to catastrophic incidents. To reduce these effects, I did averaging of road widths and averaging if the lanes detected after polyfit. In the end, I gave up trying to fix the convolution method and adopted the histogram peak search algorithm

I also noticed that increasing the length of the segment chosen for the perspective transform has a very positive effect on making detections robust and smooth. Earlier I was choosing a smaller section of the lane in the y direction. The white lane marker would have large gaps and very few pixels would be captured. Increasing the length of the road taken for the perspective transformation made the strips closer and hence the markers were easier to capture.

The other problem I faced was with shadows. I ended up lowering the L channel threshold to improve lane detection. However, this has a negative imapct when the lane has dark patches in the road. This is visible on certain areas where the road is light colored with black patches. It also fails in the challenge videos where the old lane markers have been erased using tar

It was also challenging to detect the white markers on the gray road or where the image was very bright. I was able to solve it adjusting color threshold 

The harder challenge video completely fails for my code. The lighting conditions are a problem. Adjusting contrast and playing with other color space and thresholds may help.

I also think a dynamic gradient and threshold adjustment is required to make the code to work under many dynamically changing conditions
