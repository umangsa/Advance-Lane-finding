import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
import glob
from PIL import Image
from moviepy.editor import VideoFileClip
from collections import deque

# Global variables 
# chess board features
nx = 9 # horizontal corners
ny = 6  # vertical corners

# Define conversions in x and y from pixels space to meters
ym_per_pix = 3/72.0 # meters per pixel in y dimension
xm_per_pix = 3.7/675.0 # meters per pixel in x dimension
camera_center = 640.0 # center of the image


# road_radius = deque(maxlen=1)


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = deque(maxlen=5)
        #radius of curvature of the line in some units
        self.radius_of_curvature = deque(maxlen=5) 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


# Routine for Camera Calibration
# Input paramters - None
# Output - Calibration parameters
def calibrate_camera():
	objpoints = []
	imgpoints = []
	objp = np.zeros((nx*ny, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

	for filename in glob.iglob("camera_cal/calibration*.jpg"):
		img = mpimg.imread(filename)
		# print("IMage {} shape {}".format(filename, img.shape))
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		if ret:
			img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			# plt.imshow(img)
			# plt.show()
			imgpoints.append(corners)
			objpoints.append(objp)

	return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def fix_perspective(img, mtx, dist, src, dest):
	img_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dest)
	Minv = cv2.getPerspectiveTransform(dest, src)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, Minv

def direction_threshold(gray, thresh, ksize):
	# Calculate the x and y gradients
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output

def abs_sobel_thresh(gray, orient='x', thresh=(0, 255)):
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	else:
		sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

	abs_sobel = np.absolute(sobel)
	max_value = np.max(abs_sobel)
	binary_output = np.uint8(255*abs_sobel/max_value)
	threshold = np.zeros_like(binary_output)
	threshold[(binary_output >= thresh[0]) & (binary_output <= thresh[1])] = 1
	return threshold

def full_search(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


	return True, out_img, left_fit, right_fit


def margin_search(binary_warped):
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Set the width of the windows +/- margin
	margin = 100

	left_fit = left_lane.best_fit
	right_fit = right_lane.best_fit

	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & \
		(nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & \
		(nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  


	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 


	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	status = sanity_check(left_fit, right_fit, leftx, lefty, rightx, righty)
	if status:
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	else:
		left_fit = []
		right_fit = []

	return status, out_img, left_fit, right_fit

def sanity_check(left_fit, right_fit, leftx, lefty, rightx, righty):
	# check if lanes are parallel
	# abs (dist1) > 0.3
	# dist2 < 705 or > 850
	if (abs(right_fit[1] - left_fit[1]) > 0.3) and (((right_fit[2] - left_fit[2]) > 900) or ((right_fit[2] - left_fit[2] < 705))):

		return False
	return True

def search_lanes(binary_warped):
	if left_lane.detected and right_lane.detected:
		status, out_img, left_fit, right_fit = margin_search(binary_warped)
		if status == False:
			status, out_img, left_fit, right_fit = full_search(binary_warped)
	else:
		print("Perform full search of lanes")
		status, out_img, left_fit, right_fit = full_search(binary_warped)

	if status:
		left_lane.detected = True
		right_lane.detected = True
		left_lane.current_fit.append(left_fit)
		right_lane.current_fit.append(right_fit)
		left_lane.best_fit = np.mean(left_lane.current_fit, axis=0)
		right_lane.best_fit = np.mean(right_lane.current_fit, axis=0)
	else:
		left_lane.detected = False
		right_lane.detected = False
		# left_lane.current_fit = None
		# right_lane.current_fit = None

	return out_img, left_lane.best_fit, right_lane.best_fit



def lane_pipeline(img):
	s_thresh=(100, 240)
	sx_thresh=(30, 230)
	l_thresh=(150, 225)
	y_thresh=(50, 250)
	color_threshold=(100, 220)

	# R & G helps with white and yellow lines
	R = img[:,:,0]
	G = img[:,:,1]
	color_combined = np.zeros_like(R)
	rg_binary = (R > color_threshold[0]) & (G > color_threshold[1])
	
	# Convert to HSV color space and separate the V channel
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hsv[:,:,1]
	s_channel = hsv[:,:,2]

	# Sobel x
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	
	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# L Channel to filter shadows
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	l_channel = hls[:,:,1]
	l_binary = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])

	#equalize Y channel from YUV
	yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float)
	y_channel = yuv[:,:,0]
	y_binary = (y_channel > y_thresh[0]) & (y_channel <= y_thresh[1])

	# Stack each channel
	# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# be beneficial to replace this channel with something else.
	yl = np.zeros_like(sxbinary)
	yl[((l_binary == 1) & (y_binary == 1))] = 1
	
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[((s_binary == 1) | (sxbinary == 1) | (rg_binary == 1)) & ((l_binary == 1) | (y_binary == 1))] = 1

	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary, l_binary, rg_binary, y_binary))
	return color_binary, combined_binary




def project_lanes(warped, Minv, left_fitx, right_fitx, ploty):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 

	return newwarp

def process_image(image, is_test = False):
	img = np.copy(image)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	
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

	warped, Minv = fix_perspective(dst, mtx, dist, corners, dest)

	# Get the Color and Gradient  threshold
	color_image, gray_gradient = lane_pipeline(warped)

	# Perform sliding window search using histogram peak to locate the lanes
	output, left_fit, right_fit = search_lanes(gray_gradient)

	# Generate x and y values for plotting
	ploty = np.linspace(0, gray_gradient.shape[0]-1, gray_gradient.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Calculate radius of the road curvature
	radius = calculate_radius(ploty, left_fitx, right_fitx)

	# Calculate the offet of car center w.r.t. lane center
	# assumption - camera is mounted on the lane center
	lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
	car_offset = (camera_center - lane_center) * xm_per_pix


	# Combine the result with the original image
	newwarp = project_lanes(gray_gradient, Minv, left_fitx, right_fitx, ploty)
	result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)
	cv2.putText(result,"Radius: {:.2f} m".format(radius), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)
	if car_offset < 0:
		cv2.putText(result,"Car right of center: {:.2f} m".format(abs(car_offset)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)
	elif car_offset > 0:
		cv2.putText(result,"Car left of center: {:.2f} m".format(abs(car_offset)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)
	else:
		cv2.putText(result,"Car exactly centered", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)

	if is_test == True:	
		# plot and save the transformations. Used for the test images only
		tkns = filename.split("/")
		fname = "output_images/warped_" + tkns[1]

		fig = plt.figure(figsize=(16, 9))

		ax1 = fig.add_subplot(231)
		ax1.imshow(img)
		ax1.set_title('Original Image')

		ax2 = fig.add_subplot(233)
		ax2.imshow(gray_gradient, cmap='gray')
		ax2.set_title("After Thresholding")

		ax3 = fig.add_subplot(232)
		ax3.imshow(warped, cmap='gray')
		ax3.set_title('After Perspective transform')
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		cursor = Cursor(ax1, useblit=True, color='red', linewidth=2)

		# histogram_peak_search()
		ax4 = fig.add_subplot(234)		
		ax4.imshow(output)
		ax4.set_title('Detected lanes')

		mark_size = 3
		ax5 = fig.add_subplot(235)
		ax5.imshow(output)
		ax5.plot(left_fitx, ploty, color='yellow')
		ax5.plot(right_fitx, ploty, color='yellow')

		
		ax6 = fig.add_subplot(236)
		ax6.imshow(result)

		fig.savefig(fname)

	return result



def calculate_radius(ploty, left, right):
	# scale to real world space from pixel space
	left_fit = np.polyfit(ploty * ym_per_pix, left * xm_per_pix, 2)
	right_fit = np.polyfit(ploty * ym_per_pix, right * xm_per_pix, 2)

	# Define y-value where we want radius of curvature
	# I'll choose the minimum y-value, corresponding to the bottom of the image
	y_eval = np.min(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	left_lane.radius_of_curvature.append(left_curverad)
	right_lane.radius_of_curvature.append(right_curverad)

	radius = (np.mean(left_lane.radius_of_curvature) + np.mean(right_lane.radius_of_curvature))/2.0

	return radius


# First Calibrate the camera
# Use all the calibration images in the camera_cal folder
# to get the final mtx and dist factors
global mtx, dist
left_lane = Line()
right_lane = Line()

ret, mtx, dist, rvecs, tvecs = calibrate_camera()

# Undistort and perform a perspective transform of the images
# using the mtx and dist factors
# Save the final images
# This will allow us to verify that the calibration is done correctly

for filename in glob.iglob("camera_cal/calibration*.jpg"):
	img = mpimg.imread(filename)
	img_size = img.shape
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	tkns = filename.split("/")
	fname = "output_images/warped_" + tkns[1]
	Image.fromarray(dst).save(fname)

# Use the calibration factors to work on the test images
left_lane.current_fit = deque(maxlen=1)
left_lane.radius_of_curvature = deque(maxlen=1)
right_lane.current_fit = deque(maxlen=1)
right_lane.radius_of_curvature = deque(maxlen=1)

for filename in glob.iglob("test_images/*.jpg"):
	# print("File = {}".format(filename))
	img = mpimg.imread(filename)
	process_image(img, is_test=True)

# Process the project video

# Reset lane detections
left_lane.detected = False
left_lane.best_fit = None
left_lane.current_fit = deque(maxlen=5)
left_lane.radius_of_curvature = deque(maxlen=5)

right_lane.detected = False
right_lane.best_fit = None
right_lane.current_fit = deque(maxlen=5)
right_lane.radius_of_curvature = deque(maxlen=5)

output = 'output_videos/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)

'''
output = 'output_videos/challenge_video.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)


output = 'output_videos/harder_challenge_video.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)
'''
