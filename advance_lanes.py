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

window_width = 15
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 75 # How much to slide left and right for searching

# Define conversions in x and y from pixels space to meters
road_width = deque(maxlen=5)
road_width.append(600)
left_lane = deque(maxlen=3)
right_lane = deque(maxlen=3)

ym_per_pix = 27/720 # meters per pixel in y dimension
xm_per_pix = 3.7/750 # meters per pixel in x dimension


class Lane():
	def __init__(self):
		self.detected = False
		self.l_center = 0
		self.r_center = 0


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


def image_gradient(img, s_thresh=(170, 255), l_thresh=(20, 100), sx_thresh=(20, 100), dir_thresh=(0, np.pi/2), color_threshold = (100, 100), ksize=9):

	# Work on S channel
	# Threshold the color gradient
	# S channel for yellow and white lines
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

	# L Channel to filter shadows
	l_channel = hls[:,:,1]
	l_binary = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])

	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# Sobel x
	sxbinary = abs_sobel_thresh(gray, thresh=sx_thresh)

	# Directional threshold
	dir_binary = direction_threshold(gray, dir_thresh, ksize)

	# R & G helps with white and yellow lines
	R = img[:,:,0]
	G = img[:,:,1]
	color_combined = np.zeros_like(R)
	rg_binary = (R > color_threshold[0]) & (G > color_threshold[1])

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	# color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 1
	combined_binary[(rg_binary &  l_binary) & ((s_binary == 1) | ((sxbinary == 1)  & (dir_binary == 1)))] = 1
	return combined_binary



def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions
	
	conv_thresh = 50
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template 

	# Sum quarter bottom of image to get slice, could use a different ratio
	l_sum = np.sum(warped[int(3 * warped.shape[0] / 8):, :int(warped.shape[1] / 2)], axis=0)
	l_center = np.argmax(np.convolve(window,l_sum)) - window_width / 2
	r_sum = np.sum(warped[int(3 * warped.shape[0] / 8):, int(warped.shape[1] / 2):], axis=0)
	r_center = np.argmax(np.convolve(window,r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

	if(r_center - l_center >= np.mean(road_width) + 50) or (r_center - l_center <= np.mean(road_width) - 50):
		if lane.detected:
			l_center = lane.l_center
			r_center = lane.r_center
		else:
			
			offset = window_width / 2
			l_min_index = int(max(l_center + offset - margin, 0))
			l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
			roi_zone = np.convolve(window,l_sum)
			conv_strength_l = np.max(roi_zone)

			r_min_index = int(max(r_center + offset - margin, 0))
			r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
			roi_zone = np.convolve(window,r_sum)
			conv_strength_r = np.max(roi_zone)

			if conv_strength_r >= conv_thresh and conv_strength_l >= conv_thresh:
				print("CRITICAL SITUATION")
			elif conv_strength_r < conv_thresh and conv_strength_l < conv_thresh:
				print("CRITICAL SITUATION 2")
			elif conv_strength_l < conv_thresh :
				l_center = r_center - np.mean(road_width)
			elif conv_strength_r < conv_thresh :
				r_center = l_center + np.mean(road_width)


	# Add what we found for the first layer
	window_centroids.append((l_center,r_center))
	l_center_prev = l_center
	r_center_prev = r_center

	# Go through each layer looking for max pixel locations
	for level in range(1, (int)(warped.shape[0] / window_height)):
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height) : int(warped.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		
		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width / 2
		l_min_index = int(max(l_center + offset - margin, 0))
		l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
		roi_zone = conv_signal[l_min_index : l_max_index]

		# Filter out zones which have weak conv signal
		# most likely there are stray pixels or no signal at all
		conv_strength_l = np.max(roi_zone)
		l_center = np.argmax(roi_zone) + l_min_index - offset

		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center + offset - margin, 0))
		r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
		roi_zone = conv_signal[r_min_index : r_max_index]
		
		# Filter out zones which have weak conv signal
		# most likely there are stray pixels or no signal at all
		conv_strength_r = np.max(roi_zone)
		r_center = np.argmax(roi_zone) + r_min_index - offset
		# window_centroids.append((l_center, r_center))

		if conv_strength_r >= conv_thresh and conv_strength_l >= conv_thresh:
			window_centroids.append((l_center, r_center))
			l_center_prev = l_center
			r_center_prev = r_center
			road_width.append(r_center - l_center)
		elif conv_strength_r < conv_thresh and conv_strength_l < conv_thresh:
			window_centroids.append((l_center_prev, r_center_prev))
			road_width.append(r_center_prev - l_center_prev)
		elif conv_strength_l < conv_thresh :
			l_center = r_center - np.mean(road_width)
			window_centroids.append((l_center, r_center))
			r_center_prev = r_center
			l_center_prev = l_center
			road_width.append(r_center - l_center)
		elif conv_strength_r < conv_thresh :
			r_center = l_center + np.mean(road_width)
			window_centroids.append((l_center, r_center))
			l_center_prev = l_center
			r_center_prev = r_center
			road_width.append(r_center - l_center)

		lane.detected = True
		lane.l_center = l_center
		lane.r_center = r_center

	return window_centroids

def sliding_window_search_convolution(warped, window_width, window_height, margin):
	# print("New image \n\n")
	window_centroids = find_window_centroids(warped, window_width, window_height, margin)

	# If we found any window centers
	if len(window_centroids) > 0:

		# Points used to draw all the left and right windows
		l_points = np.zeros_like(warped)
		r_points = np.zeros_like(warped)
		leftx = []
		rightx = []

		# Go through each level and draw the windows 	
		for level in range(0, len(window_centroids)):
			# get the centroid points to the lanes points
			leftx.append(window_centroids[level][0])
			rightx.append(window_centroids[level][1])

			# Window_mask is a function to draw window areas
			if window_centroids[level][0]:
				l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
				# Add graphic points from window mask here to total pixels found 
				l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
			if window_centroids[level][1]:
				r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
				# Add graphic points from window mask here to total pixels found 
				r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

		# Draw the results
		template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
		zero_channel = np.zeros_like(template) # create a zero color channel
		template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
		warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 color channels
		output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

	# If no window centers found, just display orginal road image
	else:
		print("no lane detected")
		output = np.array(cv2.merge((warped, warped, warped)),np.uint8)

	

	return output, leftx, rightx



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
	
	# Get the Color and Gradient  threshold
	gray_gradient = image_gradient(dst, s_thresh=(120, 255), \
		l_thresh=(20, 255), \
		sx_thresh=(20, 255), \
		dir_thresh=(np.pi/6, np.pi/2), \
		color_threshold = (80, 80),\
		ksize=3)

	# Perspective transform to get top view
	bottom_left = [220,720]
	bottom_right = [1110, 720]
	top_left = [570, 470]
	top_right = [722, 470]
	corners = np.float32([top_left, top_right, bottom_right, bottom_left])
	
	bottom_left_dst = [320,720]
	bottom_right_dst = [920, 720]
	top_left_dst = [320, 1]
	top_right_dst = [920, 1]
	dest = np.float32([top_left_dst, top_right_dst, bottom_right_dst, bottom_left_dst])

	warped, Minv = fix_perspective(gray_gradient, mtx, dist, corners, dest)

	# Perform sliding window search using convolution to locate the lanes
	output, leftx, rightx = sliding_window_search_convolution(warped, window_width, window_height, margin)
	# output, left_fitx, right_fitx, ploty = sliding_window_search_histogram(warped)


	# Fit the lane lines

	# Find the y center of the box
	ploty = range(0, warped.shape[0])

	res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, - window_height)

	# Polynomial fitting of the lane points
	left_fit = np.polyfit(res_yvals, leftx, 2)
	left_fitx = left_fit[0] * ploty * ploty + left_fit[1] * ploty + left_fit[2]
	left_fitx = np.array(left_fitx, np.int32)

	right_fit = np.polyfit(res_yvals, rightx, 2)
	right_fitx = left_fit[0] * ploty * ploty + right_fit[1] * ploty + right_fit[2]
	right_fitx = np.array(right_fitx, np.int32)

	left_lane.append(left_fitx)
	right_lane.append(right_fitx)


	final_left_lane = np.mean(left_lane, axis=0)
	final_right_lane = np.mean(right_lane, axis=0)
	# radius_l, radius_r = calculate_radius(np.array(res_yvals), final_left_lane, final_right_lane)

	# Combine the result with the original image
	newwarp = project_lanes(warped, Minv, final_left_lane, final_right_lane, ploty)
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

	if is_test == True:	
		# plot and save the transformations
		tkns = filename.split("/")
		fname = "output_images/warped_" + tkns[1]

		fig = plt.figure(figsize=(16, 9))

		# f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 3, figsize=(24, 9))
		# fig.tight_layout()

		ax1 = fig.add_subplot(231)
		ax1.imshow(img)
		ax1.set_title('Original Image')

		ax2 = fig.add_subplot(232)
		ax2.imshow(gray_gradient, cmap='gray')
		ax2.set_title("After Thresholding")

		ax3 = fig.add_subplot(233)
		ax3.imshow(warped, cmap='gray')
		ax3.set_title('After Perspective transform')
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		cursor = Cursor(ax1, useblit=True, color='red', linewidth=2)

		# histogram_peak_search()
		ax4 = fig.add_subplot(235)		
		ax4.imshow(output)
		ax4.set_title('Detected lanes')

		mark_size = 3
		ax5 = fig.add_subplot(236)
		ax5.imshow(warped, cmap='gray')
		ax5.plot(leftx, res_yvals, 'o', color='red', markersize=mark_size)
		ax5.plot(rightx, res_yvals, 'o', color='blue', markersize=mark_size)
		ax5.plot(left_fitx, ploty, color='green', linewidth=1)
		ax5.plot(right_fitx, ploty, color='green', linewidth=1)
		
		ax6 = fig.add_subplot(234)
		ax6.imshow(result)

		fig.savefig(fname)
		plt.show()

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

	# print(left_curverad, right_curverad)
	return left_curverad, right_curverad


# First Calibrate the camera
# Use all the calibration images in the camera_cal folder
# to get the final mtx and dist factors
global mtx, dist
lane = Lane()

ret, mtx, dist, rvecs, tvecs = calibrate_camera()

# Undistort and perform a perspective transform of the images
# using the mtx and dist factors
# Save the final images
# This will allow us to verify that the calibration is done correctly

'''
TEMP
for filename in glob.iglob("camera_cal/calibration*.jpg"):
	img = mpimg.imread(filename)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	if ret:
		src = np.float32(
			[corners[0],
			corners[nx-1],
			corners[-1],
			corners[-nx]])

		dest = np.float32(
			[[100, 100], 
			[img_size[0] - 100, 100], 
			[img_size[0]-100, img_size[1]-100], 
			[100, img_size[1]-100]])

		warped = fix_perspective(dst, mtx, dist, src, dest)
		tkns = filename.split("/")
		fname = "output_images/warped_" + tkns[1]
		Image.fromarray(warped).save(fname)
'''

# Use the calibration factors to work on the test images

'''
for filename in glob.iglob("test_images/test*.jpg"):
	img = mpimg.imread(filename)
	process_image(img, is_test=True)

'''

output = 'output_videos/project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
output_clip.write_videofile(output, audio=False)
