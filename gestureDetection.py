'''
This program is used to detect gestures using OpenCv in Python
Author: Vivan Bhalla
Date Created: Feb 25th 2018
'''

# Import libraries
import cv2
import numpy as np

# Function to be called when trackbars change
#def nothing(x):
 #   pass # IT simply does nothing
#cv2.namedWindow('image')
#cv2.createTrackbar('Threshold','image',0,255,nothing)



# Capture the video
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	ret,frame=cap.read() # Get the frame

	#thr=cv2.getTrackbarPos('Threshold','image')
	
	# Perform Background subtraction
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	blur=cv2.GaussianBlur(gray,(5,5),0) # Blurs an image i.e it soothens an image. It performs smoothing using gaussian function

	ret,threshold=cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	cv2.imshow('input',threshold)

	k = cv2.waitKey(10)
	if k == 27:
		break