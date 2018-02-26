'''
This program is used to detect gestures using OpenCv in Python
Author: Vivan Bhalla
Date Created: Feb 25th 2018
'''

# Import libraries
import cv2
import numpy as np

# Capture the video
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
	ret,frame=cap.read() # Get the frame
	
	# Perform Background subtraction
	k = cv2.waitKey(10)
	if k == 27:
		break