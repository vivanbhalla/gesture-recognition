'''
Gesture Detection algorithm using NN
Author: Vivan Bhalla
We will be using Keras library for creating CNN
'''

#from keras.model import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.utils import np_utils
#from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D

import numpy as np
#import theano
import os
#from PIL import Image
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
#import json

import cv2
import matplotlib
from matplotlib import pyplot as plt

THRESHOLD=70
imgHeight=200
imgWidth=200

# Rectangle coordinates. This rectanlge will define the space on which the hand has to be placed
xRect=400
yRect=200

# Define kernels for skin mode 
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Capture the video
cap = cv2.VideoCapture(0)


'''
Function to create a skin mask

Parameters: frame - The frame captured from webcam
			xR - x coordinates for the rectangle
			yR - y coordinates for the rectangle
			width - width for the rectangle
			height - height for the rectanlge

Note: The rectanlge has to be the size of the imsage that we are inputting

'''
def skinMask(frame,xR,yR,width,height):

	# HSV values for skin color ( Modify according to the color of skin)
	lower_hsv=np.array([0,50,80])
	upper_hsv=np.array([30,200,255])

	# Create the recgtangle with size of image and of thickness 1
	cv2.rectangle(frame,(xR,yR),(xR+width,yR+height),(0,0,255),1)

	# Get the region of interest from the frame i.e the image of hand
	roi=frame[yR:yR+height,xR:xR+width]

	# Convert the image to hsv format
	hsv= cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

	# Apply mask for skin color
	mask = cv2.inRange(hsv,lower_hsv,upper_hsv)
	mask = cv2.erode(mask,skinkernel,iterations=1)
	mask = cv2.dilate(mask, skinkernel, iterations = 1)

	# Blur the image
	mask=cv2.GaussianBlur(mask,(15,15),1)

	# Bitwise the image
	result=cv2.bitwise_and(roi,roi,mask=mask)
	result=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)

	return result

	


while(True):
	ret, frame = cap.read()
	result = skinMask(frame,xRect,yRect,imgWidth,imgHeight)
	cv2.imshow("Result",result)

	k = cv2.waitKey(5) & 0xFF
	if k ==27:
		break
