import numpy as np
import cv2
import sys
import argparse

# Creating the parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to the model used to make the prediction and generate the class activation maps")
# Parsing the arguments
args = vars(ap.parse_args())

from utilities.gapModels import MobileNetGAP
from utilities.classifier import Classifier
from utilities.helpers import *

print("⏳" + BLUE + " Loading model ... " + END)

model = MobileNetGAP(path=args["model"])
clf = Classifier(model, name='mobilenet')

print("💾" + BLUE + " Model loaded." + END)

def addContours(input_img, output_img, draw_bounding_box=True, draw_contours=False, threshold=100):
    """
	>>> Work In Progress <<<
    Detects the bounding boxes and/or contours in the input image and adds them to the output image
    Returns the modified output_img
	>>> Work In Progress <<<
    """
	# Convert image to gray
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, threshed_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Get the external contours
    _, contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if draw_contours:
        # Draw the contours
        cv2.drawContours(output_img , contours, -1, (0, 255, 0), 5)

    if draw_bounding_box:
        # Draw the bounding boxes
        for c in contours:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(c)
            # Draw it
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return output_img

def show_detection(img, prediction):
	height, width, _ = img.shape
	middle = (int(height//2), int(width//2))
	if prediction>0.6: cv2.rectangle(img,(10,10),(width-10, height-10),(255, 255, 255),thickness=40)
	cv2.rectangle(img,(0,0),(width, 40),(56, 38, 50),thickness=-1)
	cv2.rectangle(img,(0,0),(int(width*prediction), 40),(118, 230, 0),thickness=-1)

	return img


cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()

	# Get the cam and prediction made by the model
	cam, prediction = clf.cam(img, class_number=1)

	# Detect the contours and or bounding boxes in the cam
	# img = addContours(input_img=cam, output_img=img, draw_bounding_box=True, draw_contours=False, threshold=100)

	# Add the cam to the original image
	img = cv2.addWeighted(cam, 0.5, img, 0.8, 0)

	# Indicators of the probability of presence of a human
	img = show_detection(img, prediction[1])

	cv2.imshow('img',img)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
