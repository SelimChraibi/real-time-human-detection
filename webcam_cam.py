import numpy as np
import cv2
import sys

from utilities.gapModels import MobileNetGAP
from utilities.classifier import Classifier
from utilities.helpers import *

print("⏳" + BLUE + " Loading model ... " + END)

model = MobileNetGAP(path='saved_model/mobilenet.h5')
clf = Classifier(model, name='mobilenet')

print("💾" + BLUE + " Model loaded." + END)


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

	img, prediction = cam, prediction = clf.cam(img, class_number=1)
	img = show_detection(img, prediction[1])
	cv2.imshow('img',img)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
