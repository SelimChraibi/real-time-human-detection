import numpy as np
import cv2
from tools import *
from models import mobilenet
import sys


def init(model_name):
	"""
	Loading the model and its saved weights
	"""
	model, name_of_final_conv_layer, name_of_dense_layer = mobilenet.get_model(nb_classes=2, summary=False)
	model.load_weights('saved_model_weights/' + model_name + '.hdf5')
	return model, name_of_final_conv_layer, name_of_dense_layer


def show_detection(img, prediction):
	height, width, _ = img.shape
	middle = (int(height//2), int(width//2))
	if prediction>0.6: cv2.rectangle(img,(0,0),(width, height),(255, 255, 255),thickness=40)
	cv2.rectangle(img,(0,0),(width, 40),(56, 38, 50),thickness=-1)
	cv2.rectangle(img,(0,0),(int(width*prediction), 40),(118, 230, 0),thickness=-1)
	return img

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


model_name = 'MobileNet alpha=0.5 unfrozen=2 optimizer=adam epochs=20 batch_size=32 data_augmentation=False'
model, name_of_final_conv_layer, name_of_dense_layer = init(model_name)


cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()

	img, prediction = cam(model, img, name_of_final_conv_layer, name_of_dense_layer, 1)
	img = show_detection(img, prediction)
	cv2.imshow('img',img)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
