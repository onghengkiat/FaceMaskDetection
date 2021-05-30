# Import packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# default parameters
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

def predict(face_props, image, model):
	"""
    Parameters
    ----------
    face_props : ndarray
        Numpy array which contains the coordinates and size of the facee
	image: ndarray
		The image to extract the face from 
	model: Keras model
		The trained model used to classify whether the face is wearing a mask or not

    Returns
    -------
    pred : ndarray
        2D Numpy array that contains the predicted probability 
		for each of the labels for each of the input images 

	Description
	-----------
	Extract out the face from the image and classifiy whether the face
	is wearing a mask or not
    """

	(x, y, w, h) = face_props
 
	face = image[y:y+h, x:x+w]
	# Convert the image to RGB scale
	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

	# Resize the face to 224*224 which is the input size
	# for the predictive model
	face = cv2.resize(face, (224, 224))

	# Conver it to an array
	face = img_to_array(face)

	# Preprocess the image
	face = preprocess_input(face)

	# Expand one more dimension to be fit into the model for prediction
	face = np.expand_dims(face, axis=0)

	# Used the trained model to predict whether it is wearing mask or not
	pred = model.predict(face)

	return pred
	


def detect_faces(image, face_cascade):
	"""
	Parameters
	----------
	image: ndarray
		The image to detect faces from 
	face_cascade: Cascade Classifier
		Pretrained cascade classifier model that is used to detect the faces

	Return
	------
	faces: ndarray
		A 2d array which consists of the coordinates
		and size for each of the face(s) detected in the image

	Description
	-----------
	Detect the faces exist in the image by using the pretrained cascade classifier
	model 
	"""

	# Change the image to gray scale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Identify the face(s) exist in the image
	faces = face_cascade.detectMultiScale(gray_image, minNeighbors=6, minSize=[30,30])

	return faces

def label_face(face):	
	"""
	Parameters
	----------
	face: ndarray
		Numpy array which contains the coordinates and size of the face

	Description
	-----------
	Label the face on the image whether is wearing mask or not and 
	draw a rectangle surrounding it
	"""
	(x,y,w,h) = face

	# Get the probability of wearing mask or not wearing mask that are
	# predicted using the trained model
	[[wear_mask, not_wear_mask]] = predict(face, image, maskNet)

	# Identify label text and colour on the image
	if wear_mask > not_wear_mask:
		label = "Wearing Mask"
		label_color = GREEN_COLOR
	else:
		label = "Not Wearing Mask"
		label_color = RED_COLOR
	
	# Label the face on the image whether is wearing mask or not
	cv2.putText(image, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 2)

	# Draw the rectangle around the face detected on the image
	cv2.rectangle(image, (x,y), (x+w, y+h), label_color, 2)

# Load the trained model
maskNet = load_model("mask_detector.model")

# Load the pretrained model that uses cascade classifier to detect the faces
# Source of the pretrained model: 
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Image files to be classified
images = [
	cv2.imread("dataset/with_mask/0-with-mask.jpg"), 
	cv2.imread("dataset/without_mask/0.jpg"), 
	cv2.imread("Ong Heng Kiat(without mask).jpg"), 
	cv2.imread("Ong Heng Kiat(with mask).jpg")
]

for image in images:
	# Detect the face(s) exist in the image
	faces = detect_faces(image, face_cascade)

	# Label each of the faces in the image
	for face in faces:
		label_face(face)

	# Show the labelled image
	cv2.imshow("Face Mask Detector", image)

	# Do not close the image until user clicks any click on the image 
	# so that we as the users will have time to observe the image
	cv2.waitKey(0)
