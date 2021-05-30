# import packages
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# the default parameters
DATASET_DIR = "dataset"
PLOT_IMG_NAME = "accuracy_and_loss_curve.png"
MODEL_NAME = "mask_detector.model"
INIT_LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

def load_data(dataset_dir):
	"""
    Parameters
    ----------
	dataset_dir: string
		The name of the directory to import the dataset from
		
	Return
	------
	data: ndarray
		2D numpy array that contains the image pixel values for each of the images
	labels: ndarray 
		2D numpy array which contains the one hot representation for each of the labels

	Description
	-----------
	Load the data from the dataset directory
	"""
	# Obtain all the image pathnames
	imagePaths = list(paths.list_images(dataset_dir))
	
	# Create an empty list to store the image data and labels
	data = []
	labels = []

	# Iterate through all the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		# Load the image and resize it to 224*224
		image = load_img(imagePath, target_size=(224, 224))

		# Convert the image to array object
		image = img_to_array(image)

		# Preprocess the image
		image = preprocess_input(image)

		# Append the image data and label
		data.append(image)
		labels.append(label)

	# convert the data and labels to numpy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels)
	return data, labels

def one_hot_encode(labels):
	"""
    Parameters
    ----------
    epochs : int
        The total number of epochs to train the model
	History : Keras History
		Contains the value of loss and accuracy for each epoch
		
	Return
	------
	labels: ndarray 
		2D numpy array which contains the one hot representation for each of the labels
	lb: Label Binarizer
		The object that is used to transform the labels between one hot representation
		and string labels

	Description
	-----------
	Perform one hot encoding on the labels
	"""

	# Build the label binarizer object
	lb = LabelBinarizer()

	# Fit the label to the label binarizer object and transform the labels
	# into the index of the label
	labels = lb.fit_transform(labels)
	
	# Transform the label into one-hot representation
	labels = to_categorical(labels)

	return labels, lb

def build_model(learning_rate, epoch):
	"""
    Parameters
    ----------
    epochs : int
        The total number of epochs to train the model
	learning_rate : int
		The learning rate to train the model 
		
	Return
	------
	model: Keras model
		The keras model that is already compiled

	Description
	-----------
	Build and return the model 
	"""

	# Construct the MobileNetV2 network, ensuring the head FC layer sets are left off
	baseModel = MobileNetV2(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(224, 224, 3)))

	# Construct the model to be trained
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# place the head FC model on top of the base model (this will become the actual model we will train)
	model = Model(inputs=baseModel.input, outputs=headModel)
	
	# Make sure the base model is not trainable 
	for layer in baseModel.layers:
		layer.trainable = False

	# Define the optimizer and compile the model
	opt = Adam(learning_rate=learning_rate, decay=learning_rate / epoch)
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
	return model

def plot_acc_and_loss_graph(epochs, History, plot_img_name):
	"""
    Parameters
    ----------
    epochs : int
        The total number of epochs to train the model
	History : Keras History
		Contains the value of loss and accuracy for each epoch
	plot_img_name : string
		The filename of the graph when it is saved

	Description
	-----------
	Plot and save the the loss and accuracy graph over epochs
	"""
	
	plt.style.use("ggplot")
	plt.figure()

	# plot the data according to x and y values, also define the label name for it
	plt.plot(np.arange(0, epochs), History.history["loss"], label="Train Loss")
	plt.plot(np.arange(0, epochs), History.history["val_loss"], label="Validation Loss")
	plt.plot(np.arange(0, epochs), History.history["accuracy"], label="Train Accuracy")
	plt.plot(np.arange(0, epochs), History.history["val_accuracy"], label="Validation Accuracy")

	# title of the graph
	plt.title("Training Loss and Accuracy")

	# Label for the graph on x and y axis
	plt.xlabel("Epoch")
	plt.ylabel("Loss/Accuracy")

	# Put the legend on the lower left of the graph
	plt.legend(loc="lower left")

	# Save the graph as image
	plt.savefig(plot_img_name)


# Load the dataset 
data, labels = load_data(DATASET_DIR)

# Encode the labels into one hot representation
labels, lb = one_hot_encode(labels)

# Split 80% train set and 20% test set
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Generate more data as training set
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Build and compile the model
model = build_model(INIT_LEARNING_RATE, EPOCHS)

# Train the model
History = model.fit(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=int(len(trainX) / BATCH_SIZE),
	validation_data=(testX, testY),
	validation_steps=int(len(testX) / BATCH_SIZE),
	epochs=EPOCHS)

# Make predictions using the trained model on the test set
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)


# Find out the index of the label predicted by finding the index
# that is having highest probability value for each image predictions
predIdxs = np.argmax(predIdxs, axis=1)

# Find out the index of the actual label by finding the index
# that is having highest value which will be 1 because it is in one hot representation
testIdxs = np.argmax(testY, axis=1)

# Calculate and print out the accuracy of the model predicting the test set by
# comparing the predicted indexes and the actual indexes
accuracy = round((np.sum(predIdxs == testIdxs) / len(predIdxs)) * 100, 2)
print(f"Accuracy: {accuracy}%")

# Show the classification report including the precision, recall and F1 score values
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Save the model
model.save(MODEL_NAME, save_format="h5")

# Plot the lost and accuracy graph for the model
plot_acc_and_loss_graph(EPOCHS, History, PLOT_IMG_NAME)