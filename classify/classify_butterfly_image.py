# Adapted from pyimagesearch.com
# http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
# USAGE
# python classify_butterfly_image.py --image images/butterfly.jpg --model inception
# download models from https://drive.google.com/drive/folders/0B5V_P8iI1LUeMzBTQzlWLTZwVTg
# store the candidate butterfly image or images in the images folder

# import the necessary packages
import os, sys
import numpy as np
import argparse
from pathlib import Path
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help = "name of model to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": "butterflies_vgg16_model_ft.h5",
	"vgg19": "butterflies_vgg19_model_ft.h5",
	"inception": "butterflies_inception_model_ft.h5",
	"xception": "butterflies_xception_model_ft.h5"
	}

# check for valid image
print("[INFO] checking image - {}...".format(args["image"]))
if os.path.exists(args["image"]) ==  False:
	print("Image does not exist")
	sys.exit()

# ensure a valid model name was supplied via command line argument
print("[INFO] checking model file -  {}...".format(args["model"]))
if args["model"] not in MODELS.keys():
	print("The --model command line argument should "
		"be a key in the 'MODELS' dictionary. "
		"Choices are:\n" + "vgg16\n" + "vgg19\n" + "inception\n" + "xception")
	sys.exit()
model_file = "butterflies_" + args["model"] + "_model_ft.h5"
if os.path.exists(model_file) == False:
	print(args["model"] + " model file is missing")
	sys.exit()

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# load the model from disk (NOTE: if this is the
# first time this script is used for a given model, the
# weights will need to be downloaded first -- depending on which
# model is being used because of the size of the model.
print("[INFO] loading {}...".format(args["model"]))
model_file = MODELS[args["model"]]
loaded_model = load_model(model_file)

# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")

img = load_img(args["image"])
if os.path.exists(args["image"]):
	image = load_img(args["image"], target_size=inputShape)
else:
	print("Image does not exist")
	sys.exit()
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through thenetwork
image = np.expand_dims(image, axis=0)

# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)

# classify the image
print("[INFO] classifying image with {}...".format(args["model"]))
#preds = loaded_model.predict(image)
print("[INFO] close Image to display result...")
preds = [np.argmax(loaded_model.predict(image))]
print(preds)

# display the image
plt.imshow(img)
plt.axis('off')
plt.show()

# display the classification results
for preds in preds:
	if preds == 0:
		print("[RESULT] Image predicted to be Danaus Plexippus")
	elif preds == 1:
		print("[RESULT] Image predicted to be Heliconius Charitonius")
	elif preds == 2:
		print("[RESULT] Image predicted to be Heliconius Erato")
	elif preds == 3:
		print("[RESULT] Image predicted to be Junonia Coenia")
	elif preds == 4:
		print("[RESULT] Image predicted to be Lycaena Phlaeas")
	elif preds == 5:
		print("[RESULT] Image predicted to be Nymphalis Antiopa")
	elif preds == 6:
		print("[RESULT] Image predicted to be Papilio Cresphontes")
	elif preds == 7:
		print("[RESULT] Image predicted to be Pieris Rapae")
	elif preds == 8:
		print("[RESULT] Image predicted to be Vanessa Atalanta")
	elif preds == 9:
		print("[RESULT] Image predicted to be Vanessa Cardui")
	else:
		print("[RESULT] Image does not fall into any of these ten birds categories:\n"
		+ "Danaus Plexippus\n" + "Heliconius Charitonius\n" + "Heliconius Erato\n"
		+ "Junonia Coenia\n" + "Lycaena Phlaeas\n" + "Nymphalis Antiopa\n" + "Papilio Cresphontest\n"
		+ "Pieris Rapae\n" + "Vanessa Atalanta\n" + "Vanessa Cardui")
