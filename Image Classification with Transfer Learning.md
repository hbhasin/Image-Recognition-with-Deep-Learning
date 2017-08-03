# Image Classification with Transfer Learning

<p align="center"> 
<img src= "https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/splash.JPG">
</p>

Deep Learning is an emerging field of research and Transfer Learning is one of its benefits. In image classification, for example, Transfer Learning makes use of features learned from one domain and used on another through feature extraction and fine-tuning. Convolutional Neural Network (also known as ConvNet) models trained on the [ImageNet's](http://www.image-net.org) million images with 1000 categories have been successfully used on other similar or dissimilar datasets, large or small, with great success. In particular, given the fact that data acquisition is expensive, small datasets can benefit from these pre-trained networks because the lower layers of these pre-trained networks already contain many generic features such as edge and color blob detectors and only the higher layers need to be trained on the new datasets.

According to [Pan, et al](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf), “research on transfer learning has attracted more and more attention since 1995 in different names: learning to learn, life-long learning, knowledge transfer, inductive transfer, multi-task learning, knowledge consolidation, context sensitive learning, knowledge-based inductive bias, meta learning, and incremental/cumulative learning”. They describe the difference between the learning processes of traditional and transfer learning techniques in the figure below.

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Traditional%20vs.%20Transfer%20Learning.PNG"></a> 

Figure 1: Different learning processes between traditional machine learning and Transfer Learning [Pan, et al](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)


A classic demonstration of Transfer Learning is in image classification using [Kaggle’s](https://www.kaggle.com/datasets) Dogs versus Cats dataset. Using 1000 cats and 1000 dogs from this dataset of 12,500 cats and 12,500 dogs, a [three-layer ConvNet](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) model has been shown to be capable of achieving [79-81%](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) accuracy after 50 epochs. With a pre-trained [ImageNet VGG16 model](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), the accuracy improves to [90-91%](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). Finally, with fine-tuning, the accuracy improves further to [94%](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html). The VGG architecture is shown in the figure below.


<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Figure%202%20-%20VGG16%20Architecture.PNG"></a>

Figure 2: [VGG16/VGG19 Architecture](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5)

Other applications that have benefited from Transfer Learning include [object detection](http://arxiv.org/abs/1311.2524), [zero-shot learning](http://arxiv.org/abs/1312.5650), [image captioning](http://googleresearch.blogspot.com/2014/11/a-picture-is-worth-thousand-coherent.html) and [video analysis](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6751448).

There are several libraries in use today that support deep learning. Each has its own pros and cons. Some of the most commonly used ones for the Python Language are as follows:

-  [Theano](http://deeplearning.net/software/theano) - A Python library that allows one to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays.
-  [Tensorflow](https://www.tensorflow.org/) – Uses data flow graphs for numerical computation.
-  [Lasagne](https://github.com/Lasagne/Lasagne) – A lightweight library to build and train neural networks in Theano.
-  [Caffe](http://caffe.berkeleyvision.org/) -  A deep learning framework made with expression, speed, and modularity in mind.
-  [Caffe2](https://caffe2.ai/) – A new lightweight, modular, and scalable deep learning framework.
-  [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/) – Uses directed graphs for describing neural networks. Previously known as CNTK.
-  [PyTorch](http://pytorch.org/) – Builds neural networks using a technique called reverse-mode auto-differentiation.
-  [Keras](http://keras.io/) - Keras is a high-level neural networks API, written in Python and capable of running on top of [Tensorflow](https://www.tensorflow.org/), [Microsoft Cognitive Toolkit](https://www.microsoft.com/en-us/cognitive-toolkit/), or [Theano](http://deeplearning.net/software/theano). It was developed with a focus on enabling fast experimentation.


The [Keras](https://keras.io/applications/#usage-examples-for-image-classification-models) Deep Learning library for Python currently supports five models that have been pre-trained on ImageNet:
-  [VGG16](https://arxiv.org/abs/1409.1556) – A  convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. 
-  VGG19 – It is essentially the VGG16 model with three additional weight layers.
-  [Inception V3](https://arxiv.org/abs/1512.00567) - 

-  [ResNet50](https://arxiv.org/abs/1512.03385) – A residual learning framework that eases the training of deep networks. It reformulates the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.
-  Xception


This study will use the Keras library to explore ImageNet’s pre-trained VGG16, VGG19, Inception V3 and Xception models to perform image classification on a variety of small datasets with different domains.

## Hardware Details
Training Deep Learning networks require tremendous processing power to handle multiple matrix multiplications. GPUs are  ideal for performing these operations.  Facebook recently reported that its scientits were able to train nearly 1.3 million images in under one hour using [256 Tesla P100 GPUs](https://news.developer.nvidia.com/facebook-trains-imagenet-in-1-hour/) that previously took days on a single system. In 2012 the ImageNet ILSVRC model was trained on 1.2 million images over the period of 2–3 weeks across multiple GPUs. For the small datasets used in this project having a Titan GTX 1080 GPU would have been able to train 8X faster than an i7 Intel CPU running at 3.5GHz.

However, no GPU was available for this project so the datasets were trained on two systems with Core i7 CPUs and on two systems with Core i5 CPUs.

## Training, Validating and Testing Datasets Process
The following steps were used in to train, validate and test the datasets in this project:

1. Quick check Keras ImageNet pre-trained model's capability
2. Load Data
3. Define Model
4. Compile Model
5. Fit (Train) Model
6. Evaluate Model
7. Test Model

As a comparison, a simple one convolutional layer model was built to train and validate the two-class (Fried Noodles and Noodle Soup) image classifier and check it against the pre-trained models on training time and accuracy.

## Quick Checkout on Keras ImageNet pre-trained Models

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Noodles%20-%20Initial%20Checkout.PNG"></a> 

Top Prediction | VGG16 - Fried Noodles | VGG16 - Noodle Soup | VGG19- Fried Noodles | VGG19 - Noodle Soup
-------------- | --------------- | ------------- | --------------- | -------------
Carbonara | 0.61 | |
Soup Bowl | | 0.51 | | 0.55
Plate     | | |0.54 |

Top Prediction | InceptionV3 - Fried Noodles | InceptionV3 - Noodle Soup | Xception- Fried Noodles | Xception - Noodle Soup
-------------- | --------------- | ------------- | --------------- | -------------
Carbonara | 0.81 | | 0.73
Soup Bowl | | 0.75 | | 0.62

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Leaves%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Leaf 01 | VGG16 - Leaf 02 | VGG16- Leaf 03 | VGG19 - Leaf 01 | VGG19 - Leaf 02 | VGG19 - Leaf 03
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Crossword Puzzle | 0.61 | |
Lacewing | | 0.24 | | | 0.30
Tray     | | | 0.23 |
Picket Fence | | | | 0.09 | | |
Envelope | | | | | | 0.096

Top Prediction | InceptionV3 - Leaf 01 | InceptionV3 - Leaf 02 | InceptionV3- Leaf 03 | Xception - Leaf 01 | Xception - Leaf 02 | Xception - Leaf 03
-------------- | --------------- | ------------- | --------------- | ------------- | ----- | ----- |
Computer Keyboard | 0.81 | | | 0.73
Pot | | 0.19 | 0.1
Head Cabbage | | | | | 0.1
Packet | | | | | | 0.05

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Dogs%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Bull Dog | VGG16 - Pit Bull | VGG16- Foxhound | VGG19 - Bull Dog | VGG19 - Pit Bull | VGG19 - Foxhound
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
French Bull Dog | 0.63
Staffordshire Bull Terrier | | 0.97 | | | 0.93
Walker Hound     | | | 0.47
Bull Mastiff | | | | 0.37
Basset | | | | | | 0.69

Top Prediction | InceptionV3 - Bull Dog | InceptionV3 - Pit Bull | InceptionV3- Foxhound | Xception - Bull Dog | Xception - Pit Bull | Xception - Foxhound
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
French Bull Dog | 0.42 | | | 0.15
Staffordshire Bull Terrier | | 0.90 | | | 0.51
Basset     | | | 0.96 | | | 0.94

Top Prediction | VGG16 - Border Collie | VGG16 - Siberian Husky | VGG16 - Bernese Mountain Dog | VGG19 - Border Collie | VGG19 - Siberian Husky | VGG19 - Bernese Mountain Dog
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Border Collie | 0.33 | | | 0.5|
Siberian Husky | | 0.71 | | | 0.65
Bernese Mountain Dog | | | 0.86 | | | 0.98

Top Prediction | InceptionV3 - Border Collie | InceptionV3 - Siberian Husky | InceptionV3- Bernese Mountain Dog | Xception - Border Collie | Xception - Siberian Husky | Xception- Bernese Mountain Dog
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Border Collie | 0.35 | | | 0.5|
Siberian Husky | | 0.69 | | | 0.65
Bernese Mountain Dog | | | 0.98 | | | 0.93

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Birds_01%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Common Yellowthroat | VGG16 - Winter Wren | VGG16 - Pine Warbler | VGG16 - Common Yellowthroat | VGG16 - Winter Wren | VGG16 - Pine Warbler
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Gold Finch | 0.95 | | | 0.8
Water Ouzel | | 0.36 | | | 0.66
Jay | | | 0.39 | | |
Litte Blue Heron | | | | | | 0.49

Top Prediction | InceptionV3 - Common Yellowthroat | InceptionV3 - Winter Wren | InceptionV3 - Pine Warbler | Xception - Common Yellowthroat | Xception - Winter Wren | Xception - Pine Warbler
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Bee Eater | 0.78
House Finch | | 0.22
Brambling | | | 0.37 | | | 0.49
Robin | | | | 0.53
Water Ouzel | | | | | 0.4

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Birds_02%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Sage Thrasher | VGG16 - Lincoln Sparrow | VGG16 - Brown Pelican | VGG16 - Sage Thrasher | VGG16 - Lincoln Sparrow | VGG16 - Brown Pelican
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Hummingbird | 0.35
Ruffed Grouse | | 0.27 | | | 0.56
Limpkin | | | 0.84 | | | 0.43
Redshank | | | | 0.53

Top Prediction | InceptionV3 - Sage Thrasher | InceptionV3 - Lincoln Sparrow | InceptionV3 - Brown Pelican | Xception - Sage Thrasher | Xception - Lincoln Sparrow | Xception - Brown Pelican
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Bulbul | 0.29
Bittern | | 0.31 | | 0.56
Pelican | | | 0.64 | | | 0.56
Humminbird | | | | 0.29
Ruffed Grouse | | | | | 0.22

Top Prediction | VGG16 - Green Jay | VGG16 - Eared Grebe | VGG16 - Pelagic Cormorant | VGG19 - Green Jay | VGG19 - Eared Grebe | VGG19 - Pelagic Cormorant
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Toucan | 0.81
American Coot | | 0.74 | | 0.35
Black Stork | | | 0.18
Goldfinch | | | | 0.73
Red-breasted Merganser | | | 0.4
Water Ouzel | | | | | | 0.33

Top Prediction | InceptionV3 - Green Jay | InceptionV3 - Eared Grebe | InceptionV3 - Pelagic Cormorant | Xception - Green Jay | Xception - Eared Grebe | Xception - Pelagic Cormorant
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Jay | 0.19 | | | 0.33
European Gallinule | | 0.35
American Coot | | | 0.35 | | 0.12
Black Grouse | | | | | | 0.26

Top Prediction | VGG16 - Black-footed Albatross | VGG19 - Black-footed Albatross
-------------- | --------------- | ------------- 
Albatross | 0.99 | 0.98

Top Prediction | InceptionV3 - Black-footed Albatross | Xception - Black-footed Albatross
-------------- | --------------- | ------------- 
Albatross | 0.68 | 0.82

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Butterflies_01%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Heliconius Erato | VGG16 - Heliconius Charitonius | VGG16 - Junonia Coenia | VGG19 - Heliconius Erato | VGG19 - Heliconius Charitonius | VGG19 - Junonia Coenia
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Admiral | 0.53 | | | 0.63
Bee | | 0.25 | | | 0.23
Ringlet | | | 0.97 | | | 0.98

Top Prediction | InceptionV3 - Heliconius Erato | InceptionV3 - Heliconius Charitonius | InceptionV3 - Junonia Coenia | Xception - Heliconius Erato | Xception - Heliconius Charitonius | Xception - Junonia Coenia
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Admiral | 0.82 | | | 0.81
Sulphur Butterfly | | 0.8
Ringlet | | | 0.98 | |0.39 | 0.53

Top Prediction | VGG16 - Danaus Plexippus | VGG16 - Lycaena Phlaeas | VGG16 - Nymphalis Antiopa | VGG19 - Danaus Plexippus | VGG16 - Lycaena Phlaeas | VGG16 - Nymphalis Antiopa
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Monarch | 0.99 | | | 0.99
Lycaenid | | 0.56
Admiral | | | 0.99 | | | 0.97
Ringlet | | | | | 0.58

Top Prediction | InceptionV3 - Danaus Plexippus | InceptionV3 - Lycaena Phlaeas | InceptionV3 - Nymphalis Antiopa | Xception - Danaus Plexippus | InceptionV3 - Lycaena Phlaeas | InceptionV3 - Nymphalis Antiopa
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Monarch | 0.96 | | | 0.92
Lycaenid | | | | | | 0.31
Admiral | | | 0.71 | | |
Ringlet | | 0.46 | | | 0.55

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Butterflies_02%20-%20Initial%20Checkout.PNG"></a>

Top Prediction | VGG16 - Papilo Cresphontes | VGG16 - Pieris Rapae | VGG16 - Vanessa Atalanta | VGG19 - Papilo Cresphontes | VGG19 - Pieris Rapae | VGG19 - Vanessa Atalanta
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
King Snake | 0.32
Cabbage Butterfly | | 0.99 | | | 0.99
Admiral | | | 0.99 | | | 0.99
Black and Gold Garden Spider | | | | 0.36

Top Prediction | InceptionV3 - Papilo Cresphontes | InceptionV3 - Pieris Rapae | InceptionV3 - Vanessa Atalanta | Xception - Papilo Cresphontes | Xception - Pieris Rapae | Xception - Vanessa Atalanta
-------------- | --------------- | ------------- | --------------- | ------------- | --------- | --------- |
Admiral | 0.89 | | 0.95 | 0.33 | | 0.94
Cabbage Butterfly | | 0.91 | | | 0.96

Top Prediction | VGG16 - Vanessa Cardui | VGG19 - Vanessa Cardui
-------------- | --------------- | -------------
Admiral | 0.66 | 0.63

Top Prediction | InceptionV3 - Vanessa Cardui | Xception - Vanessa Cardui
-------------- | --------------- | -------------
Monarch | 0.47 | 0.66

## Code Details

The following code applies to all the Keras pre-trained models except as noted otherwise.

## Transfer Learning Phase
In the Transfer Learning phase only the new fully connected layer was trained on the features extracted from the pre-trained models.

### Import Keras Libraries
Keras supports VGG16, VGG19, ResNet50, InceptionV3 and Xception models that have been pre-trained on ImageNet. 

The Dense layer is the densely-connected Neural Network layer and the GlobalAveragePooling2D layer provides an average pooling for spatial data. 

Keras has support for several optimizers that include SGD (Stochastic Gradient Descent), [RMSprop](https://keras.io/optimizers/), [Adagrad](https://keras.io/optimizers/), [Adadelta](https://keras.io/optimizers/), [Adam](https://keras.io/optimizers/), [Adamax](https://keras.io/optimizers/), [Nadam](https://keras.io/optimizers/) and [TFOptimizer](https://keras.io/optimizers/) (for Tensorflow). 

The ImageDataGenerator module generates batches of tensor image data with real-time data augmentation to manage memory.

```
# import pertinent libraries
import os
import sys
import datetime
import glob as glob
import numpy as np
import cv2
# [Keras Models]
# import the Keras implementations of VGG16, VGG19, InceptionV3 and Xception models
# the model used here is VGG16
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
import tensorflow
from scipy.interpolate import spline
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

### Datasets
The datasets used in this project ranged from a simple two-class (Fried Noodles and Noodle Soup) image classification to a ten-class (Variety of butterflies) image classification.

[Noodles Data](https://github.com/openimages/dataset) - Open Images is a dataset of ~9 million URLs to images that have been annotated with labels spanning over 6000 categories. "Krasin I., Duerig T., Alldrin N., Veit A., Abu-El-Haija S., Belongie S., Cai D., Feng Z., Ferrari V., Gomes V., Gupta A., Narayanan D., Sun C., Chechik G, Murphy K. OpenImages: A public dataset for large-scale multi-label and multi-class image classification, 2016. Available from https://github.com/openimages".

[Leaves Data](http://www.vision.caltech.edu/Image_Datasets/leaves/leaves.tar) - Leaves dataset taken by Markus Weber. California Institute of Technology PhD student under Pietro Perona. 186 images of leaves against different backgrounds. Approximate scale normalisation. Jpeg format. Taken in and around Caltech. 896 x 592 jpg format.

[Dogs Data](https://github.com/openimages/dataset) - Open Images is a dataset of ~9 million URLs to images that have been annotated with labels spanning over 6000 categories. "Krasin I., Duerig T., Alldrin N., Veit A., Abu-El-Haija S., Belongie S., Cai D., Feng Z., Ferrari V., Gomes V., Gupta A., Narayanan D., Sun C., Chechik G, Murphy K. OpenImages: A public dataset for large-scale multi-label and multi-class image classification, 2016. Available from https://github.com/openimages".

[Birds Data](http://www.vision.caltech.edu/visipedia/CUB-200.html) Welinder P., Branson S., Mita T., Wah C., Schroff F., Belongie S., Perona, P. “Caltech-UCSD Birds 200”. California Institute of Technology. CNS-TR-2010-001. 2010.

[Butterflies Data](http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/) Josiah Wang, Katja Markert, and Mark Everingham, Learning Models for Object Recognition from Natural Language Descriptions, In Proceedings of the 20th British Machine Vision Conference (BMVC2009)

### Load Training and Validation Datasets
The raw dataset was first split into a training dataset with 80% of the images and a validation dataset with the remaining 20%. Then, 10% of this training dataset was allocated for prediction purposes. This test dataset was not part of the training or validating datasets

```
# split raw dataset into 80% for training and 20% for validating
train, validate = train_test_split(x, test_size = 0.2, random_state = 42)

# split training dataset into 90% for training and 10% for predicting
train, test = train_test_split(x, test_size = 0.1, random_state = 42)
```

The training dataset is kept in the 'data/train' folder and the validation dataset in the 'data/validate' folder.

Typical input image sizes are 224×224, 227×227, 256×256, and 299×299. VGG16, VGG19 accept 224×224 input images while Inception V3 and Xception require 299×299 pixel inputs.

The number of epochs can range from as low as 5 to as high as 40,000 depeneding upon the dataset in question. For this project, 20 epochs were used.

Batch size is defined as the number of samples propagating through the network. A typical batch size of 32 is a good default value.

```
# [Dataset]
# image dimensions for VGG16, VGG19 are 224, 224
# image dimensions for InceptionV3 and Xception are 299, 299
img_width, img_height = 224, 224

train_dir = 'data/train'
validate_dir = 'data/validate'
nb_epochs = 20
batch_size = 32
nb_classes = len(glob.glob(train_dir + '/*'))

# get number of images in training directory
nb_train_samples = 0
for r, dirs, files in os.walk(train_dir):
    for dr in dirs:
        nb_train_samples += len(glob.glob(os.path.join(r, dr + "/*")))
# get number of images in validation directory
nb_validate_samples = 0
for r, dirs, files in os.walk(validate_dir):
    for dr in dirs:
        nb_validate_samples += len(glob.glob(os.path.join(r, dr + "/*")))
```
### Preprocessing and augmenting the Datasets
Data preparation is almost always required when working with any data analysis, machine learning, neural networkk or deep learning models. It becomes even more important to augment data in the case of image recognition. Keras provides the [ImageDataGenerator](https://keras.io/preprocessing/image/) class that defines the configuration for image data preparation and augmentation. It defines the 
arguments of the ImageDataGenerator class as follows:

**rotation_range** is a value in degrees (0-180), a range within which to randomly rotate pictures

**width_shift and height_shift** are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally

**rescale** is a value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. factor.

**shear_range** is for randomly applying shearing transformations

**zoom_range** is for randomly zooming inside pictures
horizontal_flip is for randomly flipping half of the images horizontally --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures).

**fill_mode** is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.

The figure below displays the effect of applying rotation, width and height shifts, shear, zoom , fill mode and horizontal flip on a random butterfly image:

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Image%20Augmentation.PNG"></a>

The images used in the datasets underwent the following data augmentation when using VGG16 and VGG19 models. The InceptionV3 and Xception models have a built-in preprocessing function and do not need the rescaling feature.

```
# data pre-processing for training
train_datagen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip = True)

# data pre-processing for validation
validate_datagen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip = True)
```
The train_datagen.flow and validate_datagen.flow methods read the images found in the subfolders of the the 'data/train' and 'data/validate' folders and generate batches of augmented image data for use in the training and validating process. Batch size used is 32 for all datasets unless otherwise specified.

```
# generate and store training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size)

# generate and store validation data
validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size)
```
### Define the Model
Since ImageNet pre-trained models were used on the datasets in this project the model definition process was relatively simple. The top layer of the pre-trained model was removed and replaced with a new fully connected layer with a [Softmax](http://cs231n.github.io/linear-classify/#softmax) classifier. 

GlobalAveragePooling2D progressively reduces the spatial size and the amount of parameters and computation in the network as well as control overfitting. The [Dense](https://keras.io/layers/core/#dense) layer is the densely-connected Neural Network layer of size 1024 with the [Rectified Linear Unit](http://cs231n.github.io/neural-networks-1/) (relu) as the activator.

```
# set up transfer learning on pre-trained ImageNet VGG19 model - remove fully connected layer and replace
# with softmax for classifying the number of classes in the dataset
vgg19_model = VGG19(weights = 'imagenet', include_top = False)
x = vgg19_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation = 'softmax')(x)
model = Model(input = vgg19_model.input, output = predictions)
```

### Freeze the Layers of the Model
Prior to compiling the model, all the layers of the pre-trained model were frozen. Only the Dense layer of the new model needed to be trained.

```
# freeze all layers of the pre-trained InceptionV3 model
for layer in vgg19_model.layers:
    layer.trainable = False
```
### Learning Process Animation
* * *
<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/opt2.gif" align="left" height="280" width="360" >

</a> <a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/opt1.gif" align="right" height="360" width="480" ></a>

Animations that may help your intuitions about the learning process dynamics. Left: Contours of a loss surface and time evolution of different optimization algorithms. Notice the "overshooting" behavior of momentum-based methods, which make the optimization look like a ball rolling down the hill. Right: A visualization of a saddle point in the optimization landscape, where the curvature along different dimension has different signs (one dimension curves up and another down). Notice that SGD has a very hard time breaking symmetry and gets stuck on the top. Conversely, algorithms such as RMSprop will see very low gradients in the saddle direction. Due to the denominator term in the RMSprop update, this will increase the effective learning rate along this direction, helping RMSProp proceed. [Source: CS231n](http://cs231n.github.io/neural-networks-3/#ada). Images credit: [Alec Radford](https://twitter.com/alecrad).
* * *

### Compile the Model
RMSProp with its default values was the optimizer used on all the datasets during the transfer learning phase. Keras recommends leaving the parameters at their default values:

keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

lr: float >= 0. Learning rate.

rho: float >= 0.

epsilon: float >= 0. Fuzz factor.

decay: float >= 0. Learning rate decay over each update.

For the two class Noodles dataset, loss was defined as 'binary_crossentropy'. For all other datasets the loss was defined as 'categorical_crossentropy' which is a one-hot vector of the number of classes used in the classification. The metrics argument provided the classification accuracy.

```
# compile the new model using a RMSProp optimizer
model.compile(optimizer = 'rmsprop',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
```

### Fit (Train) the Model
Since the datasets were trained and validated on systems with different CPUs and RAM, the training and validating times were collected and reported. Number of epochs was set to 20 for all datasets unless otherwise specified. All training and validation images were used for training and validating, respectively.

```
# fit the model, log the results and the training time
now = datetime.datetime.now
t = now()
transfer_learning_history = model.fit_generator(
    train_generator,
    nb_epoch = nb_epochs,
    samples_per_epoch = nb_train_samples,
    validation_data = validate_generator,
    nb_val_samples = nb_validate_samples,
    class_weight='auto')
print('Training time: %s' % (now() - t))
```
```
Epoch 1/20
592/592 [==============================] - 439s - loss: 2.1416 - acc: 0.2686 - val_loss: 1.9227 - val_acc: 0.1716
Epoch 2/20
592/592 [==============================] - 433s - loss: 1.5334 - acc: 0.5034 - val_loss: 1.5641 - val_acc: 0.4083
Epoch 3/20
592/592 [==============================] - 435s - loss: 1.2112 - acc: 0.6351 - val_loss: 1.2631 - val_acc: 0.5562
Epoch 4/20
592/592 [==============================] - 437s - loss: 0.9724 - acc: 0.7196 - val_loss: 0.9148 - val_acc: 0.7456
Epoch 5/20
592/592 [==============================] - 434s - loss: 0.8360 - acc: 0.7568 - val_loss: 0.8543 - val_acc: 0.7337
.
.
.
Epoch 16/20
592/592 [==============================] - 435s - loss: 0.2825 - acc: 0.9206 - val_loss: 0.4937 - val_acc: 0.8225
Epoch 17/20
592/592 [==============================] - 481s - loss: 0.2692 - acc: 0.9274 - val_loss: 0.4280 - val_acc: 0.8521
Epoch 18/20
592/592 [==============================] - 434s - loss: 0.2565 - acc: 0.9189 - val_loss: 0.4483 - val_acc: 0.8698
Epoch 19/20
592/592 [==============================] - 433s - loss: 0.2495 - acc: 0.9172 - val_loss: 0.4029 - val_acc: 0.8521
Epoch 20/20
592/592 [==============================] - 435s - loss: 0.2510 - acc: 0.9155 - val_loss: 0.3521 - val_acc: 0.8935
Training time: 2:25:45.271227
```

### Evaluate the Model
```
# evaluate the performance the new model and report the results
score = model.evaluate_generator(validate_generator, nb_validate_samples/batch_size)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

Test Score: 0.336898863316
Test Accuracy: 0.90625
```
### Save the model
```
# save transfer learning model for offline prediction purposes
model.save('butterflies_vgg19_model_tl.h5')
```

### Plot the test results
The model.fit_generator function returns a history object that contains information on the training and validating accuracy and loss. This information was plotted to provide a graphical representation of the training versus validating test results. 
```
xfer_acc = transfer_learning_history.history['acc']
val_acc = transfer_learning_history.history['val_acc']
xfer_loss = transfer_learning_history.history['loss']
val_loss = transfer_learning_history.history['val_loss']
epochs = range(len(xfer_acc))

x = np.array(epochs)
y = np.array(xfer_acc)
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 500)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Transfer Learning - Training and Validation Accuracy')
plt.legend(loc = 'lower left', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1.05)

plt.figure()
x = np.array(epochs)
y = np.array(xfer_loss)
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_loss)
x1_smooth = np.linspace(x1.min(), x1.max(), 500)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Transfer Learning - Training and Validation Loss')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,max(y1))
plt.show()
```

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Sample%20TL%20Plot.PNG"></a>

### Predicting Unseen Images
The model.predict function generates output predictions for the input images which are retrieved from the dataset's test folder.

```
num_images = len(glob.glob("butterflies_test/*.jpg"))
predict_files = glob.glob("butterflies_test/*.jpg")

im = cv2.imread(predict_files[0])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (256, 256)).astype(np.float32)
im = np.expand_dims(im, axis = 0)/255

predictor, image_id = [], []
for i in predict_files:
    im = cv2.imread(i)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)
    outcome = [np.argmax(model.predict(im))]
    predictor.extend(list(outcome))
    image_id.extend([i.rsplit("\\")[-1]])
    
final = pd.DataFrame()
final["id"] = image_id
final["Butterfly"] = predictor

classes = train_generator.class_indices
classes = {value : key for key, value in classes.items()}

final["Butterfly"] = final["Butterfly"].apply(lambda x: classes[x])
final.head(num_images)
```
<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Prediction%20TL%20Results.PNG"></a>

### Save the Prediction Results
```
final.to_csv("csv/butterflies_with_pretrained_vgg19_model_tl_test.csv", index=False)
```


## Fine Tuning Phase
In the Fine Tuning phase some or none of the lower convolutional layers of the model were frozen depending upon the results from the Transfer Learning phase.

### Train Layers, Compile Model, Fit Model
```
# Step 1 - Set up fine tuning on pre-trained ImageNet vgg19 model - train all lower 94 layers
for layer in model.layers:
    layer.trainable = True
    
# Step 2 - Compile the revised model using SGD optimizer with a learing rate of 0.0001 and a momentum of 0.9
model.compile(optimizer = SGD(lr = 0.0001, momentum=0.9), 
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

# Step 3 - Fit the revised model, log the results and the training time
now = datetime.datetime.now
t = now()
fine_tuning_history = model.fit_generator(
    train_generator,
    nb_epoch = nb_epochs,
    samples_per_epoch = nb_train_samples,
    validation_data = validate_generator,
    nb_val_samples = nb_validate_samples,
    class_weight='auto')
print('Training time: %s' % (now() - t))
```
```
Epoch 1/20
592/592 [==============================] - 1111s - loss: 0.1320 - acc: 0.9578 - val_loss: 0.2041 - val_acc: 0.9349
Epoch 2/20
592/592 [==============================] - 1105s - loss: 0.0874 - acc: 0.9713 - val_loss: 0.1727 - val_acc: 0.9172
Epoch 3/20
592/592 [==============================] - 1061s - loss: 0.0436 - acc: 0.9899 - val_loss: 0.1797 - val_acc: 0.9172
Epoch 4/20
592/592 [==============================] - 1107s - loss: 0.0427 - acc: 0.9882 - val_loss: 0.1557 - val_acc: 0.9527
Epoch 5/20
592/592 [==============================] - 1062s - loss: 0.0317 - acc: 0.9949 - val_loss: 0.0987 - val_acc: 0.9586
.
.
.
Epoch 16/20
592/592 [==============================] - 1167s - loss: 0.0065 - acc: 1.0000 - val_loss: 0.0654 - val_acc: 0.9645
Epoch 17/20
592/592 [==============================] - 1160s - loss: 0.0108 - acc: 0.9966 - val_loss: 0.0390 - val_acc: 0.9822
Epoch 18/20
592/592 [==============================] - 1107s - loss: 0.0034 - acc: 1.0000 - val_loss: 0.0606 - val_acc: 0.9704
Epoch 19/20
592/592 [==============================] - 1112s - loss: 0.0037 - acc: 1.0000 - val_loss: 0.0924 - val_acc: 0.9527
Epoch 20/20
592/592 [==============================] - 1155s - loss: 0.0071 - acc: 0.9983 - val_loss: 0.0674 - val_acc: 0.9704
Training time: 6:14:59.254681
```
### Evaluate Model
```
# evaluate the performance the new model and report the results
score = model.evaluate_generator(validate_generator, nb_validate_samples/batch_size)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
```
```
Test Score: 0.0136573100463
Test Accuracy: 1.0
```

### Save the Fine Tuning Model
```
# save fine-tuning model for offline prediction purposes
model.save('butterflies_vgg19_model_ft.h5')
```
### Plot the Test Results
```
ft_acc = fine_tuning_history.history['acc']
val_acc = fine_tuning_history.history['val_acc']
ft_loss = fine_tuning_history.history['loss']
val_loss = fine_tuning_history.history['val_loss']
epochs = range(len(ft_acc))

x = np.array(epochs)
y = np.array(xfer_acc)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Fine-Tuning - Training and Validation Accuracy')
plt.legend(loc = 'lower left', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0,1.02)

plt.figure()
x = np.array(epochs)
y = np.array(xfer_loss)
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = spline(x, y, x_smooth)
plt.plot(x_smooth, y_smooth, 'r-', label = 'Training')

x1 = np.array(epochs)
y1 = np.array(val_loss)
x1_smooth = np.linspace(x1.min(), x1.max(), 300)
y1_smooth = spline(x1, y1, x1_smooth)

plt.plot(x1_smooth, y1_smooth, 'g-', label = 'Validation')
plt.title('Fine-Tuning - Training and Validation Loss')
plt.legend(loc = 'upper right', fontsize = 9)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0,2)
plt.show()
```
<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Prediction%20FT%20Plot.PNG"></a>
