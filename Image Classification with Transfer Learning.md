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
Training Deep Learning networks require tremendous processing power to handle multiple matrix multiplications. GPUs are  ideal for performing these operations.  Facebook recently reported that its scientits were able to train nearly 1.3 million images in under one hour using [256 Tesla P100 GPUs](https://news.developer.nvidia.com/facebook-trains-imagenet-in-1-hour/) that previously took days on a single system. For the small datasets used in this project having a Titan GTX 1080 GPU would have been able to train 8X faster than an i7 Intel CPU running at 3.5GHz.

Howevere, no GPU was available for this project so the datasets were trained on two systems with Core i7 CPUs and on two systems with Core i5 CPUs.

## Training, Validating and Testing Datasets Process
The following steps were used in to train, validate and test the datasets in this project:

1. Quick check Keras ImageNet pre-trained model's capability
2. Load Data
3. Define Model
4. Compile Model
5. Fit Model
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

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Birds%20-%20Initial%20Checkout.PNG"></a>

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

## Code Details

The following code applies to all the Keras pre-trained models except as noted otherwise.

### Import Keras Libraries
Keras supports VGG16, VGG19, ResNet50, InceptionV3 and Xception models that have been pre-trained on ImageNet. 

The Dense layer is densely-connected Neural Network layer and the GlobalAveragePooling2D layer provides an average pooling for spatial data. 

Keras has support for several optimizers that include SGD (Stochastic Gradient Descent), RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam and TFOptimizer (for Tensorflow). 

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
