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

Figure 2:[VGG16 Architecture](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5)

Other applications that have benefited from Transfer Learning include [object detection](http://arxiv.org/abs/1311.2524), [zero-shot learning](http://arxiv.org/abs/1312.5650), [image captioning](http://googleresearch.blogspot.com/2014/11/a-picture-is-worth-thousand-coherent.html) and [video analysis](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6751448).

There are several libraries in use today that support deep learning. Each has its own pros and cons. Some of the most commonly used ones for the Python Language are as follows:
•	Theano - A Python library that allows one to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays.
•	Tensorflow – Uses data flow graphs for numerical computation.
•	Lasagne – A lightweight library to build and train neural networks in Theano.
•	Caffe -  A deep learning framework made with expression, speed, and modularity in mind.
•	Caffe2 – A new lightweight, modular, and scalable deep learning framework.
•	Microsoft Cognitive Toolkit – Uses directed graphs for describing neural networks. Previously known as CNTK.
•	PyTorch – Builds neural networks using a technique called reverse-mode auto-differentiation.
•	Keras - Keras is a high-level neural networks API, written in Python and capable of running on top of Tensorflow, Microsoft Cognitive Toolkit, or Theano. It was developed with a focus on enabling fast experimentation.
The Keras Deep Learning library for Python currently supports five models that have been pre-trained on ImageNet:
•	VGG16 – A  convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. 
•	VGG19 – It is essentially the VGG16 model with three additional weight layers.
•	Inception V3 - 
•	
•	ResNet50 – A residual learning framework that eases the training of deep networks. It reformulates the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.
•	Xception
This study will use the Keras library to explore ImageNet’s pre-trained VGG16, VGG19, Inception V3 and Xception models to perform image classification on a variety of small datasets with different domains.

