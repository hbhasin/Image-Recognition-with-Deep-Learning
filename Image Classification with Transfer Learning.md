# Image Classification with Transfer Learning

![](https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/splash.JPG)

<a href="url"><img src="https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/opt1.gif" align="left" height="300" width="300" ></a>

![](https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/opt2.gif)

Deep Learning is an emerging field of research and Transfer Learning is one of its benefits. In image classification, for example, Transfer Learning makes use of features learned from one domain and used on another through feature extraction and fine-tuning. Convolutional Neural Network (also known as ConvNet) models trained on the [ImageNet’s](http://www.image-net.org/) 1.2 million images with 1000 categories have been successfully used on other similar or dissimilar datasets, large or small, with great success. In particular, given the fact that data acquisition is expensive, small datasets can benefit from these pre-trained networks because the lower layers of these pre-trained networks already contain many generic features such as edge and color blob detectors and only the higher layers need to be trained on the new datasets.

According to [Pan, et al](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf), “research on transfer learning has attracted more and more attention since 1995 in different names: learning to learn, life-long learning, knowledge transfer, inductive transfer, multi-task learning, knowledge consolidation, context sensitive learning, knowledge-based inductive bias, meta learning, and incremental/cumulative learning”. They describe the difference between the learning processes of traditional and transfer learning techniques in the figure below.


![](https://github.com/hbhasin/Image-Recognition-with-Deep-Learning/blob/master/images/Figure%201.png)
<p align="center">
Figure 1: Different learning processes between traditional machine learning and Transfer Learning - [Pan, et al](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
</p>
