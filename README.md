# STL10-ResNET101

## Intuition behind Residual blocks:
### If the identity mapping is optimal, We can easily push the residuals to zero (F(x) = 0) than to fit an identity mapping (x, input=output) by a stack of non-linear layers. In simple language it is very easy to come up with a solution like F(x) =0 rather than F(x)=x using stack of non-linear cnn layers as function (Think about it). So, this function F(x) is what the authors called Residual function.


![This is an image](https://miro.medium.com/max/856/1*WVs9ywVLLKjSUBZ_mnfFrw.png)



# How does ResNet work?
### Here, we have something called Residual blocks. Many Residual blocks are stacked together to form a ResNet. We have “skipped connections” which are the major part of ResNet. The following image was provided by the authors in the original paper which denotes how a residual network works. The idea is to connect the input of a layer directly to the output of a layer after skipping a few connections. We can see here, x is the input to the layer which we are directly using to connect to a layer after skipping the identity connections and if we think the output from identity connection to be F(x). Then we can say the output will be F(x) + x.

![This is an image](https://miro.medium.com/max/1140/1*D0F3UitQ2l5Q0Ak-tjEdJg.png)

### One problem that may happen is regarding the dimensions. Sometimes the dimensions of x and F(x) may vary and this needs to be solved. Two approaches can be followed in such situations. One involves padding the input x with weights such as it now brought equal to that of the value coming out. The second way includes using a convolutional layer from x  to addition to F(x).This way we can bring down the weights same dimensions of that coming out. When following the first way, the equation turns to be F(x) + w1.x. Here w1 is the additional parameters added so that we can bring up the dimensions to that of output coming from the activation function. The skip connections in ResNet solve the problem of vanishing gradient in deep neural networks by allowing this alternate shortcut path for the gradient to flow through. It also helps the connections by allowing the model to learn the identity functions which ensures that the higher layer will perform at least as good as the lower layer, and not worse. The complete idea is to make F(x) = 0. So that at the end we have Y = X as result. This means that the value coming out from the activation function of the identity blocks is the same as the input from which we skipped the connections.

# ResNET-101
### ResNet-101 is a convolutional neural network that is 101 layers deep. 

![This is an image](https://www.researchgate.net/profile/Jiayao-Chen-5/publication/348078198/figure/fig4/AS:974997896060929@1609469021477/The-structure-of-the-ResNet-101-based-deep-feature-extractor.png)

# STL-10 Dataset

## Dataset: 
This dataset used for training [Stanford AI](https://cs.stanford.edu/~acoates/stl10/)

### The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. It is inspired by the CIFAR-10 dataset but with some modifications. In particular, each class has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples is provided to learn image models prior to supervised training. The primary challenge is to make use of the unlabeled data (which comes from a similar but different distribution from the labeled data) to build a useful prior. We also expect that the higher resolution of this dataset (96x96) will make it a challenging benchmark for developing more scalable unsupervised learning methods.

## Overview
### 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
### Images are 96x96 pixels, color.
### 500 training images (10 pre-defined folds), 800 test images per class.
### 100000 unlabeled images for unsupervised learning. These examples are extracted from a similar but broader distribution of images. For instance, it contains other types of animals (bears, rabbits, etc.) and vehicles (trains, buses, etc.) in addition to the ones in the labeled set.

## Contribution guidelines for this project
*[1](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8)
*[2](https://github.com/fchollet/deep-learning-models#readme)
