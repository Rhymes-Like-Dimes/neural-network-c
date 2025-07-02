# Neural Network - C Implementation
## Introduction
This project is a neural network written from scratch in C to recognize handwritten digits, trained on the MNIST dataset. The MNIST dataset is a collection of 60 000, 28x28 images of handwritten digits intended for training a neural network, with an additional 10 000 images intended to test a model on data it's never seen before. Each pixel is as assigned an 8-bit 'brightness' value between 0 - 255, which can then be passed into the model. The intention of the this project was to learn the internal workings and structure of a feedforward neural network using a 'learn by doing' method. The choice to use no other librariers other than the standard C libraries was explicit such that it forces the development of fully custom network functions and structure (whether or not OpenMP is an external library is debatable - It is an API for interacting with the operating system). This read me serves as a (WIP) detailed design overview for this project. 

## Project Structure
This project is comprised of several different files, each serving a specific purpose. Note that most C files have a corresponding header file, where relevant functions and structs are defined. The main training loop and subsequent test loop is contained in 'main.c'. This also includes output file handling for recording loss and accuracy data per epoch, which will be discuessed later. The core functionality of the neural network is implemented in 'nn.c', which contains functions for the feedforward pass and the backpropagation pass, among others. 'nn.h' defines the necessary structs for creating the network, and the function definitions for 'nn.c'. 'utils.c' and its corresponding 'utils.h' is a collection of helper functions including functions for priting matrices, computing various activation functions and their derivatives (sigmoid, reLU, softmax) and various other necessary functions. 'mnist.c' and 'mnist.h' define structures and functions for loading the MNIST image data into the neural network, and also provides supporting functions such as a image shuffling function to prevent the same order of images from being passed through each epoch.

## Network Structure
This project has been created such that the number of hidden layers and size of each individual hidden layer can be defined by the user, trained and tested. Other parameters such as the learning rate and decay rate can also be adjusted by the user, as will be discussed later. The most basic unit of the neural network is the layer. For this, a struct, Layer, in 'nn.h' was created containing the layer information, and all the vectors required to perform the forward pass and backpropagation pass. This inlcudes:
- Layer ID
- Number of neurons in the layer (size)
- Number of neurons in the previous layer (prevSize)


The pre-activation value is given by $z = \sum_{i=1}^{n} w_i x_i + b$.

