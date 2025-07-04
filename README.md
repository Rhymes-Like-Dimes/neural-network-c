# Neural Network - C Implementation
This README is incomplete as is a work in progress.

## Introduction
This project is a neural network written from scratch in C to recognize handwritten digits, trained on the MNIST dataset. The MNIST dataset is a collection of 60 000, 28x28 images of handwritten digits intended for training a neural network, with an additional 10 000 images intended to test a model on data it's never seen before. Each pixel is assigned an 8-bit 'brightness' value between 0 - 255, which can then be passed into the model as 28*28 = 784 neuron inputs, which feedfoward through the network to produce 10 outputs. Neuron 0 of the output neurons corresponds to a network predicted digit of 0, neuron 1 of the output neurons corresponds to a network predicted digit of 1 and so on. In the ideal case, if the input image was a 5, the output neuron corresponding to a prediction of 5 would have a so called 'activation' of 1.00, while all other output neurons would have an 'activation' of 0.00. 

The intention of the this project was to learn the internal workings and structure of a feedforward neural network using a 'learn by doing' method. The choice to use no other librariers other than the standard C libraries was explicit such that it forces the development of fully custom network functions and structure (whether or not OpenMP is an external library is debatable - It is an API for interacting with the operating system. The neural network would be fully functional with no additional required alterations if all the OpenMP lines were removed). This read me serves as a detailed design overview for this project. 

## Project Structure
This project is comprised of several different files, each serving a specific purpose. Note that most C files have a corresponding header file, where relevant functions and structs are defined. The main training loop and subsequent test loop is contained in 'main.c'. This also includes output file handling for recording loss and accuracy data per epoch, which will be discuessed later. The core functionality of the neural network is implemented in 'nn.c', which contains functions for the feedforward pass and the backpropagation pass, among others. 'nn.h' defines the necessary structs for creating the network, and the function definitions for 'nn.c'. 'utils.c' and its corresponding 'utils.h' is a collection of helper functions including functions for priting matrices, computing activation functions and their derivatives (sigmoid, reLU, softmax) and various other supporting functions. 'mnist.c' and 'mnist.h' define structures and functions for loading the MNIST image data into the neural network, and also provides supporting functions such as a image shuffling function to prevent the same order of images from being passed through each epoch.

## Network Structure
This neural network project has two implementations. One using mean squared error (MSE) and sigmoid activation, and the other using categorical cross entropy (XNTPY) loss with reLU & softmax activation. As the implementation using cross entropy loss generally performs better, this is the current architecture implementation. Note that it is relatively simple to switch between the two methods, only involving swapping some function calls from the XNTPY version to the MSE version or vice versa, and changing a line of the 'init_layer' function (TBD).

### Layers
This project has been created such that the number of hidden layers and size of each individual hidden layer can be defined by the user. Other parameters such as the learning rate and decay rate can also be adjusted by the user, as will be discussed later. The most basic unit of the neural network is the layer. For this, a struct, Layer, in 'nn.h' was created with the layer information and vectors required to perform the forward pass and backpropagation pass. This inlcudes:
- Layer ID
- Number of neurons in the layer (size)
- Number of neurons in the previous layer (prevSize)
- Weights [size x prevSize]
- Biases [size x 1]
- Ouput [size x 1]
- Activation [size x 1]
- Delta [size x 1]
- Partial derivative of loss with respect to weights of this layer [size x prevSize]
- Partial derivative of loss with respect to biases of this layer [size x 1]

The 'init_layer' function allocates memory for these layer parameters usually using malloc and returns a pointer to the layer. There are four things of note with the layer initialization: 
1. **Input Layer:** The input layer is detected by checking if the previous layer size is 0. Hence the input layer must always have a previous layer size of 0. The input layer has all layer parameters except for layer ID, size, prevSize and activation set to NULL. Although technically no activation is applied to the input layer, the input data is assigned to the activation array of the input layer as this allows for a simplification of the code during forward propagation. Since the general equation for the pre-activation neuron output is: 

$$
z_j^{(l)} = \sum_{i=1}^{n^{(l-1)}} w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}
$$

    Instead of assigned the input image data to the input layer's output vector, we assign it to the activation thus allowing us to use the same equation to calculate the output of any layer, rather than creating a special case in order to handle the input layer.

2. **Parameter Initialization:** The MSE and XNTPY versions require separate weight initializations. While the biases for both cases can be initialized as zero, the weights are initalized as a random number on [-1, 1] for MSE, and a random number on [0, 1] from a gaussian distribution for XNTPY.

3. **Memory Allocation:** Note that for most arrays, malloc is used to allocate memory. However, calloc is neccessarily used for the delta array, as it needs to be zero'd out before being accumulated (with  +=) during backpropagation to prevent faulty starting data. All other arrays are use assignment operations (=) to have values assigned to the array.

4. **Cache Locality:** As neural networks and training are computationally intensive tasks, care has been taken to optimize this model a reasonable amount. Namely, instead of using 2D arrays, the 2D arrays are 'flattened' to 1D and indexed using the equivalent: array[i][j] = array[i * cols + j]. Storing a 2D structure as a flattened 1D array improves cache performance due to better spatial locality. In a traditional 2D array, accessing elements row by row may result in cache misses when transitioning between rows, especially if the memory layout isn't contiguous or predictable. With a 1D array, elements are laid out in a single, continuous block of memory. This means that when accessing elements sequentially (even across "rows"), the CPU is more likely to find the next value already loaded in the cache. As a result, cache hits increase, and memory access becomes faster.

### Network
Using the Layer structure, we can create a neuralNetwork structure. The neuralNetwork structure contains the following parameters:
- Number of layers in the network (numLayers - not to be confused with the constant #define NUM_LAYERS which is handed to init_nn as the paramter numLayers)
- An array of pointers to each layer (layers)
- Learning rate (To be discussed later) 

To initialize a neuralNetwork, we call 'init_nn' and pass it the number of layers (numLayers), an array containing the layer sizes (layerSizes) and the learning rate. By modifying the layerSizes array and the defined NUM_LAYERS, you can change the number of layers in the network and the size of each layer. Ensure that if additional layers are inserted in the layerSizes array, NUM_LAYERS is update to match the length of the array (i.e the number of layers). The 'init_nn' function allocates memory for a neuralNetwork structure and fills the array of layer pointers by looping from 1 to numLayers, calling 'init_layer' with increasing layer ID's. Note that the for loop begins at one as the input layer is initalized before the loop as a special case (since it requires an previous size of 0). Lastly, 'init_nn' returns a pointer to the neuralNetwork structure. 

