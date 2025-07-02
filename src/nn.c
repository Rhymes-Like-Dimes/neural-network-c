#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "nn.h"
#include "utils.h" 

/**
 * @brief Initializes a neural network layer.
 *
 * Allocates and initializes a Layer structure for use in a feed forward neural network.
 * If the layer has a previous layer (i.e., prevSize > 0), it allocates memory for
 * weights, biases, output values, activations, error deltas and weight & bias gradients.
 * Weights and biases are initialized using the `rand_init()` function.
 *
 * For input layers (prevSize == 0), only activation memory is allocated. Inputs are
 * later assigned to the input layers' activation. 
 *
 * @param layerID   An integer ID to label the layer (e.g., 0 for input, 1 for hidden).
 * @param size      The number of neurons in this layer.
 * @param prevSize  The number of neurons in the previous layer. Set to 0 for input layer.
 *
 * @return A pointer to the newly allocated Layer structure.
 */
Layer* init_layer(int layerID, int size, int prevSize) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));

    //Assign these for every layer
    layer->layerID = layerID;
    layer->size = size;
    layer->prevSize = prevSize;

    //First layer does not have weights, biases, outputs, deltas, dweights or dbiases
    if(prevSize > 0) {
        layer->weights = (float*)malloc(sizeof(float) * size * prevSize);
        layer->biases = (float*)malloc(sizeof(float) * size);
        layer->output = (float*)malloc(sizeof(float) * size);
        layer->activation = (float*)malloc(sizeof(float) * size);

        layer->delta = (float*)calloc(size, sizeof(float));
        layer->dweights = (float*)malloc(sizeof(float) * size * prevSize);
        layer->dbiases = (float*)malloc(sizeof(float) * size);

        //Initialize weights as random numbers from normal distribution
        for(int i=0; i<size; i++) {
            layer->biases[i] = 0.0f;
            for(int j=0; j<prevSize; j++) {
                layer->weights[i * prevSize + j] = rand_init_XNTPY(prevSize);
            }
        }
    //First layer case. Input is set as the input layers' activation
    } else {
        layer->weights = NULL;
        layer->biases = NULL;
        layer->output = NULL;
        layer->activation = (float*)malloc(sizeof(float) * size);

        layer->delta = NULL;
        layer->dweights = NULL;
        layer->dbiases = NULL;
    }
    return layer;
}

/**
 * @brief Initializes a feedforward neural network.
 *
 * Allocates and sets up a NeuralNetwork structure with the specified number of layers,
 * layer sizes, and learning rate. Each layer is initialized using `init_layer()`, with
 * the input layer (index 0) having no previous layer and subsequent layers connected
 * to the output size of the previous layer.
 *
 * @param numLayers     The total number of layers in the network (including input and output).
 * @param layerSizes    An array of integers specifying the number of neurons in each layer. Must be of length `numLayers`.
 * @param learningRate  The learning rate to be used during training.
 *
 * @return A pointer to the initialized NeuralNetwork structure.
 */
NeuralNetwork* init_nn(int numLayers, int* layerSizes, float learningRate) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));

    nn->numLayers = numLayers;
    nn->layers = (Layer**)malloc(sizeof(Layer*) * numLayers);
    nn->learningRate = learningRate;

    nn->layers[0] = init_layer(0, layerSizes[0], 0);
    for(int i=1; i<numLayers; i++) {
        nn->layers[i] = init_layer(i, layerSizes[i], layerSizes[i-1]);
    }
    return nn;
}

/**
 * @brief Applies exponential decay to the learning rate based on the current epoch.
 *
 * Updates the neural network's learning rate according to the formula:
 * learningRate = base_lr * (decay_rate ^ epoch).
 *
 * @param nn         Pointer to the NeuralNetwork whose learning rate will be updated.
 * @param base_lr    The initial base learning rate.
 * @param decay_rate The decay factor applied per epoch (typically < 1).
 * @param epoch      The current training epoch (starting from 0).
 */
void decay_lr(NeuralNetwork* nn, float base_lr, float decay_rate, int epoch) {
     nn->learningRate = base_lr * powf(decay_rate, epoch);
}

/**
 * @brief Updates all parameters in a neural network using their gradient
 * 
 * Updates all weights and biases for all layers in a neural network using the gradient 
 * caluclated during "backpropagation()". Note delta's must be reset to zero after each
 * backpropagation() call since the weights are accumulated using a += statement rather
 * than an assigment operation. This function is called at the end of backpropagation().
 * 
 * @param nn    Pointer to the neural network to be updated
 */
void updateParameters(NeuralNetwork* nn) {
    for(int i=1; i<nn->numLayers; i++) {
        Layer* curr = nn->layers[i];

        #pragma omp parallel for
        for(int j=0; j<curr->size; j++) {
            curr->biases[j] -= nn->learningRate * curr->dbiases[j];
            curr->delta[j] = 0.0f;

            for(int k=0; k<curr->prevSize; k++) {
                curr->weights[j * curr->prevSize + k] -= nn->learningRate * curr->dweights[j * curr->prevSize + k];
            }
        }
    }
}

/**
 * @brief Performs the feedforward pass on a neural network given an input.
 * 
 * Computes the outputs and activations of each layer from the input layer to the output layer
 * using ReLU activation for hidden layers and softmax activation for the output layer.
 * 
 * The input data is first assigned to the input layer's activation vector, allowing the same 
 * feedforward logic to be used for all layers. For each hidden layer, each neuron's 
 * output is initialized as its bias and accumulated with the weighted sum of activations from the 
 * previous layer. The ReLU activation function is then applied to compute the layer's 
 * activation vector. In the final layer, softmax is applied to the output vector to produce a 
 * normalized probability distribution over output classes.
 * 
 * @param nn    The pointer to the neural network struct to apply the feedforward stage.
 * @param input Array of input training data equal in size to the input layer. 
 */
void feedforward_XNTPY(NeuralNetwork* nn, float* input) {
    //Assign traning input as input layers' activation
    for(int i=0; i<nn->layers[0]->size; i++) {
        nn->layers[0]->activation[i] = input[i];
    }

    //For each layer i after input layer, except the output layer
    for(int i=1; i<nn->numLayers; i++) {
        Layer* curr = nn->layers[i];
        Layer* prev = nn->layers[i-1];

        //For each neuron j in current layer i (parallel)
        #pragma omp parallel for
        for(int j=0; j<curr->size; j++) {

            //Initialize the output of layer i as the bias
            curr->output[j] = curr->biases[j];

            //For each neuron k in layer i-1
            for(int k=0; k<curr->prevSize; k++) {
                curr->output[j] += curr->weights[j * curr->prevSize + k] * prev->activation[k];
            }

            //Apply activation for each neuron j in layer i
            //Hidden uses reLU, output uses softmax
            if(curr->layerID != nn->numLayers - 1) {
                curr->activation[j] = reLU(curr->output[j]);
            }
        }
        //Apply softmax to output layer
        if(curr->layerID == nn->numLayers - 1) {
            softmax(curr);
        }
    }
}

/**
 * @brief Calculates the gradient of the loss function with respect to the network 
 * parameters and updates the weights and biases to minimize the loss.
 * 
 * This function performs the backpropagation algorithm to compute the gradient of the 
 * loss function with respect to the weights and biases of each layer. The delta (error
 * term) is calculated for each layer, starting from the output layer and propagating 
 * backward to the input layer. These deltas are then used to compute the gradients for
 * each weight and bias.
 * 
 * Using softmax and reLU allows for a simplification of the gradient calculation formula.
 * The output layer uses softmax, which results in an error term of (predicted - target), 
 * and the hidden layer uses reLU which results in an error term equal to the summation of
 * the next layers error vector, multiplied by the transpose of the weight matrix, then,
 * using the dot product, multiplying by the derivative of reLU (which is 0 or 1). 
 * 
 * After all gradients are computed, the function calls "updateParameters()", which updates
 * the network's weights and biases using gradient descent with the input learning rate.
 * 
 * OpenMP is used to parallelize gradient computation for improved performance.
 * 
 * @param nn     Pointer to the neural network structure on which to perform backpropagation.
 * @param target Poitner to the one hot encoded label of the actual answer of the example.
 */
void backpropagation_XNTPY(NeuralNetwork* nn, float* target) {

    //For each layer starting from the output layer, except the input layer
    for(int i=(nn->numLayers-1); i>0; i--) {
        Layer* curr = nn->layers[i];
        Layer* prev = nn->layers[i-1];

        //Calculate output layer deltas
        if(i == nn->numLayers - 1) {
            //For each neuron j in output layer (parallel)
            #pragma omp parallel for
            for(int j=0; j<curr->size; j++) {
                curr->delta[j] = (curr->activation[j] - target[j]);

                //Calculate weight & bias gradients
                for(int k=0; k<prev->size; k++) {
                    curr->dweights[j * prev->size + k] = curr->delta[j] * prev->activation[k];
                }
                curr->dbiases[j] = curr->delta[j];
            }
        } else {
            Layer* next = nn->layers[i+1];

            //Compute delta with optimized cache locality. Still needs to be multiplied by dreLU
            for(int k=0; k<next->size; k++) {
                for(int j=0; j<curr->size; j++) {
                    curr->delta[j] += next->delta[k] * next->weights[k * curr->size + j];
                }
            }

            //Finish delta calculation & compute gradients
            #pragma omp parallel for
            for(int j=0; j<curr->size; j++) {
                curr->delta[j] *= dreLU(curr->output[j]); //Multiply delta by dreLU

                //Calculate gradients
                for(int k=0; k<prev->size; k++) {
                    curr->dweights[j * prev->size + k] = curr->delta[j] * prev->activation[k];
                }
                curr->dbiases[j] = curr->delta[j];
            }
        }
    }
    updateParameters(nn);
}

/**
 * @brief Performs the feedforward pass on a neural network given an input.
 * 
 * Computes the output and subsequent activations of each layer from input layer output layer.
 * The input data is first assigned to the input layers' activation vector, which enables the 
 * reuse of the same formula and requires no special handling for the input layer. Then, for
 * each layer, the output for each neuron is parallelly initialized as the bias and the weighted 
 * sum of the previous layer activations is accumulated. The activation vector is then computed 
 * by applying the sigmoid function to the output. 
 * 
 * @param nn    The pointer to the neural network struct to apply the feedforward stage.
 * @param input Array of input training data equal in size to the input layer. 
 */
void feedforward_MSE(NeuralNetwork* nn, float* input) {

    //Assign traning input as input layers' activation
    for(int i=0; i<nn->layers[0]->size; i++) {
        nn->layers[0]->activation[i] = input[i];
    }

    //For each layer i after input layer
    for(int i=1; i<nn->numLayers; i++) {
        Layer* curr = nn->layers[i];
        Layer* prev = nn->layers[i-1];

        //For each neuron j in layer i (parallel)
        #pragma omp parallel for
        for(int j=0; j<curr->size; j++) {

            //Initialize the output of layer i as the bias
            curr->output[j] = curr->biases[j];

            //For each neuron k in layer i-1
            for(int k=0; k<curr->prevSize; k++) {
                curr->output[j] += curr->weights[j * curr->prevSize + k] * prev->activation[k];
            }

            //Apply activation for each neuron j in layer i
            curr->activation[j] = sigmoid(curr->output[j]);
        }
    }
}

/**
 * @brief Performs the backpropagation algorithm by computing the gradient of
 * the loss function with respect to the parameters (weights and biases) of the
 * neural network and uses it to update the paramters in order to reduce the loss.
 * 
 * This function peforms the backpropagation algorithm to compute the gradient of the 
 * loss function with respect to the weights and biases of each layer. The delta (error
 * term) is calculated for each layer, then backpropagated to the previous layer and
 * and used to calculate the gradient, from output layer to input layer.
 * 
 * The output layer's deltas are calculated using the partial derivative of the loss function
 * (mean squared error) and the activation (sigmoid). Hidden layers compute their deltas based
 * on the weighted deltas of the layer ahead, followed by gradient computation. The output
 * layer error & gradient calculation are handled as a special case.
 * 
 * Gradient computation is followed by a call to "updateParameters()", which updates the
 * weights and biases of the neural network using a fraction of the gradient called the
 * learning rate.
 * 
 * Open MP is used to parallelize gradient compututaion to improve time complexity.
 * 
 * @param nn    Pointer to the nerual netwrok structure to perform backpropagation on
 */
void backpropagation_MSE(NeuralNetwork* nn, float* target) {

    //For each layer starting from the output layer, except the input layer
    for(int i=(nn->numLayers-1); i>0; i--) {
        Layer* curr = nn->layers[i];
        Layer* prev = nn->layers[i-1];

        //Calculate output layer deltas
        if(i == nn->numLayers - 1) {
            //For each neuron j in output layer (parallel)
            #pragma omp parallel for
            for(int j=0; j<curr->size; j++) {
                curr->delta[j] = (curr->activation[j] - target[j]) * dsigmoid_a(curr->activation[j]);

                //Calculate weight & bias gradients
                for(int k=0; k<prev->size; k++) {
                    curr->dweights[j * prev->size + k] = curr->delta[j] * prev->activation[k];
                }
                curr->dbiases[j] = curr->delta[j];
            }
        } else {
            Layer* next = nn->layers[i+1];

            //Compute delta with optimized cache locality. Still needs to be multiplied by dsigmoid
            for(int k=0; k<next->size; k++) {
                for(int j=0; j<curr->size; j++) {
                    curr->delta[j] += next->delta[k] * next->weights[k * curr->size + j];
                }
            }

            //Finish delta calculation & compute gradients
            #pragma omp parallel for
            for(int j=0; j<curr->size; j++) {
                curr->delta[j] *= dsigmoid_a(curr->activation[j]); //Multiply delta by dsigmoid

                //Calculate gradients
                for(int k=0; k<prev->size; k++) {
                    curr->dweights[j * prev->size + k] = curr->delta[j] * prev->activation[k];
                }
                curr->dbiases[j] = curr->delta[j];
            }
        }
    }
    updateParameters(nn);
}
