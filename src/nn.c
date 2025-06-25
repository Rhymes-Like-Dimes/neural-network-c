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

        //Initialize weights and biases as random numbers between -1 and 1
        for(int i=0; i<size; i++) {
            layer->biases[i] = rand_init();
            for(int j=0; j<prevSize; j++) {
                layer->weights[i * prevSize + j] = rand_init();
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
 * @brief Performs the feedforward pass on a neural network given an input.
 * 
 * Computes the output and subsequent activations of each layer from input layer output layer.
 * The input data is first assigned to the input layers' activation vector, which enables the 
 * reuse of the same formula and requires no special handling for the input layer. Then, for
 * each layer, the output for each neuron is parallelly initialized as the bias and the weighted sum of the previous
 * layer activations is accumulated. The activation vector is then computed by applying the sigmoid
 * function to the output. 
 * 
 * @param nn    The pointer to the neural network struct to apply the feedforward stage.
 * @param input Array of input training data equal in size to the input layer. 
 */
void feedforward(NeuralNetwork* nn, float* input) {

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
    //Note: No need to reset dweights/dbiases as they are overwritten in back propagation
}

void backpropagation(NeuralNetwork* nn, float* target) {

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
