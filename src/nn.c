#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "utils.h"


Layer* init_layer(int layerID, int size, int prevSize) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));

    layer->layerID = layerID;
    layer->size = size;
    layer->prevSize = prevSize;

    if(prevSize > 0) {
        layer->weights = (float*)malloc(sizeof(float) * size * prevSize);
        layer->biases = (float*)malloc(sizeof(float) * size);
        layer->output = (float*)malloc(sizeof(float) * size);
        layer->activation = (float*)malloc(sizeof(float) * size);
        layer->delta = (float*)calloc(size, sizeof(float));

        for(int i=0; i<size; i++) {
            layer->biases[i] = rand_init();
            for(int j=0; j<prevSize; j++) {
                layer->weights[i * prevSize + j] = rand_init();
            }
        }
    } else {
        layer->weights = NULL;
        layer->biases = NULL;
        layer->output = NULL;
        layer->activation = (float*)malloc(sizeof(float) * size);
        layer->delta = NULL;
    }
    return layer;
}

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

void feedforward(NeuralNetwork* nn, float* input) {

    //Assign traning input as input layers' activation
    for(int i=0; i<nn->layers[0]->size; i++) {
        nn->layers[0]->activation[i] = input[i];
    }

    //For each layer i after input layer
    for(int i=1; i<nn->numLayers; i++) {
        Layer* curr = nn->layers[i];
        Layer* prev = nn->layers[i-1];

        //For each neuron j in layer i
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

