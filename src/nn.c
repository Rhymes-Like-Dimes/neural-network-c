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
    layer->output = (float*)calloc(size, sizeof(float));

    if(prevSize > 0) {
        layer->weights = (float**)malloc(sizeof(float*) * size);
        layer->biases = (float*)malloc(sizeof(float) * size);
        layer->activation = (float*)calloc(size, sizeof(float));
        layer->delta = (float*)calloc(size, sizeof(float));

        for(int i=0; i<size; i++) {
            layer->weights[i] = (float*)malloc(sizeof(float) * prevSize);
            layer->biases[i] = rand_init();
            for(int j=0; j<prevSize; j++) {
                layer->weights[i][j] = rand_init();
            }
        }
    } else {
        layer->weights = NULL;
        layer->biases = NULL;
        layer->activation = NULL;
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

void test(void) {
    printf("Hello 123\n");
}
