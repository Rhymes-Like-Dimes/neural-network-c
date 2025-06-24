#ifndef NN_H
#define NN_H

typedef struct {
    int layerID;
    int size;
    int prevSize;

    float* weights;
    float* biases;
    float* activation;
    float* delta;
    float* output;
} Layer;

typedef struct {
    int numLayers;
    Layer** layers;
    float learningRate;
} NeuralNetwork;

Layer* init_layer(int layerID, int size, int prevSize);
NeuralNetwork* init_nn(int numLayers, int* layerSizes, float learningRate);
void print_network(NeuralNetwork* nn);
void feedforward(NeuralNetwork* nn, float* input);

#endif
