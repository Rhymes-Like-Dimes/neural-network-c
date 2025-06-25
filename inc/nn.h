#ifndef NN_H
#define NN_H

typedef struct {
    int layerID;
    int size;
    int prevSize;

    float* weights;
    float* biases;
    float* output;
    float* activation;
    float* delta;
    float* dweights;
    float* dbiases;
} Layer;

typedef struct {
    int numLayers;
    Layer** layers;
    float learningRate;
} NeuralNetwork;

Layer* init_layer(int layerID, int size, int prevSize);
NeuralNetwork* init_nn(int numLayers, int* layerSizes, float learningRate);
void feedforward(NeuralNetwork* nn, float* input);
void updateParameters(NeuralNetwork* nn);
void backpropagation(NeuralNetwork* nn, float* target);

#endif
