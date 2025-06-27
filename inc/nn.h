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
void decay_lr(NeuralNetwork* nn, float base_lr, float decay_rate, int epoch);
void feedforward(NeuralNetwork* nn, float* input);
void updateParameters(NeuralNetwork* nn);
void backpropagation(NeuralNetwork* nn, float* target);

#endif
