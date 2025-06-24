#ifndef UTILS_H
#define UTILS_H

float sigmoid(float output);
float rand_init();
void print_matrix(float* matrix, int rows, int cols);
void print_layer(Layer* layer, int inputFlag, int weightFlag, int biasFlag, int outputFlag, int activationFlag, int deltaFlag);
void print_network(NeuralNetwork* nn, int inputFlag, int weightFlag, int biasFlag, int outputFlag, int activationFlag, int deltaFlag);

#endif