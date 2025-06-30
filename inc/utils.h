#ifndef UTILS_H
#define UTILS_H
#include "mnist.h"
#include "nn.h"

void print_matrix(float* matrix, int rows, int cols);
float sigmoid(float z);
float dsigmoid_a(float a);
float reLU(float z);
float dreLU(float z);
void softmax(Layer* layer);
float rand_init_MSE();
float randn();
float rand_init_XNTPY(int fan_in);
int grade_result(NeuralNetwork* nn, MnistLoader* loader);
float calculate_loss(NeuralNetwork* nn, MnistLoader* loader);
void print_training_summary(float* accuracy, float* loss, int num_epochs);

#endif