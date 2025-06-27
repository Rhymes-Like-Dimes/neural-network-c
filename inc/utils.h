#ifndef UTILS_H
#define UTILS_H
#include "mnist.h"

float sigmoid(float z);
float dsigmoid_a(float a);
float rand_init();
int grade_result(NeuralNetwork* nn, MnistLoader* loader);

#endif