#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "utils.h"
#include "mnist.h"

/**
 * @brief Sigmoid activation function with overflow protection.
 *
 * Projects input into a range between 0 and 1.
 * Caps values over 40 to 1.0 and under -40 to 0.0 to avoid overflow.
 *
 * @param z The input value.
 */
float sigmoid(float z) {
    float result;
    if(z > 40.0f) {
        result = 1.0f;
    } else if (z < -40.0f) {
        result = 0.0f;
    } else {
        result = 1.0f / (1.0f + expf(-z));
    }
    return result;
}

/**
 * @brief Derivative of the sigmoid function using the activation value.
 *
 * Derivative of sigmoid s(z) is s'(z) = s(z)(1 - s(z)). But s(z) is just the
 * activation. Hence s'(z) = a * (1 - a).
 *
 * @param a The output of the sigmoid function.
 */
float dsigmoid_a(float a) {
    return a * (1.0f - a);
}

/**
 * @brief Generates a random float between -1 and 1.
 *
 * Used to initialize weights and biases so they start with small random values.
 */
float rand_init() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

/**
 * @brief Checks if the neural network guessed the right digit.
 *
 * Looks at the output layer's highest value and compares it to the correct label
 * in the target array. Returns 1 if they match, 0 if not.
 *
 * @param nn      The neural network after running feedforward.
 * @param loader  The MNIST loader with the correct target label.
 */
int grade_result(NeuralNetwork* nn, MnistLoader* loader) {
    Layer* output_layer = nn->layers[nn->numLayers - 1];
    int prediction = 0;
    int answer = 0;

    for(int i=0; i<output_layer->size; i++) {
        if(output_layer->activation[i] > output_layer->activation[prediction]) {
            prediction = i;
        }
    }

    for(int i=0; i<output_layer->size; i++) {
        if(loader->target[i]) {
            answer = i;
            break;
        }
    }
    return prediction == answer;
}