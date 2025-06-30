#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "nn.h"
#include "utils.h"
#include "mnist.h"

/**
 * @brief Prints a matrix.
 * 
 * @param matrix Pointer to the matrix to print.
 * @param rows   Number of rows in matrix.
 * @param cols   Number of columns in matrix.
 */
void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%7.4f", matrix[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

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
 * @param a The activation or the output of the sigmoid function.
 */
float dsigmoid_a(float a) {
    return a * (1.0f - a);
}

/**
 * @brief ReLU activation function. Used for hidden layer activation
 * 
 * Outputs x if x>0 and 0 if x<=0.
 * 
 * @param z The input value or the output of a neuron.  
 */
float reLU(float z) {
    return (z > 0) ? z : 0;
}

/**
 * @brief Derivative of ReLU. Used during backpropagation
 * 
 * Outputs 1 if x>0 and 0 if x<=0.
 * 
 * @param z The input value. 
 */
float dreLU(float z) {
    return (z > 0) ? 1 : 0;
}

/**
 * @brief Calculates the softmax of neuron outputs. Used only in final layer.
 * 
 * Outputs e^(z_i) / sum(j=1 to k)[e^(z_j)] for neuron i. Dividied by sum of all
 * neuron outputs in that layer. The maximum value of the output vector is subtracted
 * prior to exponentiation to avoid numerical instability and infinities.
 * 
 * @param layer Pointer to the layer this will be performed on. Requires access
 * to the entire layer->output array.
 */
void softmax(Layer* layer) {
    //Find max
    float max = layer->output[0];
    for(int i=1; i<layer->size; i++) {
        if(layer->output[i] > max) {
            max = layer->output[i];
        }
    }

    //Compute softmax
    float sum = 0.0f;
    for (int i = 0; i < layer->size; i++) {
        layer->activation[i] = expf(layer->output[i] - max);  //Subtract max for stability
        sum += layer->activation[i];
    }

    for (int i = 0; i < layer->size; i++) {
        layer->activation[i] /= sum;
    }
}

/**
 * @brief Generates a random float between -1 and 1.
 *
 * Used to initialize weights and biases so they start with small random values.
 */
float rand_init_MSE() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

/**
 * Returns a random float from a normal (Gaussian) distribution with mean 0 and std dev 1.
 * 
 * This uses the Box-Muller transform to turn two random numbers into one that follows
 * a bell curve. Useful for initializing weights with normal distribution.
 */
float randn() {
    float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
    float u2 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/**
 * Initializes a neural network weight using He initialization (good for ReLU).
 * 
 * Returns a random float from a normal distribution with std dev = sqrt(2 / fan_in),
 * where fan_in is the number of inputs to the neuron. Helps keep values from
 * blowing up or shrinking too much during training.
 */
float rand_init_XNTPY(int fan_in) {
    float stddev = sqrtf(2.0f / fan_in);
    return randn() * stddev;
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

    //Find index of highest activation (i.e the model's prediction). This 
    //corresponds to the actual label as well. 
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

/**
 * @brief Calculates the cross entropy loss for a single training example.
 * 
 * This function finds the predicted probability of the correct answer by searching through
 * the target array (stored in the MnistLoader), and finding the entry that is larger than 0.
 * As the target array is one hot encoded, the correct answer will be the only non zero entry,
 * allowing us to pick out the index, and use it to index the predicted probability of the models
 * answer (regardless of whether or not it made the correct prediction).
 * 
 * @param nn     Pointer to the neural network struct.
 * @param loader Pointer to the loader struct with the target array.
 */
float calculate_loss(NeuralNetwork* nn, MnistLoader* loader) {
    Layer* output_layer = nn->layers[nn->numLayers - 1];
    int prediction = 0;

    //Find predicted probability of correct answer 
    for(int i=0; i<output_layer->size; i++) {
        if(loader->target[i] > 0.0f) {
            prediction = i;
        }
    }
    return -logf(output_layer->activation[prediction]);
}

/**
 * @brief Prints the accuracy and loss per epoch (in a pretty way)
 * 
 * This one was written by GPT not gonna lie
 * 
 * @param accuracy   Array of accuracy values per epoch
 * @param loss       Array of loss values per epoch
 * @param num_epochs Number of epochs
 */
void print_training_summary(float* accuracy, float* loss, int num_epochs) {
    printf("\n/=========== Training Summary ===========/\n");
    printf("| Epoch |   Accuracy   |     Loss     |\n");
    printf("|-------|--------------|--------------|\n");
    for (int i = 0; i < num_epochs; i++) {
        printf("|  %3d  |   %6.2f %%   |  %10.6f |\n", 
            i + 1, accuracy[i], loss[i]);
    }
    printf("/========================================/\n");
}
