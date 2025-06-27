#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "utils.h"

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

float dsigmoid_a(float a) {
    return a * (1.0f - a);
}

float rand_init() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("]\n");
    }
}

/**
 * @brief Prints a single layer of the network.
 * 
 * This function prints the layer of the layer pointer handed to it. You can
 * choose which components of each layer you want to print by setting the appropriate
 * flag. 
 * 
 * @param layer The pointer to the layer struct to be printerd.
 * @param inputFlag Set to 1 to print the input to the network in vector form.
 * @param weightFlag Set to 1 to print the weight matrix.
 * @param biasFlag Set to 1 to print the biases in vector form.
 * @param outputFlag Set to 1 to print the pre activation outputs in vector form.
 * @param activationFlag Set to 1 to print the layer activation in vector form.
 * @param deltaFlag Set to 1 to print the layer error terms in vector form.
 * 
 * @details Set flag to zero to prevent printing of specific parameter.
 */
void print_layer(Layer* layer, int inputFlag, int weightFlag, int biasFlag, int outputFlag, int activationFlag, int deltaFlag) {
    if (layer->prevSize == 0) {
        printf("Layer: %d\nSize: %d\n", layer->layerID, layer->size);

        if (inputFlag) {
            printf("Input:\n");
            print_matrix(layer->activation, layer->size, 1);   
        }
    } else {
        printf("Layer: %d\nSize: %d\nPrev Size: %d\n", layer->layerID, layer->size, layer->prevSize);

        if (layer->weights != NULL && weightFlag) {
            printf("\nWeights:\n");
            print_matrix(layer->weights, layer->size, layer->prevSize);
        }

        if (layer->biases != NULL && biasFlag) {
            printf("\nBiases:\n");
            print_matrix(layer->biases, layer->size, 1);
            printf("\n");
        }

        if (layer->output != NULL && outputFlag) {
            printf("Output:\n");
            print_matrix(layer->output, layer->size, 1);
            printf("\n");
        }

        if (layer->activation != NULL && activationFlag) {
            printf("Activation:\n");
            print_matrix(layer->activation, layer->size, 1);
            //printf("\n");
        }

        if (layer->delta != NULL && deltaFlag) {
            printf("Delta:\n");
            print_matrix(layer->activation, layer->size, 1);
            printf("\n");
        }
    }
    printf("--------------------------\n");
}

/**
 * @brief Prints each layer of the network using 'print_layer()'.
 * 
 * This function prints every layer of the neural network pointer handed to it. You can
 * choose which components of each layer you want to print by setting the appropriate
 * flag. 
 * 
 * @param nn The pointer to the neural network struct to be printerd.
 * @param inputFlag Set to 1 to print the input to the network in vector form.
 * @param weightFlag Set to 1 to print the weight matrix.
 * @param biasFlag Set to 1 to print the biases in vector form.
 * @param outputFlag Set to 1 to print the pre activation outputs in vector form.
 * @param activationFlag Set to 1 to print the layer activation in vector form.
 * @param deltaFlag Set to 1 to print the layer error terms in vector form.
 * 
 * @details Set flag to zero to prevent printing of specific parameter.
 */
void print_network(NeuralNetwork* nn, int inputFlag, int weightFlag, int biasFlag, int outputFlag, int activationFlag, int deltaFlag) {
    printf("--------------------------\n");
    for(int i = 0; i < nn->numLayers; i++) {
        Layer* layer = nn->layers[i];
        print_layer(layer, inputFlag, weightFlag, biasFlag, outputFlag, activationFlag, deltaFlag);
    }    
}

void shuffle_indices(int* indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
