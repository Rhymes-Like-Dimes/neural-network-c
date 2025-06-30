#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "nn.h"
#include "mnist.h"
#include "utils.h"

#define INPUT_SIZE 6
#define OUTPUT_SIZE 2
#define NUM_LAYERS 3


int main() {

    //Learning parameters
    float base_lr = 0.5;
    float decay_rate = 0.85f;
    
    //Layer sizes 
    int layerSizes[NUM_LAYERS] = {
        INPUT_SIZE,
        3,
        OUTPUT_SIZE
    }; 

    //Input example
    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++) {
        input[i] = fabsf(rand_init());
    }

    //Output examples
    float target[2] = {1, 0};

    //Create network
    NeuralNetwork* nn = init_nn(NUM_LAYERS, layerSizes, base_lr);

    feedforward_reLU(nn, input);

    //Print
    printf("Inputs\n");
    print_matrix(nn->layers[0]->activation, INPUT_SIZE, 1);

    printf("Layer 1\n");
    printf("Weights:\n");
    print_matrix(nn->layers[1]->weights, 3, INPUT_SIZE);
    printf("Biases:\n");
    print_matrix(nn->layers[1]->biases, 3, 1);
    printf("Output:\n");
    print_matrix(nn->layers[1]->output, 3, 1);
    printf("Activation:\n");
    print_matrix(nn->layers[1]->activation, 3, 1);

    printf("Layer 2\n");
    printf("Weights:\n");
    print_matrix(nn->layers[2]->weights, 2, 3);
    printf("Biases:\n");
    print_matrix(nn->layers[2]->biases, 2, 1);
    printf("Output:\n");
    print_matrix(nn->layers[2]->output, 2, 1);
    printf("Activation:\n");
    print_matrix(nn->layers[2]->activation, 2, 1);\

    printf("\n");
    printf("Target:\n");
    print_matrix(target, 2, 1);

    backpropagation_Xentropy(nn, target);
    
    // printf("Output deltas:\n");
    // print_matrix(nn->layers[2]->delta, 2, 1);
    // printf("Output dWeights:\n");
    // print_matrix(nn->layers[2]->dweights, 2, 3);

    return 0;
}