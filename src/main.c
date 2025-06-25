#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "nn.h"
#include "utils.h"


#define INPUT_SIZE 1000
#define HIDDEN_SIZE 500
#define OUTPUT_SIZE 10
#define NUM_LAYERS 6

int main() {
    
    //Network structure
    int layerSizes[NUM_LAYERS] = {
        INPUT_SIZE,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        OUTPUT_SIZE
    };

    //Generate example input
    float input[INPUT_SIZE];
    for(int i=0; i<INPUT_SIZE; i++) {
        input[i] = fabsf(rand_init());
    }

    //Generate example solution
    float target[OUTPUT_SIZE] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    //Create network
    NeuralNetwork* nn = init_nn(NUM_LAYERS, layerSizes, 0.05f);

    //Feed forward
    feedforward(nn, input);

    //Backpropagation
    clock_t start = clock();
    backpropagation(nn, target);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Backpropagation took %.6f seconds\n", elapsed);

    return 0;
}