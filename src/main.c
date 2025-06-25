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
    clock_t start1 = clock();
    feedforward(nn, input);
    clock_t end1 = clock();

    //Backpropagation
    clock_t start2 = clock();
    backpropagation(nn, target);
    clock_t end2 = clock();

    double elapsed = (double)(end1 - start1) / CLOCKS_PER_SEC;
    double elapsed = (double)(end2 - start2) / CLOCKS_PER_SEC;
    printf("Feedforward took %.6f seconds\n", elapsed);
    printf("Backpropagation took %.6f seconds\n", elapsed);

    return 0;
}