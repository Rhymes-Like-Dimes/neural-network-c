#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "utils.h"

int main() {

    //Network structure
    int numLayers = 3;
    int layerSizes[] = {6, 4, 2};
    float learningRate = 0.01f;

    //Generate example input
    float input[6];
    for(int i=0; i<6; i++) {
        input[i] = fabsf(rand_init());
    }

    //Create network
    NeuralNetwork* nn = init_nn(numLayers, layerSizes, learningRate);

    //Feed forward
    feedforward(nn, input);
    print_network(nn, 1, 1, 1, 1, 1, 0);

    return 0;
}