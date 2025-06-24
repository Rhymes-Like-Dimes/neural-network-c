#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "utils.h"

int main() {
    int numLayers = 3;
    int layerSizes[] = {6, 4, 2};
    float learningRate = 0.01f;

    NeuralNetwork* nn = init_nn(numLayers, layerSizes, learningRate);
    print_network(nn);
    return 0;
}