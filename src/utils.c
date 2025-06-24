#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

float sigmoid(float output) {
    float result;
    if(output > 40.0f) {
        result = 1.0f;
    } else if (output < -40.0f) {
        result = 0.0f;
    } else {
        result = 1.0f / (1.0f + expf(-output));
    }
    return result;
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
