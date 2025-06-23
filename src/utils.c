#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

float sigmoid(float output) {
    return 1 / (1 + exp(-output));
}

float rand_init() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}
