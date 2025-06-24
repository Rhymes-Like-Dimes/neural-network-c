#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

float sigmoid(float output) {
    return 1 / (1 + expf(-output));
}

float rand_init() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}
