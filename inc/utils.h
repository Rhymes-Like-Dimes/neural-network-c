#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>

float rand_init() {
    return 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
}

#endif