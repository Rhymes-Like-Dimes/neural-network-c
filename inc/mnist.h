#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint32_t magic_img;
    uint32_t magic_lbl;

    uint32_t num_img;
    uint32_t num_lbl;
    uint32_t image_size;

    int index;
    float* images;
    uint8_t* labels;
    int* indices;

    float* input;
    float* target;
} MnistLoader;

uint32_t read_uint32_big_endian(FILE* f);
MnistLoader* init_loader(const char* image_path, const char* label_path);
void mnist_shuffle(MnistLoader* loader);
int mnist_next(MnistLoader* loader);

#endif
