#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

typedef struct {
    int numImages;
    int imageSize;    // should be 784 (28x28)
    float** images;   // normalized pixel data
    uint8_t* labels;  // raw digit labels 0-9
} MNISTDataset;

int load_mnist(const char* imageFile, const char* labelFile, MNISTDataset* dataset);
void free_mnist(MNISTDataset* dataset);

#endif
