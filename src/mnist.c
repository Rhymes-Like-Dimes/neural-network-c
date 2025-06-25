// mnist.c
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_big_endian_uint32(FILE* f) {
    uint8_t bytes[4];
    fread(bytes, sizeof(uint8_t), 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

int load_mnist(const char* imageFile, const char* labelFile, MNISTDataset* dataset) {
    FILE* img = fopen(imageFile, "rb");
    FILE* lbl = fopen(labelFile, "rb");
    if (!img || !lbl) {
        fprintf(stderr, "Error opening MNIST files.\n");
        return 0;
    }

    // Read headers
    uint32_t img_magic = read_big_endian_uint32(img);
    uint32_t lbl_magic = read_big_endian_uint32(lbl);
    if (img_magic != 2051 || lbl_magic != 2049) {
        fprintf(stderr, "Invalid MNIST magic numbers.\n");
        fclose(img);
        fclose(lbl);
        return 0;
    }

    uint32_t numImages = read_big_endian_uint32(img);
    uint32_t numLabels = read_big_endian_uint32(lbl);
    uint32_t numRows = read_big_endian_uint32(img);
    uint32_t numCols = read_big_endian_uint32(img);
    uint32_t imageSize = numRows * numCols;

    if (numImages != numLabels) {
        fprintf(stderr, "Image and label counts do not match.\n");
        fclose(img);
        fclose(lbl);
        return 0;
    }

    // Allocate memory
    dataset->images = (float**)malloc(sizeof(float*) * numImages);
    dataset->labels = (uint8_t*)malloc(sizeof(uint8_t) * numLabels);
    dataset->numImages = numImages;
    dataset->imageSize = imageSize;

    uint8_t* buffer = (uint8_t*)malloc(imageSize);
    for (int i = 0; i < numImages; i++) {
        fread(buffer, sizeof(uint8_t), imageSize, img);
        dataset->images[i] = (float*)malloc(sizeof(float) * imageSize);
        for (int j = 0; j < imageSize; j++) {
            dataset->images[i][j] = buffer[j] / 255.0f;  // Normalize to [0,1]
        }
    }
    free(buffer);

    fread(dataset->labels, sizeof(uint8_t), numLabels, lbl);

    fclose(img);
    fclose(lbl);
    return 1;
}

void free_mnist(MNISTDataset* dataset) {
    for (int i = 0; i < dataset->numImages; i++) {
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->labels);
}
