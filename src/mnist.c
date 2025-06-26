#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist.h"

//Reads 32 bit big endian integer from file
uint32_t read_uint32_big_endian(FILE* f) {
    uint8_t bytes[4];
    fread(bytes, sizeof(uint8_t), 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

//Reads and validates the MNIST image file header
int read_mnist_header(FILE* f, uint32_t* num_images, uint32_t* rows, uint32_t* cols) {
    uint32_t magic = read_uint32_big_endian(f);
    if (magic != 2051) {
        printf("Invalid MNIST image magic number\n");
        return 1;
    }

    *num_images = read_uint32_big_endian(f);
    *rows = read_uint32_big_endian(f);
    *cols = read_uint32_big_endian(f);

    return 0;
}

//Reads one image and normalizes it into a float array
int read_mnist_image(FILE* f, uint8_t* buffer, float* input, uint32_t image_size) {
    size_t read_bytes = fread(buffer, sizeof(uint8_t), image_size, f);
    if (read_bytes != image_size) {
        return 1;
    }

    for (uint32_t i = 0; i < image_size; i++) {
        input[i] = buffer[i] / 255.0f;  //Normalize to [0, 1]
    }
    return 0;
}

//Print image buffer as ASCII art
void print_image_ascii(uint8_t* buffer, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint8_t pixel = buffer[r * cols + c];
            char shade = (pixel > 200) ? '#' :
                         (pixel > 100) ? '+' :
                         (pixel > 50)  ? '.' : ' ';
            printf("%c", shade);
        }
        printf("\n");
    }
}

//Read MNIST label header
int read_label_header(FILE* f, uint32_t* num_labels) {
    uint32_t magic = read_uint32_big_endian(f);
    if (magic != 2049) {
        printf("Invalid MNIST label magic numer\n");
        return 1;
    }

    *num_labels = read_uint32_big_endian(f);
    return 0;
}

//Read one MNIST label and convert to one hot encoding
int read_mnist_label(FILE* f, uint8_t* label, float* target) {
    size_t read_bytes = fread(label, sizeof(uint8_t), 1, f);
    if (read_bytes != 1) {
        printf("Failed to read label\n");
        return 1;
    }

    for(int i = 0; i < 10; i++) {
        target[i] = (i == *label) ? 1.0f : 0.0f;  //One hot encoding
    }
    return 0;
}

//Generate target and input arrays for next example
int load_next_example(FILE* f_img, FILE* f_lbl, float* input, float* target, uint32_t image_size, uint8_t* label_out) {
    uint8_t* image_buffer = malloc(image_size);
    if (!image_buffer) {
        fprintf(stderr, "Failed to allocate temporary image buffer\n");
        return 1;
    }

    //Read and normalize image
    if (read_mnist_image(f_img, image_buffer, input, image_size)) {
        fprintf(stderr, "Failed to read MNIST image\n");
        free(image_buffer);
        return 1;
    }

    //Read and encode label
    if (read_mnist_label(f_lbl, label_out, target)) {
        fprintf(stderr, "Failed to read MNIST label\n");
        free(image_buffer);
        return 1;
    }

    free(image_buffer);
    return 0;
}

void skip_headers(FILE* f_img, FILE* f_lbl) {
    for (int i = 0; i < 4; i++) read_uint32_big_endian(f_img);
    for (int i = 0; i < 2; i++) read_uint32_big_endian(f_lbl); 
}


