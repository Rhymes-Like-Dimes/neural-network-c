#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>

uint32_t read_uint32_big_endian(FILE* f);
int read_mnist_header(FILE* f, uint32_t* num_images, uint32_t* rows, uint32_t* cols);
int read_mnist_image(FILE* f, uint8_t* buffer, float* input, uint32_t image_size);
void print_image_ascii(uint8_t *buffer, uint32_t rows, uint32_t cols);
int read_label_header(FILE* f, uint32_t* num_labels);
int read_mnist_label(FILE* f, uint8_t* label, float* target);
int load_next_example(FILE* f_img, FILE* f_lbl, float* input, float* target, uint32_t image_size, uint8_t* label_out);
void skip_headers(FILE* f_img, FILE* f_lbl);

#endif
