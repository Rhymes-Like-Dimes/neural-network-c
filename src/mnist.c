#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "mnist.h"

/**
 * @brief Reads a 32-bit big-endian unsigned integer from a file.
 *
 * Grabs 4 bytes from the file and puts them together into a single
 * uint32_t, assuming big-endian byte order (used by MNIST files).
 *
 * @param f Pointer to an open binary file.
 */
uint32_t read_uint32_big_endian(FILE* f) {
    uint8_t bytes[4];
    fread(bytes, sizeof(uint8_t), 4, f);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

/**
 * @brief Loads all the MNIST data into RAM.
 *
 * Opens the image and label files, reads in all the pixels and labels,
 * allocates memory for input/target buffers, and sets up shuffling indices.
 * Optimized for cache locality by using 1D arrays.
 *
 * @param image_path Path to image file.
 * @param label_path Path to label file.
 */
MnistLoader* init_loader(const char* image_path, const char* label_path) {
    MnistLoader* loader = (MnistLoader*)malloc(sizeof(MnistLoader));

    //Open image & label files
    FILE* img_file = fopen(image_path, "rb");
    FILE* lbl_file = fopen(label_path, "rb");

    //Read magic numbers
    loader->magic_img = read_uint32_big_endian(img_file);
    loader->magic_lbl = read_uint32_big_endian(lbl_file);

    //Read image & label counts, set image size: size = rows * cols, set index = 0
    loader->num_img = read_uint32_big_endian(img_file);
    loader->num_lbl = read_uint32_big_endian(lbl_file);
    loader->image_size = read_uint32_big_endian(img_file) * read_uint32_big_endian(img_file);
    loader->index = 0;
    
    //Allocate memory
    loader->images = (float*)malloc(sizeof(float) * loader->num_img * loader->image_size);
    loader->labels = (uint8_t*)malloc(sizeof(uint8_t) * loader->num_lbl);
    loader->indices = (int*)malloc(sizeof(int) * loader->num_img);
    loader->input = (float*)malloc(sizeof(float) * loader->image_size);
    loader->target = (float*)malloc(sizeof(float) * 10);

    //Load images
    for(int i=0; i<(loader->num_img * loader->image_size); i++) {
        loader->images[i] = fgetc(img_file) / 255.0f;
    }

    //Load labels
    fread(loader->labels, sizeof(uint8_t), loader->num_lbl, lbl_file);

    //Load indices
    for(int i=0; i<loader->num_img; i++) {
        loader->indices[i] = i;
    }

    //Clean up
    fclose(img_file);
    fclose(lbl_file);

    return loader;
}

/**
 * @brief Shuffles the order of the training examples.
 *
 * Randomly reorders the indices array so that training happens in a different order each epoch.
 * Also resets the current index back to 0.
 *
 * @param loader  Pointer to the MnistLoader to shuffle.
 */
void mnist_shuffle(MnistLoader* loader) {
    srand(time(NULL));
    for(int i=loader->num_img-1; i>0; i--) {
        int j = rand() % (i + 1);
        int temp = loader->indices[i];
        loader->indices[i] = loader->indices[j];
        loader->indices[j] = temp;
    }
    loader->index = 0;
}


/**
 * @brief Loads the next example from the MNIST dataset into input and target arrays.
 *
 * Uses the current shuffled index to grab the image and label, then moves to the next one.
 * Normalizes the image pixels to [0, 1] and turns the label into one hot encoding.
 *
 * @param loader  Pointer to the MnistLoader.
 * @return 1 if successful, 0 if out of examples.
 */
int mnist_next(MnistLoader* loader) {
    if(loader->index >= loader->num_img) {
        return 0;
    }

    int i = loader->indices[loader->index++];
    for(int j=0; j<loader->image_size; j++) {
        loader->input[j] = loader->images[i * loader->image_size + j];
    }

    for(int j=0; j<10; j++) {
        loader->target[j] = loader->labels[i] == j ? 1.0f : 0.0f;
    }
    return 1;
}

