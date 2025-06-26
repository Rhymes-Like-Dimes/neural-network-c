#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <time.h>
#include "nn.h"
#include "utils.h"
#include "mnist.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define NUM_LAYERS 5

int main() {

    //Open and verify image and label files
    FILE* f_img = fopen("data/train-images.idx3-ubyte", "rb");
    FILE* f_lbl = fopen("data/train-labels.idx1-ubyte", "rb");

    if (!f_img || !f_lbl) {
        perror("Failed to open MNIST files");
        if (f_img) {
            fclose(f_img);
        } 
        if (f_lbl) {
            fclose(f_lbl);
        } 
        return 1;
    }
    
    uint32_t num_images, rows, cols, num_labels;
    if (read_mnist_header(f_img, &num_images, &rows, &cols) ||
        read_label_header(f_lbl, &num_labels)) {
        fclose(f_img);
        fclose(f_lbl);
        return 1;
    }

    //Network structure
    float base_lr = 0.2f;
    float decay_rate = 0.85f;
    
    //3 hidden layers size 128, 64, 32 
    //learning rate = 0.3, decay rate = 0.9, accuracy = 98.6%
    //learning rate = 0.5, decay rate = 0.05, accuracy = 98.6%
    //
    int layerSizes[NUM_LAYERS] = {
        INPUT_SIZE,
        128,
        64,
        32,
        OUTPUT_SIZE
    };
    
    
    //Create network & additional required variables
    NeuralNetwork* nn = init_nn(NUM_LAYERS, layerSizes, base_lr);
    Layer* output_layer = nn->layers[NUM_LAYERS - 1];

    int num_examples = 60000;
    int num_epochs = 5;
    
    int recent_correct = 0;
    int recent_window = 1000;

    float feedforward_elapsed = 0.0;
    float backpropagation_elapsed = 0.0;
    float training_elapsed = 0.0;

    float feedforward_average;
    float backpropagation_average;
    float training_average;

    //Epoch loop
    for(int epoch=0; epoch<num_epochs; epoch++) {
        nn->learningRate = base_lr * powf(decay_rate, epoch);

        printf("\n----Epoch %d----\n\n", epoch + 1);
        printf("Learning rate: %0.5f\n", nn->learningRate);

        rewind(f_img);
        rewind(f_lbl);
        skip_headers(f_img, f_lbl);
        int num_correct = 0;
    
        //Training loop
        for(int i=0; i<num_examples; i++) {

            //Required variables
            float input[INPUT_SIZE];
            float target[OUTPUT_SIZE];
            uint8_t label;
            clock_t start0 = clock();
            
            //Learning rate decay
            // float progress = (float)i / (float)num_examples;
            // float next_lr = min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cosf(progress * M_PI));
            // nn->learningRate = next_lr;

            //Load next example
            if (load_next_example(f_img, f_lbl, input, target, INPUT_SIZE, &label)) {
                fprintf(stderr, "Failed to load example %d\n", i);
                break;
            }

            //Feed forward
            clock_t start1 = clock();
            feedforward(nn, input);
            clock_t end1 = clock();

            //Backpropagation
            clock_t start2 = clock();
            backpropagation(nn, target);
            clock_t end2 = clock();

            //Calculate accuracy over all examples
            int prediction = 0;
            for(int j=0; j<output_layer->size; j++) {
                if(output_layer->activation[j] > output_layer->activation[prediction]) {
                    prediction = j;
                }
            }
            if(prediction == label) {
                num_correct++;
                recent_correct++;
            }

            //Print loss, culmulative accuracy and accuracy over 1000 examples every 12 000 examples
            if(i % 20000 == 0) {
                float MSE = 0.0f;

                for(int j=0; j<output_layer->size; j++) {
                    MSE += 0.5 * pow(output_layer->activation[j] - target[j], 2.0);
                }
                printf("Training example (epoch %d): %d\n", epoch + 1, i);
                printf("Mean squared error: %.6f\n", MSE / 10);
                if(i != 0) {
                    printf("Cumulative Epoch Accuracy: %0.2f%%\n", 100.0f * (float)num_correct / (float)i);
                    printf("Accuracy (last %d examples): %0.2f%%\n", recent_window, 100.0f * (float)recent_correct / (float)recent_window);
                }
                printf("\n");
            }

            //Resets every 1000 although it is only needed/printed every 12000
            if(i % recent_window == 0) {
                recent_correct = 0;
            }

            //Used to measure average training times
            feedforward_elapsed += (float)(end1 - start1) / CLOCKS_PER_SEC;
            backpropagation_elapsed += (float)(end2 - start2) / CLOCKS_PER_SEC;
            training_elapsed += (float)(end2 - start0) / CLOCKS_PER_SEC;
        }
    

        //Training summary
        feedforward_average = feedforward_elapsed / num_examples;
        backpropagation_average = backpropagation_elapsed / num_examples;
        training_average = training_elapsed / num_examples;

        printf("\n------------------------\n");
        printf("Epoch training accuracy (final %d examples): %.2f%%\n", recent_window, 100.0f * (float)recent_correct / (float)recent_window);
        printf("Average feedforward time: %.4f\n", feedforward_average);
        printf("Average backpropagation time: %.4f\n", backpropagation_average);
        printf("Average training example time: %.4f\n", training_average);
        printf("------------------------\n");
    }

    //Clean up
    fclose(f_img);
    fclose(f_lbl);

    return 0;
}