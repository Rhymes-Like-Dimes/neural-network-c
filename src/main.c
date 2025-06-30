#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "nn.h"
#include "mnist.h"
#include "utils.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

//These are customizable. If the number of layers is changed, the layer sizes array must be changed below
//Also experiment with the learning rate and decay rate
#define NUM_LAYERS 4
#define NUM_EPOCHS 15

/**
 * 784 - 128 - 128 - 10
 * Best Accuracy (Cross entropy): 98.04%
 * NUM_EPOCHS = 15
 * base_lr = 0.01
 * decay_rate = 0.98
 * 
 * 784 - 128 - 10
 * Accuracy (Cross entropy): 98.00%
 * NUM_EPOCHS = 10
 * base_lr = 0.01
 * decay_rate = 0.98
 */

int main() {

    //File paths
    const char* training_images = "data/train-images.idx3-ubyte";
    const char* training_labels = "data/train-labels.idx1-ubyte";
    const char* testing_images = "data/t10k-images.idx3-ubyte";
    const char* testing_labels = "data/t10k-labels.idx1-ubyte";

    //Load images
    MnistLoader* training_loader = init_loader(training_images, training_labels);
    MnistLoader* testing_loader = init_loader(testing_images, testing_labels);

    //Learning parameters
    float base_lr = 0.01;
    float decay_rate = 0.98f;
    
    //Layer sizes 
    int layerSizes[NUM_LAYERS] = {
        INPUT_SIZE,
        128,
        128,
        OUTPUT_SIZE
    }; 

    //Create network
    NeuralNetwork* nn = init_nn(NUM_LAYERS, layerSizes, base_lr);

    //For tracking accuracy and loss over each epoch
    float accuracy[NUM_EPOCHS];
    float loss[NUM_EPOCHS];


    /*---------------- TRAINING LOOP ----------------*/


    //Multiple epochs
    for(int i=0; i<NUM_EPOCHS; i++) {
        
        //Shuffle training data & decay learning rate
        mnist_shuffle(training_loader);
        decay_lr(nn, base_lr, decay_rate, i);

        //Additional Variables
        int training_correct = 0;
        int training_iteration = 0;
        float training_loss = 0.0f;

        //Print epoch information
        printf("\n/-------------- Epoch %d --------------/\n", i + 1);
        printf("Learning Rate: %.5f\n", nn->learningRate);

        //Stochastic training loop
        while(mnist_next(training_loader)) {

            //Feedforward
            feedforward_XNTPY(nn, training_loader->input);

            //Backpropagation
            backpropagation_XNTPY(nn, training_loader->target);

            //Check prediction vs answer (for accuracy) and accumulate loss
            if(grade_result(nn, training_loader)) {
                training_correct++;
            }
            training_iteration++;
            training_loss += calculate_loss(nn, training_loader);

            //Print accuracy every 20000 iterations, 60000 iteration accuracy printed in summary
            if(training_iteration % 20000 == 0 && training_iteration != 60000) {
                printf("\nIteration: %d\n", training_iteration);
                printf("Accuracy: %.2f%%\n", 100.0f * (float)training_correct / (float)training_iteration);
                printf("Loss: %0.6f\n", training_loss / (float)training_iteration);
            }
        }
        accuracy[i] = 100.0f * (float)training_correct / (float)training_iteration;
        loss[i] = training_loss / (float)training_iteration;

        //Print epoch summary
        printf("\n/---------- Epoch %d Summary ----------/\n", i + 1);
        printf("Accuracy: %.2f%%\n", accuracy[i]);
        printf("Loss: %.6f\n", loss[i]);
    }
    print_training_summary(accuracy, loss, NUM_EPOCHS);

    /*---------------- Output to CSV ---------------*/

    FILE* csv = fopen("training_metrics.csv", "w");

    // Write header info
    fprintf(csv, "# Training configuration\n");
    fprintf(csv, "# Layer Sizes: ");
    for (int i = 0; i < NUM_LAYERS; i++) {
        fprintf(csv, "%d", layerSizes[i]);
        if (i < NUM_LAYERS - 1) fprintf(csv, "-");
    }
    fprintf(csv, "\n");
    fprintf(csv, "# Base Learning Rate: %.5f\n", base_lr);
    fprintf(csv, "# Decay Rate: %.5f\n\n", decay_rate);

    // Write column headers
    fprintf(csv, "Epoch,Accuracy,Loss\n");

    // Write data rows
    for (int i = 0; i < NUM_EPOCHS; i++) {
        fprintf(csv, "%d,%.4f,%.6f\n", i + 1, accuracy[i], loss[i]);
    }

    fclose(csv);
    printf("Saved training metrics to training_metrics.csv\n");


    /*---------------- TESTING LOOP ----------------*/


    //Additional variables
    int test_correct = 0;
    int test_iteration = 0;
    float test_loss = 0.0f;

    //Shuffle
    mnist_shuffle(testing_loader);

    //Feedforward loop
    printf("\n/---------------- TEST BEGIN ----------------/\n");
    while(mnist_next(testing_loader)) {

        //Feedfoward
        feedforward_XNTPY(nn, testing_loader->input);

        //Check prediction vs answer
        if(grade_result(nn, testing_loader)) {
            test_correct++;
        }
        test_iteration++;
        test_loss +=calculate_loss(nn, testing_loader);
        
        //Print accuracy every 20000 iterations
        if(test_iteration % 1000 == 0) {
            printf("\nIteration: %d\n", test_iteration);
            printf("Accuracy: %.2f%%\n", 100.0f * (float)test_correct / (float)test_iteration);
        }
    }

    //Print test summary
    printf("\n/------------- Test Summary -------------/\n");
    printf("Test Accuracy: %.2f%%\n", 100.0f * (float)test_correct / (float)test_iteration);
    printf("Test Loss: %.6f\n", test_loss / test_iteration);
    printf("/----------------------------------------/\n");

    return 0;
}