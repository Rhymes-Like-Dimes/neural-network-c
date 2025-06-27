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
#define NUM_LAYERS 5
#define NUM_EPOCHS 5

int main() {

    //File paths
    const char* training_images = "data/train-images.idx3-ubyte";
    const char* training_labels = "data/train-labels.idx1-ubyte";
    const char* testing_images = "data/t10k-images.idx3-ubyte";
    const char* testing_labels = "data/t10k-labels.idx1-ubyte";

    //Load images
    MnistLoader* training_loader = init_loader(training_images, training_labels);
    MnistLoader* testing_loader = init_loader(testing_images, testing_labels);

    //Network structure
    float base_lr = 0.5;
    float decay_rate = 0.85f;
    
    //Layer sizes 
    int layerSizes[NUM_LAYERS] = {
        INPUT_SIZE,
        128,
        64,
        32,
        OUTPUT_SIZE
    }; 

    //Create network
    NeuralNetwork* nn = init_nn(NUM_LAYERS, layerSizes, base_lr);


    /*---------------- TRAINING LOOP ----------------*/


    //Multiple epochs
    for(int i=0; i<NUM_EPOCHS; i++) {
        
        //Shuffle training data & decay learning rate
        mnist_shuffle(training_loader);
        decay_lr(nn, base_lr, decay_rate, i);

        //Additional Variables
        int training_correct = 0;
        int training_iteration = 0;

        //Print epoch information
        printf("\n/-------------- Epoch %d --------------/\n", i + 1);
        printf("Learning Rate: %.5f\n", nn->learningRate);

        //Stochastic training loop
        while(mnist_next(training_loader)) {

            //Feedforward
            feedforward(nn, training_loader->input);

            //Backpropagation
            backpropagation(nn, training_loader->target);

            //Check prediction vs answer
            if(grade_result(nn, training_loader)) {
                training_correct++;
            }
            training_iteration++;

            //Print accuracy every 20000 iterations, 60000 iteration accuracy printed in summary
            if(training_iteration % 20000 == 0 && training_iteration != 60000) {
                printf("\nIteration: %d\n", training_iteration);
                printf("Accuracy: %.2f%%\n", 100.0f * (float)training_correct / (float)training_iteration);
            }
        }

        //Print epoch summary
        printf("\n/---------- Epoch %d Summary ----------/\n", i + 1);
        printf("Accuracy: %.2f%%\n", 100.0f * (float)training_correct / (float)training_iteration);
    }


    /*---------------- TESTING LOOP ----------------*/


    //Additional variables
    int test_correct = 0;
    int test_iteration = 0;

    //Feedforward loop
    printf("\n/---------------- TEST BEGIN ----------------/\n");
    while(mnist_next(testing_loader)) {

        //Feedfoward
        feedforward(nn, testing_loader->input);

        //Check prediction vs answer
        if(grade_result(nn, testing_loader)) {
            test_correct++;
        }
        test_iteration++;
        
        //Print accuracy every 20000 iterations
        if(test_iteration % 1000 == 0) {
            printf("\nIteration: %d\n", test_iteration);
            printf("Accuracy: %.2f%%\n", 100.0f * (float)test_correct / (float)test_iteration);
        }
    }

    //Print test summary
    printf("\n/------------- Test Summary -------------/\n");
    printf("Test Accuracy: %.2f%%\n", 100.0f * (float)test_correct / (float)test_iteration);
    printf("/----------------------------------------/\n");

    return 0;
}