#include <stdio.h>
#include <stdlib.h>
#include "highway_lstm_kernel.cu"

int input_size = 3; // input size
int seqLength = 5; // timesteps
int numLayers = 2; // num_layers
int hiddenSize = 11; // output size
int timeSteps = 5; // == seqLength
int miniBatch = 5; // batch size
//int numElements = hiddenSize * miniBatch;

void populateVariables(
        float* x, unsigned int* lengths, float* h_data, 
        float* c_data, float* tmp_i, float* tmp_h, 
        float* T, float* bias, float* dropout) {
    srand(0);
    printf("Initializing x...\n");
    for(int i = 0; i < miniBatch * timeSteps * input_size; i++) {
        x[i] = rand() % 100;
    }
    printf("Initializing lengths...\n");
    for(int i = 0; i < miniBatch; i++) { // ?
        lengths[i] = timeSteps - (i / 2);
    }
    printf("Initializing h_data...\n");
    for(int i = 0; i < numLayers * (seqLength + 1) * miniBatch * hiddenSize; i++) {
        h_data[i] = 0;
    }
    printf("Initializing c_data...\n");
    for(int i = 0; i < numLayers * (seqLength + 1) * miniBatch * hiddenSize; i++) {
        c_data[i] = 0;
    }
    printf("Initializing tmp_i...\n");
    for(int i = 0; i < (miniBatch * 6 * hiddenSize); i++) {
        tmp_i[i] = rand() % 100;
    }
    printf("Initializing tmp_h...\n");
    for(int i = 0; i < (miniBatch * 5 * hiddenSize); i++) {
        tmp_h[i] = rand() % 100;
    }
    printf("Initializing weights...\n");
    for(int i = 0; i < 6 * hiddenSize * hiddenSize; i++) {
        T[i] = rand() % 100;
    }
    printf("Initializing bias...\n");
    for(int i = 0; i < 5 * hiddenSize + (5 * hiddenSize); i++) {
        bias[i] = rand() % 100;
    }
}

int main() {
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();
   // Host variables
    float * x = (float*)malloc((miniBatch * timeSteps * input_size + 1) * sizeof(float));
    unsigned int * lengths = (unsigned int*)malloc(miniBatch * sizeof(unsigned int));

    // Unified Variables
    //cudaErrCheck(cudaMallocManaged(&x, miniBatch * timeSteps * input_size * sizeof(float)));
    //cudaErrCheck(cudaMallocManaged(&lengths, miniBatch * sizeof(unsigned int)));

    // Device variables
    float * h_data_ptr = NULL; // (seqLength + 1) * numLayers * numElements
    float * c_data_ptr = NULL; // (seqLength) * (numLayers + 1) * numElements
    float * tmp_i_ptr = NULL; // batch_size x 6 * hidden_size
    float * tmp_h_ptr = NULL; // batch_size x 5 * hidden_size
    float * T_ptr = NULL; // numLayers * hiddenSize * hiddenSize * 8
    float * bias_ptr = NULL; // numLayers * hiddenSize * 8
    float * dropout_ptr = NULL; // input_size * input_size

    float * x_ptr = NULL;
    //unsigned int * lengths_ptr = NULL;

    printf("Allocating device variables\n");
    // Allocate device variables
    cudaErrCheck(cudaMalloc(&h_data_ptr, (numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float)));
    cudaErrCheck(cudaMalloc(&c_data_ptr, (numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float)));
    cudaErrCheck(cudaMalloc(&tmp_i_ptr, (miniBatch * (6 * hiddenSize)) * sizeof(float)));
    cudaErrCheck(cudaMalloc(&tmp_h_ptr, (miniBatch * (5 * hiddenSize)) * sizeof(float)));
    // HARDCODED VALUES AHEAD
    cudaErrCheck(cudaMalloc(&T_ptr, 2134 * sizeof(float)));
    cudaErrCheck(cudaMalloc(&bias_ptr, 110 * sizeof(float)));
    //cudaErrCheck(cudaMalloc((void**)&dropout_ptr, numLayers * miniBatch * hiddenSize * sizeof(float)));
    // Transfer host variables to device variables
    //cudaErrCheck(cudaMemcpy(x_ptr, x, input_size * sizeof(float), cudaMemcpyHostToDevice));
    //cudaErrCheck(cudaMemcpy(lengths_ptr, lengths, input_size * input_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&x_ptr, (miniBatch * timeSteps * input_size + 1) * sizeof(float)));
    //cudaErrCheck(cudaMalloc(&lengths_ptr, miniBatch * sizeof(unsigned int)));

    printf("Allocating host variables");
    float * h_data = (float*)malloc((numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float));
    float * c_data = (float*)malloc((numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float));
    float * tmp_i = (float*)malloc((miniBatch * (6 * hiddenSize) * sizeof(float)));
    float * tmp_h = (float*)malloc((miniBatch * (5 * hiddenSize) * sizeof(float)));
    float * T = (float*)malloc(2134 * sizeof(float));
    float * bias = (float*)malloc(110 * sizeof(float));
    float * dropout;
    // Populate host variables
    printf("Populating variables\n");
    //populateVariables(x, lengths, h_data, c_data, tmp_i, tmp_h, T, bias, dropout);

    printf("Copying device variables\n");
    cudaErrCheck(cudaMemcpy(h_data_ptr, h_data, (numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(c_data_ptr, c_data, (numLayers * (seqLength + 1 * miniBatch) * hiddenSize) * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(tmp_i_ptr, tmp_i, miniBatch * (6 * hiddenSize) * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(tmp_h_ptr, tmp_h, miniBatch * (5 * hiddenSize) * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(T_ptr, T, 2134 * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(bias_ptr, bias, 110 * sizeof(float), cudaMemcpyHostToDevice));
    //cudaErrCheck(cudaMemcpy(dropout_ptr, dropout, numLayers * miniBatch * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(x_ptr, x, (miniBatch * timeSteps * input_size + 1) * sizeof(float), cudaMemcpyHostToDevice));
    //cudaErrCheck(cudaMemcpy(lengths_ptr, lengths, miniBatch * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    printf("Running device kernel\n");
    highway_lstm_forward_ongpu(input_size, hiddenSize, miniBatch, numLayers,
            seqLength, x, lengths, h_data_ptr, c_data_ptr, tmp_i_ptr,
            tmp_h_ptr, T_ptr, bias_ptr, NULL, NULL,
            false, stream, handle);

    printf("Successfully ran!");
    return 0;
}
