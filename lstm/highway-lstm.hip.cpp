#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <rocblas.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))
//#define rocblasAssert(x) (assert((x)==rocblas_status_success))

void rocblasAssert(rocblas_status status) {
    switch(status) {
        case rocblas_status_success:
            return;
        case rocblas_status_invalid_handle:
            printf("ERROR: Invalid handle\n");
            exit(0);
        case rocblas_status_not_implemented:
            printf("ERROR: Function not implemented\n");
            exit(0);
        case rocblas_status_invalid_pointer:
            printf("ERROR: Invalid Pointer\n");
            exit(0);
        case rocblas_status_invalid_size:
            printf("ERROR: Invalid Size\n");
            exit(0);
        case rocblas_status_memory_error:
            printf("ERROR: Bad memory alloc/copy/dealloc!\n");
            exit(0);
        case rocblas_status_internal_error:
            printf("ERROR: Internal Error\n");
            exit(0);
        case rocblas_status_perf_degraded:
            printf("ERROR: Low device memory, performance degraded\n");
            exit(0);
        case rocblas_status_size_query_mismatch:
            printf("ERROR: Unmatched start / stop size query\n");
            exit(0);
        case rocblas_status_size_increased:
            printf("ERROR: Device memory size increased\n");
            exit(0);
        case rocblas_status_size_unchanged:
            printf("ERROR: Device memory size unchanged\n");
            exit(0);
        case rocblas_status_invalid_value:
            printf("ERROR: Arguments invalid!\n");
            exit(0);
        case rocblas_status_continue:
            printf("WARNING: nothing preventing function to proceed\n");
            exit(0);
        case rocblas_status_check_numerics_fail:
            printf("ERROR: NaN / Infinity present\n");
            exit(0);
        default:
            printf("ERROR: Unknown\n");
            exit(0);
    }
}

__device__ float sigmoidf(float in) {
    return 1.f / (1.f + expf(-in));
}

__global__ void elementWise_fp(int hiddenSize, int miniBatch, int numCovered,
                                float *tmp_h,
                                float *tmp_i,
                                float *bias,
                                float *linearGates,
                                float *h_out,
                                float *dropout_in,
                                float *c_in,
                                float * c_out,
                                bool training) {
    int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if(index >= numCovered * hiddenSize) return;

    int batch = index / hiddenSize;
    int h_gateIndex = (index % hiddenSize) + 5 * batch * hiddenSize;
    int i_gateIndex = (index % hiddenSize) + 6 * batch * hiddenSize;

    float g[6];

    for(int i = 0; i < 5; i++) {
        g[i] = tmp_i[i * hiddenSize + i_gateIndex] + tmp_h[i * hiddenSize + h_gateIndex];
        g[i] += bias[i * hiddenSize + index % hiddenSize];
    }

    g[5] = tmp_i[5 * hiddenSize + i_gateIndex];

    float in_gate       = sigmoidf(g[0]);
    float forget_gate   = sigmoidf(g[1]);
    float act_gate      = tanhf(g[2]);
    float out_gate      = sigmoidf(g[3]);
    float r_gate        = sigmoidf(g[4]);
    float lin_gate      = g[5];
/*
    if(training) { // Remove training
        linearGates[i_gateIndex] = in_gate;
        linearGates[i_gateIndex + 1 * hiddenSize] = forget_gate;
        linearGates[i_gateIndex + 2 * hiddenSize] = act_gate;
        linearGates[i_gateIndex + 3 * hiddenSize] = out_gate;
        linearGates[i_gateIndex + 4 * hiddenSize] = r_gate;
        linearGates[i_gateIndex + 5 * hiddenSize] = lin_gate;
    }
*/
    float val = (forget_gate * c_in[index]) + (in_gate * act_gate);

    c_out[index] = val;

    val = out_gate * tanhf(val);
    val = val * r_gate + (1. - r_gate) * lin_gate;
    val = val * dropout_in[index];
    h_out[index] = val;
}
// Don't worry about hip graph for now
void highway_lstm_forward_ongpu(int inputSize, int hiddenSize, int miniBatch, /*batch size */
        int numLayers, int seqLength, float *x, int *lengths, float *h_data,
        float *c_data, float *tmp_i, float *tmp_h, float *T, float *bias,
        float *dropoutMask, float *gates, bool is_training, hipStream_t stream, rocblas_handle handle) {
/*
    hipGraph_t graph;
    hipGraphExec_t graphExec;
*/    
    const int numElements = hiddenSize * miniBatch;
    
    float zero = 0.f;
    float one = 1.f;
   
//    HIP_ASSERT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    hipStream_t stream_i;
    hipStream_t stream_h;

    HIP_ASSERT(hipStreamCreate(&stream_i));
    HIP_ASSERT(hipStreamCreate(&stream_h));

    for(int layer = 0; layer < numLayers; layer++) { // Remove outer for loop
        int direction;
        int startInd;
        int currNumCovered;
        if(layer % 2 == 0) {
            direction = 1;
            startInd = 0;
            currNumCovered = miniBatch;
        } else {
            direction = -1;
            startInd = seqLength - 1;
            currNumCovered = 0;
        }
        rocblasAssert(rocblas_set_stream(handle, stream));
        
        for(int t = startInd; t < seqLength && t >= 0; t = t + direction) {
            int prevIndex;
            if(direction == 1) {
                while(lengths[currNumCovered-1] <= t) {
                    currNumCovered--;
                }
                prevIndex = t;
            } else {
                while((currNumCovered < miniBatch) && (lengths[currNumCovered] > t)) {
                    currNumCovered++;
                }
                prevIndex = (t + 2) % (seqLength + 1);
            }

            int inSize;
            int weightStart;
            float *inputPtr;
            if(layer == 0) {
                inSize = inputSize;
                weightStart = 0;
                inputPtr = x + t * inputSize * miniBatch;
                prevIndex = t;
            } else {
                inSize = hiddenSize;
                weightStart = 6 * hiddenSize * inputSize + 5 * hiddenSize * hiddenSize + (layer - 1) * 11 * hiddenSize * hiddenSize;
                inputPtr = h_data + (t+1) * numElements + (layer - 1) * (seqLength+1) * numElements;
            }
    
    dim3 threadsPerBlock = (256);
    dim3 blocks = ((currNumCovered * hiddenSize) + 256 - 1) / 256;
 
            rocblasAssert(rocblas_set_stream(handle, stream_i));

            rocblasAssert(rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        6*hiddenSize, currNumCovered, inSize,
                        &one,
                        &T[weightStart],
                        6 * hiddenSize,
                        inputPtr,
                        inSize,
                        &zero,
                        tmp_i,
                        6 * hiddenSize));
            
            rocblasAssert(rocblas_set_stream(handle, stream_h));

            rocblasAssert(rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        5*hiddenSize, currNumCovered, hiddenSize,
                        &one,
                        &T[6 * hiddenSize * inSize + weightStart],
                        5 * hiddenSize,
                        h_data + prevIndex * numElements + layer * (seqLength + 1) * numElements,
                        hiddenSize,
                        &zero,
                        tmp_h,
                        5 * hiddenSize));

            HIP_ASSERT(hipDeviceSynchronize());

            hipLaunchKernelGGL(elementWise_fp, blocks, threadsPerBlock, 0, stream, 
                    hiddenSize, miniBatch, currNumCovered,
                    tmp_h,
                    tmp_i,
                    bias + 5 * layer * hiddenSize,
                    /*is_training ? gates + 6 * (t * numElements + layer * seqLength * numElements) :*/ NULL,
                    h_data + (t + 1) * numElements + layer * (seqLength + 1) * numElements,
                    dropoutMask + layer * numElements,
                    c_data + prevIndex * numElements + layer * (seqLength + 1) * numElements,
                    c_data + (t + 1) * numElements + layer * (seqLength + 1) * numElements,
                    is_training);
            HIP_ASSERT(hipGetLastError());

        HIP_ASSERT(hipDeviceSynchronize());
        }
    }
/*    
    HIP_ASSERT(hipStreamEndCapture(stream, &graph));
    HIP_ASSERT(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    if(!graphExec) {
        printf("ERROR: Could not create graph!");
        exit(0);
    }

    HIP_ASSERT(hipGraphLaunch(graphExec, stream)); 
*/
    HIP_ASSERT(hipStreamSynchronize(stream));

    rocblasAssert(rocblas_set_stream(handle, stream));
    HIP_ASSERT(hipStreamDestroy(stream_i));
    HIP_ASSERT(hipStreamDestroy(stream_h));

    HIP_ASSERT(hipDeviceSynchronize());
}

int main() {
    hipStream_t graph_stream;
    HIP_ASSERT(hipStreamCreate(&graph_stream));

    rocblas_handle handle;
    rocblasAssert(rocblas_create_handle(&handle));

    bool TRAINING = false;

    int inputSize, hiddenSize, batchSize, numLayers, seqLength;
    
    inputSize = 3;
    seqLength = 5;
    numLayers = 2; // Default is 4
    hiddenSize = 11;
    batchSize = 5; 

    const int numElements = hiddenSize * batchSize;

    float *x = (float*)malloc(miniBatch * timeSteps * input_size * sizeof(float)); // Input
    int *lengths = (int*)malloc(miniBatch * sizeof(int)); // What size should lengths be???
    float *h_data = (float*)malloc(numLayers * (seqLength + 1 * miniBatch) * hiddenSize * sizeof(float)); // State Accumulator
    float *c_data = (float*)malloc(numLayers * (seqLength + 1 * miniBatch) * hiddenSize * sizeof(float)); // Memory Accumulator

    // Workspace Variables
    float *tmp_h = (float*)malloc(miniBatch * (6 * hiddenSize) * sizeof(float));
    float *tmp_i = (float*)malloc(miniBatch * (6 * hiddenSize) * sizeof(float));
    // HARDCODED AHEAD
    float *T = (float*)malloc(2134 * sizeof(float)); // weight
    float *bias = (float*)malloc(110 * sizeof(float));
    float *linearGates; // only used during training
    float *dropoutMask = (float*)malloc(numLayers * numElements * sizeof(float)); // what size should dropout mask be???
    srand(1);
    for(int i = 0; i < sizeof(x) / sizeof(x[0]); i++) x[i] = rand() % 10000;
    for(int i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) lengths[i] = miniBatch + i;
    for(int i = 0; i < sizeof(h_data) / sizeof(h_data[0]); i++) h_data[i] = 0;
    for(int i = 0; i < sizeof(c_data) / sizeof(c_data[0]); i++) c_data[i] = 0;
    for(int i = 0; i < sizeof(T) / sizeof(T[0]); i++) T[i] = rand() % 10000;
    for(int i = 0; i < sizeof(bias) / sizeof(bias[0]); i++) bias[i] = rand() % 10000;
    for(int i = 0; i < sizeof(dropoutMask) / sizeof(dropoutMask[0]); i++) dropoutMask[i] = rand() % 10000;

    float* d_x;
    float* d_h_data;
    float* d_c_data;
    float* d_T;
    float* d_bias;
    float* d_dropout;
    float* d_tmp_h;
    float* d_tmp_i;
    // Input Data
    HIP_ASSERT(hipMalloc((void**)&d_x, inputSize * sizeof(float)));
    // Accumulator Data
    HIP_ASSERT(hipMalloc((void**)&d_h_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_c_data, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));

    HIP_ASSERT(hipMalloc((void**)&d_T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_bias, numLayers * hiddenSize * 8 * sizeof(float)));
    // Workspace Data
    HIP_ASSERT(hipMalloc((void**)&d_tmp_h, 4 * numLayers * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_tmp_i, 4 * seqLength * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_dropout, numLayers * numElements * sizeof(float)));

    HIP_ASSERT(hipMemcpyAsync(d_x, x, inputSize * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_h_data, h_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_c_data, c_data, (seqLength) * (numLayers + 1) * numElements * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_T, T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_bias, bias, numLayers * hiddenSize * 8 * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_tmp_h, tmp_h, 4 * numLayers * numElements * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_tmp_i, tmp_i, 4 * seqLength * numElements * sizeof(float), hipMemcpyHostToDevice, graph_stream));
    HIP_ASSERT(hipMemcpyAsync(d_dropout, dropoutMask, numLayers * numElements * sizeof(float), hipMemcpyHostToDevice, graph_stream));
/*
    if(TRAINING) {
        HIP_ASSERT(hipMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
    }
*/
    highway_lstm_forward_ongpu(
            inputSize, hiddenSize, batchSize, 
            numLayers, seqLength,
            d_x, lengths,
            d_h_data, d_c_data, 
            d_tmp_i, d_tmp_h, 
            d_T, d_bias,
            d_dropout, linearGates, 
            TRAINING, 
            graph_stream, 
            handle);
// Checksums + Outputs here
    printf("Original H Data: \n");
    for(int i = 0; i < 25; i++) {
        printf("%f ", h_data[i]);
    }
    printf("\nNew H Data: \n");
    HIP_ASSERT(hipMemcpyAsync(
                h_data, 
                d_h_data, 
                (seqLength + 1) * (numLayers) * numElements * sizeof(float), 
                hipMemcpyDeviceToHost, graph_stream));
    HIP_ASSERT(hipStreamSynchronize(graph_stream));
    for(int i = 0; i < 25; i++) {
        printf("%f ", d_h_data[i]);
    }
    return 0;
}
