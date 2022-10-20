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

__global__ void elementWise_fp(int hiddenSize, int miniBatch,
                               float *tmp_h,
                               float *tmp_i,
                               float *bias,
                               float *linearGates,
                               float *h_out,
                               float *i_out,
                               float *c_in,
                               float *c_out,
                               bool training) {
    int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int numElements = miniBatch * hiddenSize;

    if(index >= numElements) return;

    int batch = index / hiddenSize;
    int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

    float g[4];

    for(int i = 0; i < 4; i++) {
        g[i] = tmp_i[i * hiddenSize * gateIndex] + tmp_h[i * hiddenSize * gateIndex];
        g[i] = bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];
        //disregard training
    }

    float in_gate       = sigmoidf(g[0]);
    float forget_gate   = sigmoidf(g[1]);
    float act_gate      = tanhf(g[2]);
    float out_gate      = sigmoidf(g[3]);

    float val = (forget_gate * c_in[index]) + (in_gate * act_gate);

    c_out[index] = val;

    val = out_gate + tanhf(val);
    
    h_out[index] = val;
    i_out[index] = val;
}

void populateInputs(float* h_data, float* i_data, float* c_data, float* T, float* bias,
                    int seqLength, int hiddenSize, int numLayers, int numElements) {
    srand(0);
    h_data = (float*)malloc((seqLength + 1) * (numLayers) * numElements * sizeof(float));
    c_data = (float*)malloc((seqLength + 1) * (numLayers) * numElements * sizeof(float));
    i_data = (float*)malloc((seqLength) * (numLayers + 1) * numElements * sizeof(float));
    T = (float*)malloc(numLayers * hiddenSize * hiddenSize * 8 * sizeof(float));
    bias = (float*)malloc(numLayers * hiddenSize * 8 * sizeof(float));
    for(int i = 0; i < (seqLength + 1) * (numLayers) * numElements; i++) {
        h_data[i] = rand() % 1000;
    }
    for(int i = 0; i < (seqLength + 1) * (numLayers) * numElements; i++) {
        c_data[i] = rand() % 1000;
    }
    for(int i = 0; i < (seqLength) * (numLayers + 1) * numElements; i++) {
        i_data[i] = rand() % 1000;
    }
    for(int i = 0; i < numLayers * hiddenSize * hiddenSize * 8; i++) {
        T[i] = rand() % 1000;
    }
    for(int i = 0; i < numLayers * hiddenSize * 8; i++) {
        bias[i] = rand() % 1000;
    }
}

void lstm_ongpu(int hiddenSize, int miniBatch, int seqLength, int numLayers, hipStream_t stream) {
    float *h_data;
    float *i_data;
    float *c_data;
    
    float *T;
    float *T_f;
    
    float *bias;
    
    float *tmp_h;
    float *tmp_i;
    float *linearGates;

    hipStream_t stream_i;
    hipStream_t stream_h;

    HIP_ASSERT(hipStreamCreate(&stream_i));
    HIP_ASSERT(hipStreamCreate(&stream_h));

    rocblas_handle handle;
    rocblasAssert(rocblas_create_handle(&handle));

    // Input/output data
    int numElements = hiddenSize * miniBatch;

    populateInputs(h_data, i_data, c_data, T, bias, seqLength, hiddenSize, numLayers, numElements);

    HIP_ASSERT(hipMalloc((void**)&h_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&i_data, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&c_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    
    HIP_ASSERT(hipMalloc((void**)&T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&T_f, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));

    HIP_ASSERT(hipMalloc((void**)&bias, numLayers * hiddenSize * 8 * sizeof(float)));

    // Workspace
    HIP_ASSERT(hipMalloc((void**)&tmp_h, 4 * numLayers * numElements * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&tmp_i, 4 * seqLength * numElements * sizeof(float)));

    // Initialize with random values

    float alpha = 1.f;
    float beta = 0.f;

    T_f = T; // Non optimized
    
    int lStart = 0;
    int lEnd = 0;
    int rStart = 0;
    int rEnd = 0;

    int recurBatchSize = 1;

    while(true) {
        if(lEnd == 0) {
            lStart = 0;
            lEnd = 1;
            rStart = 0;
        }
        else {
            // Move "up" and "left"
            lStart++;
            lEnd++;

            rStart -= recurBatchSize;

            // Over the top or off the left, reset to layer 0
            if(lEnd > numLayers || rStart < 0) {
                rStart += (lStart + 1) * recurBatchSize;

                lStart = 0;
                lEnd = 1;
            }

            // Off the right, step up
            while(rStart >= seqLength && lEnd <= numLayers) {
                lStart++;
                lEnd++;

                rStart -= recurBatchSize;
            }


            // Over the top or off the left, done!
            if(lEnd > numLayers || rStart < 0) {
                break;
            }
        }

        rEnd = rStart + recurBatchSize;
        if(rEnd > seqLength) rEnd = seqLength;

        for(int layer = lStart; layer < lEnd; layer++) {
            rocblasAssert(rocblas_set_stream(handle, stream_i));

            rocblasAssert(rocblas_sgemm(handle,
                    rocblas_operation_none, rocblas_operation_none,
                    hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                    &alpha,
                    &T_f[layer * 8 * hiddenSize * hiddenSize],
                    4 * hiddenSize,
                    i_data + rStart * numElements + layer * seqLength * numElements,
                    hiddenSize,
                    &beta,
                    tmp_i + 4 * rStart * numElements,
                    4 * hiddenSize));
            for(int i = rStart; i < rEnd; i++) {
                rocblasAssert(rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        hiddenSize, miniBatch, hiddenSize,
                        &alpha,
                        &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize],
                        4 * hiddenSize,
                        h_data + i * numElements + layer * (seqLength + 1) * numElements,
                        hiddenSize,
                        &beta,
                        tmp_h + 4 * layer * numElements,
                        4 * hiddenSize));
                HIP_ASSERT(hipDeviceSynchronize());

                dim3 threadsPerBlock = (256);
                dim3 blocks = ((numElements * hiddenSize) + 256 - 1) / 256;

                hipLaunchKernelGGL(elementWise_fp, blocks, threadsPerBlock, 0, stream, 
                        hiddenSize, miniBatch,
                        tmp_h + 4 * layer * numElements,
                        tmp_i + 4 * i * numElements,
                        bias + 8 * layer * hiddenSize,
                        NULL,
                        h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                        i_data + i * numElements + (layer + 1) * seqLength * numElements,
                        c_data + i * numElements + layer * (seqLength + 1) * numElements,
                        c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                        false);
                HIP_ASSERT(hipDeviceSynchronize());
            }
        }
    }
    HIP_ASSERT(hipDeviceSynchronize());

    // Checksum / verification against test outputs
    
    // Cleanup
    HIP_ASSERT(hipFree(h_data));
    HIP_ASSERT(hipFree(i_data));
    HIP_ASSERT(hipFree(c_data));
    if(T != T_f) HIP_ASSERT(hipFree(T));
    HIP_ASSERT(hipFree(T_f));
    
    HIP_ASSERT(hipFree(bias));

    HIP_ASSERT(hipFree(tmp_h));
    HIP_ASSERT(hipFree(tmp_i));

    HIP_ASSERT(hipStreamDestroy(stream_i));
    HIP_ASSERT(hipStreamDestroy(stream_h));
    HIP_ASSERT(hipStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    int seqLength = 100;
    int numLayers = 1;
    int hiddenSize = 512;
    int miniBatch = 64;
    hipStream_t graph_stream;
    HIP_ASSERT(hipStreamCreate(&graph_stream));

    lstm_ongpu(hiddenSize, miniBatch, seqLength, numLayers, graph_stream);
}
