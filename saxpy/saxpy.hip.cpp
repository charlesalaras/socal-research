#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "kernel.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


void gpu_saxpy(float alpha, float* __restrict__ h_x, float* __restrict__ h_y, float* __restrict__ d_x, float* __restrict__ d_y, int n) {
    if(n == 1) {
        h_y[0] = alpha * h_x[0] + h_y[0]; 
    }
    else if(n == 2) {
        h_y[0] = alpha * h_x[0] + h_y[0];
        h_y[1] = alpha * h_x[0] + h_y[0];
    }
    else {
        float* d_ax;
        HIP_ASSERT(hipHostMalloc(&d_ax, sizeof(float)*n));
        // X, Y, A*X
        hipStream_t streamForGraph;
        hipGraph_t graph;
        hipGraphExec_t graphExec;

        dim3 threadsPerBlock = (1024);
        dim3 blocks = ((n - 1) / 1024 + 1);

        HIP_ASSERT(hipStreamCreate(&streamForGraph));

        HIP_ASSERT(hipStreamBeginCapture(streamForGraph, hipStreamCaptureModeGlobal));

        HIP_ASSERT(hipMemcpyAsync(d_x, h_x, sizeof(float) * n, hipMemcpyHostToDevice, streamForGraph));

        hipLaunchKernelGGL(multKernel, blocks, threadsPerBlock, 0, streamForGraph, alpha, d_x, d_ax, n);
       
        HIP_ASSERT(hipMemcpyAsync(d_y, h_y, sizeof(float) * n, hipMemcpyHostToDevice, streamForGraph));

        HIP_ASSERT(hipStreamSynchronize(streamForGraph));

        hipLaunchKernelGGL(addKernel, blocks, threadsPerBlock, 0, streamForGraph, d_ax, d_y, n);

        HIP_ASSERT(hipMemcpyAsync(h_y, d_y, sizeof(float) * n, hipMemcpyDeviceToHost, streamForGraph));
        
        HIP_ASSERT(hipStreamSynchronize(streamForGraph));
        
        HIP_ASSERT(hipStreamEndCapture(streamForGraph, &graph));

        HIP_ASSERT(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
        if(!graphExec) {
            printf("ERROR! Could not make graph!");
            exit(0);
        }
        HIP_ASSERT(hipGraphLaunch(graphExec, streamForGraph));

        HIP_ASSERT(hipStreamSynchronize(streamForGraph));
/*
        float* A_2; // X
        HIP_ASSERT(hipHostMalloc(&A_2, sizeof(float)*n));
        for(int i = 0; i < n; i++) {
            A_2[i] = x[i];
        }
        float* A_3; // Y
        HIP_ASSERT(hipHostMalloc(&A_3, sizeof(float)*n));
        for(int i = 0; i < n; i++) {
            A_3[i] = y[i];
        }

        // B_1 = Alpha * A_2
        float* B_1;
        HIP_ASSERT(hipHostMalloc(&B_1, sizeof(float)*n));
        // C_1 = B_1 * A_3
        float* C_1;
        HIP_ASSERT(hipHostMalloc(&C_1, sizeof(float)*n));


        hipStream_t streamForGraph;
        hipGraph_t graph;
        hipGraphExec_t graphExec;

        hipGraphNode_t multKernelNode, addKernelNode;
        // Build Graph Manually
        HIP_ASSERT(hipGraphCreate(&graph, 0));

        hipKernelNodeParams multKernelParams = {0};
        hipKernelNodeParams addKernelParams = {0};
        void *multArgs[4] = {&alpha, (void *)&A_2, (void *)&B_1, &n};
        void *addArgs[4] = {(void *)&B_1, (void *)&A_3, (void *)&C_1, &n};

        multKernelParams.func = (void *)multKernel;
        multKernelParams.gridDim = blocks;
        multKernelParams.blockDim = threadsPerBlock;
        multKernelParams.sharedMemBytes = 0;
        multKernelParams.kernelParams = multArgs;
        multKernelParams.extra = NULL;

        HIP_ASSERT(hipGraphAddKernelNode(&multKernelNode, graph, NULL, 0, &multKernelParams));

        addKernelParams.func = (void *)addKernel;
        addKernelParams.gridDim = blocks;
        addKernelParams.blockDim = threadsPerBlock;
        addKernelParams.sharedMemBytes = 0;
        addKernelParams.kernelParams = addArgs;
        addKernelParams.extra = NULL;

        HIP_ASSERT(hipGraphAddKernelNode(&addKernelNode, graph, &multKernelNode, 1, &addKernelParams));

        // Create Executable Graph
        HIP_ASSERT(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

        // Launch Graph
        HIP_ASSERT(hipGraphLaunch(graphExec, streamForGraph));
        HIP_ASSERT(hipStreamSynchronize(streamForGraph));

        for(int i = 0; i < n; i++) {
            output[i] = C_1[i];            
        }
*/
        HIP_ASSERT(hipGraphExecDestroy(graphExec));
        HIP_ASSERT(hipGraphDestroy(graph));
        HIP_ASSERT(hipStreamDestroy(streamForGraph));
    }
}

int main(int argc, char * argv[]) {
    float a = (argc < 3) ? 0.5 : atof(argv[2]);
    int n = (argc < 4) ? 8 : atoi(argv[3]);

    float *h_x;
    float *h_y;
    float *d_x;
    float *d_y;

    for(int i = 0; i < n; i++) {
        h_x[i] = i;
        h_y[i] = i * 10.0f;
    }

    gpu_saxpy(a, d_x, d_y, h_x, h_y, n);    

    // Error Checking
    const float relativeTolerance = 1e-2;
    for(int i = 0; i < n; i++) {
        float difference = h_y[i] - (a * i + i * 10.0f);
        float relativeError = (difference - (a * i + i * 10.0f)) / difference;
        if(relativeError > relativeTolerance || -relativeError < -relativeTolerance) {
            printf("\n%d: TEST FAILED %f / %f\n\n", i, h_y[i], (a * i + i * 10.0f));
            exit(0);
        }
    }
    printf("\nTEST PASSED\n\n");

   
/*
    int n;
    float a;

    float* d_x;
    float* d_y;

    float* h_x;
    float* h_y;
    
    if(argc == 1) { return -1; }
    a = atof(argv[1]);

    n = argc > 2 ? size_t(atoi(argv[2])) : 1<<20;

    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));

    for(int i = 0; i < n; i++) {
        h_x[i] = (float)i;
        h_y[i] = (float)i*100.0f;
    }

    HIP_ASSERT(hipMalloc((void**)&d_x, n * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&d_y, n * sizeof(float)));

    HIP_ASSERT(hipMemcpy(d_x, h_x, n * sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_y, h_y, n * sizeof(float), hipMemcpyHostToDevice));


    hipLaunchKernelGGL(saxpy, 
            dim3((n-1) / THREADS_PER_BLOCK + 1, 1), 
            dim3(THREADS_PER_BLOCK, 1), 
            0, 0, 
            n, a, d_x, d_y);


    HIP_ASSERT(hipMemcpy(h_y, d_y, n * sizeof(float), hipMemcpyDeviceToHost));

    // Error Checking
    const float relativeTolerance = 1e-2;
    for(int i = 0; i < n; i++) {
        float difference = h_y[i] - (a * h_x[i]);
        float relativeError = (difference - (float)i*100.0f) / difference;
        if(relativeError > relativeTolerance || relativeError < -relativeTolerance) {
            printf("\n%d: TEST FAILED %f / %f\n\n", i, h_y[i], (i*100.0f + a * h_x[i]));
            exit(0);
        }
    }
    printf("\nTEST PASSED\n\n");

    hipFree(d_x);
    hipFree(d_y);

    free(h_x);
    free(h_y);
*/

    return 0;
}
