#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define THREADS_PER_BLOCK 512

__global__ void saxpy(int n, float a, float * __restrict__ x, float * __restrict__ y) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        y[i] = a*x[i] + y[i];
    }
}

int main(int argc, char * argv[]) {

    int n;
    float a;

    float* d_x;
    float* d_y;

    float* h_x;
    float* h_y;
    
    if(argc == 1) { return -1; }
    a = atof(argv[1]);

    n = argc > 2 ? atoi(argv[2]) : 1<<20;

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

    return 0;
}
