#ifndef __SAXPY_H__
#define __SAXPY_H__

__global__ void multKernel(float a, float * x, float * output, int n) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        output[i] = a * x[i];
    }
}

__global__ void addKernel(float * __restrict__ ax, float * __restrict__ y, int n) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        y[i] += ax[i];
    }
}

#endif
