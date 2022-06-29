#ifndef __SAXPY_H__
#define __SAXPY_H__

__global__ void multKernel(int n, int a, float * x, dagee::AllocManagerAtmi &bufMgr) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        x[i] = a * x[i];
    }
}

__global__ void addKernel(int n, float * __restrict__ x, float * __restrict__ y, dagee::AllocManagerAtmi &bufMgr) {
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        y[i] = x[i] + y[i];
    }
}

#endif
