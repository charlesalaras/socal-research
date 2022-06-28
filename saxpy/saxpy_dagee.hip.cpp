#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <iostream>

#include <hip/hip_runtime.h>
#include <dagee/ATMIalloc.h>
#include <dagee/ATMIdagExecutor.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define THREADS_PER_BLOCK 512

__global__ void multKernel(int n, int a, float * x) { // Buffer Manager?
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        x[i] = a * x[i];
    }
}

__global__ void addKernel(int n, float * __restrict__ x, float * __restrict__ y) { // Buffer Manager?
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(i < n) {
        y[i] = x[i] + y[i];
    }
}

void gpu_saxpy_dagee(float * x, float * y, int n, BufMgr &bufMgr) { // Buffer Manager?
    // if else for smaller sizes
    dim3 threadsPerBlock = (1024);
    dim3 blocks = ((n - 1) / THREADS_PER_BLOCK + 1);

    float empty[n] = {0};

    float *x_1 = bufMgr.makeSharedCopy(empty);
    float *y_1 = bufMgr.makeSharedCopy(empty);

    for(int i = 0; i < n; i++) {
        x_1[i] = x[i];
        y_1[i] = y[i];
    }

    auto *k_1 = bufMgr.makeSharedCopy(empty);
    auto *k_2 = bufMgr.makeSharedCopy(empty);

    using GpuExec = dagee::GpuExecutorAtmi;
    using DagExec = dagee::ATMIdagExecutor<GpuExec>;

    GpuExec gpuEx;
    DagExec dagEx(gpuEx);

    auto *dag = dagEx.makeDAG();

    auto multFunc = gpuEx.registerKernel<int, int, float *>(&multKernel);
    auto addFunc = gpuEx.registerKernel<int, int, float *>(&addKernel);

    auto M_Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, multFunc, a, x_1, k_1, n);
    auto A_Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, x_1, y_1, k_2, n);

    dag->addEdge(M_Task, A_Task);

    dagEx.execute(dag);
    float *temp;
    temp = hipHostMalloc(&temp, n*sizeof(float));

    for(auto i = 0; i < n; i++) {
        temp[i] = y[i];
    }
}

int main(int argc, char * argv[]) {
    int n = (argc < 2) ? 8 : atoi(argv[1]);

    float *x;
    float *y;

    for(int i = 0; i < n; i++) {
        x[i] = rand() % 10;
        y[i] = rand() % 10;
    }

    dagee::AllocManagerAtmi bufMgr;
    auto *d_x = bufMgr.makeSharedCopy(x);
    auto *d_y = bufMgr.makeSharedCopy(y);

    gpu_saxpy_dagee(d_x, d_y, n, bufMgr);

    std::cout << "Done" << std::endl;

    return 0;
}
