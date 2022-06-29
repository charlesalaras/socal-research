#include <cmath>
#include <cstdlib>
#include <assert.h>
#include <iostream>

#include <hip/hip_runtime.h>
#include <dagee/ATMIalloc.h>
#include <dagee/ATMIdagExecutor.h>

#include "kernel.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


void gpu_saxpy_dagee(float alpha, float * x, float * y, int n, BufMgr &bufMgr) { // Buffer Manager?
if(n == 1) {
    y[0] = alpha * x[0] + y[0]; 
}
else if(n == 2) {
    y[0] = alpha * x[0] + y[0];
    y[1] = alpha * x[0] + y[0];
}
else {
    dim3 threadsPerBlock = (1024);
    dim3 blocks = ((n - 1) / THREADS_PER_BLOCK + 1);

    float empty[n] = {0};

    float *a_1 = bufMgr.makeSharedCopy(alpha);
    float *a_2 = bufMgr.makeSharedCopy(empty);
    float *a_3 = bufMgr.makeSharedCopy(empty);

    for(int i = 0; i < n; i++) {
        a_2[i] = x[i];
        a_3[i] = y[i];
    }

    float *b_1 = bufMgr.makeSharedCopy(empty);
    float *c_1 = bufMgr.makeSharedCopy(empty);

    using GpuExec = dagee::GpuExecutorAtmi;
    using DagExec = dagee::ATMIdagExecutor<GpuExec>;

    GpuExec gpuEx;
    DagExec dagEx(gpuEx);

    auto *dag = dagEx.makeDAG();

    auto multFunc = gpuEx.registerKernel(&multKernel);
    auto addFunc = gpuEx.registerKernel(&addKernel);

    auto B_1Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, multFunc, a_1, a_2, b_1, n);
    auto C_1Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, addFunc, b_1, a_3, c_1, n);

    dag->addEdge(M_Task, A_Task);

    dagEx.execute(dag);
    
    float *temp = HIP_ASSERT(hipHostMalloc(&temp, n*sizeof(float)));

    for(auto i = 0; i < n; i++) {
        temp[i] = c_1[i];
    }
    for(auto i = 0; i < n; i++) {
        y[i] = temp[i];
    }
}
}

int main(int argc, char * argv[]) {
    float a = (argc < 3) ? 0.5 : atof(argv[2]);
    int n = (argc < 4) ? 8 : atoi(argv[3]);

    float *h_x;
    float *h_y;

    for(int i = 0; i < n; i++) {
        h_x[i] = i;
        h_y[i] = i * 10.0f;
    }

    dagee::AllocManagerAtmi bufMgr;
    auto *x = bufMgr.makeSharedCopy(h_x);
    auto *y = bufMgr.makeSharedCopy(h_y);

    gpu_saxpy_dagee(a, x, y, n, bufMgr);

    std::cout << "Done" << std::endl;

    return 0;
}
