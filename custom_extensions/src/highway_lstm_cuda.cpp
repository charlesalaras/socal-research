//#include <THC/THC.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "highway_lstm_kernel.h"

//extern THCState *state;

int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch,
        int numLayers, int seqLength,
        Tensor &x,
        Tensor &lengths,
        Tensor &h_data,
        Tensor &c_data,
        Tensor &tmp_i,
        Tensor &tmp_h,
        Tensor &T,
        Tensor &bias,
        Tensor &dropout,
        Tensor &gates,
        int isTraining) {

    float * x_ptr = x.data_ptr();
    int * lengths_ptr = lengths.data_ptr();
    float * h_data_ptr = h_data.data_ptr();
    float * c_data_ptr = c_data.data_ptr();
    float * tmp_i_ptr = tmp_i.data_ptr();
    float * tmp_h_ptr = tmp_h.data_ptr());
    float * T_ptr = T.data_ptr());
    float * bias_ptr = bias.data_ptr();
    float * dropout_ptr = dropout.data_ptr();
    float * gates_ptr = NULL;
    if (isTraining == 1) {
        gates_ptr = THCudaTensor_data(state, gates);
    } else {
        gates_ptr = NULL;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    highway_lstm_forward_ongpu(inputSize, hiddenSize, miniBatch, numLayers, 
            seqLength, x_ptr, lengths_ptr, h_data_ptr, c_data_ptr, tmp_i_ptr,
            tmp_h_ptr, T_ptr, bias_ptr, dropout_ptr, gates_ptr,
            isTraining, stream, handle);

    return 1;

}

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength,
        Tensor &out_grad,
        Tensor &lengths,
        Tensor &h_data_grad,
        Tensor &c_data_grad,
        Tensor &x,
        Tensor &h_data,
        Tensor &c_data,
        Tensor &T,
        Tensor &gates_out,
        Tensor &dropout_in,
        Tensor &h_gates_grad,
        Tensor &i_gates_grad,
        Tensor &h_out_grad,
        Tensor &x_grad,
        Tensor &T_grad,
        Tensor &bias_grad,
        int isTraining,
        int do_weight_grad) {
	/*
    float * out_grad_ptr = THCudaTensor_data(state, out_grad);
    int * lengths_ptr = THIntTensor_data(lengths);
    float * h_data_grad_ptr = THCudaTensor_data(state, h_data_grad);
    float * c_data_grad_ptr = THCudaTensor_data(state, c_data_grad);
    float * x_ptr = THCudaTensor_data(state, x);
    float * h_data_ptr = THCudaTensor_data(state, h_data);
    float * c_data_ptr = THCudaTensor_data(state, c_data);
    float * T_ptr = THCudaTensor_data(state, T);
    float * gates_out_ptr = THCudaTensor_data(state, gates_out);
    float * dropout_in_ptr = THCudaTensor_data(state, dropout_in);
    float * h_gates_grad_ptr = THCudaTensor_data(state, h_gates_grad);
    float * i_gates_grad_ptr = THCudaTensor_data(state, i_gates_grad);
    float * h_out_grad_ptr = THCudaTensor_data(state, h_out_grad);
    float * x_grad_ptr = THCudaTensor_data(state, x_grad);
    float * T_grad_ptr = THCudaTensor_data(state, T_grad);
    float * bias_grad_ptr = THCudaTensor_data(state, bias_grad);

    cudaStream_t stream = THCState_getCurrentStream(state);
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);

    highway_lstm_backward_ongpu(inputSize, hiddenSize, miniBatch, numLayers,
            seqLength, out_grad_ptr, lengths_ptr, h_data_grad_ptr, c_data_grad_ptr,
            x_ptr, h_data_ptr, c_data_ptr, T_ptr, gates_out_ptr, dropout_in_ptr,
            h_gates_grad_ptr, i_gates_grad_ptr, h_out_grad_ptr,
            x_grad_ptr, T_grad_ptr, bias_grad_ptr, isTraining, do_weight_grad,
            stream, handle);
	*/
    return 1;

}
