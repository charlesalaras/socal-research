int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength,
    Tensor &x, Tensor &lengths, Tensor &h_data,
    Tensor &c_data, Tensor &tmp_i,
    Tensor &tmp_h, Tensor &T, Tensor &bias,
    Tensor &dropout, Tensor &gates, int isTraining);

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch, 
        int numLayers, int seqLength, Tensor &out_grad, Tensor &lengths,
        Tensor &h_data_grad, Tensor &c_data_grad, Tensor &x, 
        Tensor &h_data, Tensor &c_data, Tensor &T,
        Tensor &gates_out, Tensor &dropout_in,
        Tensor &h_gates_grad, Tensor &i_gates_grad,
        Tensor &h_out_grad, Tensor &x_grad,  Tensor &T_grad,
        Tensor &bias_grad, int isTraining, int do_weight_grad);
