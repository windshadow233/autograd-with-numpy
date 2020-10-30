import numpy as np
from nptorch.tensor import Tensor
from nptorch.random import uniform
from .. import functional as F
from nptorch.functional import zeros, stack
from ..parameter import Parameter
from .module import Module


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_bias=True, activation='tanh', batch_first=False, dropout=0.):
        """
        Simple RNN,输入数据形状默认为(L,B,D)
        @param input_size: 输入节点数量,即 'D'
        @param hidden_size: 隐藏层的节点数量
        @param num_layers: RNN层数,默认是1
        @param use_bias: 使用偏置
        @param activation: 激活函数,支持 'tanh' 与 'relu'
        @param batch_first: 若输入形状为(B,L,D),则将此项置为True
        @param dropout: 隐藏层节点的dropout率,默认为0
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'unsupported activation function {activation} for RNN')
        self.batch_first = batch_first
        assert 0. <= dropout <= 1., f'dropout probability has to be between 0 and 1, but got {dropout}'
        if dropout != 0. and self.num_layers == 1:
            raise UserWarning('dropout rate may be useless when num_layers is 1')
        self.dropout = dropout

        k = 1. / np.sqrt(hidden_size)
        self.weight_ih_l0 = Parameter(uniform((hidden_size, input_size), low=-k, high=k))
        self.weight_hh_l0 = Parameter(uniform((hidden_size, input_size), low=-k, high=k))
        if use_bias:
            self.bias_ih_l0 = Parameter(zeros(hidden_size))
            self.bias_hh_l0 = Parameter(zeros(hidden_size))
        for i in range(1, num_layers):
            self.__setattr__(f'weight_ih_l{i}', Parameter(uniform((hidden_size, input_size), low=-k, high=k)))
            self.__setattr__(f'weight_hh_l{i}', Parameter(uniform((hidden_size, input_size), low=-k, high=k)))
            if use_bias:
                self.__setattr__(f'bias_ih_l{i}', Parameter(zeros(hidden_size)))
                self.__setattr__(f'bias_hh_l{i}', Parameter(zeros(hidden_size)))

    def forward(self, x: Tensor):
        """

        @param x: (L, B, D)
        @return:
        """
        cache = [None] * self.num_layers
        output = []
        if self.batch_first:
            x = x.swapaxes(0, 1)  # (B, L, D) => (L, B, D)
        for t, xt in enumerate(x):
            hidden = xt
            for i in range(self.num_layers):
                weight_ih = self.__getattribute__(f'weight_ih_l{i}')
                weight_hh = self.__getattribute__(f'weight_hh_l{i}')
                hidden = hidden.matmul(weight_ih.T)
                if self.use_bias:
                    hidden += self.__getattribute__(f'bias_ih_l{i}')
                if t > 0:
                    hidden += cache[i].matmul(weight_hh.T)
                if self.use_bias:
                    hidden += self.__getattribute__(f'bias_hh_l{i}')
                hidden = self.activation(hidden)
                cache[i] = hidden
                if self.dropout > 0. and i < self.num_layers - 1:
                    hidden = F.dropout(hidden, self.dropout, self.training)
            output.append(hidden)
        output = stack(output)
        hidden = stack(cache)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden
