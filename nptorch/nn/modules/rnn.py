import numpy as np
from nptorch.tensor import Tensor
from nptorch.random import normal
from .. import functional as F
from nptorch.functional import zeros, stack
from ..parameter import Parameter
from .module import Module


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 use_bias=True, activation='tanh', batch_first=True, dropout=0):
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
        self.dropout = dropout

        self.weight_ih_l0 = Parameter(normal((hidden_size, input_size), mean=0., std=np.sqrt(2. / input_size)))
        self.weight_hh_l0 = Parameter(normal((hidden_size, hidden_size), mean=0., std=np.sqrt(2. / input_size)))
        if use_bias:
            self.bias_l0 = Parameter(zeros(hidden_size))
        for i in range(1, num_layers):
            self.__setattr__(f'weight_ih_l{i}',
                             Parameter(normal((hidden_size, hidden_size), mean=0., std=np.sqrt(2. / input_size))))
            self.__setattr__(f'weight_hh_l{i}',
                             Parameter(normal((hidden_size, hidden_size), mean=0., std=np.sqrt(2. / input_size))))
            if use_bias:
                self.__setattr__(f'bias_l{i}', Parameter(zeros(hidden_size)))

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
                if t > 0:
                    hidden += cache[i].matmul(weight_hh.T)
                if self.use_bias:
                    hidden += self.__getattribute__(f'bias_l{i}')
                hidden = self.activation(hidden)
                cache[i] = hidden
            output.append(hidden)
        output = stack(output)
        hidden = stack(cache)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden
