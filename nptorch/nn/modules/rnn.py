import numpy as np
from nptorch.tensor import Tensor
from nptorch.random import uniform
from .. import functional as F
from nptorch.functional import zeros, stack
from ..parameter import Parameter
from .module import Module


class RNNBase(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, use_bias=True, activation='tanh',
                 batch_first=False, dropout=0.):
        """
        RNN,输入数据形状默认为(L,B,D)
        @param input_size: 输入节点数量,即 'D'
        @param hidden_size: 隐藏层的节点数量
        @param num_layers: RNN层数,默认是1
        @param use_bias: 使用偏置
        @param activation: 激活函数,支持 'tanh' 与 'relu'
        @param batch_first: 若输入形状为(B,L,D),则将此项置为True
        @param dropout: 隐藏层节点的dropout率,默认为0
        """
        super(RNNBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        if activation == 'tanh':
            self.activation_fcn = F.tanh
        elif activation == 'relu':
            self.activation_fcn = F.relu
        else:
            raise ValueError(f'unsupported activation function {activation} for RNN')
        self.batch_first = batch_first
        assert 0. <= dropout <= 1., f'dropout probability has to be between 0 and 1, but got {dropout}'
        if dropout != 0. and self.num_layers == 1:
            raise UserWarning('dropout rate may be useless when num_layers is 1')
        self.dropout = dropout
        self._init_params()

    def extra_repr(self):
        return f'{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, use_bias={self.use_bias}, ' \
               f'\nactivation={self.activation_fcn.__name__}, batch_first={self.batch_first}, dropout={self.dropout}'

    def _init_params(self):
        k = 1. / np.sqrt(self.hidden_size)
        gate_size = {'RNN': 1, 'LSTM': 4}.get(self.__class__.__name__) * self.hidden_size
        for i in range(self.num_layers):
            ih_input_size = self.input_size if i == 0 else self.hidden_size
            self.__setattr__(f'weight_ih_l{i}',
                             Parameter(uniform(low=-k, high=k, size=(gate_size, ih_input_size))))
            self.__setattr__(f'weight_hh_l{i}',
                             Parameter(uniform(low=-k, high=k, size=(gate_size, self.hidden_size))))
            if self.use_bias:
                self.__setattr__(f'bias_ih_l{i}', Parameter(zeros(gate_size)))
                self.__setattr__(f'bias_hh_l{i}', Parameter(zeros(gate_size)))

    def forward(self, *args):
        raise NotImplementedError


class RNN(RNNBase):
    """
    Simple RNN
    """
    def __init__(self, input_size, hidden_size, num_layers=1, use_bias=True, activation='tanh',
                 batch_first=False, dropout=0.):
        super(RNN, self).__init__(input_size, hidden_size, num_layers, use_bias, activation, batch_first, dropout)

    def forward(self, x: Tensor, hidden: Tensor = None) -> (Tensor, Tensor):
        """
        @param x: (L, B, D)
        @param hidden: (num_layers, B, hidden_size) initial hidden value, default None
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)  # (B, L, D) => (L, B, D)
        hiddens = [zeros(x.shape[1], self.hidden_size)] * self.num_layers if hidden is None else list(hidden)
        output = []
        for t, xt in enumerate(x):
            hidden = xt
            for i in range(self.num_layers):
                weight_ih = self.__getattribute__(f'weight_ih_l{i}')
                weight_hh = self.__getattribute__(f'weight_hh_l{i}')
                hidden @= weight_ih.T
                hidden += hiddens[i].matmul(weight_hh.T)
                if self.use_bias:
                    hidden += self.__getattribute__(f'bias_ih_l{i}')
                    hidden += self.__getattribute__(f'bias_hh_l{i}')
                hidden = self.activation_fcn(hidden)
                hiddens[i] = hidden
                if self.dropout > 0. and i < self.num_layers - 1:
                    hidden = F.dropout(hidden, self.dropout, self.training)
            output.append(hidden)
        output = stack(output)
        hidden = stack(hiddens)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden


class LSTM(RNNBase):
    """
    Long Short-Term Memory
    """
    def __init__(self, input_size, hidden_size, num_layers=1, use_bias=True, activation='tanh',
                 batch_first=False, dropout=0.):
        super(LSTM, self).__init__(input_size, hidden_size, num_layers, use_bias, activation, batch_first, dropout)

    def forward(self, x: Tensor, initial=(None, None)) -> (Tensor, (Tensor, Tensor)):
        """
        @param x: (L, B, D)
        @param initial: Tuple (hidden, cache) initial hidden and cache value, default (None, None)
                if given, shape of each one is like: (num_layers, B, hidden_size)
        """
        if self.batch_first:
            x = x.swapaxes(0, 1)  # (B, L, D) => (L, B, D)
        hidden, cache = initial
        hiddens = [zeros(x.shape[1], self.hidden_size)] * self.num_layers if hidden is None else list(hidden)
        caches = [zeros(x.shape[1], self.hidden_size)] * self.num_layers if cache is None else list(cache)
        output = []
        for t, xt in enumerate(x):
            hidden = xt
            for i in range(self.num_layers):
                weight_ih = self.__getattribute__(f'weight_ih_l{i}')
                weight_hh = self.__getattribute__(f'weight_hh_l{i}')
                hidden @= weight_ih.T
                hidden += hiddens[i].matmul(weight_hh.T)
                if self.use_bias:
                    hidden += self.__getattribute__(f'bias_ih_l{i}')
                    hidden += self.__getattribute__(f'bias_hh_l{i}')
                it = hidden[:, : 4].sigmoid()
                ft = hidden[:, 4: 8].sigmoid()
                gt = self.activation_fcn(hidden[:, 8: 12])
                ot = hidden[:, 12: 16].sigmoid()
                ct = ft * caches[i] + it * gt
                hidden = ot * self.activation_fcn(ct)
                caches[i] = ct
                hiddens[i] = hidden
                if self.dropout > 0. and i < self.num_layers - 1:
                    hidden = F.dropout(hidden, self.dropout, self.training)
            output.append(hidden)
        output = stack(output)
        hidden = stack(hiddens)
        cache = stack(caches)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, (hidden, cache)

