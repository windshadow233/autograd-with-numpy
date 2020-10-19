import numpy as np
from nptorch.tensor import Tensor
from .module import Module
from nptorch.random import normal
from nptorch.functional import zeros


class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = normal((out_features, in_features), mean=0, std=np.sqrt(2 / in_features), requires_grad=True)
        if use_bias:
            self.bias = zeros(out_features, requires_grad=True)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias}'

    def forward(self, x: Tensor):
        result = x.matmul(self.weight.T)
        if self.use_bias:
            result = result + self.bias
        return result
