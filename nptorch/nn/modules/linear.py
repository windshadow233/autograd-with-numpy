import numpy as np
from nptorch.tensor import Tensor
from ..parameter import Parameter
from .module import Module
from nptorch.random import normal
from nptorch.functional import zeros
from .. import functional as F


class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = Parameter(normal((out_features, in_features), mean=0, std=np.sqrt(2 / in_features)))
        if use_bias:
            self.bias = Parameter(zeros(out_features))

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, use_bias={self.use_bias}'

    def forward(self, x: Tensor):
        if self.use_bias:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight)
