import numpy as np
from nptorch.tensor import Tensor
from ..parameter import Parameter
from .module import Module
from nptorch.random import normal
from nptorch.functional import zeros
from .. import functional as F


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(normal(mean=0., std=np.sqrt(2. / in_features), size=(out_features, in_features)))
        self.bias = Parameter(zeros(out_features)) if bias else None

    def extra_repr(self):
        return ('{in_features}, {out_features},' + f'bias={self.bias is not None}').format(**self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
