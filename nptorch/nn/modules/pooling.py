from nptorch.tensor import *
from .. import functional as F
from .module import Module


class _PoolNd(Module):
    def __init__(self, kernel_size, stride=None):
        super(_PoolNd, self).__init__()
        self.stride = stride or kernel_size
        self.kernel_size = kernel_size

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}'

    def forward(self, *args):
        raise NotImplementedError


class MeanPool2d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MeanPool2d, self).__init__(kernel_size, stride)

    def forward(self, x: Tensor):
        return F.mean_pool2d(x, self.kernel_size, self.stride)


class MaxPool2d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool2d, self).__init__(kernel_size, stride)

    def forward(self, x: Tensor):
        return F.max_pool2d(x, self.kernel_size, self.stride)