from nptorch.tensor import Tensor
from ..functional import max_pool, mean_pool
from .module import Module


class MeanPool(Module):
    def __init__(self, kernel_size, stride=None):
        super(MeanPool, self).__init__()
        self.stride = stride or kernel_size
        self.kernel_size = kernel_size

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}'

    def forward(self, x: Tensor):
        return mean_pool(x, self.kernel_size, self.stride)


class MaxPool(Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool, self).__init__()
        self.stride = stride or kernel_size
        self.stride = kernel_size
        self.kernel_size = kernel_size

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}'

    def forward(self, x: Tensor):
        return max_pool(x, self.kernel_size, self.stride)