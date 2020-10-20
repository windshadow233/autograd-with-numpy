from nptorch.tensor import Tensor
from .module import Module


class MaxPool(Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool, self).__init__()
        self.stride = stride or kernel_size
        self.kernel_size = kernel_size

    def forward(self, x: Tensor):
        pass