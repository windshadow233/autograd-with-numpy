import numpy as np
import nptorch
from nptorch.tensor import Tensor
from nptorch.random import normal
from ..functional import conv
from .module import Module


class Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0, 0), use_bias=True):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        n = out_channels * kernel_size ** 2
        self.kernels = normal((out_channels, in_channels, kernel_size, kernel_size), mean=0., std=np.sqrt(2 / n),
                              requires_grad=True)
        if self.use_bias:
            self.bias = nptorch.zeros(out_channels, requires_grad=True)

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels},' \
               f' kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, use_bias={self.use_bias}'

    def forward(self, x: Tensor):
        b, c, h, w = x.shape
        oc, ic, kh, kw = self.kernels.shape
        assert c == ic, 'Conv channels not equal'
        if self.use_bias:
            result = conv(x, self.kernels, self.bias, stride=self.stride, padding=self.padding)
        else:
            return conv(x, self.kernels, stride=self.stride, padding=self.padding)
        return result
