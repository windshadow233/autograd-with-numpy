import nptorch
from nptorch.tensor import *
from ..parameter import Parameter
from nptorch.random import normal
from .. import functional as F
from .module import Module


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias):
        super(_ConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},' \
               f' stride={self.stride}, padding={self.padding}, use_bias={self.use_bias}'

    def forward(self, *args):
        raise NotImplementedError


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), use_bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, use_bias)
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
        if not isinstance(padding, tuple):
            raise TypeError(f"padding must be a 'tuple', got '{type(padding)}'")
        if len(padding) != 2 or padding[0] < 0 or padding[1] < 0:
            raise ValueError(f"Invalid padding value: {padding}")
        n = out_channels * self.kernel_size ** 2
        self.kernels = Parameter(
            normal((out_channels, in_channels, self.kernel_size, self.kernel_size), mean=0., std=np.sqrt(2. / n)))
        if self.use_bias:
            self.bias = Parameter(nptorch.zeros(out_channels))

    def forward(self, x: Tensor):
        b, c, h, w = x.shape
        oc, ic, kh, kw = self.kernels.shape
        assert c == ic, 'Conv2d channels not equal'
        if self.use_bias:
            result = F.conv2d(x, self.kernels, self.bias, stride=self.stride, padding=self.padding)
        else:
            return F.conv2d(x, self.kernels, stride=self.stride, padding=self.padding)
        return result
