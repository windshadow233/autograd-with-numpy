import nptorch
from nptorch.tensor import *
from ..parameter import Parameter
from nptorch.random import normal
from .. import functional as F
from .module import Module


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias):
        super(_ConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = use_bias

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},' \
               f' stride={self.stride}, padding={self.padding}, dilation={self.dilation}, use_bias={self.use_bias}'

    def forward(self, *args):
        raise NotImplementedError


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(0, 0), use_bias=True):
        """
        2维卷积,输入数据形状为(B,C,H,W)
        @param in_channels: 输入通道数
        @param out_channels: 输出通道数
        @param kernel_size: 卷积核尺寸
        @param stride: H,W方向的步长
        @param padding: H,W方向的padding
        @param dilation: 卷积核膨胀尺寸
        @param use_bias: 使用偏置
        """
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias)
        if not isinstance(stride, tuple):
            self.stride = (stride, stride)
        if not isinstance(padding, tuple):
            self.padding = (padding, padding)
        if not isinstance(dilation, tuple):
            self.padding = (dilation, dilation)
        assert len(padding) == 2 and padding[0] >= 0 and padding[1] >= 0, f"Invalid padding value: {padding}"
        n = out_channels * self.kernel_size ** 2
        self.kernels = Parameter(
            normal((out_channels, in_channels, self.kernel_size, self.kernel_size), mean=0., std=np.sqrt(2. / n)))
        if self.use_bias:
            self.bias = Parameter(nptorch.zeros(out_channels))

    def forward(self, x: Tensor):
        if self.use_bias:
            result = F.conv2d(x, self.kernels, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            return F.conv2d(x, self.kernels, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return result
