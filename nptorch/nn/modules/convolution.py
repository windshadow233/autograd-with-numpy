import numpy as np
import nptorch
from nptorch.tensor import Tensor
from ..parameter import Parameter
from nptorch.random import normal
from .. import functional as F
from .module import Module


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(_ConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def extra_repr(self):
        return ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride},'
                '\npadding={padding}, dilation={dilation}, bias=' + f'{self.bias is None}').format(**self.__dict__)

    def forward(self, *args) -> Tensor:
        raise NotImplementedError


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=0, bias=True):
        """
        1维卷积,输入数据形状为(B,C,W)
        @param in_channels: 输入通道数
        @param out_channels: 输出通道数
        @param kernel_size: 卷积核尺寸
        @param stride: 步长
        @param padding: padding
        @param dilation: 卷积核膨胀尺寸
        @param bias: 是否使用偏置
        """
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        assert isinstance(self.stride, int) and self.stride >= 1 or len(self.stride) == 1 and self.stride[0] >= 1, \
            f"Invalid stride value: {self.stride}"
        assert isinstance(self.padding, int) and self.stride >= 1 or len(self.padding) == 1 and self.padding[0] >= 0, \
            f"Invalid padding value: {self.padding}"
        assert isinstance(self.dilation, int) and self.dilation >= 0 or len(self.dilation) == 1 and self.dilation[0] >= 0, \
            f"Invalid dilation value: {self.dilation}"
        n = out_channels * self.kernel_size
        self.kernels = Parameter(
            normal(mean=0., std=np.sqrt(2. / n), size=(out_channels, in_channels, self.kernel_size)))
        self.bias = Parameter(nptorch.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.kernels, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(0, 0), bias=True):
        """
        2维卷积,输入数据形状为(B,C,H,W)
        @param in_channels: 输入通道数
        @param out_channels: 输出通道数
        @param kernel_size: 卷积核尺寸
        @param stride: H,W方向的步长
        @param padding: H,W方向的padding
        @param dilation: 卷积核膨胀尺寸
        @param bias: 是否使用偏置
        """
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        assert len(self.stride) == 2 and self.stride[0] >= 1 and self.stride[1] >= 1, \
            f"Invalid stride value: {self.stride}"
        assert len(self.padding) == 2 and self.padding[0] >= 0 and self.padding[1] >= 0, \
            f"Invalid padding value: {self.padding}"
        assert len(self.dilation) == 2 and self.dilation[0] >= 0 and self.dilation[1] >= 0, \
            f"Invalid padding value: {padding}"
        n = out_channels * self.kernel_size ** 2
        self.kernels = Parameter(
            normal(mean=0., std=np.sqrt(2. / n), size=(out_channels, in_channels, self.kernel_size, self.kernel_size)))
        self.bias = Parameter(nptorch.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.kernels, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
