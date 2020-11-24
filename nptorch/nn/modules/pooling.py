from nptorch.tensor import Tensor
from .. import functional as F
from .module import Module
from .utils import _pair


class _PoolNd(Module):
    def __init__(self, kernel_size, stride):
        super(_PoolNd, self).__init__()
        self.stride = stride or kernel_size
        self.kernel_size = kernel_size

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}'.format(**self.__dict__)

    def forward(self, *args) -> Tensor:
        raise NotImplementedError


class MeanPool2d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MeanPool2d, self).__init__(kernel_size, stride)
        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)

    def forward(self, x: Tensor) -> Tensor:
        return F.mean_pool2d(x, self.kernel_size, self.stride)


class MaxPool2d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool2d, self).__init__(kernel_size, stride)
        self.kernel_size = _pair(self.kernel_size)
        self.stride = _pair(self.stride)

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride)


class MeanPool1d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MeanPool1d, self).__init__(kernel_size, stride)
        assert isinstance(self.kernel_size, int) and self.kernel_size >= 1, f"Invalid stride value: {self.stride}"
        assert isinstance(self.stride, int) and self.stride >= 1, f"Invalid stride value: {self.stride}"

    def forward(self, x: Tensor) -> Tensor:
        return F.mean_pool1d(x, self.kernel_size, self.stride)


class MaxPool1d(_PoolNd):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool1d, self).__init__(kernel_size, stride)
        assert isinstance(self.kernel_size, int) and self.kernel_size >= 1, f"Invalid stride value: {self.stride}"
        assert isinstance(self.stride, int) and self.stride >= 1, f"Invalid stride value: {self.stride}"

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool1d(x, self.kernel_size, self.stride)
