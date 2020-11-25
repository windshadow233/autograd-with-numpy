import numpy as np
from nptorch.tensor import Tensor
from ..parameter import Parameter
from .module import Module
from nptorch.functional import ones, zeros
from .. import functional as F


class _BatchNormNd(Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats):
        """

        @param num_features: 通道数
        @param eps: 运算稳定性因子,防止除以0
        @param momentum: 动量,用于更新统计特征
        @param affine: 是否训练线性映射系数
        @param track_running_stats: 是否更新统计特征
        """
        super(_BatchNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.gamma = Parameter(ones(num_features))
            self.beta = Parameter(zeros(num_features))
        else:
            self.gamma = ones(num_features)
            self.beta = zeros(num_features)
        if track_running_stats:
            self.running_mean = Tensor(0.)
            self.running_var = Tensor(1.)

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, ' \
               'affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

    @staticmethod
    def _check_dim(x: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        self._check_dim(x)
        axis = (0, -1, -2)[:x.ndim - 1]
        if self.training:
            batch_mean = Tensor(np.mean(x.data, axis=axis, keepdims=True))
            batch_var = Tensor(np.var(x.data, axis=axis, keepdims=True))
            if self.track_running_stats:
                self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1. - self.momentum) * self.running_var + self.momentum * batch_var
            return F.batch_norm(x, batch_mean, batch_var, self.gamma, self.beta, self.eps)
        if self.track_running_stats:
            return F.batch_norm(x, self.running_mean, self.running_var, self.gamma, self.beta, self.eps)
        batch_mean = Tensor(np.mean(x.data, axis=axis, keepdims=True))
        batch_var = Tensor(np.var(x.data, axis=axis, keepdims=True))
        return F.batch_norm(x, batch_mean, batch_var, self.gamma, self.beta, self.eps)


class BatchNorm1d(_BatchNormNd):
    """
    1d批归一化层
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    @staticmethod
    def _check_dim(x: Tensor):
        assert x.ndim in {2, 3}, 'x must be 2 or 3 dimensional'


class BatchNorm2d(_BatchNormNd):
    """
    2d批归一化层
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    @staticmethod
    def _check_dim(x: Tensor):
        assert x.ndim == 4, 'x must be 4 dimensional'
