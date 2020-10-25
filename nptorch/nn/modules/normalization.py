from nptorch.tensor import *
from ..parameter import Parameter
from .module import Module
from nptorch.functional import ones, zeros
from .. import functional as F


class BatchNorm2d(Module):
    """
    批归一化层
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        """

        @param num_features: 通道数
        @param eps: 运算稳定性因子,防止除以0
        @param momentum: 动量,用于更新统计特征
        @param affine: 是否训练线性映射系数
        """
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.gamma = Parameter(ones(num_features))
            self.beta = Parameter(zeros(num_features))
        else:
            self.gamma = Tensor(ones(num_features))
            self.beta = Tensor(zeros(num_features))

        self.running_mean = Tensor(0.)
        self.running_var = Tensor(1.)

    def extra_repr(self):
        return f'num_feature={self.num_features}, momentum={self.momentum}, affine={self.affine}'

    def forward(self, x: Tensor):
        """

        @param x: 四维张量
        @return: 批归一化结果
        """
        if self.training:
            batch_mean = Tensor(np.mean(x.data, axis=(0, -1, -2), keepdims=True))
            batch_var = Tensor(np.var(x.data, axis=(0, -1, -2), keepdims=True))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            return F.batch_norm2d(x, batch_mean, batch_var, self.gamma, self.beta, self.eps)
        return F.batch_norm2d(x, self.running_mean, self.running_var, self.gamma, self.beta, self.eps)


