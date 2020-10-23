from nptorch.tensor import Tensor
from ..parameter import Parameter
from .module import Module
from .. import functional as F


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.gamma = Parameter(Tensor(1.))
            self.beta = Parameter(Tensor(0.))
        else:
            self.gamma = Tensor(1.)
            self.beta = Tensor(1.)

        self.running_mean = Tensor(0.)
        self.running_var = Tensor(1.)

    def extra_repr(self):
        return f'num_feature={self.num_features}, momentum={self.momentum}, affine={self.affine}'

    def forward(self, x: Tensor):
        if self.training:
            batch_mean = x.mean(0)
            batch_var = x.var(0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            return F.batch_norm(x, self.gamma, self.beta, batch_mean, batch_var, self.eps)
        return F.batch_norm(x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps)


