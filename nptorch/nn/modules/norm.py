from nptorch.tensor import Tensor
from .module import Module
from ..functional import batch_norm


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.gamma = Tensor(1., requires_grad=affine)
        self.beta = Tensor(0., requires_grad=affine)
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
            return batch_norm(x, self.gamma, self.beta, batch_mean, batch_var, self.eps)
        return batch_norm(x, self.gamma, self.beta, self.running_mean, self.running_var, self.eps)


