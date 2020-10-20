from nptorch.tensor import Tensor
from .module import Module
from ..functional import batch_norm


class BatchNorm(Module):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = Tensor(1., requires_grad=True)
            self.bias = Tensor(0., requires_grad=True)
            self.running_mean = Tensor(0.)
            self.running_var = Tensor(1.)

    def extra_repr(self):
        return f'in_channels={self.in_channels}, momentum={self.momentum}, affine={self.affine}'

    def forward(self, x: Tensor):
        batch_mean = x.mean(0)
        batch_var = x.var(0)
        if self.training and self.affine:
            self.running_mean = self.momentum * self.running_mean + (1. - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1. - self.momentum) * batch_var
            return batch_norm(x, self.weight, self.bias, batch_mean, batch_var, self.eps)
        else:
            if self.affine:
                return batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var, self.eps)
            else:
                return batch_norm(x, self.weight, self.bias, batch_mean, batch_var, self.eps)
