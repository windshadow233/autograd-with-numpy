from nptorch.tensor import *
from .module import Module
from .. import functional as F


class Softmax(Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.softmax(self.dim)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: Tensor):
        return x.sigmoid()


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: Tensor):
        return x.relu()


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x: Tensor):
        return x.tanh()


class LeakyReLU(Module):
    def __init__(self, leaky_rate=0.01):
        super(LeakyReLU, self).__init__()
        self.leaky_rate = leaky_rate

    def extra_repr(self):
        return f'leaky_rate={self.leaky_rate}'

    def forward(self, x: Tensor):
        return F.leaky_relu(x, self.leaky_rate)


class ELU(Module):
    def __init__(self, alpha=1.):
        super(ELU, self).__init__()
        self.alpha = alpha

    def extra_repr(self):
        return f'alpha={self.alpha}'

    def forward(self, x: Tensor):
        return F.elu(x, self.alpha)
