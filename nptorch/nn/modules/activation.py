from nptorch.tensor import Tensor
from ..functional import cross_entropy
from .module import Module


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


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor):
        return cross_entropy(x, target)
