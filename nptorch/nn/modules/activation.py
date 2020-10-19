from nptorch.tensor import Tensor
from ..functional import cross_entropy
from .module import Module


class Softmax(Module):
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __repr__(self):
        return f'Softmax(dim={self.dim})'

    def forward(self, x: Tensor):
        return x.softmax(self.dim)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def __repr__(self):
        return 'Sigmoid()'

    def forward(self, x: Tensor):
        return x.sigmoid()


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def __repr__(self):
        return 'ReLU()'

    def forward(self, x: Tensor):
        return x.relu()


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def __repr__(self):
        return 'Tanh()'

    def forward(self, x: Tensor):
        return x.tanh()


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def __repr__(self):
        return 'CrossEntropyLoss()'

    def forward(self, x: Tensor, target: Tensor):
        return cross_entropy(x, target)
