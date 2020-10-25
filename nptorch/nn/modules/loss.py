from nptorch.tensor import *
from .module import Module
from .. import functional as F


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor):
        return F.cross_entropy(x, target)


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor):
        return F.mse_loss(x, target)


class NLLLoss(Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor):
        return F.nll_loss(x, target)
