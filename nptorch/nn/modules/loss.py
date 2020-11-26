from nptorch.tensor import Tensor
from . import Module
from .. import functional as F


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(x, target)


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(x, target)


class NLLLoss(Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(x, target)


class BCELoss(Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return F.binary_cross_entropy(x, y)
