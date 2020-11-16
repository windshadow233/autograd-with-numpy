from nptorch.tensor import Tensor
from .. import functional as F
from .module import Module


class _DropoutNd(Module):
    def __init__(self, p):
        super(_DropoutNd, self).__init__()
        assert 0. <= p <= 1., f'dropout probability has to be between 0 and 1, but got {p}'
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def forward(self, *args) -> Tensor:
        raise NotImplementedError


class Dropout(_DropoutNd):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__(p=p)

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)


class Dropout2d(_DropoutNd):
    def __init__(self, p=0.5):
        super(Dropout2d, self).__init__(p=p)

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout2d(x, self.p, self.training)

