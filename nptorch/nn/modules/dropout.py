from nptorch.tensor import Tensor
from .. import functional as F
from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if not 0. <= p <= 1.:
            raise ValueError(f'dropout probability has to be between 0 and 1, but got {p}')
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def forward(self, x: Tensor):
        return F.dropout(x, self.p, self.training)

