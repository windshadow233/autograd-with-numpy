from nptorch.tensor import Tensor
from nptorch.nn.functional import dropout
from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if not 0. <= p <= 1.:
            raise ValueError(f'dropout probability has to be between 0 and 1, but got {p}')
        self.p = p

    def __repr__(self):
        return f'Dropout(p={self.p})'

    def forward(self, x: Tensor):
        return dropout(x, self.p, self.training)

