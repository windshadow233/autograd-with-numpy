from nptorch.tensor import Tensor
from .module import Module
from nptorch.random import normal
from ..parameter import Parameter
from .. import functional as F


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(normal(size=(num_embeddings, embedding_dim)))
        if padding_idx is not None:
            self.weight[padding_idx] = 0.

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        return s.format(**self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight, self.padding_idx)
