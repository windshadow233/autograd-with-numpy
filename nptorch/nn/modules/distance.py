from nptorch.tensor import Tensor
from .. import functional as F
from .module import Module


class PairwiseDistance(Module):
    def __init__(self,  p=2., keepdims=False, eps=1e-12):
        super(PairwiseDistance, self).__init__()
        self.p = p
        self.keepdims = keepdims
        self.eps = eps

    def extra_repr(self):
        return f'p={self.p}'

    def forward(self, x1: Tensor, x2: Tensor):
        return F.pairwise_distance(x1, x2, p=self.p, keepdims=self.keepdims, eps=self.eps)


class CosineSimilarity(Module):
    def __init__(self, axis=1, eps=1e-12):
        super(CosineSimilarity, self).__init__()
        if not isinstance(axis, int):
            raise TypeError(f"argument 'axis' must be int. Got {type(axis)}")
        self.axis = axis
        self.eps = eps

    def extra_repr(self):
        return f'axis={self.axis}'

    def forward(self, x1: Tensor, x2: Tensor):
        return F.cosine_similarity(x1, x2, axis=self.axis, eps=self.eps)
