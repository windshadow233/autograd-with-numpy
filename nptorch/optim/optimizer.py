import numpy as np
from ..nn.modules import Module


class Optimizer(object):
    def __init__(self, params: Module.parameters, lr=1e-3, alpha=0., weight_decay=0.):
        assert lr >= 0., f"Invalid learning rate: {lr}"
        assert alpha >= 0., f"Invalid alpha value: {alpha}"
        assert weight_decay >= 0., f"Invalid weight_decay value: {weight_decay}"
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay

    def _regularization(self):
        if self.alpha > 0.:
            for p in self.params:
                p.grad += self.alpha * (2. * (p.data > 0.).astype(np.float32) - 1.)
        if self.weight_decay > 0.:
            for p in self.params:
                p.grad += self.weight_decay * p.data

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
