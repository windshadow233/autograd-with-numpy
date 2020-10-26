import numpy as np
from ..nn.parameter import Parameters
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params: Parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, alpha=0., weight_decay=0.):
        super(Adam, self).__init__(params, lr, alpha, weight_decay)
        if eps < 0.:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0. <= betas[0] < 1. or not 0. <= betas[1] < 1.:
            raise ValueError(f"Invalid betas value: {betas}")
        self.betas = betas
        self.eps = eps

        self.s = [0.] * len(self.params)
        self.r = [0.] * len(self.params)
        self.t = 0

    def step(self):
        self.t += 1
        if self.alpha > 0.:
            for p in self.params:
                p.grad += self.alpha * (2. * (p.data > 0.).astype(np.float32) - 1.)
        if self.weight_decay > 0.:
            for p in self.params:
                p.grad += self.weight_decay * p.data
        for i, p in enumerate(self.params):
            self.s[i] = self.betas[0] * self.s[i] + (1 - self.betas[0]) * p.grad.data
            self.r[i] = self.betas[1] * self.r[i] + (1 - self.betas[1]) * np.sum(p.grad.data ** 2)
            s = self.s[i] / (1 - self.betas[0] ** self.t)
            r = self.r[i] / (1 - self.betas[1] ** self.t)
            p.data -= self.lr * s / (np.sqrt(r) + self.eps)


