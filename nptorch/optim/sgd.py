import numpy as np
from ..nn.modules import Parameters


class SGD:
    def __init__(self, params: Parameters, lr, momentum=0., alpha=0., weight_decay=0.):
        """

        @param params: 需要优化的模型参数
        @param lr: 学习率
        @param momentum: 动量
        @param alpha: L1正则化系数
        @param weight_decay: L2正则化系数
        """
        if lr < 0.:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0. <= momentum < 1.:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if alpha < 0.:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.weight_decay = weight_decay

        if momentum != 0.:
            self.v = [0.] * len(self.params)

    def step(self):
        if self.alpha > 0.:
            for p in self.params:
                p.grad += self.alpha * (2. * (p.data > 0.).astype(np.float32) - 1.)
        if self.weight_decay > 0.:
            for p in self.params:
                p.grad += self.weight_decay * p.data
        if self.momentum > 0.:
            for i, p in enumerate(self.params):
                self.v[i] = self.momentum * self.v[i] + self.lr * p.grad.data
                p.data = p.data - self.v[i]
        else:
            for p in self.params:
                p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
