import numpy as np
from ..nn.parameter import Parameters
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params: Parameters, lr=1e-3, momentum=0., alpha=0., weight_decay=0.):
        """
        随机梯度下降优化器
        @param params: 需要优化的模型参数
        @param lr: 学习率
        @param momentum: 动量
        @param alpha: L1正则化系数
        @param weight_decay: L2正则化系数
        """
        super(SGD, self).__init__(params, lr, alpha, weight_decay)
        if not 0. <= momentum < 1.:
            raise ValueError(f"Invalid momentum value: {momentum}")
        self.momentum = momentum

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
                self.v[i] = self.momentum * self.v[i] - self.lr * p.grad.data
                p.data = p.data + self.v[i]
        else:
            for p in self.params:
                p.data -= self.lr * p.grad.data
