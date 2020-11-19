from ..nn.modules import Module
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params: Module.parameters, lr=1e-3, momentum=0., alpha=0., weight_decay=0.):
        """
        SGD优化器
        @param params: 需要优化的模型参数
        @param lr: 学习率
        @param momentum: 动量
        @param alpha: L1正则化系数
        @param weight_decay: L2正则化系数
        """
        super(SGD, self).__init__(params, lr, alpha, weight_decay)
        assert 0. <= momentum < 1., f"Invalid momentum value: {momentum}"
        self.momentum = momentum

        if momentum != 0.:
            self.v = [0.] * len(self.params)

    def step(self):
        self._regularization()
        if self.momentum > 0.:
            for i, p in enumerate(self.params):
                self.v[i] = self.momentum * self.v[i] - self.lr * p.grad.data
                p.data += self.v[i]
        else:
            for p in self.params:
                p.data -= self.lr * p.grad.data
