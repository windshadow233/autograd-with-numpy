from ..nn.modules import Parameters


class SGD:
    def __init__(self, params: Parameters, lr, momentum=0.0, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        if momentum != 0.0:
            self.v = [0.0] * len(self.params)

    def step(self):
        if self.weight_decay > 0.0:
            for p in self.params:
                p.grad += self.weight_decay * p.data
        if self.momentum > 0.0:
            for i, p in enumerate(self.params):
                self.v[i] = self.momentum * self.v[i] + self.lr * p.grad.data
                p.data = p.data - self.v[i]
        else:
            for p in self.params:
                p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
