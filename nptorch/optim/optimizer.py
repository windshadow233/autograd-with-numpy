class Optimizer(object):
    def __init__(self, params, lr=1e-3, alpha=0., weight_decay=0.):
        if lr <= 0.:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()