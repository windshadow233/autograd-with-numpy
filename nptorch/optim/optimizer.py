class Optimizer(object):
    def __init__(self, params, lr=1e-3, alpha=0., weight_decay=0.):
        assert lr >= 0., f"Invalid learning rate: {lr}"
        assert alpha >= 0., f"Invalid alpha value: {alpha}"
        assert weight_decay >= 0., f"Invalid weight_decay value: {weight_decay}"
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
