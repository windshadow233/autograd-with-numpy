from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.length = len(args)
        for i, module in enumerate(args):
            self.__setattr__(str(i), module)

    def __len__(self):
        return self.length

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

