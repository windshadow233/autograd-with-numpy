from nptorch.tensor import Tensor


class Module:
    def __init__(self):
        self.training = True

    def __repr__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        self.train(False)

    def forward(self, *args):
        raise NotImplementedError

    def parameters(self, is_first_call=True):
        params = [v for _, v in self.__dict__.items() if isinstance(v, Tensor) and v.requires_grad]
        for module in self.children():
            params.extend(module.parameters(False))
        if is_first_call:
            return Parameters(params)
        else:
            return params


class Parameters:
    def __init__(self, params):
        self.params = params
        self.number = len(params)
        self.counter = 0

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        self.counter += 1
        if self.counter > self.number:
            raise StopIteration
        return self.params[self.counter - 1]

    def __len__(self):
        return self.number
