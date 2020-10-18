import nptorch


class Module:
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        pass

    def parameters(self, is_first_call=True):
        params = [v for _, v in self.__dict__.items() if isinstance(v, nptorch.Tensor) and v.requires_grad]
        for _, v in self.__dict__.items():
            if isinstance(v, Module):
                params.extend(v.parameters(False))
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
