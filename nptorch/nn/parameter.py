import numpy as np
from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, tensor: Tensor):
        super(Parameter, self).__init__(tensor.data, dtype=np.float32, requires_grad=True)

    def __repr__(self):
        return f'Parameter:\n{super(Parameter, self).__repr__()}'


class Parameters:
    def __init__(self, params):
        self.params = params
        self.number = len(params)

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        self._counter += 1
        if self._counter > self.number:
            raise StopIteration
        return self.params[self._counter - 1]

    def __len__(self):
        return self.number
