import numpy as np
from nptorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, tensor: Tensor):
        super(Parameter, self).__init__(tensor.data, dtype=np.float32, requires_grad=True)

    def __repr__(self):
        return f'Parameter:\n{super(Parameter, self).__repr__()}'
