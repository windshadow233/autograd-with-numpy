import numpy as np
from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, tensor: Tensor):
        if not tensor.is_leaf:
            raise ValueError('cannot assign a non-leaf variable to Parameter')
        super(Parameter, self).__init__(tensor.data, dtype=np.float32, requires_grad=True)

    def __repr__(self):
        return f'Parameter Containing:\n{super(Parameter, self).__repr__()}'
