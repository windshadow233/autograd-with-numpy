import nptorch
import numpy as np
from .tensor import array


def zeros(shape, dtype=None, requires_grad=False):
    return array(np.zeros(shape=shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=None, requires_grad=False):
    return array(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def zeros_like(x: nptorch.tensor.Tensor, dtype=None, requires_grad=False):
    return array(np.zeros_like(x.data), dtype=dtype, requires_grad=requires_grad)


def ones_like(x: nptorch.tensor.Tensor, dtype=None, requires_grad=False):
    return array(np.ones_like(x.data), dtype=dtype, requires_grad=requires_grad)


def abs(x: nptorch.tensor.Tensor):
    return x.abs()


def sqrt(x: nptorch.tensor.Tensor):
    return x.sqrt()


def sin(x: nptorch.tensor.Tensor):
    return x.sin()


def cos(x: nptorch.tensor.Tensor):
    return x.cos()


def tan(x: nptorch.tensor.Tensor):
    return x.tan()


def sinh(x: nptorch.tensor.Tensor):
    return x.sinh()


def cosh(x: nptorch.tensor.Tensor):
    return x.cosh()


def tanh(x: nptorch.tensor.Tensor):
    return x.tanh()


def log(x: nptorch.tensor.Tensor, base=None):
    return x.log(base)


def exp(x: nptorch.tensor.Tensor):
    return x.exp()


def pow(x: nptorch.tensor.Tensor, exponent):
    return x.pow(exponent)


def floor(x: nptorch.tensor.Tensor):
    return x.floor()


def ceil(x: nptorch.tensor.Tensor):
    return x.ceil()


def norm(x: nptorch.tensor.Tensor, p=2.0):
    return x.norm(p)
