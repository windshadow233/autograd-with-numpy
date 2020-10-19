import numpy as np
from .tensor import array, Tensor


def zeros(shape, dtype=None, requires_grad=False):
    return array(np.zeros(shape=shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=None, requires_grad=False):
    return array(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def zeros_like(x: Tensor, dtype=None, requires_grad=False):
    return array(np.zeros_like(x.data), dtype=dtype, requires_grad=requires_grad)


def ones_like(x: Tensor, dtype=None, requires_grad=False):
    return array(np.ones_like(x.data), dtype=dtype, requires_grad=requires_grad)


def abs(x: Tensor):
    return x.abs()


def sqrt(x: Tensor):
    return x.sqrt()


def sin(x: Tensor):
    return x.sin()


def cos(x: Tensor):
    return x.cos()


def tan(x: Tensor):
    return x.tan()


def sinh(x: Tensor):
    return x.sinh()


def cosh(x: Tensor):
    return x.cosh()


def tanh(x: Tensor):
    return x.tanh()


def log(x: Tensor, base=None):
    return x.log(base)


def exp(x: Tensor):
    return x.exp()


def pow(x: Tensor, exponent):
    return x.pow(exponent)


def floor(x: Tensor):
    return x.floor()


def ceil(x: Tensor):
    return x.ceil()


def norm(x: Tensor, p=2.0):
    return x.norm(p)
