from .tensor import *


def eye(n, m, dtype=float32, requires_grad=False):
    return array(np.eye(n, m), dtype=dtype, requires_grad=requires_grad)


def zeros(shape, dtype=float32, requires_grad=False):
    return array(np.zeros(shape=shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=float32, requires_grad=False):
    return array(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def zeros_like(x: Tensor, dtype=float32, requires_grad=False):
    return array(np.zeros_like(x.data), dtype=dtype, requires_grad=requires_grad)


def ones_like(x: Tensor, dtype=float32, requires_grad=False):
    return array(np.ones_like(x.data), dtype=dtype, requires_grad=requires_grad)


def max(x: Tensor, axis=None, keepdims=False):
    return x.max(axis, keepdims)


def min(x: Tensor, axis=None, keepdims=False):
    return x.min(axis, keepdims)


def mean(x: Tensor, axis=None, keepdims=False):
    return x.mean(axis, keepdims)


def var(x: Tensor, axis=None, keepdims=False):
    return x.var(axis, keepdims)


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


def asin(x: Tensor):
    return x.asin()


def acos(x: Tensor):
    return x.acos()


def atan(x: Tensor):
    return x.atan()


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


def norm(x: Tensor, p=2.):
    return x.norm(p)
