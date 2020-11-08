from .tensor import Tensor, float32
from .autograd.backward import StackBackward, CatBackward, SortBackward
from .return_types import *
import numpy as np


def eye(n, m, dtype=float32, requires_grad=False):
    return Tensor(np.eye(n, m), dtype=dtype, requires_grad=requires_grad)


def zeros(shape, dtype=float32, requires_grad=False):
    return Tensor(np.zeros(shape=shape), dtype=dtype, requires_grad=requires_grad)


def ones(shape, dtype=float32, requires_grad=False):
    return Tensor(np.ones(shape), dtype=dtype, requires_grad=requires_grad)


def zeros_like(x: Tensor, dtype=float32, requires_grad=False):
    return Tensor(np.zeros_like(x.data), dtype=dtype, requires_grad=requires_grad)


def ones_like(x: Tensor, dtype=float32, requires_grad=False):
    return Tensor(np.ones_like(x.data), dtype=dtype, requires_grad=requires_grad)


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


def abs_(x: Tensor):
    x.abs_()
    return x


def sqrt(x: Tensor):
    return x.sqrt()


def sqrt_(x: Tensor):
    x.sqrt_()
    return x


def sin(x: Tensor):
    return x.sin()


def sin_(x: Tensor):
    x.sin_()
    return x


def cos(x: Tensor):
    return x.cos()


def cos_(x: Tensor):
    x.cos_()
    return x


def tan(x: Tensor):
    return x.tan()


def tan_(x: Tensor):
    x.tan_()
    return x


def sinh(x: Tensor):
    return x.sinh()


def sinh_(x: Tensor):
    x.sinh_()
    return x


def cosh(x: Tensor):
    return x.cosh()


def cosh_(x: Tensor):
    x.cosh_()
    return x


def tanh(x: Tensor):
    return x.tanh()


def tanh_(x: Tensor):
    x.tanh_()
    return x


def asin(x: Tensor):
    return x.asin()


def asin_(x: Tensor):
    x.asin_()
    return x


def acos(x: Tensor):
    return x.acos()


def acos_(x: Tensor):
    x.acos_()
    return x


def atan(x: Tensor):
    return x.atan()


def atan_(x: Tensor):
    x.atan_()
    return x


def log(x: Tensor, base=None):
    return x.log(base)


def log_(x: Tensor, base=2.):
    x.log_(base=base)
    return x


def exp(x: Tensor):
    return x.exp()


def exp_(x: Tensor):
    x.exp_()
    return x


def pow(x: Tensor, exponent):
    return x.pow(exponent)


def floor(x: Tensor):
    return x.floor()


def floor_(x: Tensor):
    x.floor_()
    return x


def ceil(x: Tensor):
    return x.ceil()


def ceil_(x: Tensor):
    x.ceil_()
    return x


def reshape(x: Tensor, *shape):
    return x.reshape(shape)


def flatten(x: Tensor):
    return x.flatten()


def squeeze(x: Tensor, *axes):
    return x.squeeze(axes)


def unsqueeze(x: Tensor, *axes):
    return x.unsqueeze(axes)


def transpose(x: Tensor, *axes):
    return x.transpose(axes)


def swapaxes(x: Tensor, axis1, axis2):
    return x.swapaxes(axis1, axis2)


def sum(x: Tensor, axes=None, keepdims=False):
    return x.sum(axes, keepdims)


def norm(x: Tensor, p=2.):
    return x.norm(p)


def outer(x: Tensor, y: Tensor):
    return x.outer(y)


def matmul(x: Tensor, y: Tensor):
    return x.matmul(y)


def stack(tensors, axis=0):
    dtype = tensors[0].dtype
    arrays = []
    requires_grad = []
    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            raise RuntimeError('dtype of tensors are not same')
        arrays.append(tensor.data)
        if tensor.grad_enable:
            requires_grad.append((tensor, axis, i))
    result = np.stack(arrays, axis=axis)
    result = Tensor(result, requires_grad=bool(requires_grad))
    if result.grad_enable:
        result.children = requires_grad
        result.grad_fn = StackBackward()
    return result


def cat(tensors, axis=0):
    dtype = tensors[0].dtype
    arrays = []
    requires_grad = []
    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            raise RuntimeError('dtype of tensors are not same')
        arrays.append(tensor.data)
        if tensor.grad_enable:
            requires_grad.append((tensor, axis, i))
    result = np.concatenate(arrays, axis=axis)
    result = Tensor(result, requires_grad=bool(requires_grad))
    if result.grad_enable:
        result.children = requires_grad
        result.grad_fn = CatBackward()
    return result


def argsort(x: Tensor, axis=-1, descending=False):
    return x.argsort(axis, descending)


def sort(x: Tensor, axis=-1, descending=False):
    return x.sort(axis, descending)
