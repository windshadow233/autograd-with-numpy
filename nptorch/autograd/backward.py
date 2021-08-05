import math
import numpy as np
from numbers import Number
from itertools import product
from .broadcast import get_tile_dims
from ..nn.conv_operations import padding_zeros, unwrap_padding, dilate, erode, \
    reverse_conv2d, as_strided


class BackwardFcn:
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '<' + self.__class__.__name__ + '>'

    def calculate_grad(self, grad, children, place):
        raise NotImplementedError


class CopyBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return grad


class SliceBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, slices = children[0]
        result = np.zeros_like(x.data)
        result[slices] = grad
        return result


class IndexBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, item = children[0]
        result_grad = np.zeros_like(x.data)
        if isinstance(item, list):
            item = (item,)
        item = list(item)
        try:
            first_index = (slice(None),) * [not isinstance(index, slice) for index in item].index(True)
        except ValueError:
            first_index = ()
        for i, index in enumerate(item):
            if not isinstance(index, (tuple, list)):
                item[i] = (index,)
        max_length = max([len(index) for index in item])
        for i, index in enumerate(item):
            if len(index) == 1:
                item[i] = index * max_length
        for i, index in enumerate(zip(*item)):
            slices = first_index + (i,)
            result_grad[index] += grad[slices]
        return result_grad


class AddBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        grad = np.ones_like(x) * grad
        if isinstance(a, Number) or x.shape == a.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class NegBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return - np.ones_like(children[place][0].data) * grad


class SubBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        grad = (-2. * place + 1.) * np.ones_like(x) * grad
        if isinstance(a, Number) or x.shape == a.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class MulBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        if isinstance(a, Number):
            return a * np.ones_like(x) * grad
        a = a.data
        grad = a * grad
        if x.shape == a.shape:
            return grad
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class DivBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        if place == 0:
            if isinstance(a, Number):
                return np.ones_like(x) * grad / a
            a = a.data
            grad = grad / a
            if x.shape == a.shape:
                return grad
            x_tiles, _ = get_tile_dims(x, a)
            if x_tiles:
                grad = np.array(grad.sum(x_tiles))
            return grad.reshape(x.shape)
        if isinstance(a, Number):
            return -a / x ** 2. * grad
        a = a.data
        grad = -a / x ** 2. * grad
        if x.shape == a.shape:
            return grad
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class FloordivBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[place][0].data) * grad


class RemainderBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        a = children[1][0]
        grad = np.ones_like(x) * grad
        if isinstance(a, Number) or x.shape == a.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class PowerBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[place][0].data.astype(float)
        a = children[1 - place][0]
        if place == 0:
            if isinstance(a, Number):
                grad = a * np.power(x, a - 1.) * grad
                return grad
            a = a.data.astype(float)
            grad = a * np.power(x, a - 1.) * grad
            if x.shape == a.shape:
                return grad
            x_tile, _ = get_tile_dims(x, a)
            if x_tile:
                grad = grad.sum(x_tile)
            return grad.reshape(x.shape)
        if isinstance(a, Number):
            grad = children[1][1] * math.log(a) * grad
            return grad
        a = a.data.astype(float)
        grad = children[1][1] * np.log(a) * grad
        if x.shape == a.shape:
            return grad
        x_tile, _ = get_tile_dims(x, a)
        if x_tile:
            grad = grad.sum(x_tile)
        return grad.reshape(x.shape)


class SumBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return np.ones_like(children[0][0].data) * grad


class MaxBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        values, max_values, axis, keepdims = children[0]
        if max_values.size == 1:
            grad = grad * (values.data == max_values).astype(int)
            return grad / np.sum(grad)
        shape = np.array(values.shape)
        tiles = np.ones_like(shape)
        tiles[axis] = shape[axis]
        if not keepdims:
            max_values = np.expand_dims(max_values, axis)
            grad = np.expand_dims(grad, axis)
        max_values = np.tile(max_values, tiles)
        grad = grad * (values.data == max_values).astype(int)
        return grad / np.sum(grad, axis, keepdims=True)


class MinBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        values, min_values, axis, keepdims = children[0]
        if min_values.size == 1:
            grad = grad * (values.data == min_values).astype(int)
            return grad / np.sum(grad)
        shape = np.array(values.shape)
        tiles = np.ones_like(shape)
        tiles[axis] = shape[axis]
        if not keepdims:
            min_values = np.expand_dims(min_values, axis)
            grad = np.expand_dims(grad, axis)
        min_values = np.tile(min_values, tiles)
        grad = grad * (values.data == min_values).astype(int)
        return grad / np.sum(grad, axis, keepdims=True)


class MeanBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, axis, keepdims = children[0]
        x = x.data
        if axis is None:
            return grad * np.ones_like(x) / x.size
        if not keepdims:
            grad = np.expand_dims(grad, axis)
        tiles = np.ones_like(np.array(x.shape))
        tiles[axis] = x.shape[axis]
        grad = np.tile(grad, tiles)
        grad = grad / x.shape[axis]
        return grad


class VarBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, axis, keepdims = children[0]
        x = x.data
        if axis is None:
            return grad * 2. * (x - np.mean(x)) / x.size
        if not keepdims:
            grad = np.expand_dims(grad, axis)
        tiles = np.ones_like(np.array(x.shape))
        tiles[axis] = x.shape[axis]
        grad = np.tile(grad, tiles)
        grad = grad * 2. * (x - np.mean(x, axis, keepdims=True)) / x.shape[axis]
        return grad


class AbsBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        return (2. * (x >= 0.).astype(float) - 1.) * (x != 0) * grad


class TBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return grad.T


class FillBackward(BackwardFcn):
    def __init__(self):
        super(FillBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0].data)


class TransposeBackward(BackwardFcn):
    def __init__(self):
        super(TransposeBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        axes = children[0][1]
        inverse_axes = np.arange(len(axes))[np.argsort(axes)]
        return grad.transpose(inverse_axes)


class SwapaxesBackward(BackwardFcn):
    def __init__(self):
        super(SwapaxesBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        axis1, axis2 = children[0][1:]
        return grad.swapaxes(axis1, axis2)


class ReshapeBackward(BackwardFcn):
    def __init__(self):
        super(ReshapeBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad.reshape(children[0][0].shape)


class RepeatBackward(BackwardFcn):
    def __init__(self):
        super(RepeatBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x, tile_shape = children[0]
        x_shape = x.shape
        if isinstance(tile_shape, int):
            tile_shape = tile_shape,
        if len(x_shape) > len(tile_shape):
            tile_shape = (1,) * (len(x_shape) - len(tile_shape)) + tile_shape
        grad_strides = grad.strides
        split_shape = tile_shape + x_shape
        mid_strides = [grad_strides[-i] * x_shape[-i] for i in range(x.ndim, 0, -1)]
        split_strides = (*grad_strides[:-x.ndim], *mid_strides, *grad_strides[-x.ndim:])
        result = as_strided(grad, split_shape, split_strides)
        sum_axes = tuple(range(result.ndim))[:result.ndim - x.ndim]
        return result.sum(axis=sum_axes)


class SqrtBackward(BackwardFcn):
    def __init__(self):
        super(SqrtBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        result = 0.5 * grad / children[0][1]
        return result


class OuterBackward(BackwardFcn):
    def __init__(self):
        super(OuterBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        a = children[1 - place][0].data
        return grad.dot(a)


class DotBackward(BackwardFcn):
    def __init__(self):
        super(DotBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        a = children[1 - place][0].data
        return grad * a


class MvBackward(BackwardFcn):
    def __init__(self):
        super(MvBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0].data
        if x.ndim == 1:
            if place == 1:
                grad = np.matmul(np.expand_dims(grad, -2), a)
            else:
                grad = np.matmul(a, np.expand_dims(grad, -1))
            grad = np.sum(grad, tuple(range(a.ndim - 2))).reshape(x.shape)
            return grad
        if place == 0:
            grad = np.matmul(grad[..., None], a[None])
        else:
            grad = np.matmul(a[..., None], np.expand_dims(grad, -2))
        return grad


class MmBackward(BackwardFcn):
    def __init__(self):
        super(MmBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0].data
        grad = np.matmul(grad, a.swapaxes(-1, -2)) if place == 0 else np.matmul(a.swapaxes(-1, -2), grad)
        if x.shape[:-2] == a.shape[:-2]:
            return grad
        x_tiles, _ = get_tile_dims(x[..., 0, 0], a[..., 0, 0])
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class SinBackward(BackwardFcn):
    def __init__(self):
        super(SinBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return np.cos(children[0][0].data) * grad


class CosBackward(BackwardFcn):
    def __init__(self):
        super(CosBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return - np.sin(children[0][0].data) * grad


class TanBackward(BackwardFcn):
    def __init__(self):
        super(TanBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * (1. + children[0][1] ** 2)


class SinhBackward(BackwardFcn):
    def __init__(self):
        super(SinhBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * np.cosh(children[0][0].data)


class CoshBackward(BackwardFcn):
    def __init__(self):
        super(CoshBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * np.sinh(children[0][0].data)


class TanhBackward(BackwardFcn):
    def __init__(self):
        super(TanhBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * (1. - children[0][1] ** 2)


class ASinBackward(BackwardFcn):
    def __init__(self):
        super(ASinBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad / np.sqrt(1. - children[0][0].data ** 2)


class ACosBackward(BackwardFcn):
    def __init__(self):
        super(ACosBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return - grad / np.sqrt(1. - children[0][0].data ** 2)


class ATanBackward(BackwardFcn):
    def __init__(self):
        super(ATanBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad / (1. + children[0][0].data ** 2)


class LogBackward(BackwardFcn):
    def __init__(self):
        super(LogBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        base = children[0][1]
        return grad / (x * math.log(base))


class ExpBackward(BackwardFcn):
    def __init__(self):
        super(ExpBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * children[0][1]


class ReluBackward(BackwardFcn):
    def __init__(self):
        super(ReluBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return grad * (children[0][0].data > 0.).astype(np.float32)


class SigmoidBackward(BackwardFcn):
    def __init__(self):
        super(SigmoidBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        y = children[0][1]
        return grad * y * (1. - y)


class SoftplusBackward(BackwardFcn):
    def __init__(self):
        super(SoftplusBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x, y = children[0]
        grad *= y / (1. + y) * (x.data < 0.) + 1. / (1. + y) * (x.data >= 0)
        return grad


class CrossEntropyBackward(BackwardFcn):
    def __init__(self):
        super(CrossEntropyBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        x_softmax, target = children[0][1:]
        return grad * (x_softmax - target) / len(target)


class NLLLossBackward(BackwardFcn):
    def __init__(self):
        super(NLLLossBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return - grad * children[0][1]


class FloorBackward(BackwardFcn):
    def __init__(self):
        super(FloorBackward, self).__init__()

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0].data)


class CeilBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0].data)


class UniformBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return grad * np.ones_like(children[0][0].data)


class NormalBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return grad * np.ones_like(children[0][0].data)


class NormBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, p, y, axis, keepdims = children[0]
        if not keepdims and axis is not None:
            grad = np.expand_dims(grad, axis)
            y = np.expand_dims(y, axis)
        return grad * y * x.data ** (p - 1.)


class Conv2dBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, padding = children[0][1:]
        if place == 0:
            dilated_weight, stride = children[1][1: 3]
            grad = dilate(grad, (stride[0] - 1, stride[1] - 1))
            delta_x_shape = x.shape[-2:]
            add_rows, add_cols = np.array(delta_x_shape) + dilated_weight.shape[-2:] - 1 - np.array(grad.shape[-2:])
            padding_x = np.floor(add_rows / 2).astype(int), np.ceil(add_rows / 2).astype(int)
            padding_y = np.floor(add_cols / 2).astype(int), np.ceil(add_cols / 2).astype(int)
            grad = padding_zeros(grad, (padding_x, padding_y))
            return unwrap_padding(reverse_conv2d(grad, dilated_weight, rotate=True, invert=False), padding)
        elif place == 1:
            stride, dilation = children[1][2:]
            grad = dilate(grad, (stride[0] - 1, stride[1] - 1))
            return erode(reverse_conv2d(x, grad, rotate=False, invert=True), dilation)
        else:
            return np.sum(grad, (0, -1, -2))


class MeanPool2dBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, kernel_size, stride = children[0]
        new_grad = np.zeros_like(x.data)
        grad = grad / (kernel_size[0] * kernel_size[1])
        B, C, H, W = grad.shape
        for b, c, h, w in product(range(B), range(C), range(H), range(W)):
            new_grad[b, c, h * stride[0]: h * stride[0] + kernel_size[0], w * stride[1]: w * stride[1] + kernel_size[1]]\
                += grad[b, c, h, w]
        return new_grad


class AvgPool2dBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, argmax, kernel_size, stride = children[0]
        new_grad = np.zeros_like(x.data)
        B, C, H, W = grad.shape
        for index, m in zip(product(range(B), range(C), range(H), range(W)), argmax):
            b, c, h, w = index
            mh, mw = m // kernel_size[1], m % kernel_size[1]
            new_grad[b, c, h * stride[0] + mh, w * stride[1] + mw] += grad[b, c, h, w]
        return new_grad


class BatchNormBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, x_hat, mean, var, eps = children[0]
        axis = (0, -1, -2)[:x.ndim - 1]
        gamma = children[1][0]
        grad_x_hat = grad * np.expand_dims(gamma.data, axis)
        if place == 0:
            n = grad.size / grad.shape[1]
            dx = n * grad_x_hat - np.sum(grad_x_hat, axis=axis, keepdims=True) - \
                x_hat * np.sum(x_hat * grad_x_hat, axis=axis, keepdims=True)
            dx = dx / (n * np.sqrt(var + eps))
            return dx
        elif place == 1:
            return np.sum(x_hat * grad, axis=axis)
        else:
            return np.sum(grad, axis=axis)


class LeakyReLUBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, leaky_rate = children[0]
        data = x.data
        return grad * ((data > 0.) + leaky_rate * (data <= 0.)).astype(np.float32)


class ELUBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, alpha = children[0]
        data = x.data
        return grad * ((data > 0.) + (data <= 0.) * alpha * np.exp(data)).astype(np.float32)


class StackBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        _, axis, i = children[place]
        slices = [slice(None)] * grad.ndim
        slices[axis] = i
        return grad[tuple(slices)]


class CatBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        _, axis, i = children[place]
        slices = [slice(None)] * grad.ndim
        slices[axis] = i
        return grad[tuple(slices)]


class PadSequenceBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, i, batch_first = children[place]
        length = x.shape[0]
        if batch_first:
            return grad[i, :length]
        else:
            return grad[:length, i]


class SortBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        _, sorted_indices, axis = children[0]
        unsorted_indices = np.argsort(sorted_indices, axis)
        shape = list(grad.shape)
        shape.pop(axis)
        iteration = [range(i) for i in shape]
        for index in product(*iteration):
            slices = list(index)
            if axis == -1:
                slices.append(slice(None))
            else:
                slices.insert(axis, slice(None))
            slices = tuple(slices)
            grad[slices] = grad[slices][unsorted_indices[slices]]
        return grad


class DiagBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        if grad.ndim == 2:
            k = children[0][1]
            return np.diag(grad, k=k)
        x, k = children[0]
        result = np.zeros_like(x.data)
        indices = np.arange(len(grad))
        if k >= 0:
            result[indices, indices + k] = grad
        else:
            result[indices - k, indices] = grad
        return result


class TraceBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        return np.eye(*children[0][0].shape) * grad


class ClampBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, min, max = children[0]
        data = x.data
        indices = np.where((data < min) | (data > max))
        grad[indices] = 0.
        return grad


class WhereBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, condition = children[place]
        x_tiles, _ = get_tile_dims(x.data, grad)
        grad = (grad * condition.astype(np.float32)).sum(x_tiles)
        return grad.reshape(x.shape)


class SplitBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, i, step, axis = children[0]
        result = np.zeros_like(x.data)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(i * step, (i + 1) * step, 1)
        result[tuple(slices)] = grad
        return result


class EmbeddingBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, index, padding_idx = children[0]
        result_grad = np.zeros_like(x.data)
        for i, idx in enumerate(index):
            result_grad[idx] += grad[i]
        if padding_idx is not None:
            result_grad[padding_idx] = 0.
        return result_grad


class FlipBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        axis = children[0][1]
        return np.flip(grad, axis)


class BinaryCrossEntropyBackward(BackwardFcn):
    def calculate_grad(self, grad, children, place):
        x, target = children[0]
        x = x.data
        return (x - target) / x / (1. - x) / target.size
