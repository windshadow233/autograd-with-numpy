import numpy as np
import math
from nptorch.broadcast import get_tile_dims


class BackwardFcn:
    def __init__(self):
        self.name = ''
        self.operands = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '<' + self.name + '>'

    def calculate_grad(self, grad, children, place):
        raise NotImplementedError


class CopyBackward(BackwardFcn):
    def __init__(self):
        super(CopyBackward, self).__init__()
        self.name = 'CopyBackward'

    def calculate_grad(self, grad, children, place):
        return grad


class SliceBackward(BackwardFcn):
    def __init__(self):
        super(SliceBackward, self).__init__()
        self.name = 'SliceBackward'

    def calculate_grad(self, grad, children, place):
        result = np.zeros_like(children[0][0].data)
        result[children[0][1]] = grad
        return result


class AddBackward(BackwardFcn):
    def __init__(self):
        super(AddBackward, self).__init__()
        self.name = 'AddBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        grad = np.ones_like(x) * grad
        if isinstance(a, (int, float)) or x.shape == a.data.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class NegBackward(BackwardFcn):
    def __init__(self):
        super(NegBackward, self).__init__()
        self.name = 'NegBackward'

    def calculate_grad(self, grad, children, place):
        return - np.ones_like(children[place][0]) * grad


class SubBackward(BackwardFcn):
    def __init__(self):
        super(SubBackward, self).__init__()
        self.name = 'SubBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        grad = (-2 * place + 1) * np.ones_like(children[place][0]) * grad
        if isinstance(a, (int, float)) or x.shape == a.data.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class MulBackward(BackwardFcn):
    def __init__(self):
        super(MulBackward, self).__init__()
        self.name = 'MulBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        if isinstance(a, (int, float)):
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
    def __init__(self):
        super(DivBackward, self).__init__()
        self.name = 'DivBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0]
        if place == 0:
            grad = grad / a
            if isinstance(a, (int, float)):
                return np.ones_like(x) * grad
            a = a.data
            if x.shape == a.shape:
                return grad
            x_tiles, _ = get_tile_dims(x, a)
            if x_tiles:
                grad = np.array(grad.sum(x_tiles))
            return grad.reshape(x.shape)
        grad = -a / x ** 2 * grad
        if isinstance(a, (int, float)):
            return grad
        a = a.data
        if x.shape == a.shape:
            return grad
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class FloordivBackward(BackwardFcn):
    def __init__(self):
        super(FloordivBackward, self).__init__()
        self.name = 'FloordivBackward'

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[place][0]) * grad


class RemainderBackward(BackwardFcn):
    def __init__(self):
        super(RemainderBackward, self).__init__()
        self.name = 'RemainderBackward'

    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        a = children[1][0]
        grad = np.ones_like(x) * grad
        if isinstance(a, (int, float)) or x.shape == a.data.shape:
            return grad
        a = a.data
        x_tiles, _ = get_tile_dims(x, a)
        if x_tiles:
            grad = np.array(grad.sum(x_tiles))
        return grad.reshape(x.shape)


class PowerBackward(BackwardFcn):
    def __init__(self):
        super(PowerBackward, self).__init__()
        self.name = 'PowerBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data.astype(float)
        a = children[1 - place][0]
        if place == 0:
            if isinstance(a, (int, float)):
                grad = a * children[0][1] / x * grad
                return grad
            a = a.data.astype(float)
            grad = a * children[0][1] / x * grad
            if x.shape == a.shape:
                return grad
            x_tile, _ = get_tile_dims(x, a)
            if x_tile:
                grad = grad.sum(x_tile)
            return grad.reshape(x.shape)
        if isinstance(a, (int, float)):
            grad = children[1][1] * np.log(a) * grad
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
    def __init__(self):
        super(SumBackward, self).__init__()
        self.name = 'SumBackward'

    def calculate_grad(self, grad, children, place):
        return np.ones_like(children[0][0]) * grad


class MaxBackward(BackwardFcn):
    def __init__(self):
        super(MaxBackward, self).__init__()
        self.name = 'MaxBackward'

    def calculate_grad(self, grad, children, place):
        values, max_values, axis, keepdims = children[0]
        if max_values.size == 1:
            return grad * (values.data == max_values).astype(int)
        shape = np.array(values.shape)
        tiles = np.ones_like(shape)
        tiles[axis] = shape[axis]
        if not keepdims:
            max_values = np.expand_dims(max_values, axis)
            grad = np.expand_dims(grad, axis)
        max_values = np.tile(max_values, tiles)
        return grad * (values.data == max_values).astype(int)


class MinBackward(BackwardFcn):
    def __init__(self):
        super(MinBackward, self).__init__()
        self.name = 'MinBackward'

    def calculate_grad(self, grad, children, place):
        values, min_values, axis, keepdims = children[0]
        if min_values.size == 1:
            return grad * (values.data == min_values).astype(int)
        shape = np.array(values.shape)
        tiles = np.ones_like(shape)
        tiles[axis] = shape[axis]
        if not keepdims:
            min_values = np.expand_dims(min_values, axis)
            grad = np.expand_dims(grad, axis)
        min_values = np.tile(min_values, tiles)
        return grad * (values.data == min_values).astype(int)


class MeanBackward(BackwardFcn):
    def __init__(self):
        super(MeanBackward, self).__init__()
        self.name = 'MeanBackward'

    def calculate_grad(self, grad, children, place):
        x, axis, keepdims = children[0]
        if axis is None:
            return grad * np.ones_like(x.data) / x.data.size
        if not keepdims:
            grad = np.expand_dims(grad, axis)
        tiles = np.ones_like(np.array(x.shape))
        tiles[axis] = x.data.shape[axis]
        grad = np.tile(grad, tiles)
        grad = grad / x.data.shape[axis]
        return grad


class AbsBackward(BackwardFcn):
    def __init__(self):
        super(AbsBackward, self).__init__()
        self.name = 'AbsBackward'

    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        return (2 * (x >= 0).astype(float) - 1) * grad


class TBackward(BackwardFcn):
    def __init__(self):
        super(TBackward, self).__init__()
        self.name = 'TBackward'

    def calculate_grad(self, grad, children, place):
        return grad.T


class FillBackward(BackwardFcn):
    def __init__(self):
        super(FillBackward, self).__init__()
        self.name = 'FillBackward'

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0])


class TransposeBackward(BackwardFcn):
    def __init__(self):
        super(TransposeBackward, self).__init__()
        self.name = 'TransposeBackward'

    def calculate_grad(self, grad, children, place):
        axes = children[0][1]
        inverse_axes = np.arange(len(axes))[np.argsort(axes)]
        result = grad.transpose(inverse_axes)
        return result


class ReshapeBackward(BackwardFcn):
    def __init__(self):
        super(ReshapeBackward, self).__init__()
        self.name = 'ReshapeBackward'

    def calculate_grad(self, grad, children, place):
        return grad.reshape(children[0][1])


class SqrtBackward(BackwardFcn):
    def __init__(self):
        super(SqrtBackward, self).__init__()
        self.name = 'SqrtBackward'

    def calculate_grad(self, grad, children, place):
        result = 0.5 * grad / children[0][1]
        return result


class DotBackward(BackwardFcn):
    def __init__(self):
        super(DotBackward, self).__init__()
        self.name = 'DotBackward'

    def calculate_grad(self, grad, children, place):
        a = children[1 - place][0].data
        return grad * a


class MvBackward(BackwardFcn):
    def __init__(self):
        super(MvBackward, self).__init__()
        self.name = 'MvBackward'

    def calculate_grad(self, grad, children, place):
        x = children[place][0].data
        a = children[1 - place][0].data
        grad = np.expand_dims(grad, -1)
        if x.ndim == 1:
            grad = np.matmul(a.swapaxes(-1, -2), grad)
            grad = np.sum(grad, tuple(np.arange(a.ndim - 2))).reshape(x.shape)
            return grad
        grad = np.matmul(grad, np.expand_dims(a, 0))
        return grad


class MmBackward(BackwardFcn):
    def __init__(self):
        super(MmBackward, self).__init__()
        self.name = 'MmBackward'

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
        self.name = 'SinBackward'

    def calculate_grad(self, grad, children, place):
        return np.cos(children[0][0].data) * grad


class CosBackward(BackwardFcn):
    def __init__(self):
        super(CosBackward, self).__init__()
        self.name = 'CosBackward'

    def calculate_grad(self, grad, children, place):
        return - np.sin(children[0][0].data) * grad


class TanBackward(BackwardFcn):
    def __init__(self):
        super(TanBackward, self).__init__()
        self.name = 'TanBackward'

    def calculate_grad(self, grad, children, place):
        return grad * (1.0 + children[0][1] ** 2)


class SinhBackward(BackwardFcn):
    def __init__(self):
        super(SinhBackward, self).__init__()
        self.name = 'SinhBackward'

    def calculate_grad(self, grad, children, place):
        return grad * np.cosh(children[0][0].data)


class CoshBackward(BackwardFcn):
    def __init__(self):
        super(CoshBackward, self).__init__()
        self.name = 'CoshBackward'

    def calculate_grad(self, grad, children, place):
        return grad * np.sinh(children[0][0].data)


class TanhBackward(BackwardFcn):
    def __init__(self):
        super(TanhBackward, self).__init__()
        self.name = 'TanhBackward'

    def calculate_grad(self, grad, children, place):
        return grad * (1.0 - children[0][1] ** 2)


class LogBackward(BackwardFcn):
    def __init__(self):
        super(LogBackward, self).__init__()
        self.name = 'LogBackward'

    def calculate_grad(self, grad, children, place):
        x = children[0][0].data
        base = children[0][1]
        return grad / (x * math.log(base))


class ExpBackward(BackwardFcn):
    def __init__(self):
        super(ExpBackward, self).__init__()
        self.name = 'ExpBackward'

    def calculate_grad(self, grad, children, place):
        return grad * children[0][1]


class ReluBackward(BackwardFcn):
    def __init__(self):
        super(ReluBackward, self).__init__()
        self.name = 'ReluBackward'

    def calculate_grad(self, grad, children, place):
        return grad * (children[0][0].data > 0).astype(np.float32)


class SigmoidBackward(BackwardFcn):
    def __init__(self):
        super(SigmoidBackward, self).__init__()
        self.name = 'SigmoidBackward'

    def calculate_grad(self, grad, children, place):
        y = children[0][1]
        return grad * y * (1 - y)


class SoftmaxBackward(BackwardFcn):
    def __init__(self):
        super(SoftmaxBackward, self).__init__()
        self.name = 'SoftmaxBackward'

    def calculate_grad(self, grad, children, place):
        _, dim, y = children[0]


class CrossEntropyBackward(BackwardFcn):
    def __init__(self):
        super(CrossEntropyBackward, self).__init__()
        self.name = 'CrossEntropyBackward'

    def calculate_grad(self, grad, children, place):
        x_softmax, target = children[0][1:]
        target = target.data
        return grad * (x_softmax - target) / target.shape[0]


class FloorBackward(BackwardFcn):
    def __init__(self):
        super(FloorBackward, self).__init__()
        self.name = 'FloorBackward'

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0])


class CeilBackward(BackwardFcn):
    def __init__(self):
        super(CeilBackward, self).__init__()
        self.name = 'CeilBackward'

    def calculate_grad(self, grad, children, place):
        return np.zeros_like(children[0][0])


class UniformBackward(BackwardFcn):
    def __init__(self):
        super(UniformBackward, self).__init__()
        self.name = 'UniformBackward'

    def calculate_grad(self, grad, children, place):
        return grad * np.ones_like(children[0][0])


class NormalBackward(BackwardFcn):
    def __init__(self):
        super(NormalBackward, self).__init__()
        self.name = 'NormalBackward'

    def calculate_grad(self, grad, children, place):
        return grad * np.ones_like(children[0][0])


class NormBackward(BackwardFcn):
    def __init__(self):
        super(NormBackward, self).__init__()
        self.name = 'NormBackward'

    def calculate_grad(self, grad, children, place):
        child, p, y = children[0]
        return grad * y * child.data ** (p - 1)
