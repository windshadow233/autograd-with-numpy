from copy import deepcopy, copy
from .backward import *

# np.set_printoptions(precision=4, suppress=True)


def array(data, dtype=None, requires_grad=False):
    return Tensor(data, dtype, requires_grad)


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False):
        if dtype is None:
            self.data = np.array(data)
            if self.dtype == np.float64:
                self.data = self.data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=dtype)
        if 'float' not in self.dtype.name and requires_grad:
            raise RuntimeError('Only Arrays of floating point dtype can require gradients')
        self.requires_grad = requires_grad
        self.grad_fn = None
        self.children = None
        self.grad = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = repr(self.data)[:-1]
        if 'dtype' not in s:
            s += f', dtype={self.data.dtype}'
        if self.requires_grad:
            if self.grad_fn:
                s += ', grad_fn=' + str(self.grad_fn) + ')'
            else:
                s += ', ' + 'requires_grad=True)'
        else:
            s += ')'
        return s

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[key] = value

    def __getitem__(self, item):
        result = Tensor(self.data[item], dtype=self.dtype, requires_grad=self.requires_grad)
        if not result.requires_grad:
            return result
        result.children = [(self, item)]
        result.grad_fn = SliceBackward()
        return result

    def __bool__(self):
        return self.all().item()

    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other, dtype=np.bool_)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other, dtype=np.bool_)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other, dtype=np.bool_)

    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other, dtype=np.bool_)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other, dtype=np.bool_)

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other, dtype=np.bool_)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        result = Tensor(self.data.T, dtype=self.dtype, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = TBackward()
            result.children = [(self, None)]
        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def is_leaf(self):
        return not self.grad_fn

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def strides(self):
        return self.data.strides

    def _check_inplace(self, other=None):
        if other is self:
            raise RuntimeError('prohibit doing inplace operations with self')
        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad has been used in an in-place operation.')

    def _check_type(self, operation, excludes=('int8', 'int16', 'int32', 'int64', 'bool')):
        if self.dtype.name in excludes:
            raise RuntimeError(f"'{operation}' not implemented for '{self.dtype.name}'")

    def half(self):
        result = Tensor(self.data.astype(np.float16), dtype=np.float16, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = CopyBackward()
        return result

    def float(self):
        result = Tensor(self.data.astype(np.float32), dtype=np.float32, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = CopyBackward()
        return result

    def double(self):
        result = Tensor(self.data.astype(np.float64), dtype=np.float64, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = CopyBackward()
        return result

    def char(self):
        return Tensor(self.data.astype(np.int8), dtype=np.int8)

    def short(self):
        return Tensor(self.data.astype(np.int16), dtype=np.int16)

    def int(self):
        return Tensor(self.data.astype(np.int32), dtype=np.int32)

    def long(self):
        return Tensor(self.data.astype(np.int64), dtype=np.int64)

    def bool(self):
        return Tensor(self.data.astype(np.bool), dtype=np.bool_)

    def all(self, axis=None, keepdims=False):
        return Tensor(self.data.all(axis=axis, keepdims=keepdims))

    def any(self, axis=None, keepdims=False):
        return Tensor(self.data.any(axis=axis, keepdims=keepdims))

    def item(self):
        return self.data.item()

    def argmax(self, axis=None):
        return Tensor(self.data.argmax(axis), dtype=np.int64)

    def argmin(self, axis=None):
        return Tensor(self.data.argmin(axis), dtype=np.int64)

    def detach(self):
        return Tensor(self.data)

    def numpy(self):
        return self.data

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data + other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data + other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = AddBackward()
        return result

    def __iadd__(self, other):
        self._check_inplace(other)
        if isinstance(other, (int, float)):
            y = self.data + other
        else:
            y = self.data + other.data
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = AddBackward()
        self.data = np.array(y)
        return self

    def __neg__(self):
        result = Tensor(- self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = NegBackward()
        return result

    def __rsub__(self, other):
        return - self.__sub__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data - other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data - other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = SubBackward()
        return result

    def __isub__(self, other):
        self._check_inplace(other)
        if isinstance(other, (int, float)):
            y = self.data - other
        else:
            y = self.data - other.data
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = SubBackward()
        self.data = np.array(y)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data * other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data * other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = MulBackward()
        return result

    def __imul__(self, other):
        self._check_inplace(other)
        if isinstance(other, (int, float)):
            y = self.data * other
        else:
            y = self.data * other.data
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = MulBackward()
        self.data = np.array(y)
        return self

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(other / self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(other.data / self.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(other, None), (self, None)]
            result.grad_fn = DivBackward()
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data / other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data / other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = DivBackward()
        return result

    def __itruediv__(self, other):
        self._check_inplace(other)
        if isinstance(other, (int, float)):
            y = self.data / other
        else:
            y = self.data / other.data
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = DivBackward()
        self.data = np.array(y)
        return self

    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(other // self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(other.data // self.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(other, None), (self, None)]
            result.grad_fn = FloordivBackward()
        return result

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            result = Tensor(self.data // other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data // other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = FloordivBackward()
        return result

    def __ifloordiv__(self, other):
        self._check_inplace(other)
        if isinstance(other, (int, float)):
            y = self.data // other
        else:
            y = self.data // other.data
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = FloordivBackward()
        self.data = np.array(y)
        return self

    def __mod__(self, other):
        if isinstance(other, Tensor):
            if other.requires_grad:
                raise RuntimeError("the derivative for 'other' is not implemented")
            other = other.data
        result = Tensor(self.data % other, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (other, None)]
            result.grad_fn = RemainderBackward()
        return result

    def __imod__(self, other):
        self._check_inplace(other)
        if isinstance(other, Tensor):
            if other.requires_grad:
                raise RuntimeError("the derivative for 'other' is not implemented")
            other = other.data
        y = self.data % other
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (other, None)]
            self.grad_fn = RemainderBackward()
        self.data = np.array(y)
        return self

    def __rpow__(self, power):
        if isinstance(power, (int, float)):
            y = np.power(power, self.data)
            result = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = np.power(power.data, self.data)
            result = Tensor(power, dtype=self.dtype, requires_grad=self.requires_grad or power.requires_grad)
        if result.requires_grad:
            result.children = [(power, None), (self, y)]
            result.grad_fn = PowerBackward()
        return result

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            y = np.power(self.data, power)
            result = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = np.power(self.data, power.data)
            result = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad or power.requires_grad)
        if result.requires_grad:
            result.children = [(self, None), (power, y)]
            result.grad_fn = PowerBackward()
        return result

    def __ipow__(self, power):
        self._check_inplace(power)
        if isinstance(power, (int, float)):
            y = np.power(self.data, power)
        else:
            y = np.power(self.data, power.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (power, y)]
            self.grad_fn = PowerBackward()
        self.data = np.array(y)
        return self

    def max(self, axis=None, keepdims=False):
        result = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        indices = Tensor(np.argmax(self.data, axis=axis), dtype=np.int64)
        if result.requires_grad:
            result.children = [(self, result.data, axis, keepdims)]
            result.grad_fn = MaxBackward()
        return {'values': result, 'indices': indices}

    def min(self, axis=None, keepdims=False):
        result = Tensor(np.min(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        indices = Tensor(np.argmin(self.data, axis=axis), dtype=np.int64)
        if result.requires_grad:
            result.children = [(self, result.data, axis, keepdims)]
            result.grad_fn = MinBackward()
        return {'values': result, 'indices': indices}

    def mean(self, axis=None, keepdims=False):
        result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, axis, keepdims)]
            result.grad_fn = MeanBackward()
        return result

    def var(self, axis=None, keepdims=False):
        result = Tensor(np.var(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        if not axis:
            result.data *= self.data.size / (self.data.size - 1)
        else:
            result.data *= self.data.shape[axis] / (self.data.shape[axis] - 1)
        if result.requires_grad:
            result.children = [(self, axis, keepdims)]
            result.grad_fn = VarBackward()
        return result

    def abs(self):
        y = Tensor(np.abs(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = AbsBackward()
        return y

    def abs_(self):
        self._check_inplace()
        y = np.abs(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = AbsBackward()
        self.data = np.array(y)

    def sqrt(self):
        self._check_type('sqrt')
        y = Tensor(np.sqrt(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, y.data)]
            y.grad_fn = SqrtBackward()
        return y

    def sqrt_(self):
        self._check_type('sqrt')
        self._check_inplace()
        y = np.sqrt(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, y)]
            self.grad_fn = SqrtBackward()
        self.data = np.array(y)

    def sin(self):
        self._check_type('sin')
        y = Tensor(np.sin(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = SinBackward()
        return y

    def sin_(self):
        self._check_type('sin')
        self._check_inplace()
        y = np.sin(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = SinBackward()
        self.data = np.array(y)

    def cos(self):
        self._check_type('cos')
        y = Tensor(np.cos(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = CosBackward()
        return y

    def cos_(self):
        self._check_type('cos')
        self._check_inplace()
        y = np.cos(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = CosBackward()
        self.data = np.array(y)

    def tan(self):
        self._check_type('tan')
        y = Tensor(np.tan(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, y.data)]
            y.grad_fn = TanBackward()
        return y

    def tan_(self):
        self._check_type('tan')
        self._check_inplace()
        y = np.tan(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, y)]
            self.grad_fn = TanBackward()
        self.data = np.array(y)

    def sinh(self):
        self._check_type('sinh')
        y = Tensor(np.sinh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = SinhBackward()
        return y

    def sinh_(self):
        self._check_type('sinh')
        self._check_inplace()
        y = np.sinh(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = SinhBackward()
        self.data = np.array(y)

    def cosh(self):
        self._check_type('cosh')
        y = Tensor(np.cosh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = CoshBackward()
        return y

    def cosh_(self):
        self._check_type('cosh')
        self._check_inplace()
        y = np.cosh(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = CoshBackward()
        self.data = np.array(y)

    def tanh(self):
        self._check_type('tanh')
        y = Tensor(np.tanh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, y.data)]
            y.grad_fn = TanhBackward()
        return y

    def tanh_(self):
        self._check_type('tanh')
        self._check_inplace()
        y = np.tanh(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, y)]
            self.grad_fn = TanhBackward()
        self.data = np.array(y)

    def asin(self):
        self._check_type('asin')
        y = Tensor(np.arcsin(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = ASinBackward()
        return y

    def asin_(self):
        self._check_type('asin')
        self._check_inplace()
        y = np.arcsin(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = ASinBackward()
        self.data = np.array(y)

    def acos(self):
        self._check_type('acos')
        y = Tensor(np.arccos(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = ACosBackward()
        return y

    def acos_(self):
        self._check_type('acos')
        self._check_inplace()
        y = np.arccos(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = ACosBackward()
        self.data = np.array(y)

    def atan(self):
        self._check_type('atan')
        y = Tensor(np.arctan(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = ATanBackward()
        return y

    def atan_(self):
        self._check_type('atan')
        self._check_inplace()
        y = np.arctan(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = ATanBackward()
        self.data = np.array(y)

    def log(self, base=None):
        self._check_type('log')
        if base is not None:
            if not isinstance(base, (int, float)) or base <= 0 or base == 1:
                raise TypeError("param 'base' must be a non-1 positive number")
            y = Tensor(np.log(self.data) / math.log(base), dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = Tensor(np.log(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
            base = math.e
        if y.requires_grad:
            y.children = [(self, base)]
            y.grad_fn = LogBackward()
        return y

    def log_(self, base=None):
        self._check_type('log')
        self._check_inplace()
        if base is not None:
            if not isinstance(base, (int, float)) or base <= 0 or base == 1:
                raise TypeError("param 'base' must be a non-1 positive number")
            y = np.log(self.data) / math.log(base)
        else:
            y = np.log(self.data)
            base = math.e
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, base)]
            self.grad_fn = LogBackward()
        self.data = np.array(y)

    def exp(self):
        self._check_type('exp')
        y = Tensor(np.exp(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, y.data)]
            y.grad_fn = ExpBackward()
        return y

    def exp_(self):
        self._check_type('exp')
        self._check_inplace()
        y = np.exp(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, y)]
            self.grad_fn = ExpBackward()
        self.data = np.array(y)

    def relu(self):
        self._check_type('relu', ('bool',))
        y = Tensor(np.maximum(self.data, 0), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, None)]
            y.grad_fn = ReluBackward()
        return y

    def relu_(self):
        self._check_type('relu', ('bool',))
        self._check_inplace()
        y = np.maximum(self.data, 0)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = ReluBackward()
        self.data = np.array(y)

    def sigmoid(self):
        self._check_type('sigmoid')
        y = Tensor(1.0 / (1 + np.exp(- self.data)), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, y.data)]
            y.grad_fn = SigmoidBackward()
        return y

    def sigmoid_(self):
        self._check_type('sigmoid')
        self._check_inplace()
        y = 1.0 / (1.0 + np.exp(- self.data))
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, y)]
            self.grad_fn = SigmoidBackward()
        self.data = np.array(y)

    def softmax(self, dim):
        self._check_type('softmax', ('int8', 'int16', 'int32', 'int64', 'float16', 'bool'))
        if dim is None:
            dim = -1
        data = self.data
        maximum = np.max(data, dim, keepdims=True)
        data = data - maximum
        data = np.exp(data)
        data = data / data.sum(dim, keepdims=True)
        y = Tensor(data, dtype=self.dtype, requires_grad=self.requires_grad)
        if y.requires_grad:
            y.children = [(self, dim, y.data)]
            y.grad_fn = SoftmaxBackward()
        return y

    def softmax_(self, dim):
        self._check_type('softmax', ('int8', 'int16', 'int32', 'int64', 'float16', 'bool'))
        self._check_inplace()
        data = self.data
        maximum = np.max(data, dim, keepdims=True)
        data = data - maximum
        data = np.exp(data)
        y = data / data.sum(dim, keepdims=True)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, dim, y)]
            self.grad_fn = SoftmaxBackward()
        self.data = np.array(y)

    def pow(self, power):
        return self.__pow__(power)

    def pow_(self, power):
        self._check_inplace(power)
        if isinstance(power, (int, float)):
            y = np.power(self.data, power)
        else:
            y = np.power(self.data, power.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None), (power, y)]
            self.grad_fn = PowerBackward()
        self.data = np.array(y)

    def floor(self):
        self._check_type('floor')
        result = Tensor(np.floor(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = FloorBackward()
        return result

    def floor_(self):
        self._check_type('floor')
        self._check_inplace()
        y = np.floor(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = FloordivBackward()
        self.data = np.array(y)

    def ceil(self):
        self._check_type('ceil')
        result = Tensor(np.ceil(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = CeilBackward()
        return result

    def ceil_(self):
        self._check_type('ceil')
        self._check_inplace()
        y = np.ceil(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = CeilBackward()
        self.data = np.array(y)

    def uniform_(self, low=0, high=1):
        self._check_type('uniform')
        self._check_inplace()
        y = np.random.uniform(low, high, size=self.data.shape)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = UniformBackward()
        self.data = np.array(y)

    def normal_(self, mean=0, std=1):
        self._check_type('normal')
        self._check_inplace()
        y = np.random.normal(mean, std, size=self.data.shape)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = NormalBackward()
        self.data = np.array(y)

    def reshape(self, *shape):
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = ReshapeBackward()
        return result

    def flatten(self):
        result = Tensor(self.data.flatten(), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = ReshapeBackward()
        return result

    def squeeze(self, *axes):
        if axes:
            result = Tensor(self.data.squeeze(axes), dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data.squeeze(), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = ReshapeBackward()
        return result

    def unsqueeze(self, *axes):
        result = Tensor(np.expand_dims(self.data, axes), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, None)]
            result.grad_fn = ReshapeBackward()
        return result

    def transpose(self, *axes):
        result = Tensor(self.data.transpose(axes), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, axes)]
            result.grad_fn = TransposeBackward()
        return result

    def sum(self, axes=None, keepdims=False):
        result = Tensor(np.sum(self.data, axis=axes, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, axes)]
            result.grad_fn = SumBackward()
        return result

    def zero_(self):
        self._check_inplace()
        y = np.zeros_like(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = FillBackward()
        self.data = np.array(y)

    def fill_(self, value):
        self._check_inplace()
        y = value * np.ones_like(self.data)
        if self.requires_grad:
            child = deepcopy(self)
            child.children = self.children
            self.children = [(child, None)]
            self.grad_fn = FillBackward()
        self.data = np.array(y)

    def norm(self, p=2.):
        self._check_type('norm')
        data = self.data
        s = (np.abs(data) ** p).sum()
        y = s ** (1. / p - 1.)
        result = Tensor(y * s, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.requires_grad:
            result.children = [(self, p, y)]
            result.grad_fn = NormBackward()
        return result

    def matmul(self, other):
        """
        矩阵乘法,支持broadcast
        """
        if not isinstance(other, Tensor):
            raise TypeError(f"argument 'other' (position 1) must be Tensor, not {type(other)}")
        result = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if not result.requires_grad:
            return result
        result.children = [(self, None), (other, None)]
        if self.data.ndim == 1 and other.data.ndim == 1:
            result.grad_fn = DotBackward()
        elif self.data.ndim == 1 or other.data.ndim == 1:
            result.grad_fn = MvBackward()
        else:
            result.grad_fn = MmBackward()
        return result

    def backward(self, grad=1.0, is_last=True):
        if self.size > 1 and is_last:
            raise RuntimeError('grad can be implicitly created only for scalar outputs')
        if self.is_leaf:
            return
        if not isinstance(grad, Tensor):
            grad = Tensor(grad)
        for i, child in enumerate(self.children):
            child_tensor = child[0]
            if isinstance(child_tensor, Tensor) and child_tensor.requires_grad:
                if child_tensor.grad is None:
                    child_tensor.grad = Tensor(np.zeros_like(child_tensor.data))
                child_tensor.grad = child_tensor.grad + Tensor(self.grad_fn.calculate_grad(grad.data, self.children, i),
                                                               dtype=np.float32)
        for child in self.children:
            child_tensor = child[0]
            if isinstance(child_tensor, Tensor) and child_tensor.requires_grad:
                child_tensor.backward(child_tensor.grad, False)
