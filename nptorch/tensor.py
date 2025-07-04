from copy import copy
from numbers import Number
from .autograd.backward import *
from .autograd.grad_mode import is_grad_enabled
from .return_types import Max, Min, Sort
from numpy import float16, float32, float64, int8, int16, int32, int64, bool_,\
    uint8, uint16, uint32, uint64, inf

# np.set_printoptions(precision=4, suppress=True)


def tensor(data, dtype=None, requires_grad=False):
    return Tensor(data, dtype, requires_grad)


class Tensor(object):
    def __init__(self, data, dtype=None, requires_grad=False):
        if isinstance(data, Tensor):
            data = data.data
        if dtype is None:
            self.data = np.array(data)
            if self.dtype == float64:
                self.data = self.data.astype(float32)
        else:
            self.data = np.array(data, dtype=dtype)
        assert 'float' in self.dtype.name or not requires_grad, \
            'Only Arrays of floating point dtype can require gradients'
        self.requires_grad = requires_grad
        self.grad_fn = None
        self.children = []
        self.grad = None
        self._retain = False

    def __hash__(self):
        return id(self)

    @property
    def retains_grad(self):
        return self._retain

    @property
    def grad_enable(self):
        return self.requires_grad and is_grad_enabled()

    @property
    def is_leaf(self):
        return self.grad_fn is None

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        result = Tensor(self.data.T, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.grad_fn = TBackward()
            result.children = [(self,)]
        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def strides(self):
        return self.data.strides

    def __repr__(self):
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
        if isinstance(item, int):
            item = (item,)
        result = Tensor(self.data[item], dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, item)]
            if isinstance(item, list) or (isinstance(item, tuple) and any([isinstance(index, (tuple, list)) and len(index) > 1 for index in item])):
                result.grad_fn = IndexBackward()
            else:
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

    def equal(self, other):
        return (self == other).all().item()

    def t(self):
        return self.T

    def _check_inplace(self):
        assert not (self.grad_enable and self.is_leaf), \
            'a leaf Variable that requires grad has been used in an in-place operation.'

    def _check_type(self, operation, excludes=('uint8', 'uint16', 'uint32', 'uint64',
                                               'int8', 'int16', 'int32', 'int64', 'bool')):
        assert self.dtype.name not in excludes, f"'{operation}' not implemented for '{self.dtype.name}'"

    def half(self):
        result = Tensor(self.data.astype(float16), dtype=float16, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = CopyBackward()
        return result

    def float(self):
        result = Tensor(self.data.astype(np.float32), dtype=float32, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = CopyBackward()
        return result

    def double(self):
        result = Tensor(self.data.astype(float64), dtype=float64, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = CopyBackward()
        return result

    def char(self):
        return Tensor(self.data.astype(int8), dtype=int8)

    def short(self):
        return Tensor(self.data.astype(int16), dtype=int16)

    def int(self):
        return Tensor(self.data.astype(int32), dtype=int32)

    def long(self):
        return Tensor(self.data.astype(int64), dtype=int64)

    def bool(self):
        return Tensor(self.data.astype(bool_), dtype=bool_)

    def tolist(self):
        return self.data.tolist()

    def all(self, axis=None, keepdims=False):
        return Tensor(self.data.all(axis=axis, keepdims=keepdims))

    def any(self, axis=None, keepdims=False):
        return Tensor(self.data.any(axis=axis, keepdims=keepdims))

    def item(self):
        return self.data.item()

    def argmax(self, axis=None):
        return Tensor(self.data.argmax(axis), dtype=int64)

    def argmin(self, axis=None):
        return Tensor(self.data.argmin(axis), dtype=int64)

    def detach(self):
        return Tensor(self.data, dtype=self.dtype, requires_grad=False)

    def retain_grad(self):
        assert self.requires_grad, 'you can only retain gradients of Tensors that require them'
        self._retain = True

    def requires_grad_(self, mode=True):
        assert self.is_leaf, 'you can only change requires_grad flags of leaf variables'
        assert 'float' in self.dtype.name, 'only Tensors of floating point dtype can require gradients'
        self.requires_grad = mode
        return self

    def numpy(self):
        return self.data

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Number):
            result = Tensor(self.data + other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data + other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = AddBackward()
        return result

    def __iadd__(self, other):
        self._check_inplace()
        return self.__add__(other)

    def __neg__(self):
        result = Tensor(- self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = NegBackward()
        return result

    def __rsub__(self, other):
        return - self.__sub__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            result = Tensor(self.data - other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data - other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = SubBackward()
        return result

    def __isub__(self, other):
        self._check_inplace()
        return self.__sub__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            result = Tensor(self.data * other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data * other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = MulBackward()
        return result

    def __imul__(self, other):
        self._check_inplace()
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            result = Tensor(other / self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(other.data / self.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(other,), (self,)]
            result.grad_fn = DivBackward()
        return result

    def __truediv__(self, other):
        if isinstance(other, Number):
            result = Tensor(self.data / other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data / other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = DivBackward()
        return result

    def __itruediv__(self, other):
        self._check_inplace()
        return self.__truediv__(other)

    def __rfloordiv__(self, other):
        if isinstance(other, Number):
            result = Tensor(other // self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(other.data // self.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(other,), (self,)]
            result.grad_fn = FloordivBackward()
        return result

    def __floordiv__(self, other):
        if isinstance(other, Number):
            result = Tensor(self.data // other, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data // other.data, dtype=self.dtype,
                            requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = FloordivBackward()
        return result

    def __ifloordiv__(self, other):
        self._check_inplace()
        return self.__ifloordiv__(other)

    def __mod__(self, other):
        if isinstance(other, Tensor):
            if other.requires_grad:
                raise RuntimeError("the derivative for 'other' is not implemented")
            other = other.data
        result = Tensor(self.data % other, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = RemainderBackward()
        return result

    def __imod__(self, other):
        self._check_inplace()
        return self.__mod__(other)

    def __rpow__(self, power):
        if isinstance(power, Number):
            y = np.power(power, self.data)
            result = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = np.power(power.data, self.data)
            result = Tensor(power, dtype=self.dtype, requires_grad=self.requires_grad or power.requires_grad)
        if result.grad_enable:
            result.children = [(power,), (self, y)]
            result.grad_fn = PowerBackward()
        return result

    def __pow__(self, power):
        if isinstance(power, Number):
            y = np.power(self.data, power)
            result = Tensor(self.data ** power, dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = np.power(self.data, power.data)
            result = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad or power.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (power, y)]
            result.grad_fn = PowerBackward()
        return result

    def __ipow__(self, power):
        self._check_inplace()
        return self.__pow__(power)

    def max(self, axis=None, keepdims=False):
        values = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        indices = Tensor(np.argmax(self.data, axis=axis), dtype=int64)
        if values.grad_enable:
            values.children = [(self, values.data, axis, keepdims)]
            values.grad_fn = MaxBackward()
        return Max(values, indices)

    def min(self, axis=None, keepdims=False):
        values = Tensor(np.min(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        indices = Tensor(np.argmin(self.data, axis=axis), dtype=int64)
        if values.grad_enable:
            values.children = [(self, values.data, axis, keepdims)]
            values.grad_fn = MinBackward()
        return Min(values, indices)

    def mean(self, axis=None, keepdims=False):
        result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axis, keepdims)]
            result.grad_fn = MeanBackward()
        return result

    def var(self, axis=None, keepdims=False):
        result = Tensor(np.var(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype,
                        requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axis, keepdims)]
            result.grad_fn = VarBackward()
        return result

    def abs(self):
        y = Tensor(np.abs(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = AbsBackward()
        return y

    def argsort(self, axis=-1, descending=False):
        if descending:
            return Tensor(np.argsort(- self.data, axis=axis))
        return Tensor(np.argsort(self.data, axis=axis))

    def sort(self, axis=-1, descending=False):
        if descending:
            sorted_values = Tensor(- np.sort(- self.data, axis=axis), dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            sorted_values = Tensor(np.sort(self.data, axis=axis), dtype=self.dtype, requires_grad=self.requires_grad)
        sorted_indices = self.argsort(axis, descending)
        if sorted_values.grad_enable:
            sorted_values.children = [(self, sorted_indices.data, axis)]
            sorted_values.grad_fn = SortBackward()
        return Sort(sorted_values, sorted_indices)

    def index_select(self, axis, index: list):
        if isinstance(index, Tensor):
            assert index.dtype == int64, f'index is supposed to have type int64, got {index.dtype}'
            assert index.ndim == 1, f'index is supposed to be 1-dimensional, got {index.ndim}'
            index = index.tolist()
        slices = [slice(None)] * self.ndim
        slices[axis] = index
        return self[tuple(slices)]

    def diagonal(self, k=0):
        result = Tensor(np.diag(self.data, k=k), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, k)]
            result.grad_fn = DiagBackward()
        return result

    def abs_(self):
        self._check_inplace()
        y = np.abs(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = AbsBackward()
        self.data = np.array(y)

    def sqrt(self):
        self._check_type('sqrt')
        y = Tensor(np.sqrt(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self, y.data)]
            y.grad_fn = SqrtBackward()
        return y

    def sqrt_(self):
        self._check_type('sqrt')
        self._check_inplace()
        y = np.sqrt(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, y)]
            self.grad_fn = SqrtBackward()
        self.data = np.array(y)

    def sin(self):
        self._check_type('sin')
        y = Tensor(np.sin(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = SinBackward()
        return y

    def sin_(self):
        self._check_type('sin')
        self._check_inplace()
        y = np.sin(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = SinBackward()
        self.data = np.array(y)

    def cos(self):
        self._check_type('cos')
        y = Tensor(np.cos(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = CosBackward()
        return y

    def cos_(self):
        self._check_type('cos')
        self._check_inplace()
        y = np.cos(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = CosBackward()
        self.data = np.array(y)

    def tan(self):
        self._check_type('tan')
        y = Tensor(np.tan(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self, y.data)]
            y.grad_fn = TanBackward()
        return y

    def tan_(self):
        self._check_type('tan')
        self._check_inplace()
        y = np.tan(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, y)]
            self.grad_fn = TanBackward()
        self.data = np.array(y)

    def sinh(self):
        self._check_type('sinh')
        y = Tensor(np.sinh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = SinhBackward()
        return y

    def sinh_(self):
        self._check_type('sinh')
        self._check_inplace()
        y = np.sinh(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = SinhBackward()
        self.data = np.array(y)

    def cosh(self):
        self._check_type('cosh')
        y = Tensor(np.cosh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = CoshBackward()
        return y

    def cosh_(self):
        self._check_type('cosh')
        self._check_inplace()
        y = np.cosh(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = CoshBackward()
        self.data = np.array(y)

    def tanh(self):
        self._check_type('tanh')
        y = Tensor(np.tanh(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self, y.data)]
            y.grad_fn = TanhBackward()
        return y

    def tanh_(self):
        self._check_type('tanh')
        self._check_inplace()
        y = np.tanh(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, y)]
            self.grad_fn = TanhBackward()
        self.data = np.array(y)

    def asin(self):
        self._check_type('asin')
        y = Tensor(np.arcsin(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = ASinBackward()
        return y

    def asin_(self):
        self._check_type('asin')
        self._check_inplace()
        y = np.arcsin(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = ASinBackward()
        self.data = np.array(y)

    def acos(self):
        self._check_type('acos')
        y = Tensor(np.arccos(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = ACosBackward()
        return y

    def acos_(self):
        self._check_type('acos')
        self._check_inplace()
        y = np.arccos(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = ACosBackward()
        self.data = np.array(y)

    def atan(self):
        self._check_type('atan')
        y = Tensor(np.arctan(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = ATanBackward()
        return y

    def atan_(self):
        self._check_type('atan')
        self._check_inplace()
        y = np.arctan(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = ATanBackward()
        self.data = np.array(y)

    def log(self, base=None):
        self._check_type('log')
        if base is not None:
            if not isinstance(base, Number) or base <= 0. or base == 1.:
                raise TypeError("param 'base' must be a non-1 positive number")
            y = Tensor(np.log(self.data) / math.log(base), dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            y = Tensor(np.log(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
            base = math.e
        if y.grad_enable:
            y.children = [(self, base)]
            y.grad_fn = LogBackward()
        return y

    def log_(self, base=None):
        self._check_type('log')
        self._check_inplace()
        if base is not None:
            if not isinstance(base, Number) or base <= 0. or base == 1.:
                raise TypeError("param 'base' must be a non-1 positive number")
            y = np.log(self.data) / math.log(base)
        else:
            y = np.log(self.data)
            base = math.e
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, base)]
            self.grad_fn = LogBackward()
        self.data = np.array(y)

    def exp(self):
        self._check_type('exp')
        y = Tensor(np.exp(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self, y.data)]
            y.grad_fn = ExpBackward()
        return y

    def exp_(self):
        self._check_type('exp')
        self._check_inplace()
        y = np.exp(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, y)]
            self.grad_fn = ExpBackward()
        self.data = np.array(y)

    def relu(self):
        self._check_type('relu', ('bool',))
        y = Tensor(np.maximum(self.data, 0.), dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self,)]
            y.grad_fn = ReluBackward()
        return y

    def relu_(self):
        self._check_type('relu', ('bool',))
        self._check_inplace()
        y = np.maximum(self.data, 0.)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = ReluBackward()
        self.data = np.array(y)

    def sigmoid(self):
        self._check_type('sigmoid')
        exp = np.exp(- np.abs(self.data))
        y = 1. / (1. + exp) * (self.data >= 0.) + exp / (exp + 1.) * (self.data < 0.)
        y = Tensor(y, dtype=self.dtype, requires_grad=self.requires_grad)
        if y.grad_enable:
            y.children = [(self, y.data)]
            y.grad_fn = SigmoidBackward()
        return y

    def sigmoid_(self):
        self._check_type('sigmoid')
        self._check_inplace()
        exp = np.exp(- np.abs(self.data))
        y = 1. / (1. + exp) * (self.data >= 0.) + exp / (exp + 1.) * (self.data < 0.)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, y)]
            self.grad_fn = SigmoidBackward()
        self.data = np.array(y)

    def softmax(self, axis):
        self._check_type('softmax')
        # data = self.data
        # maximum = np.max(data)
        # data = data - maximum
        # data = np.exp(data)
        # data = data / data.sum(axis, keepdims=True)
        # y = Tensor(data, dtype=self.dtype, requires_grad=self.requires_grad)
        # if y.grad_enable:
        #     y.children = [(self, axis, y.data)]
        #     y.grad_fn = SoftmaxBackward()
        # return y
        m = self.max().values
        data = (self - m).exp()
        data /= data.sum(axis, keepdims=True)
        return data

    def softplus(self):
        self._check_type('softplus')
        y = np.exp(- np.abs(self.data))
        result = Tensor(np.log(1. + y) + self.data * (self.data >= 0.), requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, y)]
            result.grad_fn = SoftplusBackward()
        return result

    def floor(self):
        self._check_type('floor')
        result = Tensor(np.floor(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = FloorBackward()
        return result

    def floor_(self):
        self._check_type('floor')
        self._check_inplace()
        y = np.floor(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = FloordivBackward()
        self.data = np.array(y)

    def ceil(self):
        self._check_type('ceil')
        result = Tensor(np.ceil(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = CeilBackward()
        return result

    def ceil_(self):
        self._check_type('ceil')
        self._check_inplace()
        y = np.ceil(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = CeilBackward()
        self.data = np.array(y)

    def uniform_(self, low=0., high=1.):
        self._check_type('uniform')
        self._check_inplace()
        y = np.random.uniform(low, high, size=self.shape)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = UniformBackward()
        self.data = y

    def normal_(self, mean=0., std=1.):
        self._check_type('normal')
        self._check_inplace()
        y = np.random.normal(mean, std, size=self.shape)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = NormalBackward()
        self.data = y

    def reshape(self, *shape):
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = ReshapeBackward()
        return result

    def repeat(self, *shape):
        result = Tensor(np.tile(self.data, shape), requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, shape)]
            result.grad_fn = RepeatBackward()
        return result

    def flatten(self):
        result = Tensor(self.data.flatten(), requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = ReshapeBackward()
        return result

    def squeeze(self, *axes):
        if axes:
            result = Tensor(self.data.squeeze(axes), dtype=self.dtype, requires_grad=self.requires_grad)
        else:
            result = Tensor(self.data.squeeze(), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = ReshapeBackward()
        return result

    def unsqueeze(self, *axes):
        result = Tensor(np.expand_dims(self.data, axes), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = ReshapeBackward()
        return result

    def transpose(self, *axes):
        result = Tensor(self.data.transpose(axes), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axes)]
            result.grad_fn = TransposeBackward()
        return result

    def swapaxes(self, axis1, axis2):
        result = Tensor(self.data.swapaxes(axis1, axis2), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axis1, axis2)]
            result.grad_fn = SwapaxesBackward()
        return result

    def sum(self, axes=None, keepdims=False):
        result = Tensor(np.sum(self.data, axis=axes, keepdims=keepdims), requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axes)]
            result.grad_fn = SumBackward()
        return result

    def zero_(self):
        self._check_inplace()
        y = np.zeros_like(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = FillBackward()
        self.data = y

    def fill_(self, value):
        self._check_inplace()
        y = value * np.ones_like(self.data)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child,)]
            self.grad_fn = FillBackward()
        self.data = y

    def norm(self, axis=None, p=2., keepdims=False, eps=1e-12):
        self._check_type('norm')
        data = self.data
        if p == inf:
            return self.abs().max(axis=axis, keepdims=keepdims).values
        s = (np.abs(data) ** p).sum(axis, keepdims=keepdims) + eps
        y = s ** (1. / p - 1.)
        result = Tensor(y * s, dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, p, y, axis, keepdims)]
            result.grad_fn = NormBackward()
        return result

    def trace(self):
        self._check_type('trace', excludes=('bool',))
        assert self.ndim == 2, f"ndim must be 2, got {self.ndim}"
        result = Tensor(np.trace(self.data), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self,)]
            result.grad_fn = TraceBackward()
        return result

    def outer(self, other):
        """
        向量外积
        """
        assert self.ndim == 1 and other.ndim == 1, 'only vectors allow outer product operations'
        result = Tensor(np.outer(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        if result.grad_enable:
            result.children = [(self,), (other,)]
            result.grad_fn = OuterBackward()
        return result

    def __matmul__(self, other):
        """
        矩阵乘法,支持broadcast
        """
        assert isinstance(other, Tensor), f"argument 'other' must be Tensor, not {type(other)}"
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        if not result.grad_enable:
            return result
        result.children = [(self,), (other,)]
        if self.ndim == 1 and other.ndim == 1:
            result.grad_fn = DotBackward()
        elif self.ndim == 1 or other.ndim == 1:
            result.grad_fn = MvBackward()
        else:
            result.grad_fn = MmBackward()
        return result

    def __imatmul__(self, other):
        self._check_inplace()
        return self.__matmul__(other)

    def clamp(self, min=math.inf, max=math.inf):
        self._check_type('clamp', excludes=('bool',))
        if min == max == math.inf:
            raise ValueError("At least one of 'min' or 'max' must not be infinity")
        result = Tensor(np.clip(self.data, a_min=min, a_max=max), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, min, max)]
            result.grad_fn = ClampBackward()
        return result

    def clamp_(self, min=math.inf, max=math.inf):
        self._check_type('clamp', excludes=('bool',))
        self._check_inplace()
        y = np.clip(self.data, a_min=min, a_max=max)
        if self.grad_enable:
            child = copy(self)
            self.children = [(child, min, max)]
            self.grad_fn = ClampBackward()
        self.data = y

    def where(self, condition, y):
        assert isinstance(y, Tensor), f"arguments 'y' must be Tensor, got {type(y)}"
        assert self.dtype == y.dtype, f"expected scalar type {self.dtype}, got {y.dtype}"
        if isinstance(condition, Tensor):
            condition = condition.data
        if not isinstance(condition, np.ndarray):
            condition = np.array(condition)
        result = Tensor(np.where(condition, self.data, y.data), dtype=self.dtype,
                        requires_grad=self.requires_grad or y.requires_grad)
        if result.grad_enable:
            result.children = [(self, condition.astype(float)), (y, 1. - condition.astype(float))]
            result.grad_fn = WhereBackward()
        return result

    def split(self, split_size, axis=0):
        tensors = [Tensor(x, dtype=self.dtype, requires_grad=self.requires_grad) for x in np.split(self.data, split_size, axis)]
        if self.grad_enable:
            for i, tensor in enumerate(tensors):
                tensor.children = [(self, i, self.shape[axis] // split_size, axis)]
                tensor.grad_fn = SplitBackward()
        return tuple(tensors)

    def unbind(self, axis=0):
        return tuple(tensor.squeeze(axis) for tensor in self.split(self.shape[axis], axis=axis))

    def flip(self, axis=None):
        result = Tensor(np.flip(self.data, axis=axis), dtype=self.dtype, requires_grad=self.requires_grad)
        if result.grad_enable:
            result.children = [(self, axis)]
            result.grad_fn = FlipBackward()
        return result

    def __reversed__(self):
        if self.ndim == 0:
            return self
        else:
            return self.flip(0)

    def backward(self, grad=None):
        if not self.grad_enable:
            return
        if grad is None:
            assert self.size == 1, 'grad can be implicitly created only for scalar outputs'
            grad = np.ones_like(self.data)
            if self.retains_grad or (self.is_leaf and self.requires_grad):
                self.grad = Tensor(grad)
        stack = [(self, grad)]
        while stack:
            item, grad = stack.pop()
            for i, child in enumerate(item.children):
                child_tensor = child[0]
                if isinstance(child_tensor, Tensor) and child_tensor.requires_grad:
                    child_grad = item.grad_fn.calculate_grad(grad, item.children, i)
                    if child_tensor.retains_grad or child_tensor.is_leaf:
                        if child_tensor.grad is None:
                            child_tensor.grad = Tensor(np.zeros_like(child_tensor.data), dtype=child_tensor.dtype)
                        child_tensor.grad = child_tensor.grad + Tensor(child_grad, dtype=float32)
                    if not child_tensor.is_leaf:
                        stack.append((child_tensor, child_grad))

    add = __add__
    sub = __sub__
    mul = __mul__
    div = __truediv__
    mod = __mod__
    neg = __neg__
    pow = __pow__
    matmul = __matmul__
