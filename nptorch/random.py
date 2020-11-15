from nptorch.tensor import *


def seed(n):
    np.random.seed(n)


def rand(*size, dtype=float32, requires_grad=False):
    return Tensor(np.random.random(size), dtype=dtype, requires_grad=requires_grad)


def uniform(low=0., high=1., size=None, dtype=float32, requires_grad=False):
    return Tensor(np.random.uniform(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def normal(mean=0., std=1., size=None, dtype=float32, requires_grad=False):
    return Tensor(np.random.normal(loc=mean, scale=std, size=size), dtype=dtype, requires_grad=requires_grad)


def randint(low, high, size=None, dtype=int32, requires_grad=False):
    return Tensor(np.random.randint(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def rand_like(x, dtype=float32, requires_grad=False):
    return rand(*x.shape, dtype=dtype, requires_grad=requires_grad)


def uniform_like(x, low=0., high=1., dtype=float32, requires_grad=False):
    return uniform(low=low, high=high, size=x.data.shape, dtype=dtype, requires_grad=requires_grad)


def normal_like(x, mean=0., std=1., dtype=float32, requires_grad=False):
    return normal(mean=mean, std=std, size=x.shape, dtype=dtype, requires_grad=requires_grad)


def randint_like(x, low, high, dtype=int32, requires_grad=False):
    return randint(low=low, high=high, size=x.shape, dtype=dtype, requires_grad=requires_grad)
