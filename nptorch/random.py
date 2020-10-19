import numpy as np
from nptorch.tensor import tensor


def seed(n):
    np.random.seed(n)


def rand(size, dtype=np.float32, requires_grad=False):
    return tensor(np.random.random(size), dtype=dtype, requires_grad=requires_grad)


def uniform(size, low=0, high=1, dtype=np.float32, requires_grad=False):
    return tensor(np.random.uniform(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def normal(size, mean=0, std=1, dtype=np.float32, requires_grad=False):
    return tensor(np.random.normal(loc=mean, scale=std, size=size), dtype=dtype, requires_grad=requires_grad)


def randint(size, low, high, dtype=np.int64, requires_grad=False):
    return tensor(np.random.randint(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def rand_like(x, dtype=np.float32, requires_grad=False):
    return rand(x.data.shape, dtype=dtype, requires_grad=requires_grad)


def uniform_like(x, dtype=np.float32, requires_grad=False):
    return uniform(x.data.shape, dtype=dtype, requires_grad=requires_grad)


def normal_like(x, mean=0, std=1, dtype=np.float32, requires_grad=False):
    return normal(x.data.shape, mean=mean, std=std, dtype=dtype, requires_grad=requires_grad)


def randint_like(x, low, high, dtype=np.int64, requires_grad=False):
    return randint(x.data.shape, low=low, high=high, dtype=dtype, requires_grad=requires_grad)
