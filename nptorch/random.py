from nptorch.tensor import *


def seed(n):
    np.random.seed(n)


def rand(size, dtype=float32, requires_grad=False):
    return array(np.random.random(size), dtype=dtype, requires_grad=requires_grad)


def uniform(size, low=0., high=1., dtype=float32, requires_grad=False):
    return array(np.random.uniform(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def normal(size, mean=0., std=1., dtype=float32, requires_grad=False):
    return array(np.random.normal(loc=mean, scale=std, size=size), dtype=dtype, requires_grad=requires_grad)


def randint(size, low, high, dtype=int32, requires_grad=False):
    return array(np.random.randint(low=low, high=high, size=size), dtype=dtype, requires_grad=requires_grad)


def rand_like(x, dtype=float32, requires_grad=False):
    return rand(x.data.shape, dtype=dtype, requires_grad=requires_grad)


def uniform_like(x, dtype=float32, requires_grad=False):
    return uniform(x.data.shape, dtype=dtype, requires_grad=requires_grad)


def normal_like(x, mean=0., std=1., dtype=float32, requires_grad=False):
    return normal(x.data.shape, mean=mean, std=std, dtype=dtype, requires_grad=requires_grad)


def randint_like(x, low, high, dtype=int32, requires_grad=False):
    return randint(x.data.shape, low=low, high=high, dtype=dtype, requires_grad=requires_grad)
