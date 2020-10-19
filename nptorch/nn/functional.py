import numpy as np
from ..tensor import tensor
from ..backward import CrossEntropyBackward


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def softmax(x, dim):
    return x.softmax(dim)


def one_hot(n, x):
    """
    编码one_hot向量
    :param n: 类别数
    :param x: 标签,行张量
    :return: ont_hot张量
    """
    return tensor(np.eye(n)[x.data])


def cross_entropy(x, target):
    """
    交叉熵
    :param x: 二维张量,每一行表示一个数据
    :param target: 标签,行张量,数量与x的行数相同
    :return: cross_entropy
    """
    x_softmax = x.softmax(-1)
    log_softmax = x_softmax.log()
    n = target.size
    one_hot_target = one_hot(x.shape[-1], target)
    loss = - (one_hot_target * log_softmax).sum() / n
    if x.requires_grad:
        loss.children = [(x, x_softmax.data, one_hot_target)]
        loss.grad_fn = CrossEntropyBackward()
    return loss

