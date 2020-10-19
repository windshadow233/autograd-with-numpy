import numpy as np
from ..tensor import array, Tensor
from ..random import rand_like
from ..backward import CrossEntropyBackward


def relu(x: Tensor):
    return x.relu()


def sigmoid(x: Tensor):
    return x.sigmoid()


def softmax(x: Tensor, dim):
    return x.softmax(dim)


def one_hot(n, x: Tensor):
    """
    编码one_hot向量
    :param n: 类别数
    :param x: 标签,行张量
    :return: ont_hot张量
    """
    return array(np.eye(n)[x.data])


def dropout(x: Tensor, p=0.5, training=True):
    """

    @param x: 输入的张量
    @param p: 神经元失活率
    @param training: 是否处于训练模式
    """
    if not 0. <= p <= 1.:
        raise ValueError(f'dropout probability has to be between 0 and 1, but got {p}')
    mask = (rand_like(x) > p).float()
    y = x / (1 - p) * mask if training else x
    return y


def cross_entropy(x: Tensor, target):
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

