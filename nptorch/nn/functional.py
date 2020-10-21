from ..tensor import array, Tensor
from ..random import rand_like
from ..backward import CrossEntropyBackward, ConvBackward, MeanPoolBackward
from .conv_operations import *


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


def linear(x: Tensor, w: Tensor, b: Tensor = None):
    output = x.matmul(w.T)
    if b is not None:
        output = output + b
    return output


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


def conv(x: Tensor, kernels: Tensor, bias: Tensor = None, stride=1, padding=(0, 0)):
    data = x.data
    padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    data = padding_zeros(data, padding)
    split = split_by_strides(data, *kernels.shape[-2:], stride=stride)
    output = Tensor(np.tensordot(split, kernels.data, axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2)),
                    requires_grad=x.requires_grad)
    if bias is not None:
        output = output + bias.unsqueeze(-1, -2)
    if output.requires_grad:
        output.children = [(x, padding), (kernels, stride)]
        if bias is not None:
            output.children.append((bias, None))
        output.grad_fn = ConvBackward()
    return output


def max_pool(x: Tensor, kernel_size):
    # stride = stride or kernel_size
    stride = kernel_size


def mean_pool(x: Tensor, kernel_size, stride):
    stride = stride or kernel_size
    split = split_by_strides(x.data, kernel_size, kernel_size, stride)
    mean = np.mean(split, axis=(-1, -2))
    output = Tensor(mean, requires_grad=x.requires_grad)
    if output.requires_grad:
        output.children = [(x, kernel_size, stride)]
        output.grad_fn = MeanPoolBackward()
    return output


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

