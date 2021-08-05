import numpy as np
from ..tensor import Tensor, float32
from .. import random
from ..autograd.backward import CrossEntropyBackward, Conv2dBackward, MeanPool2dBackward, AvgPool2dBackward,\
    LeakyReLUBackward, ELUBackward, BatchNormBackward, NLLLossBackward, EmbeddingBackward, BinaryCrossEntropyBackward
from .conv_operations import split_by_strides, padding_zeros, dilate
from .modules.utils import _pair


def relu(x: Tensor):
    """
    max(0, x)
    """
    return x.relu()


def relu_(x: Tensor):
    x.relu_()
    return x


def sigmoid(x: Tensor):
    """
    1 / (1 + e^{-x})
    """
    return x.sigmoid()


def softmax(x: Tensor, dim):
    """
    y_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
    """
    return x.softmax(dim)


def tanh(x: Tensor):
    return x.tanh()


def leaky_relu(x: Tensor, leaky_rate=0.01):
    data = x.data
    y = Tensor(((data > 0.) + (data <= 0.) * leaky_rate).astype(float32) * data, requires_grad=x.requires_grad)
    if y.grad_enable:
        y.children = [(x, leaky_rate)]
        y.grad_fn = LeakyReLUBackward()
    return y


def elu(x: Tensor, alpha=1.):
    data = x.data
    assert 'float' in data.dtype.name, f"'elu' operation not implement for '{data.dtype}'"
    y = Tensor(((data > 0.) * data + (data <= 0.) * alpha * (np.exp(data) - 1.)).astype(float32),
               requires_grad=x.requires_grad)
    if y.grad_enable:
        y.children = [(x, alpha)]
        y.grad_fn = ELUBackward()
    return y


def softplus(x: Tensor):
    """
    y = log(1 + e^x)
    """
    return x.softplus()


def one_hot(n, x: Tensor):
    """
    编码one_hot向量
    :param n: 类别数
    :param x: 标签,行张量
    :return: ont_hot张量
    """
    return Tensor(np.eye(n)[x.data])


def linear(x: Tensor, w: Tensor, b: Tensor = None):
    """
    线性变换, y = x w^T + b
    """
    x @= w.T
    if b is not None:
        x += b
    return x


def dropout(x: Tensor, p=0.5, training=True):
    """
    普通dropout,每个元素有p的概率置0
    @param x: 输入的张量
    @param p: 失活率
    @param training: 是否处于训练模式
    """
    assert 0. <= p <= 1., f'dropout probability has to be between 0 and 1, but got {p}'
    if not training:
        return x
    y = x / (1. - p) * (random.rand_like(x) > p).float()
    return y


def dropout2d(x: Tensor, p=0.5, training=True):
    """
    二维dropout,每个channel有p的概率置0
    @param x: 输入的张量,(B,C,H,W)
    @param p: 失活率
    @param training: 是否处于训练模式
    """
    assert x.ndim == 4, 'x must be 4 dimensional'
    assert 0. <= p <= 1., f'dropout probability has to be between 0 and 1, but got {p}'
    if not training:
        return x
    y = x / (1. - p) * (random.rand(1, x.shape[1], 1, 1) > p).float()
    return y


def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride=(1, 1), padding=(0, 0), dilation=(0, 0)):
    assert x.ndim == 4, 'x must be 4 dimensional'
    b, c, h, w = x.shape
    oc, ic, kh, kw = weight.shape
    assert c == ic, 'Conv2d channels not equal'
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    padded_data = padding_zeros(x.data, padding)
    dilated_weight = dilate(weight.data, dilation)
    split = split_by_strides(padded_data, dilated_weight.shape[-2:], stride=stride)
    output = Tensor(np.tensordot(split, dilated_weight, axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2)),
                    requires_grad=x.requires_grad)
    if bias is not None:
        output = output + bias[:, None, None]
    if output.grad_enable:
        output.children = [(x, padded_data, padding), (weight, dilated_weight, stride, dilation)]
        if bias is not None:
            output.children.append((bias, None))
        output.grad_fn = Conv2dBackward()
    return output


def conv1d(x: Tensor, weight: Tensor, bias: Tensor = None, stride=1, padding=0, dilation=0):
    assert x.ndim == 3, 'x must be 3 dimensional'
    return conv2d(x.unsqueeze(-2), weight.unsqueeze(-2), bias, (1, stride), (0, padding), (0, dilation)).squeeze(-2)


def avg_pool2d(x: Tensor, kernel_size, stride):
    assert x.ndim == 4, 'x must be 4 dimensional'
    kernel_size = _pair(kernel_size)
    stride = stride or kernel_size
    stride = _pair(stride)
    split = split_by_strides(x.data, kernel_size, stride)
    mean_data = np.mean(split, axis=(-1, -2))
    output = Tensor(mean_data, requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, kernel_size, stride)]
        output.grad_fn = MeanPool2dBackward()
    return output


def max_pool2d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 4, 'x must be 4 dimensional'
    kernel_size = _pair(kernel_size)
    stride = stride or kernel_size
    stride = _pair(stride)
    split = split_by_strides(x.data, kernel_size, stride)
    max_data = np.max(split, axis=(-1, -2))
    argmax = np.argmax(split.reshape(-1, kernel_size[0] * kernel_size[1]), axis=-1).flatten()
    output = Tensor(max_data, requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, argmax, kernel_size, stride)]
        output.grad_fn = AvgPool2dBackward()
    return output


def batch_norm(x: Tensor, mean: Tensor, var: Tensor, gamma: Tensor, beta: Tensor, eps=1e-5):
    axis = (0, -1, -2)[:x.ndim - 1]
    x_hat = (x.data - mean.data) / np.sqrt(var.data + eps)
    output = Tensor(np.expand_dims(gamma.data, axis) * x_hat + np.expand_dims(beta.data, axis),
                    requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad)
    if output.grad_enable:
        output.grad_fn = BatchNormBackward()
        output.children = [(x, x_hat, mean.data, var.data, eps), (gamma, None), (beta, None)]
    return output


def avg_pool1d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 3, 'x must be 3 dimensional'
    stride = stride or kernel_size
    return avg_pool2d(x.unsqueeze(-2), kernel_size=(1, kernel_size), stride=(1, stride)).squeeze(-2)


def max_pool1d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 3, 'x must be 3 dimensional'
    stride = stride or kernel_size
    return max_pool2d(x.unsqueeze(-2), kernel_size=(1, kernel_size), stride=(1, stride)).squeeze(-2)


def embedding(x: Tensor, weight: Tensor, padding_idx=None):
    assert 'int' in x.dtype.name
    index = x.flatten().tolist()
    result = weight.index_select(0, index).reshape(*x.shape, weight.shape[1])
    if weight.grad_enable:
        result.children = [(weight, index, padding_idx)]
        result.grad_fn = EmbeddingBackward()
    return result


def cross_entropy(x: Tensor, target):
    """
    交叉熵
    :param x: 二维张量,每一行表示一个数据
    :param target: 标签,行张量,长度与x的行数相同
    :return: 交叉熵
    """
    if 'int' not in target.dtype.name:
        raise TypeError(f"cross entropy not implement for dtype of target: '{target.dtype}'")
    x_softmax = x.softmax(-1)
    log_softmax = x_softmax.log()
    n = target.size
    one_hot_target = one_hot(x.shape[-1], target)
    loss = - (one_hot_target * log_softmax).sum() / n
    if x.grad_enable:
        loss.children = [(x, x_softmax.data, one_hot_target.data)]
        loss.grad_fn = CrossEntropyBackward()
    return loss


def binary_cross_entropy(x: Tensor, target: Tensor):
    assert not target.requires_grad, 'the derivative for `target` is not implemented'
    data = x.data
    target = target.data
    result = - np.log(data + 1e-12) * target - np.log(1 - data + 1e-12) * (1. - target)
    result = Tensor(np.mean(result), requires_grad=x.requires_grad)
    if x.grad_enable:
        result.children = [(x, target)]
        result.grad_fn = BinaryCrossEntropyBackward()
    return result


def mse_loss(x: Tensor, target: Tensor):
    """
    均方误差
    @param x: 预测值
    @param target: 真实值,与真实值形状相同
    @return: 均方误差
    """
    if x.shape != target.shape:
        raise RuntimeError(f"shape of x '{x.shape}' not equals to shape of target {target.size}")
    return ((x - target) ** 2.).mean()


def nll_loss(x: Tensor, target: Tensor):
    """
    负对数似然损失函数
    @param x: 二维张量,每一行表示一个数据
    @param target: 标签,行张量,长度与x的行数相同
    @return: NLL损失
    """
    if 'int' not in target.dtype.name:
        raise TypeError(f"nll loss not implement for dtype of target: '{target.dtype}'")
    n = target.size
    one_hot_target = one_hot(x.shape[-1], target)
    loss = - (one_hot_target * x).sum() / n
    if x.grad_enable:
        loss.children = [(x, one_hot_target.data / n)]
        loss.grad_fn = NLLLossBackward()
    return loss


def pairwise_distance(x1: Tensor, x2: Tensor, p=2., keepdims=False, eps=1e-12):
    return (x1 - x2).norm(axis=1, p=p, keepdims=keepdims, eps=eps)


def cosine_similarity(x1: Tensor, x2: Tensor, axis=1, eps=1e-12):
    return (x1 * x2).sum(axes=axis) / (x1.norm(axis=axis, eps=eps) * x2.norm(axis=axis, eps=eps))
