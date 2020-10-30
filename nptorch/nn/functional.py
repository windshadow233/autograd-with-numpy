from ..tensor import Tensor, float32
from .. import random
from ..autograd.backward import CrossEntropyBackward, Conv2dBackward, MeanPool2dBackward, MaxPool2dBackward,\
    LeakyReLUBackward, ELUBackward, BatchNorm2dBackward, MeanPool1dBackward, MaxPool1dBackward, NLLLossBackward
from .utils.conv_operations import *


def relu(x: Tensor):
    return x.relu()


def relu_(x: Tensor):
    x.relu_()
    return x


def sigmoid(x: Tensor):
    return x.sigmoid()


def softmax(x: Tensor, dim):
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
    output = x.matmul(w.T)
    if b is not None:
        output = output + b
    return output


def dropout(x: Tensor, p=0.5, training=True):
    """
    普通dropout,每个元素有p的概率置0
    @param x: 输入的张量
    @param p: 失活率
    @param training: 是否处于训练模式
    """
    assert 0. <= p <= 1., f'dropout probability has to be between 0 and 1, but got {p}'
    y = x / (1 - p) * (random.rand_like(x) > p).float() if training else x
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
    y = x / (1 - p) * (random.rand((1, x.shape[1], 1, 1)) > p).float() if training else x
    return y


def conv2d(x: Tensor, kernels: Tensor, bias: Tensor = None, stride=(1, 1), padding=(0, 0), dilation=(0, 0)):
    assert x.ndim == 4, 'x must be 4 dimensional'
    b, c, h, w = x.shape
    oc, ic, kh, kw = kernels.shape
    assert c == ic, 'Conv2d channels not equal'
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    data = x.data
    padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    data = padding_zeros(data, padding)
    dilated_kernels = dilate(kernels.data, dilation)
    split = split_by_strides(data, *dilated_kernels.shape[-2:], stride=stride)
    output = Tensor(np.tensordot(split, dilated_kernels, axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2)),
                    requires_grad=x.requires_grad)
    if bias is not None:
        output = output + bias.unsqueeze(-1, -2)
    if output.grad_enable:
        output.children = [(x, padding), (kernels, dilated_kernels, stride, dilation)]
        if bias is not None:
            output.children.append((bias, None))
        output.grad_fn = Conv2dBackward()
    return output


def mean_pool2d(x: Tensor, kernel_size, stride):
    assert x.ndim == 4, 'x must be 4 dimensional'
    stride = stride or (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    split = split_by_strides(x.data, kernel_size, kernel_size, stride)
    mean_data = np.mean(split, axis=(-1, -2))
    output = Tensor(mean_data, requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, kernel_size, stride)]
        output.grad_fn = MeanPool2dBackward()
    return output


def max_pool2d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 4, 'x must be 4 dimensional'
    stride = stride or (kernel_size, kernel_size)
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    split = split_by_strides(x.data, kernel_size, kernel_size, stride)
    max_data = np.max(split, axis=(-1, -2))
    argmax = np.argmax(split.reshape(-1, kernel_size * kernel_size), axis=-1).flatten()
    output = Tensor(max_data, requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, argmax, kernel_size, stride)]
        output.grad_fn = MaxPool2dBackward()
    return output


def batch_norm2d(x: Tensor, mean: Tensor, var: Tensor, gamma: Tensor, beta: Tensor, eps=1e-5):
    assert x.ndim == 4, 'x must be 4 dimensional'
    x_hat = (x.data - mean.data) / np.sqrt(var.data + eps)
    output = Tensor(gamma.unsqueeze(0, -1, -2).data * x_hat + beta.unsqueeze(0, -1, -2).data,
                    requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad)
    if output.grad_enable:
        output.grad_fn = BatchNorm2dBackward()
        output.children = [(x, x_hat, mean.data, var.data, eps), (gamma, None), (beta, None)]
    return output


def mean_pool1d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 3, 'x must be 3 dimensional'
    stride = stride or kernel_size
    assert isinstance(stride, int), f'stride must be int. Got{type(stride)}'
    split = split_by_strides(x.data, 1, kernel_size, (1, stride))
    output = Tensor(split.mean(-1).squeeze(-1), requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, kernel_size, stride)]
        output.grad_fn = MeanPool1dBackward()
    return output


def max_pool1d(x: Tensor, kernel_size, stride=None):
    assert x.ndim == 3, 'x must be 3 dimensional'
    stride = stride or kernel_size
    assert isinstance(stride, int), f'stride must be int. Got {type(stride)}'
    split = split_by_strides(x.data, 1, kernel_size, (1, stride))
    max_data = np.max(split, axis=-1).squeeze(-1)
    argmax = np.argmax(split.reshape(-1, kernel_size), axis=-1).flatten()
    output = Tensor(max_data, requires_grad=x.requires_grad)
    if output.grad_enable:
        output.children = [(x, argmax, kernel_size, stride)]
        output.grad_fn = MaxPool1dBackward()
    return output


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
        loss.children = [(x, x_softmax.data, one_hot_target)]
        loss.grad_fn = CrossEntropyBackward()
    return loss


def mse_loss(x: Tensor, target: Tensor):
    """
    均方误差
    @param x: 预测值
    @param target: 真实值,与真实值形状相同
    @return: 均方误差
    """
    if x.shape != target.shape:
        raise RuntimeError(f"shape of x '{x.shape}' not equals to shape of target {target.size}")
    return ((x - target) ** 2).mean()


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
        loss.children = [(x, one_hot_target / n)]
        loss.grad_fn = NLLLossBackward()
    return loss
