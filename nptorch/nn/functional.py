from ..tensor import array, Tensor
from ..random import rand_like
from ..backward import CrossEntropyBackward, ConvBackward
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


def split_by_strides(input_data: Tensor, kernel_x, kernel_y, stride):
    """
    将张量按卷积核尺寸与步长进行分割
    :param input_data: 被卷积的张量(四维)
    :param kernel_x: 卷积核的高度
    :param kernel_y: 卷积核的宽度
    :param stride: 步长
    :return: output_data: 按卷积步骤展开后的矩阵

    Example: [[1, 2, 3, 4],    2, 2, 2       [[[[1, 2],
              [5, 6, 7, 8],             =>      [5, 6]],
              [9, 10, 11, 12],                 [[3, 4],
              [13, 14, 15, 16]]                 [7, 8]]],
                                              [[[9, 10],
                                                [13, 14]],
                                               [[11, 12],
                                                [15, 16]]]]
    """
    input_data = input_data.data
    batches, channels, x, y = input_data.shape
    out_x, out_y = (x - kernel_x) // stride + 1, (y - kernel_y) // stride + 1
    shape = (batches, channels, out_x, out_y, kernel_x, kernel_y)
    strides = (*input_data.strides[:-2], input_data.strides[-2] * stride,
               input_data.strides[-1] * stride, *input_data.strides[-2:])
    output_data = as_strided(input_data, shape, strides=strides)
    return output_data


def max_pool(x: Tensor, kernel_size, stride=None):
    stride = stride or kernel_size


def conv(x: Tensor, kernels: Tensor, bias: Tensor = None, stride=1, padding=(0, 0)):
    data = x.data
    padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    data = padding_zeros(data, padding)
    split = split_by_strides(data, *kernels.shape[-2:], stride=stride)
    output = Tensor(np.tensordot(split, kernels.data, axes=[(1, 4, 5), (1, 2, 3)]).transpose((0, 3, 1, 2)), requires_grad=x.requires_grad)
    if bias is not None:
        output = output + bias.unsqueeze(-1, -2)
    if output.requires_grad:
        output.children = [(x, None), (kernels, stride)]
        output.children = [(x, padding), (kernels, stride)]
        if bias is not None:
            output.children.append((bias, None))
        output.grad_fn = ConvBackward()
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

