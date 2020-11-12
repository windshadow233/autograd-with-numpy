"""
本文件中所有函数运算对象是np.ndarray
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


def split_by_strides(x: np.ndarray, kernel_size, stride=(1, 1)):
    """
    将张量按卷积核尺寸与步长进行分割
    :param x: 被卷积的张量
    :param kernel_size: 卷积核的长宽
    :param stride: 步长
    :return: y: 按卷积步骤展开后的矩阵

    Example: [[1, 2, 3, 4],    (2, 2), (2, 2)       [[[[1, 2],
              [5, 6, 7, 8],             =>             [5, 6]],
              [9, 10, 11, 12],                        [[3, 4],
              [13, 14, 15, 16]]                        [7, 8]]],
                                                     [[[9, 10],
                                                       [13, 14]],
                                                      [[11, 12],
                                                       [15, 16]]]]
    """
    *bc, h, w = x.shape
    out_h, out_y = (h - kernel_size[0]) // stride[0] + 1, (w - kernel_size[1]) // stride[1] + 1
    shape = (*bc, out_h, out_y, kernel_size[0], kernel_size[1])
    strides = (*x.strides[:-2], x.strides[-2] * stride[0],
               x.strides[-1] * stride[1], *x.strides[-2:])
    y = as_strided(x, shape, strides=strides)
    return y


def padding_zeros(x: np.ndarray, padding):
    """
    在张量周围填补0
    @param x: 需要被padding的张量,ndarray类型
    @param padding: 一个二元组,其每个元素也是一个二元组,分别表示x,y方向需要padding的数量
    @return: padding的结果
    """
    if padding == ((0, 0), (0, 0)):
        return x
    n = x.ndim - 2
    x = np.pad(x, ((0, 0),) * n + padding, 'constant', constant_values=0)
    return x


def unwrap_padding(x: np.ndarray, padding):
    """
    padding的逆操作
    @param x:
    @param padding:
    @return:
    """
    if padding == ((0, 0), (0, 0)):
        return x
    p, q = padding
    if p == (0, 0):
        return x[..., :, q[0]:-q[1]]
    if q == (0, 0):
        return x[..., p[0]:-p[1], :]
    return x[..., p[0]:-p[1], q[0]:-q[1]]


def kernel_rotate180(kernel: np.ndarray):
    return kernel[..., ::-1, ::-1]


def dilate(x: np.ndarray, dilation=(0, 0)):
    """
    膨胀,按一定步长填充0
    Example: [[1, 2, 3], 1    [[1, 0, 2, 0, 3],
              [4, 5, 6],   =>  [0, 0, 0, 0, 0],
              [7, 8, 9]]       [4, 0, 5, 0, 6],
                               [0, 0, 0, 0, 0],
                               [7, 0, 8, 0, 9]]
    """
    if dilation == (0, 0):
        return x
    *bc, h, w = x.shape
    y = np.zeros((*bc, (h - 1) * (dilation[0] + 1) + 1, (w - 1) * (dilation[1] + 1) + 1), dtype=np.float32)
    y[..., ::dilation[0] + 1, ::dilation[1] + 1] = x
    return y


def erode(x: np.ndarray, dilation=(0, 0)):
    """
    腐蚀,与膨胀互为逆运算
    """
    if dilation == (0, 0):
        return x
    y = x[..., ::dilation[0] + 1, ::dilation[1] + 1]
    return y


def reverse_conv2d(x: np.ndarray, kernel: np.ndarray, rotate=False, invert=False):
    """
    反向卷积,求梯度时用的
    @param x: 被卷积的张量
    @param kernel: 卷积核
    @param rotate: 卷积核旋转180度
    @param invert: 该参数有点迷,不好解释,简单的说就是反向卷积有两种,视卷积结果的形状需要调整一些轴的位置
    @return: 反向卷积结果
    """
    ksize = kernel.shape
    x = split_by_strides(x, ksize[-2:])
    if rotate:
        kernel = kernel_rotate180(kernel)
    i = 0 if invert else 1
    y = np.tensordot(x, kernel, [(i, 4, 5), (0, 2, 3)])
    if invert:
        return y.transpose((3, 0, 1, 2))
    return y.transpose((0, 3, 1, 2))
