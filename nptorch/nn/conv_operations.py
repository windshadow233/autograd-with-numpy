"""
本文件中所有函数运算对象是np.ndarray
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


def split_by_strides(input_data: np.ndarray, kernel_h, kernel_w, stride=1):
    """
    将张量按卷积核尺寸与步长进行分割
    :param input_data: 被卷积的张量
    :param kernel_h: 卷积核的长度
    :param kernel_w: 卷积核的宽度
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
    *bc, h, w = input_data.shape
    out_x, out_y = (h - kernel_h) // stride + 1, (w - kernel_w) // stride + 1
    shape = (*bc, out_x, out_y, kernel_h, kernel_w)
    strides = (*input_data.strides[:-2], input_data.strides[-2] * stride,
               input_data.strides[-1] * stride, *input_data.strides[-2:])
    output_data = as_strided(input_data, shape, strides=strides)
    return output_data


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


def dilate(x: np.ndarray, stride=1):
    """
    膨胀,按一定步长填充0
    Example: [[1, 2, 3], 2    [[1, 0, 2, 0, 3],
              [4, 5, 6],   =>  [0, 0, 0, 0, 0],
              [7, 8, 9]]       [4, 0, 5, 0, 6],
                               [0, 0, 0, 0, 0],
                               [7, 0, 8, 0, 9]]
    """
    if stride == 1:
        return x
    *bc, h, w = x.shape
    result = np.zeros((*bc, (h - 1) * stride + 1, (w - 1) * stride + 1), dtype=np.float32)
    result[..., ::stride, ::stride] = x
    return result


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
    x = split_by_strides(x, *ksize[-2:])
    if rotate:
        kernel = kernel_rotate180(kernel)
    i = 0 if invert else 1
    result = np.tensordot(x, kernel, [(i, 4, 5), (0, 2, 3)])
    if invert:
        return result.transpose((3, 0, 1, 2))
    return result.transpose((0, 3, 1, 2))
