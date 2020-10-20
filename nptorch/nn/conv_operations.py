"""
本文件中所有函数运算对象是np.ndarray
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided


def split_by_strides(input_data, kernel_x, kernel_y, stride=1):
    """
    将张量按卷积核尺寸与步长进行分割
    :param input_data: 被卷积的张量(四维)
    :param kernel_x: 卷积核的长度
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
    b, c, x, y = input_data.shape
    out_x, out_y = (x - kernel_x) // stride + 1, (y - kernel_y) // stride + 1
    shape = (b, c, out_x, out_y, kernel_x, kernel_y)
    strides = (*input_data.strides[:-2], input_data.strides[-2] * stride,
               input_data.strides[-1] * stride, *input_data.strides[-2:])
    output_data = as_strided(input_data, shape, strides=strides)
    return output_data


def padding_zeros(x, padding):
    if padding == ((0, 0), (0, 0)):
        return x
    b, c, h, w = x.shape
    p, q = padding
    result = np.zeros((b, c, h + p[0] + p[1], w + q[0] + q[1]))
    result[:, :, p[0]:-p[1], q[0]:-q[1]] = x
    return result


def unwrap_padding(x, padding):
    if padding == ((0, 0), (0, 0)):
        return x
    p, q = padding
    return x[..., p[0]:-p[1], q[0]:-q[1]]


def kernel_rotate180(kernel):
    return kernel[..., ::-1, ::-1]


def insert_zero(x, stride=1):
    """
    按一定步长填充0
    Example: [[1, 2, 3], 2    [[1, 0, 2, 0, 3],
              [4, 5, 6],   =>  [0, 0, 0, 0, 0],
              [7, 8, 9]]       [4, 0, 5, 0, 6],
                               [0, 0, 0, 0, 0],
                               [7, 0, 8, 0, 9]]
    """
    if stride == 1:
        return x
    input_shape = x.shape
    result = np.zeros((input_shape[0], input_shape[1], (input_shape[2] - 1) *
                       stride + 1, (input_shape[3] - 1) * stride + 1), dtype=np.float32)
    result[..., ::stride, ::stride] = x
    return result


def batch_transposed_conv(x, kernel, rotate=False, invert=False):
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
