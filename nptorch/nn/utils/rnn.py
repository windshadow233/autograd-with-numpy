from copy import deepcopy
import numpy as np
import nptorch
from nptorch.autograd import backward


def pad_sequence(tensors, batch_first=False, padding_value=0):
    """
    输入一个batch的张量,输出将这些张量补齐至长度相等后的张量
    @param tensors: 一个张量列表
    @param batch_first: 输出张量的第0维为batch_size
    @param padding_value: 补齐时填的数字
    @return: 补齐结果
    """
    max_length = max([tensor.shape[0] for tensor in tensors])
    B = len(tensors)
    embed_dims = list(tensors[0].shape[1:])
    dtype = tensors[0].dtype
    result_shape = (B, max_length, *embed_dims) if batch_first else (max_length, B, *embed_dims)
    result_tensor = nptorch.zeros(result_shape, dtype=dtype)
    result_tensor.fill_(padding_value)
    requires_grad = []
    for i, tensor in enumerate(tensors):
        if tensor.dtype != dtype:
            raise RuntimeError('dtype of tensors are not same')
        length, *d = tensor.shape
        if d != embed_dims:
            raise RuntimeError('embed_dims of data are not same')
        if batch_first:
            result_tensor[i, :length] = tensor
        else:
            result_tensor[:length, i] = tensor
        if tensor.grad_enable:
            requires_grad.append((tensor, i, batch_first))
    result_tensor.requires_grad = bool(requires_grad)
    if result_tensor.grad_enable:
        result_tensor.children = requires_grad
        result_tensor.grad_fn = backward.PadSequenceBackward()
    return result_tensor
