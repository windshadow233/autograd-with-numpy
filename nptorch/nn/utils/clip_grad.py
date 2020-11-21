from nptorch.tensor import Tensor


def clip_grad_value_(parameters, clip_value):
    """
    基于数值的梯度截断
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.clamp_(min=-clip_value, max=clip_value)
