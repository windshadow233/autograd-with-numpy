from nptorch.tensor import Tensor, inf


def clip_grad_value_(parameters, clip_value):
    """
    基于数值的梯度截断
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    for p in filter(lambda x: x.grad is not None, parameters):
        p.grad.clamp_(min=-clip_value, max=clip_value)


def clip_grad_norm_(parameters, max_norm, norm_type=2.):
    """
    基于模长的梯度截断
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda x: x.grad is not None, parameters))
    if norm_type == inf:
        norm = max(p.grad.abs().max() for p in parameters)
    else:
        norm = sum(p.grad.norm(p=norm_type).item() ** norm_type for p in parameters)
        norm = norm ** (1. / norm_type)
    clip_coef = max_norm / (norm + 1e-12)
    if clip_coef < 1:
        for p in parameters:
            p.grad *= clip_coef
