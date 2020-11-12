import nptorch
from nptorch import nn
x = nptorch.array([1], float, 1)
with nptorch.no_grad():
    y = x + 1
    y.retain_grad()
    print(y.grad_enable)
    y.backward()