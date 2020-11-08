import nptorch
from nptorch import nn

x = nptorch.random.randint((3, 4, 5), 0, 10, dtype=float, requires_grad=True)
y = x.sort(0)