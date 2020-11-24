import nptorch
from nptorch import nn

nptorch.random.seed(0)
x = nptorch.random.randint(0, 10, size=(2,3,6), dtype=float, requires_grad=True)
conv = nn.MaxPool1d(kernel_size=3, stride=1)
y = conv(x)
y.sum().backward()
