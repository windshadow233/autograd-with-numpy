import nptorch
import numpy as np
nptorch.random.seed(0)
x = nptorch.random.rand((3,4,4), requires_grad=True)
meanpool = nptorch.nn.MaxPool1d(2, 3)
y = meanpool(x)
y.log_()
y.sum().backward()
print(x.grad)
