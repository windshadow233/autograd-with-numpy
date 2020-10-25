import nptorch

nptorch.random.seed(0)
x = nptorch.random.rand((2,3,4,4), requires_grad=True)
bn = nptorch.nn.BatchNorm2d(3)
y = bn(x)
y.sum().backward()