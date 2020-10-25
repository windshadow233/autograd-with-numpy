import nptorch

nptorch.random.seed(0)
x = nptorch.random.rand((2,3,4,4), requires_grad=True)
c = nptorch.nn.Dropout2d()
y = c(x)
y.sum().backward()
