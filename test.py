import nptorch

nptorch.random.seed(0)
x = nptorch.random.rand((2,3,4,4), requires_grad=True)
c = nptorch.nn.BatchNorm2d(3, affine=False, track_running_stats=True)
y = c(x)
y.sum().backward()
