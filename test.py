import nptorch
nptorch.random.seed(0)
x = nptorch.random.rand(size=(5), dtype=float, requires_grad=True)