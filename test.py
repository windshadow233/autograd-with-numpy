import nptorch

nptorch.random.seed(0)
x = nptorch.random.randint(size=(2, 3), low=0, high=5, dtype=float, requires_grad=True)
y = (x - x.mean(-1, keepdims=True)) / x.var(-1, keepdims=True)
y.sum().backward()
