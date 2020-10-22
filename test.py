import nptorch
nptorch.random.seed(0)
x = nptorch.random.randint((1,5),low=1,high=5, dtype=float, requires_grad=True)
b = x.pow(2)
c = b/b
c.sum().backward()
print(x.grad)