import nptorch
nptorch.random.seed(0)
x = nptorch.random.randint(1,low=0,high=5, dtype=float, requires_grad=True)
b = x.pow(x)
d = b/b
d.backward()
print(b.grad)
print(x.grad)