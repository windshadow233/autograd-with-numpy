import nptorch
nptorch.random.seed(0)
x = nptorch.random.randint((1,5),low=1,high=5, dtype=float, requires_grad=True)
m=x.mean()
v = x.var()
b = (x-m)/v.sqrt()
b.sum().backward()
print(x.grad)