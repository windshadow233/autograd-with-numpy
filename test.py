import nptorch

nptorch.random.seed(0)
x = nptorch.array([2., 3, 4, 5], requires_grad=True)
a = 2 * x
a += x
a *= 2
a.sum().backward()
print(x.grad)
