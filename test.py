import nptorch


nptorch.random.seed(0)
x = nptorch.random.randint(size=(2, 3,4,4),low=0, high=5, dtype=float, requires_grad=True)
y = nptorch.mean_pool(x, 2)
y.tan().sum().backward()
print(x.grad)