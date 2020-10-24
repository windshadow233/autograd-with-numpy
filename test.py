import nptorch

x = nptorch.array([1.,3,5], requires_grad=True)
a = x * 2
b = a.log()
b.retain_grad()
b.sum().backward()
