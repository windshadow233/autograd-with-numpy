import nptorch

x = nptorch.Tensor([2.,3], requires_grad=True)
a = x.pow(2.0)
z = a / (a.sum())
w = z.sum()
w.backward()
print(x.grad)