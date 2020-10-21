import nptorch

x = nptorch.Tensor([2.], requires_grad=True)
a = x.pow(2.0)
z = a / (a.sum() * 2)
w = z.sum()
w.backward()
print(x.grad)