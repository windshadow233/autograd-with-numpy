import nptorch

x = nptorch.Tensor([2.,3], requires_grad=True)
a = x+0
a.pow_(2)
b=a
z = a / (b.sum())
w = z.sum()+b.sum()
w.backward()
print(x.grad)