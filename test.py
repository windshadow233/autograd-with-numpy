import nptorch

x = nptorch.array([[1.,2,3],[4,5,6]],requires_grad=False)
w = nptorch.array(1., requires_grad=True)
b = nptorch.array(0., requires_grad=True)
y = nptorch.batch_norm(x, w, b, x.mean(0), x.var(0))
y.sum().backward()