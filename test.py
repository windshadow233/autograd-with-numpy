import nptorch
nptorch.random.seed(0)
x = nptorch.array(2., requires_grad=True)
a=x+x
b=a+a
c=b+b+x
c.backward()
print(x.grad)