import nptorch

x=nptorch.array(2.,requires_grad=True)
y= x+0
y**=y
y.backward()
