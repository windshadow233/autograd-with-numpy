import nptorch
x = nptorch.array([1,2,3.],requires_grad=True)
print(x.requires_grad)
with nptorch.no_grad():
    print(x.requires_grad)
    y = x + 1
    print(y)
y = x+1
print(x.requires_grad)