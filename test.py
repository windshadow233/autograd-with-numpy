import nptorch
x = nptorch.array([1., 2, 3], requires_grad=True)
print(x.requires_grad)
with nptorch.no_grad():
    y = x * 2
    print(y.build_graph)
y = x * 2
print(y.build_graph)
print(x.requires_grad)