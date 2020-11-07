import nptorch
from nptorch import nn

x = nptorch.random.rand((3, 10), requires_grad=True)
y = nptorch.random.rand((5, 10), requires_grad=True)
p = nptorch.nn.utils.rnn.pad_sequence([x, y], True)