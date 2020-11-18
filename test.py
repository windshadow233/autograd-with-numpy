import nptorch
from nptorch import nn

nptorch.random.seed(0)
x = nptorch.random.rand(5, 10, requires_grad=True)
# rnn = nn.LSTM(6, 5, 1, use_bias=True, batch_first=True)
# y = rnn(x)
# y[0].sum().backward()
