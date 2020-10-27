import nptorch
import numpy as np
nptorch.random.seed(0)
x = nptorch.random.rand((2,4,4), requires_grad=True).data

