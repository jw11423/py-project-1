if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero.utils import plot_dot_graph
from dezero import Variable
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

# plot_dot_graph(y, verbose=False, to_file='test35_tanh_0grad.png')

iters = 2

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# 계산 그래프 그리기
gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='test35_tanh_3grad.png')

# https://github.com/WegraLee/deep-learning-from-scratch-3/tree/tanh