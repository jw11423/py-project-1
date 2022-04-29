if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/selfpkg'))

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)

y.backward(create_graph = True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()

print(x.grad)