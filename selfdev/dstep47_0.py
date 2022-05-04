if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
y = F.get_item(x, 1)
print(y)

y.backward()
print(x.grad)
print(x.grad.data)
print(y.grad)

print('-----------------------------------')
x = Variable(np.array([[1, 2, 3],[4, 5, 6]]))
indices = np.array([0,0,1])
y = F.get_item(x, indices)
print(y)

print('-----------------------------------')
Variable.__getitem__= F.get_item

y = x[0]
print(y)

y = x[1]
print(y)

y = x[:, 2]
print(y)

y = x[:, 1]
print(y)

y = x[:,0]
print(y)