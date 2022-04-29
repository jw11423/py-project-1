if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/selfpkg'))

import numpy as np
from dezero import Variable

x = np.array([1,2,3])
print(x)
y = np.broadcast_to(x, (2,3))
print(y)

print('----------------------')

from dezero.utils import sum_to

x = np.array([[1,2,3],[4,5,6]])
y = sum_to(x, (1,3))
print(y)

y = sum_to(x, (2,1))
print(y)



print('1----------------------')
x0 = np.array([1,2,3])
x1 = np.array([10])
y = x0 + x1
print(y)

x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

