if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1,2,3,4,5,6]))
y = F.sum(x)
y.backward()

print(y)
print(x.grad)

print('---------------')

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
y.backward()

print(y)
print(x.grad)

print('1---------------')

x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, axis=0)
print(y)
print(x.shape, '->', y.shape)

x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, keepdims=True)
print(y)
print(x.shape, y.shape)

print('2----------------')

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
y.backward()

print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
# print(x)
y = x.sum(keepdims=True)
print(y.shape)


# print(x, x.shape, y.shape)