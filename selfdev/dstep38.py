if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/selfpkg'))

import numpy as np

x = np.array([[1,2,3],[4,5,6]])
print(x, x.shape)
y = np.reshape(x, (6,))
print(y, y.shape)


from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
print(x)
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(y)

x = Variable(np.random.randn(1,2,3))
print(x)
y = x.reshape((2,3))
print(y)
y = x.reshape(2,3)
print(y)
y = x.reshape([2,3])
print(y)

# transpose

x = np.array([[1,2,3],[4,5,6]])
y = np.transpose(x)

print(y)

x = Variable(np.array([[1,2,3],[4,5,6]]))
print(x)
y = F.transpose(x)
y.backward()
print(y)
print(x.grad)

print('-----------------------------')

x = Variable(np.random.randn(2,3))
print(x)
y = x.transpose()
print(y)
y = x.T
print(y)

print('1-----------------------------')
A, B, C, D = 1,2,3,4
x1 = np.random.rand(A,B,C,D)
print(x1)
print('2-----------------------------')
y1 = x1.transpose(1,0,3,2)
print(y1)

