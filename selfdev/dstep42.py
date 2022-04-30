if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from matplotlib import pyplot as plt
from dezero import Variable
import dezero.functions as F


#토이 데이터 셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    y = F.sum(diff ** 2) / len(diff)
    return y

lr = 0.1
iters = 100

for i in range(iters):
    print(i)
    print('------ 11 ------')
    print(type(x))
    y_pred = predict(x)
    # print(W.data, W.grad.data)
    print(type(y_pred))
    print('------ 22 ------')
    print(type(y), type(y_pred))
    loss = mean_squared_error(y, y_pred)
    print('------ 33 ------')
    print(type(loss))

    # print(x,'\n', y,'\n', y_pred)

    # print(y_pred, loss)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    print(type(W.data), type(W.grad.data), type(lr), type(b.data), type(W.grad))
    # print( W.data - lr * W.grad.data )
    # print( W.data - lr * W.grad )
    print('------ 1 ------')

    W.data = W.data - lr * W.grad.data
    b.data = b.data - lr * b.grad.data
    print(W, b, loss)
    print('------ 2 ------')


plt.scatter(x.data, y.data, s=10)
plt.ylabel('y')
plt.xlabel('x')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.savefig('../img_test/dstep42.png')