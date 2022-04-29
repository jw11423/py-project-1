
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable
from matplotlib import pyplot as plt

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx(x):
    y = 4 * x ** 3 - 4 * x 
    return y

def gx2(x):
    return 12 * x ** 2 - 4

plt.figure()


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    plt.subplot(1, 2, 2)                # nrows=2, ncols=1, index=1
    plt.plot(x.data, y.data, 'o-', color='orange')
    

    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data) ## 뉴턴 방법을 이용한 최적화 기법

x2 = np.linspace(-2.1,2.1)
y2 = x2 ** 4 - 2 * x2 ** 2

plt.title('Newton')
plt.ylabel('y')
plt.xlabel('x')
plt.plot(x2, y2)

x0 = Variable(np.array(2.0))
lr = 0.01 # 학습률
iters = 70
for i in range(iters):
    print(i, x0)
    y1 = f(x0)

    plt.subplot(1, 2, 1)                # nrows=2, ncols=1, index=2
    plt.plot(x0.data, y1.data, 'o-', color='green')

    x0.cleargrad()
    y1.backward()
    # plt.plot(x0.data, x1.data, 'o-')
    
    x0.data -= lr * x0.grad  # 경사하강법


x2 = np.linspace(-2.1,2.1)
y2 = f(x2) # x2 ** 4 - 2 * x2 ** 2
plt.title('gradiant')
plt.ylabel('y')
plt.xlabel('x')
plt.axis([-2, 2, -2, 10]) # x축은 0~3 까지, y축은 2~5 까지 보여줍니다
plt.plot(x2, y2)

plt.subplots_adjust(wspace=0.35, hspace=0.5)
plt.savefig('ztest01.png', facecolor='#eeeeee', edgecolor='blue',)

