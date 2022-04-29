
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/selfpkg'))

import numpy as np
from dezero import Variable
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 학습률
iters = 10 # 반복횟수

plt.figure()

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    print(x0.data, x1.data, y.data)

    # plt.plot(x0.data, x1.data, 'o-', color='blue')
    # plt.plot( [x0.data, y.data], [y.data,x1.data], '--', color='blue')

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

x0 = np.linspace(-2.0,2.0,50)
x1 = np.linspace(-1.0,3.0,50)



X0,X1 = np.meshgrid(x0,x1)

Y2 = rosenbrock(X0, X1) # x2 ** 4 - 2 * x2 ** 2

ax3d = plt.axes(projection='3d')
# ax3d.plot_surface(X2, X3, Y2,cmap='plasma')
ax3d.plot_surface(X1, X0, Y2, cmap='viridis')
ax3d.set_title('Surface Plot in Matplotlib')
ax3d.set_xlabel('X1')
ax3d.set_ylabel('X0')
ax3d.set_zlabel('Y2')

plt.gca().invert_xaxis()
# # y3 = rosenbrock(x3, x2)
# plt.title('fit')
# print(y2)
# # plt.axis([-2, 2, -2, 10]) # x축은 0~3 까지, y축은 2~5 까지 보여줍니다
# # plt.plot( [x2, y2], [x3, y3], '--', color='blue')
# # plt.plot( y2, x3, '--', color='red')
# # plt.plot( y2, x2, '--', color='blue')
# plt.plot_surface(x1.data, x2.data,  y2.data, '--', color='orange')

plt.savefig('ztest001.png')
