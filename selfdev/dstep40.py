if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/selfpkg'))
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))

import numpy as np
from dezero import Variable
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


np.linspace
x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

# fig = plt.figure(figsize=(7, 7))
# ax = fig.gca(projection='3d')
# ax.scatter(x0.data,x1.data,y.data, marker='o', s=15, c='darkgreen')
# ax.set_xlabel('X0')
# ax.set_ylabel('X1')
# ax.set_zlabel('Y')
# plt.gca().invert_xaxis()
# plt.savefig('test40_01.png')

y.backward()
print(x1.grad)

# import plotly.express as px

# df = px.data.gapminder().query("country=='Brazil'")
# fig = px.line_3d(df, x="gdpPercap", y="pop", z="year")
# fig.show()

# import plotly.express as px
# df = px.data.iris()
# print(df.shape)
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')
# fig.show()