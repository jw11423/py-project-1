import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.figure()

# x1 = np.linspace(0.0, 5.0)
# # x2 = np.linspace(0.0, 2.0)

# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# # y2 = np.cos(2 * np.pi * x2)

# # plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
# plt.plot(x1, y1, 'o-')
# plt.title('1st Graph')
# plt.ylabel('Damped oscillation')

# # plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
# # plt.plot(x2, y2, '.-')
# # plt.title('2nd Graph')
# # plt.xlabel('time (s)')
# # plt.ylabel('Undamped')

# # plt.show()
# plt.savefig('savefig_edgecolor.png', facecolor='#eeeeee', edgecolor='blue')

# plt.figure()

# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')

# x0 = np.linspace(-2,2)
# x1 = np.linspace(3,-1)
# y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2

# ax.plot_surface(x0, y, x1, cmap="brg_r")

# plt.savefig('ztest.png')

import numpy as np
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)
ax.plot(x, y, z)




# x = np.arange(0, 10, 0.1)
# y = np.sin(x)
# x_m, y_m = np.meshgrid(x, y)
# z = x_m + 5 * y_m
# ax.plot_surface(x, y, z, cmap="brg_r")

plt.savefig('ztest.png')