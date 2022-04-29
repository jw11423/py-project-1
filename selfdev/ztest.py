

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(x.grad)
print(y)


from matplotlib import pyplot as plt
import numpy as np

x = np.arange(1,10)
y = x*5

plt.plot(x,y)

plt.savefig('test.png')
# plt.show()
