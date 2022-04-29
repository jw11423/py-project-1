if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

from statistics import variance
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))

y = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)

print(x)
print(W)
print(y)

## 에러 이유를 모르겠다

