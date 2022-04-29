### Sum 을 아직 배우지 않음.


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([1.0, 2.0]))
y = Variable(np.array([4.0, 5.0]))

def f(x):
    t = x ** 2
    y = F