if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))


from doctest import Example
import numpy as np
from dezero import dezero


train_set = dezero.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))

batch_index = [0, 1, 2] 
batch = [train_set[i] for i in batch_index]

x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])
print(x.shape)
print(t.shape)