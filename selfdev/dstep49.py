if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import math
import numpy as np
from dezero import dezero
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F


train_set = dezero.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))

batch_index = [0, 1, 2] 
batch = [train_set[i] for i in batch_index]

x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])
print(x.shape)
print(t.shape)

max_epoch = 10000
batch_size = 30
hidden_size = 20
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0
    for i in range(max_iter):
        batch_index = index[i * batch_size:(i+1)*batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data)*len(batch_t)

    #에포크마다 출력
    avg_loss = sum_loss / data_size
    print('epoch %d loss %.2f' % ( epoch +1, avg_loss))