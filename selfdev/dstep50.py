if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
import dezero
import dezero.functions as F
from dezero.datasets import Spiral
from dezero import DataLoader
from dezero.models import MLP
from dezero import optimizers

max_epoch = 300
batch_size = 30

# batch_size = 10
# max_epoch = 1
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

history_train_loss = []
history_train_acc = []
history_test_loss = []
history_test_acc = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch:{}'.format(epoch + 1))
    print('train loss: {:4f}, accuracy: {:4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0

    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))


