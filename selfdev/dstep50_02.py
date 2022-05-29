if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

from dezero.datasets import Spiral
from dezero import DataLoader

batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape) # x, t 는 훈련 데이터 
        break

    # 에포크 끝에서 테스트 데이터를 꺼낸다.
    for x, t in test_loader:
        print(x.shape, t.shape) # x, t 는 테스트 데이터
        break
    