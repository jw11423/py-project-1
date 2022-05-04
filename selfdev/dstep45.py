if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Variable, Model
from dezero.models import MLP



#데이터 셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 하이퍼파라미터 설정
lr = 0.2
max_iters = 10000
hidden_size = 10

# 모델정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

# x = Variable(np.random.randn(5, 10), name='x')
# model = TwoLayerNet(hidden_size, 1)
model = MLP((10, 1)) # 2층
# model = MLP((100, 1)) # 3층
# model = MLP((10, 20, 1)) # 3층
# model = MLP((10, 20, 30, 40, 1)) # 5층
# model = MLP((10, 20, 30, 40, 50, 60, 1)) # 7층

#학습 시작
for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0 :
        print(loss)



