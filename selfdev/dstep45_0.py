if '__file__' in globals():
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__), '/home/jwkim/dev/venv03/deep-learning-from-scratch-3'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../selfpkg'))

from operator import mod
import dezero.functions as F
import dezero.layers as L
from dezero import Layer


model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

# 추론 수행 함수
def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

# 모든 매개변수에 접근
for p in model.params():
    print(p)


#모든 매개변수의 기울기를 재설정
model.cleargrads()

