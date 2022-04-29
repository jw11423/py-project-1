import numpy as np


# Variable 선언
class Variable:
    def __init__(self, data):
        self.data = data


# Function 선언
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()        

# 구체적인 함수는 Funtion 클래스를 상속한 클래스에서 구현
# Square 정의
class Square(Function):
    def forward(self, x):
        return x ** 2
# Exp 정의
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
'''
합성 함수 (composite function)
연속해서 사용을 하는 경우 미분을 효율적으로 계산을 할 수 있음. 변수별 미분을 계산하는 알고리즘. 역전파.
'''
a = A(x)
b = B(a)
y = C(b)

print(a.data)
print(b.data)
print(y.data)