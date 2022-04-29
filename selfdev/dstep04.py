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

# 수치 미분 (numerical_diff)
'''
eps : epsilon 의 약어
'''        

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)

print(dy)

# 합성 함수의 미분

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f,x)
print(dy)