import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data # 통상값, 다차원배열(nparray)
        self.grad = None # 미분값, 다차원배열(nparray), None에서 역전파 시 미분값 대입


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    #미분을 계산하는 역전파   
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self. input.data
        gx = np.exp(x) * gy
        return gx


## 순전파
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(x.data)
print(a.data)
print(b.data)
print(y.data)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print("-----------------------")
print(y.grad)
print(b.grad)
print(a.grad)
print(x.grad)

    