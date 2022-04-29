import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data # 통상값, 다차원배열(nparray)
        self.grad = None # 미분값, 다차원배열(nparray), None에서 역전파 시 미분값 대입
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

    # def backward(self):
    #     f = self.creator                    #1. 함수를 가져온다.
    #     if f is not None:
    #         x = f.input                     #2. 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad)  #3. 함수의 backward 메서드를 호출 한다.
    #         x.backward()                    # 하나 앞의 변수의 backward 메서드를 호출한다.

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()             #함수를 가져온다.
            x, y = f.input, f.output    #함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad) 

            if x.creator is not None:
                funcs.append(x.creator) #하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

a = A(x)
b = B(a)
y = C(b)

#역전파
y.grad = np.array(1.0)

y.backward()

print(x.grad)
