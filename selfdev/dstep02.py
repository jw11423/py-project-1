import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data


"""
class Function:
    def __call__(self, input):
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output

x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)


Fuction 클래스는 기반 클래스로서 모든 함수에 공통으로 되는 기능을 구현
구체적인 함수는 Funtion 클래스를 상속한 클래스에서 구현
"""

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

x = Variable(np.array([[10,22],[34,55]]))
# x = Variable(np.array(10,22))
f = Square()
y = f(x)

print(type(y))
print(y.data)
"""
<class '__main__.Variable'>
[[ 100  484]
 [1156 3025]]
"""