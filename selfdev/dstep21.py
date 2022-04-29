import numpy as np
import traceback
import unittest
import weakref
import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    def __init__(self, data, name = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data # 통상값, 다차원배열(nparray)
        self.name = name
        self.grad = None # 미분값, 다차원배열(nparray), None에서 역전파 시 미분값 대입
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    # def backward(self):
    #     f = self.creator                    #1. 함수를 가져온다.
    #     if f is not None:
    #         x = f.input                     #2. 함수의 입력을 가져온다.
    #         x.grad = f.backward(self.grad)  #3. 함수의 backward 메서드를 호출 한다.
    #         x.backward()                    # 하나 앞의 변수의 backward 메서드를 호출한다.

    def backward(self, retain_grad=False):
        '''
        추가
        '''
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # funcs = [self.creator]
        '''추가'''
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key = lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()             #함수를 가져온다.
            ''' 입출력이 1 개
            x, y = f.input, f.output    #함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad) 
            if x.creator is not None:
                funcs.append(x.creator) #하나 앞의 함수를 리스트에 추가한다.
            '''
            # gys = [output.grad for output in f.outputs]
            gys = [ output().grad for output in f.outputs ]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else :
                    x.grad = x.grad + gx
                if x.creator is not None:
                    # funcs.append(x.creator)
                    add_func(x.creator)

            if not retain_grad :
                for y in f.outputs:
                    y().grad = None # y는 약한 참조(weakref)
                
    
    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n'+ ' '*9)
        return 'variable('+p+')'

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):

        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs =[ Variable(as_array(y)) for y in ys ]

        if Config.enable_backprop :
            self.generation = max([ x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # self.outputs = outputs
            self.outputs = [weakref.ref(output) for output in outputs ]
            # return outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()

    #미분을 계산하는 역전파   
    def backward(self, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        # x = self.input.data
        x = self.inputs[0].data
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

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1 , gy * x0



def square(x):
    f = Square()
    return f(x)

def exp(x):
    return Exp()(x)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    # def test_gradient_check(self):
    #     print("-----test_gradient_check")
    #     x = Variable(np.random.rand(1))
    #     y = square(x)
    #     y.backward()
    #     num_grad = numerical_diff(square, x)
    #     #print(num_grad.data)
    #     # print(x.grad)
    #     flg = np.allclose(x.grad, num_grad)
    #     #print(flg)
    #     self.assertTrue(flg)

    def test_gradient_check(self):
        print("-1-")
        x = Variable(np.random.rand(1))
        print(x.data)
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        print(num_grad)
        print(x.grad)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add( x0, x1):
    x1 = as_array(x1)
    return Add() (x0, x1)


print("-20-")

Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

try:
   
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    print(y)

    x = Variable(np.array(2.0))
    y = 2.0 * x
    print(y)
    
    x = Variable(np.array(2.0))

    print(x, type(x))
    y = x + np.array(3.0) 

    print(y)

    print("-21-")

    
except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
    print('예외가 발생했습니다.', e)
    errStr = traceback.format_exc() 
    print("--errStr.s--")
    print(errStr)
    print("--errStr.e--")
