import numpy as np 
from sympy import Symbol
from math import e
import matplotlib.pyplot as plt

s = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
x = Symbol('x')

class Optimizer:
    def __init__(self):
        pass

    def derivative_calculator(self, function_fx):
        if isinstance(function_fx, str):
            function_fx = eval(function_fx)
        derivative = function_fx.diff(x)
        return derivative

    def find_extreme(self, function_fx, a, x):
        derivative_1 = self.derivative_calculator(function_fx)
        print("Đạo hàm bậc nhất: \n", derivative_1)
        print("Đạo hàm bậc nhất bằng 0 khi x = {}".format(x))

        derivative_2 = self.derivative_calculator(derivative_1)
        print("Đạo hàm bậc 2: \n", derivative_2)
        derivative_2_value = eval(str(derivative_2))
        print("Khi x = {}, giá trị của đạo hàm bậc 2: \n".format(x), derivative_2_value)

        if derivative_2_value > 0:
            # min
            print("x = {} là điểm cực tiểu".format(x))
            pass
        else:
            #max
            print("x = {} là điểm cực đại".format(x))
            pass

    def draw(self, function_fx, x):
        function_fx = str(function_fx)
        y = eval(function_fx)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        plt.xlabel('f(x) = {}'.format(function_fx), fontsize=18)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.show()
        pass

    def draw_two(self, function_fx, function_gx, x):
        function_fx = str(function_fx)
        y = eval(function_fx)
        function_gx = str(function_gx)
        y_gx = eval(function_gx)
        fig, ax = plt.subplots()

        ax.plot(x, y, label='f(x)')
        ax.plot(x, y_gx, label='g(x)')
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.legend()
        plt.show()
        pass

    def bisection_algorithm(self, function_gx):
        '''
        function_gx > 0 where x > 2
        otherwise, function_gx < 0 where x < 2
        '''
        function_gx = str(function_gx)
        # init x1, x2
        # x1, x2 = -15, 15
        x1, x2 = -2.25, 5.5
        rate = 0.0001
        while True:
            x = (x1 + x2) / 2
            y = str(self.derivative_calculator(function_gx))
            y = eval(y)
            if y > 0:
                x2 = x
            else:
                x1 = x
        
            if abs(y) <= rate:
                print("Cực trị gần đúng là:  {}".format(x))
                return x

        pass


if __name__ == "__main__":
    a = 1
    function_fx = 'x*e**(-{}*x)'.format(a)
    op = Optimizer()
    '''
    f(x) = x.e^-ax
    '''
    print("f(x) = x.e^-ax")
    _x = 1/a
    op.find_extreme(function_fx, a, _x)
    x_fx = np.linspace(-0.75,5,100)
    # x_fx = np.linspace(-5,15,200)
    # op.draw(function_fx, x_fx)

    print("*"*100)
    '''
    g(x) = f'(x)
    '''
    print("g(x) = f'(x)")
    function_gx = op.derivative_calculator(function_fx)
    _x = 2/a
    x_gx = np.linspace(-1.5,5,100)
    op.find_extreme(function_gx, a, _x)
    # op.draw(function_gx, x_gx)

    # draw both function
    op.draw_two(function_fx, function_gx, x_fx)

    # implement bisection algorithm for finding approximate extreme value
    op.bisection_algorithm(function_gx)
