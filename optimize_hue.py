import numpy as np 
from sympy import Symbol
from math import e
import matplotlib.pyplot as plt

s = ['a']
x = Symbol('x')
class Optimizer:
    def __init__(self):
        pass
#tính đạo hàm
    def derivative_calculator(self, pt):
        if isinstance(pt, str):
            pt = eval(pt) # đổi kiểu chuổi về dạng biểu thức
        return pt.diff(x) # tính đạo hàm
#tìm điểm cực trị
    def find_extreme(self, pt, a, x):
        derivative_1 = self.derivative_calculator(pt) #đạo hàm bậc 1
        print("Đạo hàm bậc nhất: y' = ", derivative_1)
        print("Đạo hàm bậc nhất bằng 0 khi x = {}".format(x))
        derivative_2 = self.derivative_calculator(derivative_1) #đạo hàm bậc 2
        print("Đạo hàm bậc 2: y'' = ", derivative_2)
        derivative_2_value = eval(str(derivative_2)) # đổi kiểu chuổi về đạng biểu thức
        print("Khi x = {}, giá trị của đạo hàm bậc 2: \n".format(x), derivative_2_value)
        if derivative_2_value > 0:
            # min
            print("x = {} là điểm cực tiểu".format(x))
            pass
        else:
            #max
            print("x = {} là điểm cực đại".format(x))
            pass
# vẽ đồ thị hàm số 2
    def draw(self, pt, x):
        pt = str(pt)
        y = eval(pt)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        plt.xlabel('f(x) = {}'.format(pt), fontsize=18)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        plt.show()
        pass
# vẽ đồ thị hàm số 2
    def draw_two(self, pt_fx, pt_gx, x):
        pt_fx = str(pt_fx)
        y = eval(pt_fx)
        pt_gx = str(pt_gx)
        y_gx = eval(pt_gx)
        fig, ax = plt.subplots()
        ax.set_title('Đồ thị hàm số', color='k',  fontsize=18)
        ax.plot(x, y, label='f(x)')
        ax.plot(x, y_gx, label='g(x)')
        ax.set_aspect('equal')
        ax.grid(True, which='both')
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.legend()
        plt.show()
        pass
# giải thuật chia đôi
    def bisection_algorithm(self, pt):
        '''
        function_gx > 0 where x > 2
        otherwise, function_gx < 0 where x < 2
        '''
        function_gx = str(pt)
        # init x1, x2
        x1, x2 = -2.25, 5.5
        rate = 0.0001
        while True:
            x = (x1 + x2) / 2
            y = str(self.derivative_calculator(pt))
            y = eval(y)
            if y > 0:
                x2 = x
            else:
                x1 = x
        
            if abs(y) <= rate:
                print("Extreme value is {} approximately".format(x))
                return x

        pass

if __name__ == "__main__":   
    '''
    f(x) = x.e^-ax
    '''
    print("f(x) = x.e^-ax")
    a = 0.1
    _x = 1/a
    pt_fx = 'x*e**(-{}*x)'.format(a)
    op = Optimizer()
    op.find_extreme(pt_fx, a, _x)
    x_fx = np.linspace(-0.75,5,100)
    # op.draw(function_fx, x_fx)

    print("*"*100)
    '''
    g(x) = f'(x)
    '''
    print("g(x) = f'(x)")
    pt_gx = op.derivative_calculator(pt_fx)
    _x = 2/a
    x_gx = np.linspace(-1.5,5,100)
    op.find_extreme(pt_gx, a, _x)
    # draw both function
    op.draw_two(pt_fx, pt_gx, x_fx)
    # implement bisection algorithm for finding approximate extreme value
    op.bisection_algorithm(pt_gx)
