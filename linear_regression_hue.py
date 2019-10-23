import sys
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt
from math import exp, expm1
from sklearn.linear_model import LinearRegression
class Linear_Regression:
    def __init__(self):
        pass
    def draw_multiple_points(self, x, y, theta):
        x_number_list = np.asarray(x[:,1].T)[0,:]
        y_number_list = np.asarray(y.T)[0,:]
        fig, ax = plt.subplots()
        # ax.scatter(x_number_list, y_number_list,c="red", s=10)
        ax.plot(x_number_list, y_number_list, 'rx', label='training data')
        ax.plot(x_number_list, x*theta,'b',label='linear regression')
        ax.set_title("Biểu diễn đám mây dữ liệu ")
        plt.xlabel("X number")
        plt.ylabel("Y number")
        plt.legend()
        plt.show()
    def calculator(self, x, y):
        '''
        (XTX)-1 XTY
        '''
        xt= x.transpose()
        xtx= xt*x
        xtx_1 = np.linalg.inv(xtx)
        result = xtx_1*xt*y
        return result
        
if __name__ == "__main__":
    ln = Linear_Regression()
    x_number_list = np.matrix([[2],[4],[9],[16],[25]])
    y_number_list = np.matrix([[1],[2],[4],[4],[5]]) 
    x_number_list = np.insert(x_number_list, 0, 1, axis=1)
    # (XTX)-1 XTY
    theta = ln.calculator(x_number_list, y_number_list)
    print(ln.calculator(x_number_list,y_number_list))
    ln.draw_multiple_points(x_number_list, y_number_list, theta)
        