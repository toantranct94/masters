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
        num_rows = 6
        num_cols = 1
        m = np.full((num_rows, num_cols), 1)
        x = np.hstack(( m,x[:,0]))
        xt= x.transpose()
        print("X-transpose = \n", xt)
        xtx= xt*x
        print("X-transpose-X = \n", xtx)
        xtx_1 = np.linalg.inv(xtx)
        print("X-transpose-X inverse = \n", xtx_1)
        xty= xt*y
        print("X-transpose-Y = \n", xty)
        beta_hat = xtx_1*xt*y
        print("X-transpose-X-inverse times X-transpose-Y = \n", beta_hat)
        y_hat = x*beta_hat
        print("X beta-hat = \n", y_hat)
        errors = np.subtract(y, y_hat) #hiệu 2 ma trận
        print("Errors = \n", errors)
        squared_errors =[]
        for x in errors:
            squared_errors.append(x*x)
        print("Squared Errors \n", np.transpose(squared_errors))
        print("Sum Squared Errors \n", sum(squared_errors))
        return beta_hat
        
if __name__ == "__main__":
    ln = Linear_Regression()
    x_number_list = np.matrix([[18],[25],[15],[22],[24],[20]])
    y_number_list = np.matrix([[45],[58],[50],[54],[62],[53]])
    print ("vector X = \n", x_number_list)
    print ("vector Y = \n", y_number_list)
    # x_number_list = np.insert(x_number_list, 0, 1, axis=1)
    # (XTX)-1 XTY
    theta = ln.calculator(x_number_list, y_number_list)
    # ln.calculator(x_number_list,y_number_list)
    # ln.draw_multiple_points(x_number_list, y_number_list, theta)
        