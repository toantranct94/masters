import sys
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt
from math import exp
class PCA_LSA:
    def __init__(self):
        pass
    def draw(self, x):
        x_number_list = np.asarray(x[:,0].T)[0,:]
        y_number_list = np.asarray(x[:,1].T)[0,:]
        fig, ax = plt.subplots()
        # ax.scatter(x_number_list, y_number_list,c="red", s=10)
        ax.plot(x_number_list, y_number_list, 'rx', label='training data')
        # ax.plot(x_number_list, x*theta,'b',label='linear regression')
        ax.set_title("Biểu diễn đám mây dữ liệu ")
        # plt.xlabel("X number")
        # plt.ylabel("Y number")
        plt.legend()
        plt.show()
    def calculator(self, x, m):
        '''
        
        '''
        count_m = len(x)
        m1m2 = (1/count_m)*((np.transpose(x))*m)
        return m1m2
if __name__ == "__main__":
    pca = PCA_LSA()
    X = np.matrix([(2, 6), (4, 8),(6, 10)])
    m = np.matrix([(1, 1), (1, 1),(1, 1)]) 
    print(pca.calculator(X, m))


   