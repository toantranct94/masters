import sys
import numpy as np
from sympy import Symbol
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
class PCA_LSA:
    def __init__(self):
        pass
    def draw_PCA(self, x, y):
        x_number_list = np.asarray(x[:,0].T)[0,:]
        y_number_list = np.asarray(y[:,0].T)[0,:]
        fig, ax = plt.subplots()
        # ax.scatter(x_number_list, y_number_list,c="red", s=10)
        ax.plot(x_number_list, y_number_list, 'rx', label='Data')
        # ax.plot(x_number_list, x*theta,'b',label='linear regression')
        ax.set_title("Original PCA data ")
        # plt.xlabel("X number")
        # plt.ylabel("Y number")
        plt.legend()
        plt.show()
    def drawFinal_PCA(self, transformed_Data):
        fig, ax = plt.subplots()
        ax.plot(transformed_Data[:,0], transformed_Data[:,1], 'rx', label='Transformed Data')
        ax.set_title("Data Transformed with 2 eigenvectors ")
        plt.legend()
        plt.show()
    def Calculate_PCA(self, x,y, m):
        '''
        
        '''
        x_number_list = np.hstack((x, y))
        count_m = len(x)
        m1m2 = (1/count_m)*(np.transpose(x_number_list))*m
        DataAdjust = np.matrix([])
        col1 = x_number_list[:,0]-m1m2[0,0]
        col2 = x_number_list[:,1]-m1m2[1,0]
        DataAdjust= np.hstack((col1, col2))
        C = np.transpose(DataAdjust) * DataAdjust
        w, v = la.eig(C)
        print("Eigenvalues: ")
        print(w)
        print("Eigenvectors: ")
        print(v)
        transformed_Data_x =[]
        transformed_Data_y =[]
        for row in DataAdjust:
            row*(v[:,1])
            row*(v[:,0])
            transformed_Data_x.append(np.squeeze(np.asarray([row*(v[:,1])])))
            transformed_Data_y.append(np.squeeze(np.asarray([row*(v[:,0])])))
        transformed_Data_ts = np.vstack([np.transpose(transformed_Data_x), np.transpose(transformed_Data_y)])
        transformed_Data = np.transpose(transformed_Data_ts)
        print("Transformed Data: ")
        print(transformed_Data)   
        return transformed_Data
if __name__ == "__main__":
    pca = PCA_LSA()
    x = np.matrix([[2.5],[0.5],[2.2],[1.9],[3.1],[2.3],[2],[1],[1.5],[1.1]])
    y = np.matrix([[2.4],[0.7],[2.9],[2.2],[3.0],[2.7],[1.6],[1.1],[1.6],[0.9]])
    m = np.matrix([])
    num_rows = 10
    num_cols = 2
    m = np.full((num_rows, num_cols), 1)
    # pca.draw_PCA(x,y)
    transformed_Data = pca.Calculate_PCA(x,y, m)
    # print(transformed_Data[:,0])
    pca.drawFinal_PCA(transformed_Data)


   