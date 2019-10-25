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
        plt.axhline(linewidth=1, color='k', linestyle='--')
        plt.axvline(linewidth=1, color='k', linestyle='--')
        plt.plot(x_number_list, y_number_list, 'rx', label='Data',linewidth=1 )
        plt.title("Original PCA data ")
        plt.legend()
        plt.show()
    def drawFinal_PCA(self, transformed_Data, DataAdjust):
        plt.axhline(linewidth=0.5, color='k', linestyle='--')
        plt.axvline(linewidth=0.5, color='k', linestyle='--')
        plt.plot(DataAdjust[:,0],DataAdjust[:,1], 'rx', label='Transformed Data',  linewidth=0.5)
        plt.plot(DataAdjust[:,0],transformed_Data[:,0],  '--', label='Data',  linewidth=0.5)
        plt.plot(DataAdjust[:,0],transformed_Data[:,1],  '--', label='Data',  linewidth=0.5)
        plt.title("Data Transformed with 2 eigenvectors ")
        plt.legend()
        plt.show()
    def DataAdjust(self, x,y, m):
        x_number_list = np.hstack((x, y))
        count_m = len(x)
        m1m2 = (1/count_m)*(np.transpose(x_number_list))*m
        DataAdjust = np.matrix([])
        col1 = x_number_list[:,0]-m1m2[0,0]
        col2 = x_number_list[:,1]-m1m2[1,0]
        DataAdjust= np.hstack((col1, col2))
        return DataAdjust
    def Calculate_PCA(self, x,y, m):
        p = PCA_LSA()
        DataAdjust= p.DataAdjust(x,y,m)
        C = np.transpose(DataAdjust) * DataAdjust
        w, v = la.eig(C)
        print("Eigenvalues: ")
        print(w)
        print("Eigenvectors: ")
        print(v)
        transformed_Data_x =[]
        transformed_Data_x1 =[]
        transformed_Data_y =[]
        transformed_Data_y1 =[]
        for row in DataAdjust:
            print(v[0,0]/v[1,0]*2)
            transformed_Data_x.append(np.squeeze(np.asarray([(v[0,0]/v[1,0])*row[0,0]])))
            transformed_Data_y.append(np.squeeze(np.asarray([(v[0,1]/v[1,1])*row[0,0]])))
            # transformed_Data_x.append(np.squeeze(np.asarray([row*(v[:,1])])))
            # transformed_Data_y.append(np.squeeze(np.asarray([row*(v[:,0])])))
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
    pca.draw_PCA(x,y)
    transformed_Data = pca.Calculate_PCA(x,y, m)
    # print(transformed_Data[:,0])
    DataAdjust= pca.DataAdjust(x,y,m)
    pca.drawFinal_PCA(transformed_Data, DataAdjust )


   