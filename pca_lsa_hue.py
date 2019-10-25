import sys
import numpy as np
from sympy import Symbol
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
import re
class PCA:
    def __init__(self):
        pass
    '''
    Hàm vẽ mô tả PCA 
    '''
    def draw_PCA(self, x, y):
        x_number_list = np.asarray(x[:,0].T)[0,:]
        y_number_list = np.asarray(y[:,0].T)[0,:]
        plt.axhline(linewidth=1, color='k', linestyle='--')
        plt.axvline(linewidth=1, color='k', linestyle='--')
        plt.plot(x_number_list, y_number_list, 'rx', label='Data',linewidth=1 )
        plt.title("Original PCA data ")
        plt.legend()
        plt.show()
    '''
    Hàm vẽ PCA sao khi tính toán
    '''
    def drawFinal_PCA(self, transformed_Data, DataAdjust):
        plt.axhline(linewidth=0.5, color='k', linestyle='--')
        plt.axvline(linewidth=0.5, color='k', linestyle='--')
        plt.plot(DataAdjust[:,0],DataAdjust[:,1], 'rx', label='Transformed Data',  linewidth=0.5)
        plt.plot(DataAdjust[:,0],transformed_Data[:,0],  '--', label='Data',  linewidth=0.5)
        plt.plot(DataAdjust[:,0],transformed_Data[:,1],  '--', label='Data',  linewidth=0.5)
        plt.title("Data Transformed with 2 eigenvectors ")
        plt.legend()
        plt.show()
    '''
    Tính trung bình
    '''
    def DataAdjust(self, x,y, m):
        x_number_list = np.hstack((x, y))
        count_m = len(x)
        m1m2 = (1/count_m)*(np.transpose(x_number_list))*m
        DataAdjust = np.matrix([])
        col1 = x_number_list[:,0]-m1m2[0,0]
        col2 = x_number_list[:,1]-m1m2[1,0]
        DataAdjust= np.hstack((col1, col2))
        return DataAdjust
    '''
    Hàm tính
    '''
    def Calculate_PCA(self, x,y, m):
        p = PCA()
        DataAdjust= p.DataAdjust(x,y,m)
        C = np.transpose(DataAdjust) * DataAdjust
        w, v = la.eig(C)
        print("Eigenvalues: ")
        print(w)
        print("Eigenvectors: ")
        print(v)
        transformed_Data_x =[]
        transformed_Data_y =[]
        transformed_Data_final =[]
        for row in DataAdjust:
            transformed_Data_x.append(np.squeeze(np.asarray([(v[0,0]/v[1,0])*row[0,0]])))
            transformed_Data_y.append(np.squeeze(np.asarray([(v[0,1]/v[1,1])*row[0,0]])))
            transformed_Data_final.append(np.squeeze(np.asarray([row*(v[:,1])])))
            # transformed_Data_y.append(np.squeeze(np.asarray([row*(v[:,0])])))
        transformed_Data_ts = np.vstack([np.transpose(transformed_Data_x), np.transpose(transformed_Data_y)])
        transformed_Data = np.transpose(transformed_Data_ts)
        print("Transformed Data final: ")
        print( np.transpose(np.vstack((transformed_Data_final))))
        return transformed_Data
class LSA:
    def __init__(self):
        pass
    def Convert_Docs_List(self, docs):
        wordList = re.sub("[^\w]", " ", docs).split()
        return wordList
    def SVD(self, A):
        k=2
        u, s, v = np.linalg.svd(A)
        s = np.diag(s)
        vT = np.transpose(v)
        print("STEP 2  ***********************************************************")
        print("U: \n", u)
        print("S: \n", s)
        print("V: \n", v)
        print("VT: \n", vT)
        u = u[:, :k]
        s = s[:k, :k]
        v = v[:, :k]
        print("STEP 3  *****************************************************************")
        print("U_k: \n", u)
        print("S_k: \n", s)
        print("V_k: \n", v)
        print("VT_k: \n", np.transpose(v))
        return u, s, v
    def Sim(self, q, d):
         # sim (q, d) = (q * d)/|q|*|d|
        result =[]
        for term in d:
            result.append(q @ term / (np.linalg.norm(q) * np.linalg.norm(term)))
        return result
    def Calculate_LSA(self, docs, query):
        d1_List = lsa.Convert_Docs_List(docs[0])
        d2_List = lsa.Convert_Docs_List(docs[1])
        d3_List = lsa.Convert_Docs_List(docs[2])
        query_list = lsa.Convert_Docs_List(query)
        d1_List_number =[]
        d2_List_number =[]
        d3_List_number =[]
        query_list_number =[]
        list_full=[]
        list_Not_Coincide =[]
        for term in docs:
            list_full += lsa.Convert_Docs_List(term)
        for word in list_full:
            if word not in list_Not_Coincide:
                list_Not_Coincide.append(word)
        for word in list_Not_Coincide:
            m1 = d1_List.count(word)
            d1_List_number.append(m1)
            m2 = d2_List.count(word)
            d2_List_number.append(m2)
            m3 = d3_List.count(word)
            d3_List_number.append(m3)
            mQuery = query_list.count(word)
            query_list_number.append(mQuery)
        A = np.vstack([d1_List_number, d2_List_number,d3_List_number])
        A = np.transpose(A)
        print("STEP 1  *****************************************************")
        print("A =  \n", A)
        print("q =  \n", query_list_number)
        u,s,v = lsa.SVD(A)
        vT = np.transpose(v)
        print("STEP 4  ****************************************************")
         # d = AT * u * s^-1
        d = np.transpose(A) @ u @ (np.linalg.inv(s))
        print("d1 = ", d[0])
        print("d2 = ", d[1])
        print("d3 = ", d[2])
        print("STEP 5  ****************************************************")
        # q = qT * u * s^-1
        q = np.transpose(query_list_number) @ u @ (np.linalg.inv(s))
        print("q = ", q)
        print("STEP 6  ****************************************************")
        result = lsa.Sim(q,d)
        print("sim (q, d1) = ", result[0])
        print("sim (q, d2) = ", result[1])
        print("sim (q, d3) = ", result[2])
        pass
'''
Đọc file csv
'''
class CSV:
    def __init__(self):
        pass
    def read(self, file_path=None):
        if file_path is not  None:
            df = pd.read_csv(file_path)
            x = df.values[:,0]
            y = df.values[:,1]
            return x, y
        pass
    def write(self, file_path=None, data=None):
        if data is not None and file_path is not None:
            df = pd.DataFrame(data, columns=['x', 'y'])
            df.to_csv(file_path, index=None, header=True)  
        pass

if __name__ == "__main__":
    '''
    PCA---------------------
    '''
    pca = PCA()
    x = np.matrix([[2.5],[0.5],[2.2],[1.9],[3.1],[2.3],[2],[1],[1.5],[1.1]])
    y = np.matrix([[2.4],[0.7],[2.9],[2.2],[3.0],[2.7],[1.6],[1.1],[1.6],[0.9]])
    m = np.matrix([])
    num_rows = 10
    num_cols = 2
    m = np.full((num_rows, num_cols), 1)
    # pca.draw_PCA(x,y)
    # transformed_Data = pca.Calculate_PCA(x,y, m)
    # DataAdjust= pca.DataAdjust(x,y,m)
    # pca.drawFinal_PCA(transformed_Data, DataAdjust )
    '''
    LSA---------------------
    '''
    lsa = LSA()
    docs = [
        'Shipment of gold damaged in a fire', 
        'Delivery of silver arrived in a silver truck', 
        'Shipment of gold arrived in a truck'
    ]
    query = 'gold silver truck'
    lsa.Calculate_LSA(docs,query)
   

   