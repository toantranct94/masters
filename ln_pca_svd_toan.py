import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matrix import Matrix, MatrixCalculator


class LinearRegression:
    def __init__(self, x=None, y=None):
        self.mt_calculator = MatrixCalculator()
        self.x = Matrix(rows=-1, cols=1, numbers=np.asarray(x)).get()
        self.y = Matrix(rows=-1, cols=1, numbers=np.asarray(y)).get()
        pass
    
    # for testing purpose
    def _set_parameters(self, a, b):
        self.a = a
        self.b = b
        pass
    # for testing purpose
    def _get_parameters(self):
        return self.a, self.b

    def visualize(self, x0=None):
        plt.plot(self.x, self.y, 'ro')
        if x0 is not None:
            a,b = self._get_parameters()
            y0 = a * x0 + b
            plt.plot(x0, y0)
        plt.show()
        pass

    def compute(self):
        # create [1,1,1...]
        x_one = np.ones((self.x.shape[0], 1))
        # stick ones to x
        x = np.concatenate((x_one, self.x), axis=1)
        # compute transpose of x
        xT = self.mt_calculator.transpose(x)
        # compute xT times x
        xTx = self.mt_calculator.multiply(xT, x)
        # compute the inverse matrix of the previous things
        xTx_inv = self.mt_calculator.inverse(xTx)
        # compute xT times y
        xTy = self.mt_calculator.multiply(xT, self.y)
        # beta_hat: coefficients matrix which includes a and b
        beta_hat = self.mt_calculator.multiply(xTx_inv, xTy)
        # predict output y
        y_hat = self.mt_calculator.multiply(x, beta_hat)
        # compute errors which compares to the real output
        errors = self.mt_calculator.subtract(self.y, y_hat)
        # dont really know what it is :v
        squared_errors = np.asarray([x*x for x in errors])
        # get a and b
        a = beta_hat[1]
        b = beta_hat[0]
        # set a and b
        self._set_parameters(a, b)

        pass

    # testing
    def predict(self, x):
        a,b = self._get_parameters()
        y = a*x + b
        print("with x = {}, y = {}".format(x, y))
        return y

class PCA:
    def __init__(self):
        pass
    def visualize(self):
        # plt.plot(self.x, self.y, 'ro')
        # plt.show()
        pass

class SVD:
    def __init__(self):
        pass

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
    file_path_ln = 'masters/data_ln.csv'

    x = [18, 25, 15, 22, 24, 20]
    y = [45, 58, 50, 54, 62, 53]
    '''
    read and write data from/to csv file
    # csv = CSV()
    # csv.write(file_path=file_path_ln, data=list(zip(x, y)))
    # csv.read(file_path=file_path_ln)
    '''

    # ln = LinearRegression(x, y)
    # ln.compute()
    # x0 = np.linspace(10, 50, 2)
    # ln.visualize(x0)
    # ln.predict(30)
    pass