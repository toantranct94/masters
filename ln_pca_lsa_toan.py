import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

from matrix import Matrix, MatrixCalculator


class LinearRegression:
    def __init__(self, x=None, y=None):
        self.mt_calculator = MatrixCalculator(is_debugging=False)
        self.x = Matrix(rows=-1, cols=1, numbers=np.asarray(x), name='x').get()
        self.y = Matrix(rows=-1, cols=1, numbers=np.asarray(y), name='y').get()
        pass
    
    # for testing purpose
    def _set_parameters(self, a, b):
        self.a = a
        self.b = b
        pass
    # for testing purpose
    def _get_parameters(self):
        return self.a, self.b

    def visualize(self, x0=None,limits=None):
        plt.plot(self.x, self.y, '+')
        plt.axhline(linewidth=1, color='k')
        plt.axvline(linewidth=1, color='k')
        plt.title("Đám mây dữ liệu", fontsize=12)
        plt.grid(True)
        if limits is None:
            limits_x = (min(self.x) // 2, max(self.x) * 1.3)
            limits_y = (min(self.y) // 2, max(self.y) * 1.3)
            plt.xlim(limits_x)
            plt.ylim(limits_y)
        if x0 is not None:
            a,b = self._get_parameters()
            y0 = a * x0 + b
            plt.plot(x0, y0, linewidth=0.5)
        plt.show()
        pass

    def process(self):
        # create [1,1,1...]
        x_one = np.ones((self.x.shape[0], 1))
        # stick ones to x
        x = np.concatenate((x_one, self.x), axis=1)
        print("X \n", x)
        print("Y \n", self.y)
        # compute transpose of x
        xT = self.mt_calculator.transpose(x)
        print("-"*100)
        print("X transpose \n", xT)
        # compute xT times x
        xTx = self.mt_calculator.multiply(xT, x)
        print("-"*100)
        print("X transpose times X\n", xTx)
        # compute the inverse matrix of the previous things
        xTx_inv = self.mt_calculator.inverse(xTx)
        print("-"*100)
        print("X transpose inserse \n", xTx_inv)
        # compute xT times y
        xTy = self.mt_calculator.multiply(xT, self.y)
        print("-"*100)
        print("X transpose times Y \n", xTy)
        # beta_hat: coefficients matrix which includes a and b
        beta_hat = self.mt_calculator.multiply(xTx_inv, xTy)
        print("-"*100)
        print("Beta hat \n", beta_hat)
        # compute extreme values
        print("-"*100)
        eigenvalues = self.mt_calculator.eigenvalues(xTx)
        if np.all(eigenvalues > 0):
            print("{} is maximize".format(beta_hat))
        elif np.all(eigenvalues < 0):
            print("{} is minimize".format(beta_hat))
        else:
            print("Not defined")

        # predict output y
        y_hat = self.mt_calculator.multiply(x, beta_hat)
        print("-"*100)
        print("X times beta hat \n", y_hat)
        # compute errors which compares to the real output
        errors = self.mt_calculator.subtract(self.y, y_hat)
        print("-"*100)
        print("Errors \n", errors)
        # dont really know what it is :v
        squared_errors = np.asarray([x*x for x in errors])
        print("-"*100)
        print("Squared Errors \n", squared_errors)
        print("Sum Squared Errors \n", sum(squared_errors))
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
    def __init__(self, x=None, y=None):
        self.mt_calculator = MatrixCalculator(is_debugging=False)
        self.x = Matrix(rows=-1, cols=1, numbers=np.asarray(x)).get()
        self.y = Matrix(rows=-1, cols=1, numbers=np.asarray(y)).get()
        pass

    def visualize(self, x=None, y=None, x_tilde=None, eigenvectors=None, limits=None, title=''):
        plt.title(title)
        plt.axhline(linewidth=0.5, color='k', linestyle='--')
        plt.axvline(linewidth=0.5, color='k', linestyle='--')
        plt.xlim(limits)
        plt.ylim(limits)
        if x is not None and y is not None:
            plt.plot(x, y, '+')
            plt.show()
        if x_tilde is not None and eigenvectors is not None:
            x = self.mt_calculator.get_cols(x_tilde, 0)
            y = self.mt_calculator.get_cols(x_tilde, 1)
            plt.plot(x, y, '+')
            for eigenvector in eigenvectors:
                x0 = np.linspace(2.5, -2.5, 10)
                y0 = eigenvector[1]/eigenvector[0] * x0
                plt.plot(x0, y0, linestyle='--', linewidth=0.5)
            plt.show()
        pass
    
    def process(self):
        # stick x, y together
        XY = np.concatenate((self.x, self.y), axis=1)
        self.visualize(x=self.x, y=self.y, limits=(-10, 60), title='Original PCA data')

        # compute mean of x and y. mean: gia tri trung binh
        m = np.mean(XY.T, axis=1)
        # compute X~
        X_tilde = XY - m
        # compute covariance matrix: C
        C = np.cov(X_tilde.T)
        print("Covariance matrix \n", C)
        # compute eigenvalues and eigenvectors of the matrix C
        eigenvalues = self.mt_calculator.eigenvalues(C)
        eigenvectors = self.mt_calculator.eigenvectors(C)
        print("-"*100)
        print("Eigenvalues \n", eigenvalues)
        print("Eigenvectors \n", eigenvectors)

        self.visualize(x_tilde=X_tilde, eigenvectors=eigenvectors, limits=(-40, 40), title='Mean adjusted data with eigenvectors overlayed')

        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        feature_vector = self.mt_calculator.get_cols(eigenvectors, 0)
        print("-"*100)
        print("Feature vector \n", feature_vector)
        final_data = self.mt_calculator.multiply(feature_vector.T, X_tilde.T)
        final_data = final_data.T
        print("-"*100)
        print("Final data \n", final_data)

        transformed_data = eigenvectors @ X_tilde.T
        transformed_data = transformed_data.T
        print("-"*100)
        print("Transformed data \n", transformed_data)
        x = transformed_data[:, 0]
        y = transformed_data[:, 1]

        self.visualize(x=x, y=y, limits=(-30, 30), title='Data transformed with 2 eigenvectors')

class LSA:
    def __init__(self, docs=None, query=None):
        self.docs = list(map(self._normalize_docs, docs))
        self.bag_of_words = self._build_bow()
        self.query = self._normalize_query(query)
        self.A = self._build_term_doc_matrix()
        self.mt_calculator = MatrixCalculator(is_debugging=False)
        pass

    def _build_bow(self):
        bow = []
        for doc in self.docs:
            bow += [word for word in doc]
        bow = list(set(bow))
        return sorted(bow)

    def _normalize_docs(self, text):
        docs = list(map(str.lower, str(text).split()))
        return docs

    def _normalize_query(self, text):
        q = np.zeros(len(self.bag_of_words))
        for word in sorted(self._normalize_docs(text)):
            if word in self.bag_of_words:
                index = self.bag_of_words.index(word)
                q[index] += 1
        print("STEP 1")
        print("Query q: \n")
        return q
    
    def _build_term_doc_matrix(self):
        # STEP 1
        # words x docs matrix
        matrix = np.zeros((len(self.bag_of_words), len(self.docs)), dtype=int)
        for i, word in enumerate(self.bag_of_words):
            for j, doc in enumerate(self.docs):
                matrix[i, j] = doc.count(word)
        print("STEP 1")
        print("Term Document matrix A: \n", matrix)
        return matrix


    def _svd(self, k=2):
        # STEP 2
        u, s, v = np.linalg.svd(self.A)
        s = np.diag(s)
        print("STEP 2")
        print("U: \n", u)
        print("S: \n", s)
        print("V: \n", v)
        # STEP 3
        # get k cols
        u = u[:, :k]
        # get k cols, k rows
        s = s[:k, :k]
        # get k cols
        v = v[:, :k]
        print("STEP 3")
        print("U_k: \n", u)
        print("S_k: \n", s)
        print("V_k: \n", v)
        return u, s, v

    def process(self):
        u, s, v = self._svd()
        inv_s = self.mt_calculator.inverse(s)
        # STEP 4
        # d = AT * u * s^-1
        d = self.A.T @ u @ inv_s
        print("STEP 4")
        print("d \n", d)
        # STEP 5
        # q = qT * u * s^-1
        q = self.query.T @ u @ inv_s
        print("STEP 5")
        print("q \n", q)
        # STEP 6
        cosines = list(map(lambda x: self._sim(q, x), d))
        print("STEP 6")
        for index, cosine in enumerate(cosines):
            print("sim{} = {}".format(index + 1, cosine))
        # print("cosine: ", cosines)
        index = np.argmax(cosines) + 1
        print("=> Document {}".format(index))
        pass

    def _sim(self, x, y):
        cosine = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
        return cosine

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
    Linear Regression
    '''
    file_path_ln = 'masters/data_ln.csv'
    # x = [18, 25, 15, 22, 24, 20]
    # y = [45, 58, 50, 54, 62, 53]
    # '''
    # read and write data from/to csv file
    # # csv = CSV()
    # # csv.write(file_path=file_path_ln, data=list(zip(x, y)))
    # # csv.read(file_path=file_path_ln)
    # '''
    # print("Hồi quy tuyến tính")

    # ln = LinearRegression(x, y)
    # ln.process()
    # # dữ liệu vẽ đường hồi quy
    # x0 = np.linspace(10, 30, 2)
    # # ln.visualize(x0)
    # ln.visualize()
    # ln.predict(30)

    print("="*100)

    print("PCA")
    '''
    PCA
    '''
    # x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
    # y = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
    # x = [39.17, 7.54, 3.41, 19.84, 56.69, 46.86]
    # y = [32.6, 46.9, 36.8, 39.9, 42.5, 40.8]
    # pca = PCA(x=x, y=y)
    # pca.process()
    # print("="*100)

    # print("LSA")
    # '''
    # LSA
    # '''
    # docs = [
    #     'Shipment of gold damaged in a fire', 
    #     'Delivery of silver arrived in a silver truck', 
    #     'Shipment of gold arrived in a truck'
    # ]
    # query = 'gold silver truck'
    docs = [
        ' Romeo and Juliet',
        ' Juliet : O Happy dagger !',
        ' Romeo died by dagger',
        "' Live free or die ' that's the New-Hampshire 's motto",
        ' Did you know , New-Hampshire is in New England'
    ]
    query = 'Dagger and Die'
    lsa = LSA(docs=docs, query=query)
    lsa.process()
    pass