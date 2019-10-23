import numpy as np 
from math import sin, cos, sqrt 

class Matrix:
    def __init__(self, rows=None,cols=None, numbers=None, name=None):
        '''
        rows: số hàng 
        cols: cố cột
        '''
        self.rows = rows
        self.cols = cols
        self.numbers = numbers
        self.name = name
        self.matrix  = self._init_matrix()
        pass

    def _init_matrix(self):
        # convert to numpy array
        matrix = np.asarray(self.numbers)
        # reshape to m rows x n cols
        matrix = np.reshape(matrix, (self.rows, self.cols))
        if self.rows == 1 or self.cols == 1:
            prefix = 'Vector'
        else:
            prefix = 'Matrix'
        print("{} {}: \n".format(prefix,self.name), matrix)
        return matrix

    def get(self):
        return self.matrix

class MatrixCalculator:
    def __init__(self):
        pass

    def plus(self, added_matrix_1=None, added_matrix_2=None):
        if added_matrix_1.shape == added_matrix_2.shape:
            result = np.add(added_matrix_1, added_matrix_2)
            print("plus: \n",result)
            return result
        else:
            print("Operands could not be broadcast together with shapes {} {}".format(added_matrix_1.shape, added_matrix_2.shape))

    def subtract(self, sub_matrix_1=None, sub_matrix_2=None):
        if sub_matrix_1.shape == sub_matrix_2.shape:
            result = np.subtract(sub_matrix_1, sub_matrix_2)
            print("subtract: \n",result)
            return result
        else:
            print("Operands could not be broadcast together with shapes {} {}".format(sub_matrix_1.shape, sub_matrix_2.shape))

    def multiply(self, matrix_1, matrix_2):
        if matrix_1.shape[-1] == matrix_2.shape[0]:
            result = np.dot(matrix_1, matrix_2)
            print("Multiply: \n", result)
            return result
        else:
            print("Operands could not be broadcast together with shapes {} {}".format(matrix_1.shape, matrix_2.shape))

    def compute_ex_3(self, matrix_A, matrix_B, matrix_C):
        matrix_A = np.dot(3, matrix_A)
        matrix_B = np.dot(2, matrix_B)
        matrix_C = np.dot(0.5, matrix_C)
        result = matrix_A + matrix_B - matrix_C
        print("Compute: \n", result)
        return result

    def get_rows(self, matrix, index): 
        rows = matrix[index]
        return rows
    
    def get_cols(self, matrix, index):
        cols = matrix[:,index].reshape(-1, 1)
        return cols

    def transpose(self, matrix):
        result = np.transpose(matrix)
        print("Transpose: \n", result)
        return result

    def trace(self, matrix):
        if matrix.shape[0] == matrix.shape[1]:
            result = np.trace(matrix)
            print("Trace: \n", result)
            return result
        else:
            print("Not defined since this matrix is not square")

    def powm(self, matrix, n):
        result = np.linalg.matrix_power(matrix, n)
        print("pow: \n", result)
        return result

    def create_indentity_matrix(self, n):
        result = np.identity(n)
        return result

    def compute_ex_4_pg(self, matrix_pow_3, matrix, indentity_matrix):
        matrix_pow_3 = np.dot(-6, matrix_pow_3)
        matrix = np.dot(10, matrix)
        indentity_matrix = np.dot(-9, indentity_matrix)
        result = matrix_pow_3 + matrix + indentity_matrix
        print("p(A) \n", result)
        return result

    def inverse(self, matrix):
        result = np.linalg.inv(matrix)
        print("Inverse: \n", result)
        return result 

    def det(self, matrix):
        result = np.linalg.det(matrix)
        print("det: \n", result)
        return result
    
    def eigenvalues(self, matrix):
        eigenvalues, _ = np.linalg.eig(matrix)
        print("eigenvalues: \n", eigenvalues)
        return eigenvalues

if __name__ == "__main__":
    calculator = MatrixCalculator()
    '''
    VD 2, trang 35
    '''
    print("VD 2, trang 35")
    numbers_A = [2, 0, -3, 2, -1, 8, 10, -5]
    numbers_B = [0, -4, -7, 2, 12, 3, 7, 9]
    numbers_C = [2, 0, 2, -4, 9, 5, 6, 0, -6]
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=2, cols=4, numbers=numbers_B, name='B').get()
    matrix_C = Matrix(rows=3, cols=3, numbers=numbers_C, name='C').get()

    print("A + B")
    calculator.plus(matrix_A, matrix_B)
    print("B - A")
    calculator.subtract(matrix_B, matrix_A)
    print("A + C")
    calculator.plus(matrix_A, matrix_C)

    print("*"*100)

    '''
    VD 3, trang 36
    '''
    print("VD 3, trang 36")
    numbers_A = [0, 9, 2, -3, -1, 1]
    numbers_B = [8, 1, -7, 0, 4, -1]
    numbers_C = [2, 3, -2, 5, 10, -6]
    matrix_A = Matrix(rows=3, cols=2, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=2, numbers=numbers_B, name='B').get()
    matrix_C = Matrix(rows=3, cols=2, numbers=numbers_C, name='C').get()
    print("3A + 2B - 1/2C")
    calculator.compute_ex_3(matrix_A, matrix_B, matrix_C)

    print("*"*100)
    '''
    VD 4,5 trang 37
    '''
    print("VD 4 trang 37")
    numbers_A = [4, -10, 3]
    numbers_B = [-4, 3, 8]
    matrix_A = Matrix(rows=1, cols=3, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=1, numbers=numbers_B, name='B').get()
    calculator.multiply(matrix_A, matrix_B)
    print("VD 5 trang 37")
    numbers_A = [1, -3, 0, 4, -2, 5, -8, 9]
    numbers_C = [8, 5, 3, -3, 10, 2, 2, 0, -4, -1, -7, 5]
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A, name='A').get()
    matrix_C = Matrix(rows=4, cols=3, numbers=numbers_C, name='C').get()
    print("A * C")
    calculator.multiply(matrix_A, matrix_C)

    print("*"*100)
    '''
    VD 7 trang 40
    '''
    print("VD 7 trang 40")
    second_row_A = calculator.get_rows(matrix_A, 1)
    print("A[1] * C")
    calculator.multiply(second_row_A, matrix_C)
    third_col_C = calculator.get_cols(matrix_C, 2)
    print("A * C[2]")
    calculator.multiply(matrix_A, third_col_C)

    print("*"*100)
    '''
    VD 8 trang 41
    '''
    print("VD 8 trang 41")
    first_row_A = calculator.get_rows(matrix_A, 0)
    calculator.multiply(first_row_A, matrix_C)
    calculator.multiply(second_row_A, matrix_C)

    first_col_C = calculator.get_cols(matrix_C, 0)
    second_col_C = calculator.get_cols(matrix_C, 1)
    third_col_C = calculator.get_cols(matrix_C, 2)
    calculator.multiply(matrix_A, first_col_C)
    calculator.multiply(matrix_A, second_col_C)
    calculator.multiply(matrix_A, third_col_C)
    calculator.multiply(matrix_A, matrix_C)

    print("*"*100)
    '''
    VD 10 trang 45
    '''
    print("VD 10 trang 45")
    numbers_A = [4, 10, -7, 0, 5, -1, 3, -2]
    numbers_B = [3, 2, -6, -9, 1, -7, 5, 0, 12]
    numbers_C = [9, -1, 8]
    numbers_D = [15]
    numbers_E = [-12, -7, -7, 10]
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=3, numbers=numbers_B, name='B').get()
    matrix_C = Matrix(rows=3, cols=1, numbers=numbers_C, name='C').get()
    matrix_D = Matrix(rows=1, cols=1, numbers=numbers_D, name='D').get()
    matrix_E = Matrix(rows=2, cols=2, numbers=numbers_E, name='E').get()

    matrices = [matrix_A, matrix_B, matrix_C, matrix_D, matrix_E]
    for matrix in matrices:
        calculator.transpose(matrix)
        calculator.trace(matrix)
        print("-"*10)
    
    print("*"*100)
    '''
    VD 4 trang 50
    '''
    print("VD 4 trang 50")
    numbers_A = [-7, 3, 5, 1]
    matrix_A = Matrix(rows=2, cols=2,  numbers=numbers_A, name='A').get()
    # A^2
    calculator.powm(matrix_A, 2)
    # A^3
    Apow3 = calculator.powm(matrix_A, 3)
    # -6x^3 + 10x - 9
    I = calculator.create_indentity_matrix(2)
    calculator.compute_ex_4_pg(Apow3, matrix_A, I)

    print("*"*100)
    '''
    VD 3 trang 54
    '''
    print("VD 3 trang 54")
    numbers_A = [-4, -2, 5, 5]
    matrix_A = Matrix(rows=2, cols=2, numbers=numbers_A, name='A').get()
    inverse_A = calculator.inverse(matrix_A)
    calculator.powm(inverse_A, 3)

    print("*"*100)
    '''
    VD 2 trang 68
    '''
    print("VD 2 trang 68")
    numbers_C = [3, 1, 0, -1, 2, 2, 5, 0, -1]
    matrix_C = Matrix(rows=3, cols=3, numbers=numbers_C, name='C').get()
    inverse_C = calculator.inverse(matrix_C)
    calculator.multiply(matrix_C, inverse_C)
    calculator.multiply(inverse_C, matrix_C)

    print("*"*100)
    '''
    VD 1 trang 84
    '''
    print("VD 1 trang 84")
    numbers_A = [3, 1, 0, -1, 2, 2, 5, 0, -1]
    numbers_B = [6, -7, 10]
    matrix_A = Matrix(rows=3, cols=3, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=1, numbers=numbers_B, name='B').get()

    inverse_A = calculator.inverse(matrix_A)
    calculator.multiply(inverse_A, matrix_B)

    print("*"*100)
    '''
    VD 7 trang 100
    '''
    print("VD 7 trang 100")
    numbers_A = [3, 2, -9, 5]
    numbers_B = [3, 5, 4, -2, -1, 8, -11, 1, 7]
    numbers_C = [2, -6, 2, 2, -8, 3, -3, 1, 1]
    matrix_A = Matrix(rows=2, cols=2, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=3, numbers=numbers_B, name='B').get()
    matrix_C = Matrix(rows=3, cols=3, numbers=numbers_C, name='C').get()
    calculator.det(matrix_A)
    calculator.det(matrix_B)
    calculator.det(matrix_C)

    print("*"*100)
    '''
    VD 1-3 trang 178
    '''
    print("VD 1 trang 178")
    print("A")
    nubmers_reflecion_xz = [1, 0, 0, 0, -1, 0, 0, 0, 1]
    matrix_reflection_xz = Matrix(rows=3, cols=3, numbers=nubmers_reflecion_xz).get()
    numbers_x = [2, -4, 1]
    vector_x = Matrix(rows=3, cols=1,numbers=numbers_x).get()
    calculator.multiply(matrix_reflection_xz, vector_x)
    print("B")
    nubmers_projection_x = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    matrix_projection_x = Matrix(rows=3, cols=3, numbers=nubmers_projection_x).get()
    numbers_x = [10, 7, -9]
    vector_x = Matrix(rows=3, cols=1,numbers=numbers_x).get()
    calculator.multiply(matrix_projection_x, vector_x)
    print("C")
    nubmers_projection_yz = [0, 0, 0, 0, 1, 0, 0, 0, 1]
    matrix_projection_yz = Matrix(rows=3, cols=3, numbers=nubmers_projection_yz).get()
    calculator.multiply(matrix_projection_yz, vector_x)
    print("VD 2 trang 178")
    print("A")
    numbers_w = [cos(np.deg2rad(30)), -sin(np.deg2rad(30)), sin(np.deg2rad(30)), cos(np.deg2rad(30))]
    numbers_x = [2, -6]
    matrix_W = Matrix(rows=2, cols=2, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=2, cols=1, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)
    print("B")
    numbers_w = [cos(np.deg2rad(90)), 0, sin(np.deg2rad(90)), 0, 1, 0, -sin(np.deg2rad(90)), 0, cos(np.deg2rad(90))]
    numbers_x = [0, 5, 1]
    matrix_W = Matrix(rows=3, cols=3, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=3, cols=1, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)
    print("C")
    numbers_w = [cos(np.deg2rad(25)), -sin(np.deg2rad(25)), 0, sin(np.deg2rad(25)), cos(np.deg2rad(25)), 0, 0, 0, 1]
    numbers_x = [-3, 4, -2]
    matrix_W = Matrix(rows=3, cols=3, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=3, cols=1, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)

    print("VD 3 trang 179")
    print("A")
    numbers_w = [0, 0, 0, 0, 2, 0, 0, 0, 0]
    numbers_x = [4, 1, -3]
    matrix_W = Matrix(rows=3, cols=3, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=3, cols=1, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)
    print("B")
    numbers_w = [2, 0, 0, 0, 2, 0, 0, 0, 2]
    numbers_x = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    matrix_W = Matrix(rows=3, cols=3, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=3, cols=3, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)
    print("C")
    numbers_w = [sqrt(2)/2, 0, sqrt(2)/2, 0]
    numbers_x = [4, 2]
    matrix_W = Matrix(rows=2, cols=2, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=2, cols=1, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)
    print("D")
    numbers_w = [1, 0, 0, 0]
    numbers_x = [sqrt(2)/2, -sqrt(2)/2, sqrt(2)/2, sqrt(2)/2]
    matrix_W = Matrix(rows=2, cols=2, numbers=numbers_w, name='W').get()
    vector_x = Matrix(rows=2, cols=2, numbers=numbers_x, name='x').get()
    calculator.multiply(matrix_W, vector_x)

    print("*"*100)
    '''
    VD 3 trang 312
    '''
    print("VD 3 trang 312")
    numbers_A = [4, 0, 1, -1, -6, -2, 5, 0, 0]
    numbers_B = [6, 3, -8, 0, -2, 0, 1, 0, -3]
    numbers_C = [0, 1, 1, 1, 0, 1, 1, 1, 0]
    numbers_D = [4, 0, -1, 0, 3, 0, 1, 0, 2]
    matrix_A = Matrix(rows=3, cols=3, numbers=numbers_A, name='A').get()
    matrix_B = Matrix(rows=3, cols=3, numbers=numbers_B, name='B').get()
    matrix_C = Matrix(rows=3, cols=3, numbers=numbers_C, name='C').get()    
    matrix_D = Matrix(rows=3, cols=3, numbers=numbers_D, name='D').get()

    calculator.eigenvalues(matrix_A)
    calculator.eigenvalues(matrix_B)
    calculator.eigenvalues(matrix_C)
    calculator.eigenvalues(matrix_D)
    