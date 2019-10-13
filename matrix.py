import numpy as np 

class Matrix:
    def __init__(self, rows=None,cols=None, numbers=None):
        '''
        rows: số hàng 
        cols: cố cột
        '''
        self.rows = rows
        self.cols = cols
        self.numbers = numbers
        self.matrix  = self._init_matrix()
        pass

    def _init_matrix(self):
        # convert to numpy array
        matrix = np.asarray(self.numbers)
        # reshape to m rows x n cols
        matrix = np.reshape(matrix, (self.rows, self.cols))
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

if __name__ == "__main__":
    calculator = MatrixCalculator()
    '''
    VD 2, trang 35
    '''
    print("VD 2, trang 35")
    numbers_A = [2, 0, -3, 2, -1, 8, 10, -5]
    numbers_B = [0, -4, -7, 2, 12, 3, 7, 9]
    numbers_C = [2, 0, 2, -4, 9, 5, 6, 0, -6]
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A).get()
    matrix_B = Matrix(rows=2, cols=4, numbers=numbers_B).get()
    matrix_C = Matrix(rows=3, cols=3, numbers=numbers_C).get()

    AplusB = calculator.plus(matrix_A, matrix_B)
    BsubA = calculator.subtract(matrix_B, matrix_A)
    AplusC = calculator.plus(matrix_A, matrix_C)

    print("*"*100)

    '''
    VD 3, trang 36
    '''
    print("VD 3, trang 36")
    numbers_A = [0, 9, 2, -3, -1, 1]
    numbers_B = [8, 1, -7, 0, 4, -1]
    numbers_C = [2, 3, -2, 5, 10, -6]
    matrix_A = Matrix(rows=3, cols=2, numbers=numbers_A).get()
    matrix_B = Matrix(rows=3, cols=2, numbers=numbers_B).get()
    matrix_C = Matrix(rows=3, cols=2, numbers=numbers_C).get()
    
    calculator.compute_ex_3(matrix_A, matrix_B, matrix_C)

    print("*"*100)
    '''
    VD 4,5 trang 37
    '''
    print("VD 4 trang 37")
    numbers_A = [4, -10, 3]
    numbers_B = [-4, 3, 8]
    matrix_A = Matrix(rows=1, cols=3, numbers=numbers_A).get()
    matrix_B = Matrix(rows=3, cols=1, numbers=numbers_B).get()
    calculator.multiply(matrix_A, matrix_B)
    print("VD 5 trang 37")
    numbers_A = [1, -3, 0, 4, -2, 5, -8, 9]
    numbers_C = [8, 5, 3, -3, 10, 2, 2, 0, -4, -1, -7, 5]
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A).get()
    matrix_C = Matrix(rows=4, cols=3, numbers=numbers_C).get()
    calculator.multiply(matrix_A, matrix_C)

    print("*"*100)
    '''
    VD 7 trang 40
    '''
    print("VD 7 trang 40")
    second_row_A = calculator.get_rows(matrix_A, 1)
    calculator.multiply(second_row_A, matrix_C)
    third_col_C = calculator.get_cols(matrix_C, 2)
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
    second_col_C = calculator.get_cols(matrix_C, 0)
    third_col_C = calculator.get_cols(matrix_C, 0)
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
    matrix_A = Matrix(rows=2, cols=4, numbers=numbers_A).get()
    matrix_B = Matrix(rows=3, cols=3, numbers=numbers_B).get()
    matrix_C = Matrix(rows=3, cols=1, numbers=numbers_C).get()
    matrix_D = Matrix(rows=1, cols=1, numbers=numbers_D).get()
    matrix_E = Matrix(rows=2, cols=2, numbers=numbers_E).get()

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
    matrix_A = Matrix(rows=2, cols=2, numbers=numbers_A).get()
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