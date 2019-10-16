import sys
import numpy as np
import math
# function Square matrix
def SqrtMatrix(matrix):
    return matrix*matrix
 # function cube matrix
def CubeMatrix(matrix):
    return SqrtMatrix(matrix)*matrix
 # function cos
def Cos(self):
    return math.cos(math.radians(self))
# function sin
def Sin(self):
    return math.sin(math.radians(self))
#Ví dụ 2, tr35
def Vd2_Tr35():
    A = np.matrix([(2, 0, -3, 2), (-1, 8, 10, -5)])
    B = np.matrix([(0, -4, -7, 2), (12, 3, 7, 9)])
    C = np.matrix([(2, 0, 2), (-4, 9, 5), (6, 0, -6)])
    lenA = A.shape
    lenB = B.shape
    lenC = C.shape
    if lenA == lenB:
        kq1 = A + B # (a)
        kq2 = A-B #(b)
        print('A+B = ')
        print(kq1)
        print('A-B =')
        print(kq2)
    if lenA == lenC :
        kq3 = A + C # (a)
        print('A+C = ',kq3)
    elif lenA != lenC :
        print("A+C = Error 'incompatibility of rows and columns A and C'")    
# Vd2_Tr35()      
#Ví dụ 3, tr36
def Vd3_Tr36():
    A = np.matrix([(0, 9), (2, -3), (-1, 1)])
    B = np.matrix([(8, 1), (-7, 0), (4, -1)])
    C = np.matrix([(2, 3), (-2, 5), (10, -6)])
    lenA = len(A)
    lenB = len(B)
    lenC = len(C)
    print("3A + 2B - 1/2C =")
    print((3*A)+(2*B)-(0.5*C))    
#Ví dụ 4, tr37
def Vd4_Tr37():
    a = np.matrix([(4, -10, 3)])
    b = np.matrix([(-4), (3), (8)]) 
    print("ab = ", a.dot(b)) #tính tích vô hướng sử dụng hàm dot
#Ví dụ 5, tr37
def Vd5_Tr37():
    A = np.matrix([(1, -3, 0, 4), (-2, 5, -8, 9)])
    C = np.matrix([(8, 5, 3), (-3, 10, 2),(2, 0, -4), (-1, -7, 5)]) 
    if A.shape[1] == C.shape[0]:
        print("AC = ", A.dot(C))
    if C.shape[1] == A.shape[0]:
        print("AC = ", C.dot(A))
    if C.shape[1] != A.shape[0]:
        print("CA --> Error: ",C.shape[1],'and',A.shape[0]," NOT the same ") 
# Vd3_Tr36()
#Ví dụ 7, tr40
def Vd7_Tr40():
    A = np.matrix([(1, -3, 0, 4), (-2, 5, -8, 9)])
    C = np.matrix([(8, 5, 3), (-3, 10, 2),(2, 0, -4), (-1, -7, 5)]) 
    print("A row 2*C = ",A[1].dot(C))
    print("A*C column 3 = ",A.dot(C[:,2]))
# Vd7_Tr40()
#Ví dụ 8, tr41
def Vd8_Tr41():
    A = np.matrix([(1, -3, 0, 4), (-2, 5, -8, 9)])
    C = np.matrix([(8, 5, 3), (-3, 10, 2),(2, 0, -4), (-1, -7, 5)])
    r1= A[0]
    r2= A[1]
    r1C = r1.dot(C)
    r2C = r2.dot(C)
    AC = np.vstack([r1C, r2C])
    print('r1C =',r1C )
    print('r2C =',r2C )
    print('AC =')
    print(AC)
    c1 = C[:,0]
    c2 = C[:,1]
    c3 = C[:,2]
    Ac1 = A.dot(c1)
    Ac2 = A.dot(c2)
    Ac3 = A.dot(c3)
    col1=np.matrix(Ac1).reshape(2,1)
    col2=np.matrix(Ac2).reshape(2,1)
    col3=np.matrix(Ac3).reshape(2,1)
    AB = np.hstack((col1, col2, col3))
    print('Ac1 =',Ac1 )
    print('Ac2 =',Ac2 )
    print('Ac3 =',Ac3 )
    print('AB =')
    print(AB)
# Vd8_Tr41()
#Ví dụ 10, tr45
def Vd10_Tr45():
    A = np.matrix([(4, 10, -7, 0), (5, -1, 3, -2)])
    B = np.matrix([(3, 2, -6), (-9, 1, -7),(5, 0, 12)])
    C =  np.matrix([[ 9],[-1],[ 8]])
    D =  np.matrix([(15)])
    E = np.matrix([(-12, -7),(-7, 10)])
    AT = np.transpose(A) #chuyển vị
    print('AT =', AT)
    BT = np.transpose(B) #chuyển vị
    print('BT =', BT)
    trB = np.sum(np.diag(B,0)) #diag(matrix, start) vector đường chéo
    print('trB =', trB)
    CT = np.transpose(C) #chuyển vị
    print('CT =', CT)
    # trC = np.sum(np.diag(C,0))
    print('trC Not defined since C is not square')
    DT = np.transpose(D) #chuyển vị
    print('DT =', DT)
    trD = np.sum(np.diag(D,0))
    print('trD =', trD)
    ET = np.transpose(E) #chuyển vị
    print('ET =', ET)
    trE = np.sum(np.diag(E,0))
    print('trE =', trE)
# Vd10_Tr45()
#Ví dụ 4, tr50
def Vd4_Tr50():
    A = np.matrix([(-7, 3), (5, 1)])
    I =  np.matrix([(1, 0), (0, 1)])
    A2 = SqrtMatrix(A) # câu a
    print("A2 = ")
    print(A2)
    A3 = CubeMatrix(A) #câu b
    print("A3 = ")
    print(A3)
    pA = -6*A3 + 10*A - 9*I # câu c
    print("p(A)= ")
    print(pA)
# Vd4_Tr50()
#Ví dụ 3, tr54
def Vd3_Tr54():
    A = np.matrix([(-4, -2), (5, 5)])
    A_1 = np.linalg.inv(A)
    A_1_3 = CubeMatrix(A_1 )
    print("A mũ -3 = ")
    print( A_1_3 )
# Vd3_Tr54()
#Ví dụ 2, tr68
def Vd2_Tr68():
    A = np.matrix([(3, 1, 0), (-1, 2, 2), (5, 0, -1)])
    An= np.linalg.inv(A)
    print("Nghịch đảo A = ")
    print(An)
# Vd2_Tr68()
#Ví dụ 1, tr84
def Vd1_Tr84():
    A = np.matrix([(3, 1, 0), (-1, 2, 2), (5, 0, -1)])
    B =  np.matrix([[ 6],[-7],[10]])
    A_1 = np.linalg.inv(A)
    A_1B = A_1*B
    print("x1 = ",float(A_1B[0]) )
    print("x2= ",float(A_1B[1] ))
    print("x3 = ",float(A_1B[2]))
# Vd1_Tr84()
#Ví dụ 7, tr100
def Vd7_Tr100():
    A = np.matrix([(3, 2), (-9, 5)])
    B = np.matrix([(3, 5, 4), (-2, -1, 8), (-11, 1, 7)])
    C = np.matrix([(2, -6, 2), (2, -8, 3), (-3, 1, 1)])
    A1 = int(np.linalg.det(A)) #a
    print('det(A) = ',A1)
    B1 = int(np.linalg.det(B)) #b
    print('det(B) = ',B1)
    C1 = int(np.linalg.det(C)) #c
    print('det(C) = ',C1)
# Vd7_Tr100()
#Ví dụ 1, tr178
def Vd1_Tr178():
    A =  np.matrix([[ 2],[-4],[1]])
    B =  np.matrix([[10],[7],[-9]])
    C =  np.matrix([[10],[7],[-9]])
    I1 = np.matrix([(1, 0, 0), (0, -1, 0), (0, 0, 1)])
    I2 = np.matrix([(1, 0, 0), (0, 0, 0), (0, 0, 0)])
    I3 = np.matrix([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
    print("The point x = (2, -4, 1) => map to w =  ")
    print(I1*A)
    print("The projection here is ")
    print(I2*B)
    print("The projection here is ")
    print(I3*C)
# Vd1_Tr178()
#Ví dụ 2, tr179
def Vd2_Tr179():
    A =  np.matrix([[ 2],[-6]])
    B =  np.matrix([[0],[5],[1]])
    C =  np.matrix([[-3],[4],[-2]])
    I1 = np.matrix([(Cos(30), -Sin(30)), (Sin(30), Cos(30) )])
    I2 = np.matrix([(0, 0, Sin(90)),(0, 1, 0),(-Sin(90), 0, 0)])
    I3 = np.matrix([(Cos(25), -Sin(25), 0),(Sin(25), Cos(25), 0),(0, 0, 1)]) 
    print('Result a) = ')
    print(I1*A)
    print('Result b) = ')
    print(I2*B)
    print('Result c) = ')
    print(I3*C)
# Vd2_Tr179()
#Ví dụ 3, tr180
def Vd3_Tr180():
    A =  np.matrix([[4],[-1],[-3]])
    B =  np.matrix([[4],[-1],[-3]])
    C =  np.matrix([[4],[2]])
    D =  np.matrix([[4],[2]]) 

# Vd3_Tr180()
#Ví dụ 3, tr312
def Vd3_Tr312():
    A = np.matrix([(4, 0, 1), (-1, -6, -2), (5, 0, 0)])
    A = np.matrix([(6, 3, -8), (0, -2, 0), (1, 0, -3)])
    A = np.matrix([(0, 1, 1), (1, 0, 1), (1, 1, 0)])
    A = np.matrix([(4, 0, -1), (0, 3, 0), (1, 0, 2)])
Vd3_Tr312()


