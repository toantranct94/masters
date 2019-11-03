import math

# Bai 1
def fibonaci(n):
    if n in [0, 1]:
        return 1
    else:
        return fibonaci(n-2) + fibonaci(n-1)
    pass

# Bai 2
def powm(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    else:
        g = powm(x, n - 1)
        return g * x
    pass

# Bai 3
def binray_search_bai3(arr, first_index, last_index, x):

    middle_index = (first_index + last_index) / 2
    
    if middle_index == last_index or first_index == last_index:
        print("KO TON TAI PHAN TU X TRONG MANG")
        return 

    if x == arr[first_index]:
        return x

    if arr[middle_index] > x:
        return binray_search_bai3(arr, first_index, middle_index, x)
    
    if arr[middle_index] < x:
        return binray_search_bai3(arr, middle_index + 1, len(arr), x)

    if arr[middle_index] == x:
        if arr[middle_index] > x - 1:
            return binray_search_bai3(arr, first_index, middle_index, x)
        return x

# Bai 4
def binray_search_bai4(arr, first_index, last_index):

    middle_index = (first_index + last_index) // 2
    
    if middle_index == last_index or first_index == last_index:
        print("KO TON TAI PHAN TU THOA DIEU KIEN A[i] = i")
        return 

    if first_index == arr[first_index]:
        return first_index

    if arr[middle_index] > middle_index:
        return binray_search_bai4(arr, first_index, middle_index)
    
    if arr[middle_index] < middle_index:
        return binray_search_bai4(arr, middle_index + 1, len(arr))

    if arr[middle_index] == middle_index:
        if arr[middle_index] > middle_index - 1:
            return binray_search_bai4(arr, first_index, middle_index)
        return middle_index

# Bai 5
def tohop(n, k):
    def nCk(n,k):
        f = math.factorial
        return f(n) / f(k) / f(n-k)
    if k in [0, n]:
        return 1
    else:
        return nCk(n-1, k-1) + nCk(n-1, k)
    pass

# Bai 6
def find_maximum(arr, first_index, last_index):

    middle_index = (first_index + last_index) / 2

    if first_index == last_index:
        return arr[first_index]
    
    max1 = find_maximum(arr, 0, middle_index)
    max2 = find_maximum(arr, middle_index + 1, last_index)

    return max1 if max1 > max2 else max2

# Bai 7
def find_UCLN(a, b):
    if b > a:
        a,b = b,a
    if b == 0:
        return a
    return find_UCLN(b, b % a)

# Bai 9
def merge_sort(arr): 
    if len(arr) >1: 
        mid = len(arr)//2 
        L = arr[:mid]  
        R = arr[mid:] 

        merge_sort(L) 
        merge_sort(R)

        i = j = k = 0
            
        while i < len(L) and j < len(R): 
            if L[i] < R[j]: 
                arr[k] = L[i] 
                i+=1
            else: 
                arr[k] = R[j] 
                j+=1
            k+=1
            
        while i < len(L): 
            arr[k] = L[i] 
            i+=1
            k+=1
            
        while j < len(R): 
            arr[k] = R[j] 
            j+=1
            k+=1

    return arr



if __name__ == "__main__":
    # Bai 1
    # print("Bai 1")
    # print(fibonaci(4))

    # Bai 2
    # print("Bai 2")
    # print(powm(3,3))

    # Bai 3
    # print("Bai 3")
    # arr = [-4, -2, 0, 3, 4]
    # print(binray_search_bai3(arr, 0, len(arr), 3))

    # Bai 4:
    # print("Bai 4")
    # arr = [0, 1, 2, 3, 4, 6]
    # print(binray_search_bai4(arr, 0, len(arr)))

    # Bai 5
    # print("Bai 5")
    # n = 4
    # k = 2
    # print(tohop(n, k))

    # Bai 6
    # print("Bai 6")
    # arr = [2, 4, 1, 3, 6, 0]
    # print(find_maximum(arr, 0, len(arr) - 1))

    # Bai 7
    # print("Bai 7")
    # print(find_UCLN(6,2))

    # Bai 9
    # arr = [2,5,1,6,3,4,0,7]
    # print(merge_sort(arr))
    pass