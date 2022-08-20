import numpy as np
import scipy.sparse as sp
from SparseMatrix import SparseMatrix

A = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12],
              [13,14,15,16]])

B = np.array([[0,1,2,0],
              [0,4,5,0],
              [0,6,7,0]])

C = np.array([[0,0,0],
              [1,2,3],
              [4,5,6],
              [0,0,0]])

D = np.array([[0,0,0],
              [0,1,0],
              [0,0,0]])

# Generate large matrices
np.random.seed(10)

E = np.zeros((10,10))
idx = np.random.choice(np.linspace(0,10*10-1,10*10), size = int(10*10*0.1), replace = False).astype(int)
E[np.unravel_index(idx, E.shape)] = np.random.randint(0, 100, int((10*10*0.1)))

F = np.zeros((100,100))
idx = np.random.choice(np.linspace(0,100*100-1,100*100), size = int(100*100*0.1), replace = False).astype(int)
F[np.unravel_index(idx, F.shape)] = np.random.randint(0, 100, int((100*100*0.1)))

G = np.zeros((10000,10000))
idx = np.random.choice(np.linspace(0,np.shape(G)[0]*np.shape(G)[0]-1,np.shape(G)[0]*np.shape(G)[0]), 
                       size = int(np.shape(G)[0]*np.shape(G)[0]*0.1), replace = False).astype(int)
G[np.unravel_index(idx, G.shape)] = np.random.randint(0, 100, int((np.shape(G)[0]*np.shape(G)[0]*0.1)))

# Function to test CSR form
def testEqual(M):
    M_sparse = SparseMatrix(M)
    M_scipy = sp.csr_matrix(M)
    M_scipy_arr = np.array([M_scipy.data, M_scipy.indices, M_scipy.indptr], dtype = object)
    
    M_equal = [np.array_equal(M_sparse.sparse_form[i], M_scipy_arr[i]) for i in range(3)]
    
    return M_equal
    
# Function to test change() function
def testChange(M, row, col, val):
    M_sparse = SparseMatrix(M).change(row, col, val)
    M_scipy = sp.csr_matrix(M)
    M_scipy[row, col] = val
    M_scipy.eliminate_zeros()
    M_scipy_arr = np.array([M_scipy.data, M_scipy.indices, M_scipy.indptr], dtype = object)
    
    M_equal = [np.array_equal(M_sparse.sparse_form[i], M_scipy_arr[i]) for i in range(3)]
    
    return M_equal

# Run tests
print("A:", testEqual(A))
print("B:", testEqual(B))
print("C:", testEqual(C))
print("D:", testEqual(D))
print("E:", testEqual(E))
print("F:", testEqual(F))
#print("G:", testEqual(G))

# Change a non-zero element to a non-zero element
print(testChange(A, 0, 0, 99))

# Change a non-zero element to a zero element
print(testChange(B, 1, 2, 0))

# Change a zero element to a non-zero element
print(testChange(C, 3, 2, 99))
