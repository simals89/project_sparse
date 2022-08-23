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

#### Test __init__() ####

# Function to test CSR format
def testEqual(M):
    M_sparse = SparseMatrix(M)
    M_scipy = sp.csr_matrix(M)
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

#### Test change() ####
    
# Function to test change() function
def testChange(M, row, col, val):
    M_sparse = SparseMatrix(M).change(row, col, val)
    M_scipy = sp.csr_matrix(M)
    M_scipy[row, col] = val
    M_scipy.eliminate_zeros()
    M_scipy_arr = np.array([M_scipy.data, M_scipy.indices, M_scipy.indptr], dtype = object)
    
    M_equal = [np.array_equal(M_sparse.sparse_form[i], M_scipy_arr[i]) for i in range(3)]
    
    return M_equal

# Change a non-zero element to a non-zero element
print(testChange(A, 0, 0, 99))

# Change a non-zero element to a zero element
print(testChange(B, 1, 2, 0))

# Change a zero element to a non-zero element
print(testChange(C, 3, 2, 99))


#### Test add() ####

A = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12],
              [13,14,0,16]])


B = np.array([[1,0,3,4],
              [0,0,0,8],
              [9,0,11,0],
              [0,14,9,0]])

C = np.array([[0,1,2,0],
              [0,4,5,0],
              [0,6,7,0]])

#Tests if function detects if two matrices with seperate dimensions have separate dimensions.
def test_wrongDimensions(M1,M2):
    try:
        SparseMatrix(M1)+SparseMatrix(M2)
    except Exception:
        return "Test Passed"
    else:
        return "Test Failed"
      
#Tests if addition with SparseMatrix gives same result as SciPy. Should return True
def test_add(M1, M2):
    
    M_sparse = SparseMatrix(M1)+SparseMatrix(M2)
    M_scipy = sp.csr_matrix(M1)+sp.csr_matrix(M2)
    M_scipy_arr = np.array([M_scipy.data, M_scipy.indices, M_scipy.indptr], dtype = object)
    
    M_equal = [np.array_equal(M_sparse.sparse_form[i], M_scipy_arr[i]) for i in range(3)]
    
    return M_equal

# Function tests if two different additions are equal. Should return False
def test_add_not_same(M1, M2):
    
     M_sparse = SparseMatrix(M1)+SparseMatrix(M2)
     M_scipy = sp.csr_matrix(M1)+sp.csr_matrix(M1) # Here "M_scipy" adds two "M1" matrices. Should not be same as "M_sparse"
     M_scipy_arr = np.array([M_scipy.data, M_scipy.indices, M_scipy.indptr], dtype = object)

     M_equal = [np.array_equal(M_sparse.sparse_form[i], M_scipy_arr[i]) for i in range(3)]
     
     return M_equal
  
# Testing addition with correct matrices and dimensions. Should return list containing three True"
print(test_add(A, B))

# Testing addition when SparseMatrix and SciPy gets different inputs. Should return at least one "False" depending on the inputs
print(test_add_not_same(A, B))

#Testing if the Exception gets raised when dimensions does not match
print(test_wrongDimensions(A, C))

  #testing if error is raised when SparseMatrix nbr of columns and array nbr of rows is different
    def test_diff_size_arr_mult(self):
        mat_1=np.matrix([[1, 2, 3], [4,5,6], [7, 8, 9]])
        sparse_1=SparseMatrix(mat_1)
        arr=np.matrix([[1], [1]])
        try: 
            sparse_1*arr
        except IndexError:
            pass
        else:
            raise AssertionError()
            
    #testing if error is raised when having int input instead of array
    def test_input_int_mult(self):  
        mat_1=np.matrix([[1, 2, 3], [4,5,6], [7, 8, 9]])
        try:
            SparseMatrix(mat_1)*5
        except TypeError:
            pass
        else:
            raise AssertionError()
    #testing if multiplication of small SparseMatrix and array gives correct output
    def test_small_array_mult(self):
        mat_1=np.matrix([[1]])
        mat_2=np.matrix([[2, 3]])
        mat_3=np.matrix([[-1, 2], [2, -1]])
        sparse_mat_1=SparseMatrix(mat_1)
        sparse_mat_2=SparseMatrix(mat_2)
        sparse_mat_3=SparseMatrix(mat_3)
        arr_1=np.matrix([[2]])
        arr_2=np.matrix([[4], [5]])
        arr_3=np.matrix([[1],[2]])
        result_1=sparse_mat_1*arr_1
        result_2=sparse_mat_2*arr_2
        result_3=sparse_mat_3*arr_3
        expected_1=SparseMatrix(np.matrix([[2]]))
        expected_2=SparseMatrix(np.matrix([[23]]))
        expected_3=SparseMatrix(np.matrix([[3], [0]]))
        errors = []

        # replace assertions by conditions
        if not result_1.sparse_form==expected_1.sparse_form:
            errors.append("Error: wrong result when array has size 1")
        if not result_2.sparse_form==expected_2.sparse_form:
            errors.append("Error: wrong result when array has size 2")
        if not result_3.sparse_form==expected_3.sparse_form:
            errors.append("Error: wrong result when array has size 3")
        assert not errors, "errors:\n{}".format("\n".join(errors))
        
    #testing if multiplication of large SparseMatrix and array gives correct output    
    def test_large_array_mult(self):
        arr_1=np.zeros((10,1))
        arr_2=np.zeros((100,1))
        arr_3=np.zeros((1000,1))
        for i in range(0, 1000):
            if i%100==0:
                # np.append(arr_1, [[i/100]], axis=0)
                arr_1[int(i/100)]=i/100+1
            if i%10==0:
                # np.append(arr_2, [[i/10]], axis=0)
                arr_2[int(i/10)]=i/10+1
            arr_3[i]=i+1
        
        expected_1=np.zeros((10, 1))
        expected_2=np.zeros((100, 1))
        expected_3=np.zeros((1000, 1))
        expected_1[1][0]=-1
        expected_2[1][0]=-1
        expected_3[1][0]=-1
        expected_1[8][0]=-10
        expected_2[98][0]=-100
        expected_3[998][0]=-1000
        expected_1[9][0]=11
        expected_2[99][0]=101
        expected_3[999][0]=1001
        exp_sparse_1=SparseMatrix(expected_1)
        exp_sparse_2=SparseMatrix(expected_2)
        exp_sparse_3=SparseMatrix(expected_3)
        mat_1=np.zeros((10,10))
        mat_2=np.zeros((100,100))
        mat_3=np.zeros((1000,1000))
        mat_list=[mat_1, mat_2, mat_3]
        for i in range(0,3):
            matrix=mat_list[i]
            matrix[len(mat_list[i])-1][len(mat_list[i])-1]=2
            matrix[0][0]=2
            matrix[0][1]=-1
            matrix[1][0]=-1
            matrix[len(mat_list[i])-2][len(mat_list[i])-1]=-1
            matrix[len(mat_list[i])-1][len(mat_list[i])-2]=-1
        result_1=SparseMatrix(mat_1)*arr_1
        result_2=SparseMatrix(mat_2)*arr_2
        result_3=SparseMatrix(mat_3)*arr_3

        
        errors = []
        if not result_1.sparse_form==exp_sparse_1.sparse_form:
            errors.append("Error: wrong result when multiplying Sparse of size (10, 10")
        if not result_2.sparse_form==exp_sparse_2.sparse_form:
            errors.append("Error: wrong result when multiplying Sparse of size (100, 100")
        if not result_3.sparse_form==exp_sparse_3.sparse_form:
            errors.append("Error: wrong result when multiplying Sparse of size (1000, 1000")
        assert not errors, "errors:\n{}".format("\n".join(errors))
        
    #testing if multiplication with empty array returns correct output    
    def test_empty_array_mult(self):
        mat_1=np.matrix([[1, 2, 3], [4,5,6], [7, 8, 9]])
        arr_1=np.matrix([[0],[0], [0]])
        result_1=SparseMatrix(mat_1)*arr_1
        expected_1=SparseMatrix(np.matrix([[0],[0], [0]]))
        if result_1.sparse_form!=expected_1.sparse_form:
            print("multiplication with array with only zero elements does not return all zero SparseMatrix")
            
    #testing if multiplication with empty SparseMatrix returns correct output  
    def test_empty_sparse_mult(self):
        mat_1=np.matrix([[0, 0, 0], [0,0,0], [0, 0, 0]])
        arr_1=np.matrix([[1],[2], [3]])
        result_1=SparseMatrix(mat_1)*arr_1
        expected_1=SparseMatrix(np.matrix([[0],[0], [0]]))
        if result_1.sparse_form!=expected_1.sparse_form:
            print("multiplication with SparseMatrix with only zero elements does not return all zero SparseMatrix")
