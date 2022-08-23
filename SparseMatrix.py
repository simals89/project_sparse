import numpy as np

class SparseMatrix:
    """
    A class to represent a sparse matrix.
    
    Attributes
    ----------
    intern_represent : string
        internal representation of the matrix (CSR or CSC)
    number_of_nonzero : int
        number of non-zero elements in the matrix
    sparse_form : numpy array
        sparse representation of the matrix
    """
    def __init__(self, *args, tol = 10**(-8)):
        """
        The constructor for class SparseMatrix.
        
        Parameters
        ----------
        *args
            a numpy matrix or a SparseMatrix
        tol : float
            threshold: all absolute values less than tol will be set to 0
        """
        if isinstance(args[0], np.ndarray) and len(args) == 1: # ÄNDRING
            mat = args[0]
            tol = 10**(-8)
            # Set attribute intern_represent to CSR
            self.intern_represent = "CSR"
            
            # Set values less than threshold to 0
            mat[abs(mat) < tol] = 0 # ÄNDRING
            
            # Number of rows and columns of matrix
            n_rows = np.shape(mat)[0]
            n_cols = np.shape(mat)[1]
            
            # Create attribute number_of_nonzero
            self.number_of_nonzero = 0
            
            # Create lists for storing values, row indices, and column indices
            sparse_mat = []
            row_ind = [0]*(n_rows + 1)
            col_ind = []
            
            for i in range(n_rows):
                for j in range(n_cols):
                    if mat[i,j] != 0:
                        sparse_mat.append(mat[i,j])
                        col_ind.append(j)
                        self.number_of_nonzero += 1
                    row_ind[i+1] = (self.number_of_nonzero)
                        
            self.sparse_form = [sparse_mat, col_ind, row_ind]
        elif len(args) == 3:
            self.sparse_form = [args[0], args[1], args[2]]
        
    def change(self, row, col, value):
        """
        Function to change a particular element in the matrix.
        
        Parameters
        ----------
            self : SparseMatrix
                a sparse matrix
            row : int
                row index of element to be changed
            col : int
                column index of element to be changed
            value : float
                new value
            
        Returns
        -------
        SparseMatrix
            an updated SparseMatrix
        """
        # Extract non-zero columns on row
        row_start = self.sparse_form[2][row]
        row_end = self.sparse_form[2][row+1]

        # If matrix element is non-zero
        if col in self.sparse_form[1][row_start:row_end]:
            #change_index = np.where(self.sparse_form[1][row_start:row_end] == col)
            change_index = self.sparse_form[1][row_start:row_end].index(col)
            
            # If new value is not 0, simply insert value
            if value != 0:
                self.sparse_form[0][row_start+change_index] = value
            # If new value is 0, remove from arrays and subtract 1 from number_of_nonzero
            elif value == 0:
                self.sparse_form[0] = np.delete(self.sparse_form[0], row_start+change_index)
                self.sparse_form[1] = np.delete(self.sparse_form[1], row_start+change_index)
                self.sparse_form[2][(row+1):] = [x-1 for x in self.sparse_form[2][(row+1):]]
                self.number_of_nonzero -= 1
        
        # If matrix element is zero
        elif col not in self.sparse_form[1][row_start:row_end] and value != 0:
            insert_index = np.where(np.sort(np.append(self.sparse_form[1][row_start:row_end], col)) == col)[0] + row_start
            self.sparse_form[0] = np.insert(self.sparse_form[0], insert_index, value)
            self.sparse_form[1] = np.insert(self.sparse_form[1], insert_index, col)
            self.sparse_form[2][(row+1):] = [x+1 for x in self.sparse_form[2][(row+1):]]
            self.number_of_nonzero += 1

        return self
    
    def CSRtoCSC(self):
        if self.intern_represent == "CSC":
            print("Already CSC")
        else:
            newV = []

            newRow = []
            rowChange = []

            ## Calculate base for new row_ind
            for i in range(len(self.sparse_form[2])-1):
                diff = self.sparse_form[2][i+1]-self.sparse_form[2][i]
                rowChange.extend([i]*diff)

            #print(rowChange)
            newCol = [-1] * len(self.sparse_form[1])
            newCol[0] = 0

            ## Calculate new sparse_mat and new col_ind
            newNonZ = 0
            for i in range(np.max(self.sparse_form[1])+1):
                for j in range(len(self.sparse_form[1])):
                    val = self.sparse_form[1][j]
                    if val == i:
                        newV.append(self.sparse_form[0][j])
                        newRow.append(rowChange[j])
                        newNonZ += 1
                        newCol[i+1] = newNonZ


            self.sparse_form[0] = newV
            self.sparse_form[1] = [value for value in newCol if value != -1]
            #self.sparse_form[1] = np.insert(np.cumsum(np.bincount(self.sparse_form[1])), 0, 0)
            self.sparse_form[2] = newRow
            
        return self
    
    def __eq__(self, other):
        """
        Function to check whether two SparseMatrix objects are exactly equal
        
        Parameters
        ----------
            
        Returns
        -------
        boolean
            True if SparseMatrix objects are exactly equal, else False
        """
        for i in range(3):
            if self.sparse_form[i] != other.sparse_form[i]:
                return False
            elif i == 2 and self.sparse_form[i] == other.sparse_form[i]:
                return True
    
    def __add__(self, other):
        
        if max(self.sparse_form[1]) != max(other.sparse_form[1]) or len(self.sparse_form[2]) != len(other.sparse_form[2]):
            raise Exception(f"Not same dimensions!")
            
        # Create new vectors
        newV = np.empty((0, 0), int)
        newCOL = np.empty((0, 0), int)
        newROW = np.array([0])
        
        # Going through each row
        for i in range(len(self.sparse_form[2]) - 1):
            # the difference between a and b & c and d is equal to the number of elements per row i
            a = self.sparse_form[2][i + 1]
            b = self.sparse_form[2][i]
            c = other.sparse_form[2][i + 1]
            d = other.sparse_form[2][i]
            
            # Checking if elements left in any of the matrices at row i
            while b < a or d < c:
                # If this hold, then only elements left in matrix "self" at row i. 
                if d == c:
                    # adding to new vectors
                    newV = np.append(newV, self.sparse_form[0][b])
                    newCOL = np.append(newCOL, self.sparse_form[1][b])
                    b += 1
                
                # If this hold, then only elements left in matrix "other" at row i. 
                elif b == a:
                    # adding to new vectors
                    newV = np.append(newV, other.sparse_form[0][d])
                    newCOL = np.append(newCOL, other.sparse_form[1][d])
                    d += 1
                    
                # If elements left at row i in both matrices, we check which of the element that comes first
                else:
                    if self.sparse_form[1][b] < other.sparse_form[1][d]:
                        # adding to new vectors
                        newV = np.append(newV, self.sparse_form[0][b])
                        newCOL = np.append(newCOL, self.sparse_form[1][b])
                        b += 1

                    elif self.sparse_form[1][b] > other.sparse_form[1][d]:
                        # adding to new vectors
                        newV = np.append(newV, other.sparse_form[0][d])
                        newCOL = np.append(newCOL, other.sparse_form[1][d])
                        d += 1
                        
                    # element at same posiiton
                    elif self.sparse_form[1][b] == other.sparse_form[1][d]:
                        # if sum is equal to zero, then we don't add to any of the matrices
                        if self.sparse_form[0][b] + other.sparse_form[0][d] == 0:
                            b += 1
                            d += 1
                        else:
                            # adding to new vectors
                            newV = np.append(newV, self.sparse_form[0][b] + other.sparse_form[0][d])
                            newCOL = np.append(newCOL, self.sparse_form[1][b])
                            b += 1
                            d += 1
                            
            # at the end of each row we add the nr of elements appended this far to the new row vector
            newROW = np.append(newROW, len(newV))
        # returning an object with the new vectors
        return SparseMatrix(newV, newCOL, newROW)
    
        def __mul__(self, arr):
            mat_arr=[]
            row_ind_arr=[0]
            nbr_row=0
            sparse_list=[]
            for i in range(len(self.sparse_form[2])-1):               
                sum_elm=0
                for j in range(self.sparse_form[2][i], self.sparse_form[2][i+1]):
                    sum_elm=sum_elm+self.sparse_form[0][j]*arr[self.sparse_form[1][j]][0]
                if sum_elm !=0:
                    nbr_row=nbr_row+1
                    mat_arr.append(sum_elm)  
                row_ind_arr.append(nbr_row)
            col_ind_arr= list(0 for i in range(0,len(mat_arr)))
            sparse_list.append(mat_arr)
            sparse_list.append(col_ind_arr)
            sparse_list.append(row_ind_arr)
            prod_sparse=SparseMatrix(*sparse_list)

            return prod_sparse

     #nbr columns in sparse_1 and nbr rows in arr should be of different length for this test
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
            
    #not_arr should be of any type other than array or list
    def test_input_type_mult(self):  
        mat_1=np.matrix([[1, 2, 3], [4,5,6], [7, 8, 9]])
        try:
            SparseMatrix(mat_1)*5
        except AttributeError:
            pass
        else:
            raise AssertionError()
    
    def test_small_array_mult(self):
        mat_1=np.ndarray([1])
        mat_2=np.ndarray([2, 3])
        mat_3=np.matrix([[-1, 2], [2, -1]])
        sparse_mat_1=SparseMatrix(mat_1)
        sparse_mat_2=SparseMatrix(mat_2)
        sparse_mat_3=SparseMatrix(mat_3)
        arr_1=np.ndarray([2])
        arr_2=np.ndarray([[4], [5]])
        arr_3=np.ndarray([[1],[2]])
        result_1=mat_1*arr_1
        result_2=mat_2*arr_2
        result_3=mat_3*arr_3
        expected_1=SparseMatrix(np.ndarray([2]))
        expected_2=SparseMatrix(np.ndarray([23]))
        expected_3=SparseMatrix(np.ndarray([[3], [0]]))
        errors = []

        # replace assertions by conditions
        if not result_1==expected_1:
            errors.append("Error: wrong result when array has size 1")
        if not result_2==expected_2:
            errors.append("Error: wrong result when array has size 2")
        if not result_3==expected_3:
            errors.append("Error: wrong result when array has size 3")
        assert not errors, "errors:\n{}".format("\n".join(errors))
        
    def test_large_array_mult(self):
        arr_1=np.arange(1, 11)
        arr_2=np.arange(1, 101)
        arr_3=np.arange(1, 1001)
        expected_1=np.zeros((10, 1))
        expected_2=np.zeros((100, 1))
        expected_3=np.zeros((1000, 1))
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
        if not result_2.sparse_form==expected_2.sparse_form:
            errors.append("Error: wrong result when multiplying Sparse of size (100, 100")
        if not result_3.sparse_form==expected_3.sparse_form:
            errors.append("Error: wrong result when multiplying Sparse of size (1000, 1000")
        assert not errors, "errors:\n{}".format("\n".join(errors))
        
    def test_empty_array_mult(self):
        mat_1=np.ndarray([[1, 2, 3], [4,5,6], [7, 8, 9]])
        arr_1=np.ndarray([[0],[0], [0]])
        result_1=mat_1*arr_1
        expected_1=SparseMatrix(np.ndarray([[0],[0], [0]]))
        
        assert allclose(result_1, expected_1, "multiplication with array with only zero elements does not return all zero SparseMatrix")
    def test_empty_sparse_mult(self):
        mat_1=np.ndarray([[0, 0, 0], [0,0,0], [0, 0, 0]])
        arr_1=np.ndarray([[1],[2], [3]])
        result_1=mat_1*arr_1
        expected_1=SparseMatrix(np.ndarray([[0],[0], [0]]))
        
        assert allclose(result_1, expected_1, "multiplication with SparseMatrix with only zero elements does not return all zero SparseMatrix")
            
