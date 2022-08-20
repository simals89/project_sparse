import numpy as np
import scipy.sparse as sp
import cProfile
from project import SparseMatrix

# Create matrix
np.random.seed(10)
A = np.zeros((10000,10000))
idx = np.random.choice(np.linspace(0,np.shape(A)[0]*np.shape(A)[0]-1,np.shape(A)[0]*np.shape(A)[0]), 
                       size = int(np.shape(A)[0]*np.shape(A)[0]*0.1), replace = False).astype(int)
A[np.unravel_index(idx, G.shape)] = np.random.randint(0, 100, int((np.shape(A)[0]*np.shape(A)[0]*0.1)))

# Create SparseMatrix
M_sparse = SparseMatrix(A)

# Create CSR matrix using SciPy
M_scipy = sp.csr_matrix(G)

# Run cProfile
cProfile.run("M_sparse.change(0,0,99)")
cProfile.run("scipyChange(M_scipy, 0, 0, 99)")
