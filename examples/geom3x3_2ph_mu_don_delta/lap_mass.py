import numpy as np
import os
import pickle
import time
import scipy
from scipy.sparse import csr_matrix

import cupy as cp
from cupy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import eigsh, spsolve
from cupy.linalg import eigh

filepath = './matrices.pkl'
with open(filepath, 'rb') as filepath:
    arquivos = pickle.load(filepath)
A = arquivos['bigA']
M = arquivos['bigM']
indices = np.squeeze(arquivos['indices'])
vertices = np.squeeze(arquivos['vertices'])
centroid = np.squeeze(arquivos['centroid'])
B_matrices = np.squeeze(arquivos['B_matrices'])
A_matrices = np.squeeze(arquivos['A_matrices'])
del arquivos


def compute_invM_A(M_gpu, A_gpu):
    """
    Computes inv(M) @ A for sparse matrices M and A using CuPy.

    Args:
        M_gpu: Cupy sparse matrix representing the matrix M.
        A_gpu: Cupy sparse matrix representing the matrix A.

    Returns:
        invM_A: Cupy sparse matrix representing inv(M) @ A.

    Note:
        - This function assumes M is invertible.
        - It uses spsolve to solve linear systems efficiently.
    """

    # Get the number of columns in A
    num_cols_A = A_gpu.shape[1]

    # Create an identity matrix of appropriate size
    I_gpu = csr_matrix(cp.eye(M_gpu.shape[0]))

    # Create an empty list to store the columns of inv(M) @ A
    invM_A_cols = []

    # Solve M * x = a_i for each column a_i of A
    for i in range(num_cols_A):
        a_i_gpu = A_gpu[:, i]  # Extract the i-th column of A
        x_i_gpu = spsolve(M_gpu, a_i_gpu)
        invM_A_cols.append(x_i_gpu)

    # Concatenate the columns to form the inv(M) @ A matrix
    invM_A_gpu = cp.hstack(invM_A_cols)

    return invM_A_gpu

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

time_start = time.time()

A_gpu = cp_csr_matrix(A)
M_gpu = cp_csr_matrix(M)

print("inverting and multiplying")
invM_A_gpu = compute_invM_A(M_gpu, A_gpu) 
time_end = time.time()
elapsed_time = time_end - time_start
print(f"Elapsed time: {elapsed_time}")

print("eigenvalues")

# eigvals, eigvecs = eigh(A_gpu, M_gpu)
# Solve for a few eigenvalues and eigenvectors (e.g., k=5)
eigvals, eigvecs = eigsh(invM_A_gpu, k=5, which='SA')


time_end = time.time()
elapsed_time = time_end - time_start
print(f"Elapsed time: {elapsed_time}")


# filepath = './matrices_delta.pkl'
# with open(filepath,"wb") as filepath:
# 	pickle.dump({"bigA":  A_sparse,
#                   "bigM": M_sparse, 
#                   "indices": indices,
#                   "vertices": vertices,
#                   "centroid": centroid,
#                   "B_matrices": B_matrices,
#                   "A_matrices": A_matrices,
#                   "elapsed_time": elapsed_time, 
#               }, filepath)