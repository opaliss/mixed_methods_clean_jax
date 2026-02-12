"""module with universal functions used for Hermite and Legendre formulations

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np
import scipy


def get_D_inv(Nx, D):
    """inverse of derivative D matrix

    :param Nx: int, number of spatial grid points
    :param D: 2d array (matrix), finite difference derivative matrix
    :return: 2d array (matrix), inverse of D
    """
    mat = np.zeros((Nx + 1, Nx + 1))
    mat[:-1, :-1] = D.toarray()
    mat[-1, :-1] = np.ones(Nx)
    mat[:-1, -1] = np.ones(Nx)
    return np.linalg.inv(mat)


def nu_func(n, Nv):
    """coefficient for hyper-collisions

    :param n: int, index of spectral term
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: float, coefficient for hyper-collisions
    """
    return n * (n - 1) * (n - 2) / (Nv - 1) / (Nv - 2) / (Nv - 3)



def A3(Nx, Nv):
    """A3 matrix in advection term with nu

    :param Nv: int, Hermite spectral resolution
    :param Nx: int, finite difference grid resolution
    :return: 2d array (matrix), A3 matrix in weak landau advection term
    """
    A = np.zeros((Nv, Nv))
    for nn in range(Nv):
        # main diagonal
        A[nn, nn] = nu_func(n=nn, Nv=Nv)
    return -scipy.sparse.kron(A, scipy.sparse.identity(n=Nx), format="csr")


def A2(D, Nv):
    """A2 matrix in advection term with u or sigma_bar

    :param D: 2d array (matrix), finite difference derivative matrix
    :param Nv: int, Hermite spectral resolution
    :return: 2d array (matrix), A3 matrix in advection term
    """
    return -scipy.sparse.kron(scipy.sparse.identity(n=Nv), D, format="csr")