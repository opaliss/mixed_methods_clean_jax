"""module to solve Poisson Equation.

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Nov 20th, 2023
"""
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lgmres
import numpy as np


def gmres_solver(rhs, D, D_inv, a_tol=1e-8, r_tol=1e-8):
    """Poisson solver using an iterative solver: GMRES

    :param D_inv: 2d array (matrix), inverse of derivative matrix
    :param a_tol: float, lgmres absolute error tolerance (default is 1e-8)
    :param r_tol: float, lgmres relative error tolerance (default is 1e-8)
    :param D: 2d array (matrix), derivative matrix
    :param rhs: array, rhs of the equation (poisson)
    :return: E that satisfies d/dx E = rho or d^2/dx^2 phi = rho
    """
    x, _ = lgmres(A=D, b=rhs - np.mean(rhs), atol=a_tol, rtol=r_tol, x0=linear_solver_v2(rhs=rhs, D_inv=D_inv))
    return x - np.mean(x)


def linear_solver_v2(rhs, D_inv):
    """Poisson solver

    :param D_inv: derivative matrix
    :param rhs: array, rhs of the equation (poisson)
    :return: E that satisfies d/dx E = rho or d^2/dx^2 phi = rho
    """
    x_ = D_inv @ np.append(rhs - np.mean(rhs), 0)
    return x_[:-1] - np.mean(x_[:-1])


def fft_solver_Ax_b(rhs, D, dx):
    """Poisson solver using fft of the equations
        A x= b
        fft(A[:, 0]) * fft(x) = fft(b)
        fft(x) = fft(b) / fft(A[:, 0])
        x = ifft(fft(b) / fft(A[:, 0]))

    :param rhs: array, rhs of the equation (poisson)
    :param D: matrix, derivative matix
    :return: E that satisfies d/dx E = rho
    """
    A = np.zeros(len(rhs))
    A[1] = -1 / (2 * dx)
    A[-1] = 1 / (2 * dx)

    rhs_fft = np.fft.fft(rhs - np.mean(rhs))
    A_fft = np.fft.fft(A)
    sol = np.divide(rhs_fft, A_fft, where=A_fft != 0)
    x = np.fft.ifft(sol).real
    L_inf = np.max(np.abs(D @ x - rhs))
    print("fft error = ", L_inf)
    if L_inf < 1e-10:
        return x - np.mean(x)
    else:
        return gmres_solver(rhs=rhs, D=D)



def fft_solver(rhs, L):
    """Poisson solver using fft

        d/dx x = b
        J @ fft(x) = fft(b)
        x = ifft(J @ fft(b))

    :param rhs: array, rhs of the equation (poisson)
    :param L: float, length of spatial domain
    :return: x that satisfies Ax = rhs
    """
    x = np.fft.fft(rhs - np.mean(rhs))
    x = np.fft.fftshift(x)
    N = len(x)

    for index, k in enumerate(np.arange(-int(np.floor(N/2)), int(np.floor(N/2)) + 1)):
        if not k == 0:
            x[index] /= 1j*2*np.pi*k/L
        else:
            x[index] = 0

    x_ = np.fft.ifft(np.fft.ifftshift(x)).real
    return x_