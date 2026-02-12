"""update the f0 and df distributions.

Author: Opal Issan (oissan@ucsd.edu)
Date: Oct 24th, 2025
"""

import numpy as np
from operators.legendre.legendre_operators import xi_legendre
from operators.aw_hermite.aw_hermite_operators import aw_psi_hermite, aw_psi_hermite_complement
import scipy
import copy
import matplotlib


def reprojection_aw_hermite_and_legendre_old(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, J):
    """

    :param cutoff: int, has to be less than Nv_e1.
    :param Nx: int, spatial resolution
    :param Nv_e1: int, velocity resolution "bulk"
    :param Nv_e2: int, velocity resolution "bump"
    :param y_curr: 1d array, solution from previous timestep
    :param v_a: float, lower bound of the velocity coordinate for Legendre
    :param v_b: float, upper bound of the velocity coordinate for Legendre
    :param J: 2d array, matrix with projection of aw hermite and legendre basis J_[n,m] = int psi_n xi_m
    :return:
    """
    new_solution = np.zeros(len(y_curr))
    new_solution[:Nx * cutoff] = y_curr[:Nx * cutoff]

    for m in range(Nv_e2):
        hermite_correction = np.zeros(Nx)
        for p in range(cutoff, Nv_e1):
            hermite_correction += y_curr[p * Nx:(p + 1) * Nx] * J[p, m] / (v_b - v_a)

        new_solution[Nx * Nv_e1 + m * Nx: Nx * Nv_e1 + (m + 1) * Nx] = y_curr[Nx * Nv_e1 + m * Nx: Nx * Nv_e1 + (
                m + 1) * Nx] + hermite_correction
    return new_solution


def reprojection_aw_hermite_and_legendre(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, Nv_int, alpha, u):
    """

    :param Nx: int, spatial resolution
    :param Nv_e1: int, velocity resolution "bulk"
    :param Nv_e2: int, velocity resolution "bump"
    :param y_curr: 1d array, solution from previous timestep
    :param v_a: float, lower bound of the velocity coordinate for Legendre
    :param v_b: float, upper bound of the velocity coordinate for Legendre
    :param J: 2d array, matrix with projection of aw hermite and legendre basis J_[n,m] = int psi_n xi_m
    :return:
    """
    new_solution = np.zeros(len(y_curr))
    f0 = np.zeros((Nx, Nv_int))
    df = np.zeros((Nx, Nv_int))
    f0_approx = np.zeros((Nx, Nv_int))
    v_ = np.linspace(v_a, v_b, Nv_int, endpoint=True)

    for nn in range(0, Nv_e1):
        f0 += np.outer(y_curr[nn*Nx: (nn+1)*Nx], aw_psi_hermite(v=v_, u_s=u, alpha_s=alpha, n=nn))
    for mm in range(0, Nv_e2):
        df += np.outer(y_curr[Nx*Nv_e1 + mm*Nx: Nx*Nv_e1 + (mm+1)*Nx], xi_legendre(n=mm, v=v_, v_a=v_a, v_b=v_b))

    # total f is the sum of the two
    psi0 = aw_psi_hermite(v=v_, u_s=u, alpha_s=alpha, n=0)
    convolved_f = scipy.ndimage.convolve1d(f0 + df, psi0, axis=1) / len(v_)

    f0_new = f0 + df
    idx = np.where(convolved_f < cutoff)
    # # for bump on tail let the bottom part stay intact
    # y_ = idx[1][np.where(idx[1] > len(v_) // 2)]
    # x_ = idx[0][np.where(idx[1] > len(v_) // 2)]
    # f0_new[x_, y_] = 0

    # or
    f0_new[idx] = 0

    for nn in range(Nv_e1//2):
        new_solution[nn*Nx: (nn+1)*Nx] = \
            scipy.integrate.trapezoid(f0_new * aw_psi_hermite_complement(n=nn, alpha_s=alpha, u_s=u, v=v_), x=v_, dx=np.abs(v_[1]-v_[0])) / alpha
        f0_approx += np.outer(new_solution[nn*Nx: (nn+1)*Nx], aw_psi_hermite(v=v_, u_s=u, alpha_s=alpha, n=nn))

    df_new = f0 + df - f0_approx
    for mm in range(Nv_e2):
        new_solution[Nx*Nv_e1 + mm*Nx: Nx*Nv_e1 + (mm+1)*Nx] = \
            scipy.integrate.trapezoid(df_new * xi_legendre(n=mm, v=v_, v_a=v_a, v_b=v_b), x=v_, dx=np.abs(v_[1]-v_[0])) / (v_b -v_a)

    return new_solution


def reprojection_adaptive_in_space_aw_hermite_and_legendre(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, J):
    """

    :param cutoff:
    :param Nx:
    :param Nv_e1:
    :param Nv_e2:
    :param y_curr:
    :param v_a:
    :param v_b:
    :param J:
    :return:
    """
    new_solution = np.zeros(len(y_curr))
    new_solution[:Nx * cutoff] = y_curr[:Nx * cutoff]

    for ii in range(Nx):
        for m in range(Nv_e2):
            hermite_correction = 0
            for p in range(cutoff, Nv_e1):
                hermite_correction += y_curr[p * Nx + ii] * J[p, m, ii] / (v_b - v_a)
            new_solution[Nx * Nv_e1 + m * Nx + ii] = y_curr[Nx * Nv_e1 + m * Nx + ii] + hermite_correction
    return new_solution
