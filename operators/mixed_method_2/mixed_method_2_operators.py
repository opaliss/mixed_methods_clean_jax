"""module with mixed method #2 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np
from operators.legendre.legendre_operators import construct_f


def extra_term_1_hermite(I_int_complement, Nv_H, D, Nx, state_legendre, Nv_L):
    """

    :param I_int_complement:
    :param state_legendre:
    :param Nx:
    :param Nv_H:
    :param Nv_L:
    :param D:
    :return:
    """
    sol_ = np.zeros(Nx * Nv_H)
    sol_[(Nv_H - 1) * Nx:] = - np.sqrt(Nv_H / 2) * summation_term(I_int_complement=I_int_complement,
                                                                  D=D, state_legendre=state_legendre, Nx=Nx, Nv_L=Nv_L)
    return sol_


def extra_term_2_hermite(E, state_legendre, Nv_H, Nv_L, Nx, gamma, v_a, v_b,
                         xi_v_a, xi_v_b,
                         psi_dual_v_a, psi_dual_v_b, alpha):
    """

    :param xi_v_a:
    :param xi_v_b:
    :param psi_dual_v_b:
    :param psi_dual_v_a:
    :param alpha:
    :param v_b:
    :param v_a:
    :param E: 1d array, electric field on finite difference mesh
    :param state_legendre: 1d array, vector of all coefficients of delta f [Legendre]
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param Nx: int, grid size in space
    :param Nv_H: int, spectral resolution in velocity [Hermite]
    :param Nv_L: int, spectral resolution in velocity [Legendre]
    :param gamma: float, penalty term
    :return: N(E, psi)
    """
    res_boundary = np.zeros(Nv_H * Nx)
    for nn in range(0, Nv_H):
        if gamma != 0:
            res_boundary[nn * Nx: (nn + 1) * Nx] = boundary_term(n=nn,
                                                                 gamma=gamma * (v_b - v_a),
                                                                 v_b=v_b,
                                                                 v_a=v_a,
                                                                 Nx=Nx,
                                                                 Nv=Nv_L,
                                                                 state_legendre=state_legendre,
                                                                 psi_dual_v_a=psi_dual_v_a,
                                                                 psi_dual_v_b=psi_dual_v_b,
                                                                 xi_v_a=xi_v_a,
                                                                 xi_v_b=xi_v_b)
    return (res_boundary.reshape(Nv_H, Nx) * E).flatten() / alpha


def summation_term(I_int_complement, D, state_legendre, Nx, Nv_L):
    """

    :param I_int_complement:
    :param D:
    :param state_legendre:
    :param Nx:
    :param Nv_L:
    :return:
    """
    sol_ = np.zeros(Nx)
    for ii in range(Nv_L):
        sol_ += I_int_complement[ii] * state_legendre[ii * Nx: (ii + 1) * Nx]
    return D @ sol_


def extra_term_2_legendre(I_int_complement, J_int, Nv_H, Nv_L, Nx, v_b, v_a, D, state_legendre):
    """

    :param I_int_complement:
    :param J_int:
    :param Nv_H:
    :param Nv_L:
    :param Nx:
    :param v_b:
    :param v_a:
    :param D:
    :param state_legendre:
    :return:
    """
    sol_ = np.zeros(Nx * Nv_L)
    sum_term = 1 / (v_b - v_a) * np.sqrt(Nv_H / 2) * summation_term(I_int_complement=I_int_complement,
                                                                    D=D, state_legendre=state_legendre,
                                                                    Nx=Nx, Nv_L=Nv_L)
    for ii in range(Nv_L):
        sol_[ii * Nx: (ii + 1) * Nx] += J_int[ii] * sum_term
    return sol_


def extra_term_3_legendre(J_int, Nv_H, Nv_L, Nx, v_b, v_a, state_legendre, psi_dual_v_b, psi_dual_v_a,
                          xi_v_b, xi_v_a, alpha, gamma, E):
    """

    :param J_int:
    :param Nv_H:
    :param Nv_L:
    :param Nx:
    :param v_b:
    :param v_a:
    :param state_legendre:
    :param psi_dual_v_b:
    :param psi_dual_v_a:
    :param xi_v_b:
    :param xi_v_a:
    :param alpha:
    :param gamma:
    :param E:
    :param q:
    :param m:
    :return:
    """
    res_boundary = np.zeros(Nv_L * Nx)
    for mm in range(0, Nv_L):
        for nn in range(0, Nv_H):
            res_boundary[mm * Nx: (mm + 1) * Nx] += boundary_term(n=nn, gamma=gamma,
                                                                  v_b=v_b, v_a=v_a,
                                                                  Nx=Nx, Nv=Nv_L,
                                                                  state_legendre=state_legendre,
                                                                  psi_dual_v_a=psi_dual_v_a,
                                                                  psi_dual_v_b=psi_dual_v_b,
                                                                  xi_v_a=xi_v_a,
                                                                  xi_v_b=xi_v_b) * J_int[nn, mm]

    return -(res_boundary.reshape(Nv_L, Nx) * E).flatten() / alpha


def boundary_term(n, gamma, v_b, v_a, Nx, Nv, state_legendre, psi_dual_v_a, psi_dual_v_b, xi_v_a, xi_v_b):
    """

    :param psi_dual_v_b:
    :param psi_dual_v_a:
    :param state_legendre:
    :param Nv:
    :param Nx:
    :param v_a:
    :param v_b:
    :param gamma:
    :param n:
    :param xi_v_b:
    :param xi_v_a:
    :return:
    """

    return gamma / (v_b - v_a) * (psi_dual_v_b[n] * construct_f(state=state_legendre, Nv=Nv, Nx=Nx, xi=xi_v_b)
                                  - psi_dual_v_a[n] * construct_f(state=state_legendre, Nv=Nv, Nx=Nx, xi=xi_v_a))
