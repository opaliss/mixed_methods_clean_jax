"""module with Hermite functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Oct 23rd, 2025
"""
import numpy as np
import scipy


def sw_psi_hermite(n, alpha_s, u_s, v):
    """SW Hermite basis function (iterative approach)
                ==> FYI this is also equal to the complement because it is the symmetric formulation =)
    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, AW aw_hermite polynomial of degree n evaluated at xi
    """
    # scaled velocity coordinate
    xi = (v - u_s) / alpha_s
    # iteratively compute psi_{n}(xi)
    if n == 0:
        return np.exp(-0.5 * (xi ** 2)) / np.sqrt(np.sqrt(np.pi))
    if n == 1:
        return np.exp(-0.5 * (xi ** 2)) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
    else:
        psi = np.zeros((n + 1, len(xi)))
        psi[0, :] = np.exp(-0.5 * (xi ** 2)) / np.sqrt(np.sqrt(np.pi))
        psi[1, :] = np.exp(-0.5 * (xi ** 2)) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def nonlinear_sw_hermite(E, psi, q, m, alpha, Nv, Nx):
    """compute acceleration term (nonlinear)

    :param E: 1d array, electric field on finite difference mesh
    :param psi: 1d array, vector of all coefficients
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param alpha: float, temperature of particles
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :return: N(E, psi)
    """
    res = np.zeros(len(psi))
    for n in range(Nv):
        if n != 0:
            res[n * Nx: (n + 1) * Nx] = (-np.sqrt(n) * psi[(n - 1) * Nx: n * Nx]
                                         +np.sqrt(n+1) * psi[(n + 1) * Nx: (n+2) * Nx]) * E
    return q / m / alpha * res / np.sqrt(2)



def integral_I0(n):
    """
    :param n: int, the order of the integral
    :return: the integral I0_{n}
    """
    if n < 0:
        return 0
    elif n == 0:
        return np.sqrt(2) * (np.pi ** (1 / 4))
    elif n % 2 == 1:
        return 0
    else:
        term = np.zeros(n+10)
        term[0] = np.sqrt(2) * (np.pi ** (1 / 4))
        for m in range(2, n+10):
            term[m] = np.sqrt((m - 1) / m) * term[m - 2]
        return term[n]


def integral_I1(n, u_s, alpha_s):
    """
    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I1_{n}
    """
    if n % 2 == 0:
        return u_s * integral_I0(n=n)
    else:
        return alpha_s * np.sqrt(2) * np.sqrt(n) * integral_I0(n=n - 1)


def integral_I2(n, u_s, alpha_s):
    """integral I2 in SW formulation

    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I2_{n}
    """
    if n % 2 == 0:
        return (alpha_s ** 2) * (0.5 * np.sqrt((n + 1) * (n + 2)) * integral_I0(n=n + 2) + (
                (2 * n + 1) / 2 + (u_s / alpha_s) ** 2) * integral_I0(n=n) + 0.5 * np.sqrt(n * (n - 1)) *
                                 integral_I0(n=n - 2))
    else:
        return 2 * u_s * integral_I1(n=n, u_s=u_s, alpha_s=alpha_s)



def density_sw(state_, Nv_e1, Nx):
    """charge density for single electron and ion species

    :param Nx: int, the number of grid points in space
    :param Nv: int, the number of spectral terms
    :param state_e: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :param q_e: float, the normalized charge of electrons
    :param q_i: float, the normalized charge of ions
    :return: L_{2}(t)
    """
    term1 = np.zeros(Nx)
    for m in range(Nv_e1):
        term1 += state_[m, :] * integral_I0(n=m)
    return term1