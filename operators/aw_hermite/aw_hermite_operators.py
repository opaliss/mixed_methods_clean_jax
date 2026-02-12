"""module with Hermite functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Nov 25th, 2025
"""
import numpy as np
import scipy


def aw_psi_hermite_vector(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach)

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
        return np.exp(-xi ** 2) / np.sqrt(np.pi)
    if n == 1:
        return np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
    else:
        psi = np.zeros((n + 1, len(v)))
        psi[0, :] = np.exp(-xi ** 2) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi


def aw_psi_hermite(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach)

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
        return np.exp(-xi ** 2) / np.sqrt(np.pi)
    if n == 1:
        return np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
    else:
        psi = np.zeros((n + 1, len(v)))
        psi[0, :] = np.exp(-xi ** 2) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-xi ** 2) * (2 * xi) / np.sqrt(2 * np.pi)
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def aw_psi_hermite_complement(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach) complement

    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, AW aw_hermite polynomial of degree n evaluated at xi
    """
    # scaled velocity coordinate
    xi = (v - u_s) / alpha_s
    # iteratively compute psi_{n}(xi)
    if isinstance(xi, np.ndarray):
        if n == 0:
            return np.ones(len(xi))
        if n == 1:
            return (2 * xi) / np.sqrt(2)
        else:
            psi = np.zeros((n + 1, len(xi)))
            psi[0, :] = 1
            psi[1, :] = (2 * xi) / np.sqrt(2)
            for jj in range(1, n):
                factor = - alpha_s * np.sqrt((jj + 1) / 2)
                psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
        return psi[n, :]
    else:
        if n == 0:
            return 1
        if n == 1:
            return (2 * xi) / np.sqrt(2)
        else:
            psi = np.zeros(n + 1)
            psi[0] = 1
            psi[1] = (2 * xi) / np.sqrt(2)
            for jj in range(1, n):
                factor = - alpha_s * np.sqrt((jj + 1) / 2)
                psi[jj + 1] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1] + u_s * psi[jj] - v * psi[jj]) / factor
        return psi[n]


def A1_hermite(D, Nv):
    """A1 matrix advection term with alpha

    :param D: 2d array (matrix), finite difference derivative matrix
    :param Nv: int, Hermite spectral resolution
    :return: 2d array (matrix), A1 matrix in advection term
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        if n != 0:
            # lower diagonal
            A[n, n - 1] = np.sqrt(n / 2)
        if n != Nv - 1:
            # upper diagonal
            A[n, n + 1] = np.sqrt((n + 1) / 2)
    return -scipy.sparse.kron(A, D, format="csr")


def B_hermite(Nv,  q, m):
    """

    :param Nv: int, spectral resolution in velocity
    :param q: float, charge of particles
    :param m: float, mass of particles
    :return:
    """
    B = np.zeros((Nv, Nv))
    for nn in range(Nv):
        B[nn, nn-1] = q / m * np.sqrt(2 * nn)
    return scipy.sparse.csr_matrix(B)


# def B_coeff_Hermite(Nv, q, m):
#     B = np.zeros(Nv-1)
#     return (q/m) * np.sqrt(2 * np.arange(1, Nv))


def nonlinear_aw_hermite(E, psi, alpha,B, Nv, Nx, q, m):
    """compute acceleration term (nonlinear)

    :param E: 1d array, electric field on finite difference mesh
    :param psi: 1d array, vector of all coefficients
    :param alpha: float, temperature of particles
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :return: N(E, psi)
    """
    coeff = q/m * np.sqrt(2 * np.arange(1, Nv)) / alpha

    psi2 = psi.reshape(Nv, Nx)
    out = np.zeros_like(psi2)

    out[1:] = coeff[:, None] * psi2[:-1]
    out *= E

    return out.ravel()

    # return (B.dot(psi.reshape(Nv, Nx)) * E).flatten() / alpha


def M1_du_dx(Nv, Nx):
    """M operator with all terms multiplying du/dx that exclude u/alpha

    :param Nv: int, Hermite spectral resolution
    :param Nx: int, grid size in space
    :return:
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        A[n, n - 2] = np.sqrt(n * (n - 1))
        A[n, n] = n
    return -scipy.sparse.kron(A, np.eye(Nx), format="csr")


def M2_du_dx(Nv, Nx):
    """M operator with all terms multiplying du/dx including u/alpha

    :param Nv: int, Hermite spectral resolution
    :param Nx: int, grid size in space
    :return:
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        A[n, n - 1] = np.sqrt(2 * n)
    return -scipy.sparse.kron(A, np.eye(Nx), format="csr")


def M1_dalpha_dx(Nv, Nx):
    """M operator with all terms multiplying dalpha/dx that exclude u/alpha

    :param Nv: int, Hermite spectral resolution
    :param Nx: int, grid size in space
    :return:
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        A[n, n - 3] = np.sqrt(n * (n - 1) * (n - 2) / 2)
        A[n, n - 1] = np.sqrt(2*n) * n
        if n != Nv - 1:
            A[n, n + 1] = np.sqrt((n + 1) / 2) * (n + 1)
    return -scipy.sparse.kron(A, np.eye(Nx), format="csr")


def M2_dalpha_dx(Nv, Nx):
    """M operator with all terms multiplying dalpha/dx that include u/alpha

    :param Nv: int, Hermite spectral resolution
    :param Nx: int, grid size in space
    :return:
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        A[n, n - 2] = np.sqrt(n * (n - 1))
        A[n, n] = n + 1
    return -scipy.sparse.kron(A, np.eye(Nx), format="csr")


def charge_density_aw_hermite(q_e, q_i, alpha_e, alpha_i, C0_e, C0_i):
    """charge density (right hand side of Poisson equation)

    :param q_e: float, charge of electrons
    :param q_i: float, charge of ions
    :param alpha_e: float, aw_hermite scaling parameter or thermal velocity of electrons
    :param alpha_i: float, aw_hermite scaling parameter or thermal velocity of ions
    :param C0_e: 1d array, density of electrons
    :param C0_i: 1d array, density of ions
    :return: change density rho(x, t=t*)
    """
    return q_e * alpha_e * C0_e + q_i * alpha_i * C0_i


def mass_aw_hermite(state):
    """mass of a single specie

    :param state: 1d array, electron or ion state
    :return: mass for the state
    """
    return state[0, :]


def momentum_aw_hermite(state, u_s, alpha_s):
    """momentum of a single specie

    :param state: 1d array, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: momentum for the state
    """
    return alpha_s * state[1, :] / np.sqrt(2) + u_s * state[0, :]


def energy_k_aw_hermite(state, u_s, alpha_s):
    """kinetic energy of a single specie

    :param state: 1d array, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: kinetic energy for the state
    """
    return (alpha_s ** 2) / np.sqrt(2) * state[2, :] + np.sqrt(2) * u_s * alpha_s * state[1, :] \
           + ((alpha_s ** 2) / 2 + u_s ** 2) * state[0, :]


def total_mass_aw_hermite(state, alpha_s, dx):
    """total mass of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :return: total mass of single electron and ion setup
    """
    return np.sum(mass_aw_hermite(state=state) * dx * alpha_s)


def total_momentum_aw_hermite(state, alpha_s, dx, m_s, u_s):
    """total momentum of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return np.sum(momentum_aw_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s)


def total_energy_k_aw_hermite(state, alpha_s, dx, m_s, u_s):
    """total kinetic energy of single electron and ion setup

    :param state: 1d array, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * np.sum(energy_k_aw_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s)


def charge_density_two_stream_aw_hermite(alpha_e1, alpha_e2, alpha_i, C0_e1, C0_e2, C0_i, q_e1=-1, q_e2=-1, q_i=1):
    """charge density (right hand side of Poisson equation)

    :param q_e1: float, charge of electrons species 1
    :param q_e2: float, charge of electrons species 2
    :param q_i: float, charge of ions
    :param alpha_e1: float, aw_hermite scaling parameter or thermal velocity of electrons species 1
    :param alpha_e2: float, aw_hermite scaling parameter or thermal velocity of electrons species 2
    :param alpha_i: float, aw_hermite scaling parameter or thermal velocity of ions
    :param C0_e1: 1d array, density of electrons species 1
    :param C0_e2: 1d array, density of electrons species 2
    :param C0_i: 1d array, density of ions
    :return: change density rho(x, t=t*)
    """
    return q_e1 * alpha_e1 * C0_e1 + q_e2 * alpha_e2 * C0_e2 + q_i * alpha_i * C0_i


def A_matrix_Opi(N, alpha):
    """

    :param N:
    :param alpha:
    :return:
    """
    A = np.zeros((N, N))
    A[0, 0] = alpha / np.sqrt(2 * np.pi)
    for i in range(1, N):
        A[i, i] = A[i - 1, i - 1] * (2 * i - 1) / (2 * i)

    for i in range(0, N):
        for j in range(1, i + 1):
            if (i - j) >= 0 and (i + j) < N:
                A[i - j, i + j] = -np.sqrt((i - (j - 1)) / (i + j)) * A[i - (j - 1), i + (j - 1)]
                A[i + j, i - j] = A[i - j, i + j]
    return A
