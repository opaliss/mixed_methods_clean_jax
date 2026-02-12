"""module to setup Legendre simulation.

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""

import numpy as np
from operators.legendre.legendre_operators import A1_legendre, sigma_bar, B_legendre, xi_legendre
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central


class SimulationSetupLegendre:
    def __init__(self, Nx, Nv_e,  epsilon, v_a, v_b, gamma, L, dt, T0, T, nu,
                 alpha_e1, alpha_e2, u_e1, u_e2, n0_e1, n0_e2, k0, Nv_int=1000,
                 m_e=1, m_i=1836, q_e=-1, q_i=1, ions=False, Nv_i=0, problem_dir=None):
        # velocity grid
        # set up configuration parameters
        # spatial resolution
        self.Nx = Nx
        # velocity resolution
        self.Nv_e = Nv_e
        self.Nv_i = Nv_i
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity boundaries
        self.v_a = v_a
        self.v_b = v_b
        # initial condition parameters of a Maxwellian
        self.alpha_e1 = alpha_e1
        self.alpha_e2 = alpha_e2
        self.u_e1 = u_e1
        self.u_e2 = u_e2
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
        # resolution of projection of initial condition on the Legendre basis
        self.Nv_int = Nv_int
        # penalty magnitude
        self.gamma = gamma
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping delta t
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e = m_e
        self.m_i = m_i
        # charge normalized
        self.q_e = q_e
        self.q_i = q_i
        # artificial collisional frequency
        self.nu = nu
        # directory name
        self.problem_dir = problem_dir
        # initial perturbed wavenumber
        self.k0 = k0

        # matrices
        # finite difference derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=True, order=2)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        self.A_e = A1_legendre(D=self.D, Nv=self.Nv_e, v_a=v_a, v_b=v_b) \
                   + sigma_bar(v_a=self.v_a, v_b=self.v_b) * A2(D=self.D, Nv=self.Nv_e) \
                   + self.nu * A3(Nx=self.Nx, Nv=self.Nv_e)

        self.B_e = B_legendre(Nv=self.Nv_e, Nx=self.Nx, v_a=self.v_a, v_b=self.v_b)

        # xi functions
        self.xi_v_a = np.zeros(max(self.Nv_e, self.Nv_i))
        self.xi_v_b = np.zeros(max(self.Nv_e, self.Nv_i))
        for nn in range(max(self.Nv_e, self.Nv_i)):
            self.xi_v_a[nn] = xi_legendre(n=nn, v=self.v_a, v_a=self.v_a, v_b=self.v_b)
            self.xi_v_b[nn] = xi_legendre(n=nn, v=self.v_b, v_a=self.v_a, v_b=self.v_b)

        if ions:
            self.A_i = A1_legendre(D=self.D, Nv=self.Nv_i, v_a=v_a, v_b=v_b) \
                   + sigma_bar(v_a=self.v_a, v_b=self.v_b) * A2(D=self.D, Nv=self.Nv_i) \
                   + self.nu * A3(Nx=self.Nx, Nv=self.Nv_i)

            self.B_i = B_legendre(Nv=self.Nv_i, Nx=self.Nx, v_a=self.v_a, v_b=self.v_b)


