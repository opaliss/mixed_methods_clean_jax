"""module to setup Hermite simulation with two electron populations

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Feb 12th, 2026
"""
import numpy as np
from operators.aw_hermite.aw_hermite_operators import A1_hermite
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central


class SimulationSetupTwoStreamHermite:
    def __init__(self, Nx,  Nv_e1, Nv_e2, epsilon, alpha_e1, alpha_e2, alpha_i, u_e1, u_e2, u_i, L,
                 dt, T0, T, u_tol, alpha_tol, nu_e1, nu_e2,  n0_e1, n0_e2, k0, FD_order=2,
                 periodic=True, nu_i=0, m_e1=1, m_e2=1, m_i=1836, q_e1=-1, q_e2=-1, q_i=1, Nv_i=0):
        # set up configuration parameters
        # resolution in space
        self.Nx = Nx
        # resolution in velocity
        self.Nv_e1 = Nv_e1
        self.Nv_e2 = Nv_e2
        self.Nv_i = Nv_i
        # parameters tolerances
        self.u_tol = u_tol
        self.alpha_tol = alpha_tol
        # total DOF for each species
        # self.N = self.Nv * self.Nx
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling of electron and ion
        self.alpha_e1 = [alpha_e1]
        self.alpha_e2 = [alpha_e2]
        self.alpha_i = [alpha_i]
        # velocity scaling
        self.u_e1 = [u_e1]
        self.u_e2 = [u_e2]
        self.u_i = [u_i]
        # average density coefficient
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e1 = m_e1
        self.m_e2 = m_e2
        self.m_i = m_i
        # charge normalized
        self.q_e1 = q_e1
        self.q_e2 = q_e2
        self.q_i = q_i
        # artificial collisional frequency
        self.nu_e1 = nu_e1
        self.nu_e2 = nu_e2
        self.nu_i = nu_i
        # order of finite difference operator
        self.FD_order = FD_order
        # initial perturbed wavenumber
        self.k0 = k0

        # matrices
        # Fourier derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=periodic, order=FD_order)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # A matrices
        # matrix of coefficients (advection)
        A_diag_e1 = A2(D=self.D, Nv=self.Nv_e1)
        A_off_e1 = A1_hermite(D=self.D, Nv=self.Nv_e1)
        A_col_e1 = A3(Nx=self.Nx, Nv=self.Nv_e1)

        A_diag_e2 = A2(D=self.D, Nv=self.Nv_e2)
        A_off_e2 = A1_hermite(D=self.D, Nv=self.Nv_e2)
        A_col_e2 = A3(Nx=self.Nx, Nv=self.Nv_e2)

        # save A
        self.A_e1 = self.u_e1[-1] * A_diag_e1 + self.alpha_e1[-1] * A_off_e1 + self.nu_e1 * A_col_e1
        self.A_e2 = self.u_e2[-1] * A_diag_e2 + self.alpha_e2[-1] * A_off_e2 + self.nu_e2 * A_col_e2

    def add_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1.append(alpha_e1_curr)

    def add_alpha_e2(self, alpha_e2_curr):
        self.alpha_e2.append(alpha_e2_curr)

    def add_alpha_i(self, alpha_i_curr):
        self.alpha_i.append(alpha_i_curr)

    def add_u_e1(self, u_e1_curr):
        self.u_e1.append(u_e1_curr)

    def add_u_e2(self, u_e2_curr):
        self.u_e2.append(u_e2_curr)

    def add_u_i(self, u_i_curr):
        self.u_i.append(u_i_curr)

    def replace_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1[-1] = alpha_e1_curr

    def replace_alpha_e2(self, alpha_e2_curr):
        self.alpha_e2[-1] = alpha_e2_curr

    def replace_alpha_i(self, alpha_i_curr):
        self.alpha_i[-1] = alpha_i_curr

    def replace_u_e1(self, u_e1_curr):
        self.u_e1[-1] = u_e1_curr

    def replace_u_e2(self, u_e2_curr):
        self.u_e2[-1] = u_e2_curr

    def replace_u_i(self, u_i_curr):
        self.u_i[-1] = u_i_curr