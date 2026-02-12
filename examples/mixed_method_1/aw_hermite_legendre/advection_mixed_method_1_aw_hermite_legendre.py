"""Module to run mixed method #1 bump-on-tail instability

Author: Opal Issan (oissan@ucsd.edu)
Date: Dec 9th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_1.mixed_method_1_operators import extra_term_1_legendre
from operators.legendre.legendre_operators import nonlinear_legendre
from operators.mixed_method_1.setup_mixed_method_1_two_stream import SimulationSetupMixedMethod1
from operators.implicit_midpoint_adaptive_single_stream import implicit_midpoint_solver_adaptive_single_stream
import time
import numpy as np
import scipy


def rhs(y):
    # electric field computed (poisson solver)
    E = np.zeros(setup.Nx)

    dydt_ = np.zeros(len(y))

    # evolving bulk aw_hermite
    A_eH = setup.u_e1[-1] * setup.A_e_H_diag + setup.alpha_e1[-1] * setup.A_e_H_off + setup.nu_H * setup.A_e_H_col
    dydt_[:setup.Nv_e1 * setup.Nx] = A_eH @ y[:setup.Nv_e1 * setup.Nx] \
                                    + scipy.sparse.kron(setup.B_e_H, scipy.sparse.diags(E, offsets=0)) @ y[:setup.Nv_e1 * setup.Nx] / setup.alpha_e1[-1]

    dydt_[setup.Nv_e1 * setup.Nx:] = setup.A_e_L @ y[setup.Nv_e1 * setup.Nx:] \
                                     + nonlinear_legendre(E=E, psi=y[setup.Nv_e1 * setup.Nx:],
                                                          Nv=setup.Nv_e2,
                                                          Nx=setup.Nx,
                                                          B_mat=setup.B_e_L,
                                                          q=setup.q_e,
                                                          m=setup.m_e,
                                                          gamma=setup.gamma,
                                                          v_a=setup.v_a,
                                                          v_b=setup.v_b,
                                                          xi_v_a=setup.xi_v_a,
                                                          xi_v_b=setup.xi_v_b) \
                                     + extra_term_1_legendre(J_int=setup.J_int[-1, :],
                                                             v_b=setup.v_b,
                                                             v_a=setup.v_a,
                                                             C_hermite_last=y[(setup.Nv_e1 - 1) * setup.Nx: setup.Nv_e1 * setup.Nx],
                                                             alpha=setup.alpha_e1[-1],
                                                             Nv_H=setup.Nv_e1,
                                                             D=setup.D,
                                                             E=E)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod1(Nx=101,
                                        Nv_e1=100,
                                        Nv_e2=100,
                                        epsilon=1,
                                        v_a=-5,
                                        v_b=5,
                                        alpha_e1=np.sqrt(2),
                                        u_e1=0,
                                        L=2 * np.pi,
                                        dt=1e-2,
                                        T0=0,
                                        T=20,
                                        nu_L=0,
                                        nu_H=0,
                                        n0_e1=1,
                                        n0_e2=0,
                                        alpha_e2=0,
                                        u_e2=0,
                                        gamma=0.5,
                                        alpha_tol=np.inf,
                                        u_tol=np.inf,
                                        k0=1,
                                        construct_integrals=True)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
    # grid
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = (1 + setup.epsilon * np.cos(setup.k0 * x_)) / setup.alpha_e1[-1]

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_adaptive_single_stream(y_0=y0,
                                                                            right_hand_side=rhs,
                                                                            a_tol=1e-10,
                                                                            r_tol=1e-10,
                                                                            max_iter=100,
                                                                            param=setup,
                                                                            adaptive=False)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save(
        "/Users/oissan/PycharmProjects/mixed_methods/data/mixed_method_1_aw_hermite_legendre/advection/sol_runtime_NvH_" + str(
            setup.Nv_e1) + "_NvL_" + str(
            setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T),
        np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save(
        "/Users/oissan/PycharmProjects/mixed_methods/data/mixed_method_1_aw_hermite_legendre/advection/sol_u_NvH_" + str(
            setup.Nv_e1) + "_NvL_" + str(
            setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save(
        "/Users/oissan/PycharmProjects/mixed_methods/data/mixed_method_1_aw_hermite_legendre/advection/sol_t_NvH_" + str(
            setup.Nv_e1) + "_NvL_" + str(
            setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
