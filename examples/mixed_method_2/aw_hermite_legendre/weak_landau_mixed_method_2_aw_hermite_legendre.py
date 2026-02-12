"""Module to run mixed method #2 weak landau testcase

Author: Opal Issan (oissan@ucsd.edu)
Last updated: Nov 13th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_0.mixed_method_0_operators import charge_density_two_stream_mixed_method_0
from operators.mixed_method_1.mixed_method_1_operators import extra_term_1_legendre
from operators.mixed_method_2.mixed_method_2_operators import extra_term_1_hermite, extra_term_2_legendre, \
    extra_term_3_legendre, extra_term_2_hermite
from operators.legendre.legendre_operators import nonlinear_legendre
from operators.aw_hermite.aw_hermite_operators import nonlinear_aw_hermite
from operators.mixed_method_2.setup_mixed_method_2_two_stream import SimulationSetupMixedMethod2
from operators.implicit_midpoint_adaptive_two_stream import implicit_midpoint_solver_adaptive_two_stream
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed
    rho = charge_density_two_stream_mixed_method_0(q_e=setup.q_e, alpha_e=setup.alpha_e1[-1],
                                                   v_a=setup.v_a, v_b=setup.v_b,
                                                   C0_e_hermite=y[:setup.Nx],
                                                   C0_e_legendre=y[setup.Nx*setup.Nv_e1: setup.Nx*setup.Nv_e1+setup.Nx])

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    dydt_ = np.zeros(len(y))

    # evolving bulk aw_hermite
    A_eH = setup.u_e1[-1] * setup.A_eH_diag + setup.alpha_e1[-1] * setup.A_eH_off + setup.nu_H * setup.A_eH_col
    dydt_[:setup.Nv_e1 * setup.Nx] = A_eH @ y[:setup.Nv_e1 * setup.Nx] \
                                     + nonlinear_aw_hermite(E=E,
                                                            psi=y[:setup.Nv_e1 * setup.Nx],
                                                            q=setup.q_e,
                                                            m=setup.m_e,
                                                            alpha=setup.alpha_e1[-1],
                                                            Nv=setup.Nv_e1,
                                                            Nx=setup.Nx) \
                                     + extra_term_1_hermite(I_int_complement=setup.I_int_complement[-1, :],
                                                            Nv_H=setup.Nv_e1,
                                                            D=setup.D,
                                                            Nx=setup.Nx,
                                                            state_legendre=y[setup.Nv_e1 * setup.Nx:],
                                                            Nv_L=setup.Nv_e2) \
                                     + extra_term_2_hermite(E=E,
                                                            state_legendre=y[setup.Nv_e1 * setup.Nx:],
                                                            Nv_H=setup.Nv_e1,
                                                            Nv_L=setup.Nv_e2,
                                                            Nx=setup.Nx,
                                                            gamma=setup.gamma,
                                                            v_a=setup.v_a,
                                                            v_b=setup.v_b,
                                                            psi_dual_v_a=setup.psi_dual_v_a,
                                                            psi_dual_v_b=setup.psi_dual_v_b,
                                                            xi_v_a=setup.xi_v_a,
                                                            xi_v_b=setup.xi_v_b,
                                                            alpha=setup.alpha_e1[-1])

    dydt_[setup.Nv_e1 * setup.Nx:] = setup.A_e_L @ y[setup.Nv_e1 * setup.Nx:] \
                                     + nonlinear_legendre(E=E,
                                                          psi=y[setup.Nv_e1 * setup.Nx:],
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
                                                             E=E,
                                                             Nv_L=setup.Nv_e2,
                                                             Nx=setup.Nx) \
                                     + extra_term_2_legendre(I_int_complement=setup.I_int_complement[-1, :],
                                                             J_int=setup.J_int[-2, :],
                                                             Nv_H=setup.Nv_e1,
                                                             D=setup.D,
                                                             Nx=setup.Nx,
                                                             state_legendre=y[setup.Nv_e1 * setup.Nx:],
                                                             Nv_L=setup.Nv_e2,
                                                             v_b=setup.v_b,
                                                             v_a=setup.v_a)\
                                    + extra_term_3_legendre(J_int=setup.J_int,
                                                            Nv_H=setup.Nv_e1,
                                                            Nv_L=setup.Nv_e2,
                                                            Nx=setup.Nx,
                                                            v_b=setup.v_b,
                                                            v_a=setup.v_a,
                                                            state_legendre=y[setup.Nv_e1 * setup.Nx:],
                                                            psi_dual_v_b=setup.psi_dual_v_b,
                                                            psi_dual_v_a=setup.psi_dual_v_a,
                                                            xi_v_b=setup.xi_v_b,
                                                            xi_v_a=setup.xi_v_a,
                                                            alpha=setup.alpha_e1[-1],
                                                            gamma=setup.gamma,
                                                            E=E)


    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod2(Nx=21,
                                        Nv_e1=11,
                                        Nv_e2=11,
                                        epsilon=1e-2,
                                        v_a=-2,
                                        v_b=2,
                                        alpha_e1=0.8,
                                        u_e1=0,
                                        L=2 * np.pi,
                                        dt=1e-2,
                                        T0=0,
                                        T=20,
                                        nu_L=0,
                                        nu_H=0,
                                        gamma=1,
                                        alpha_e2=np.nan,
                                        u_e2=np.nan,
                                        k0=1,
                                        alpha_tol=1e-2,
                                        u_tol=1e-2,
                                        n0_e1=1,
                                        n0_e2=0,
                                        construct_integrals=True)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
    # grid
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    # initial condition (only initialize Hermite zeroth coefficient)
    y0[:setup.Nx] = (1 + setup.epsilon * np.cos(x_)) / setup.alpha_e1

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_adaptive_two_stream(y_0=y0,
                                                                         right_hand_side=rhs,
                                                                         a_tol=1e-11,
                                                                         r_tol=1e-11,
                                                                         max_iter=100,
                                                                         param=setup,
                                                                         adaptive_u_and_alpha=True,
                                                                         bulk_hermite_adapt=True,
                                                                         bump_hermite_adapt=False,
                                                                         adaptive_between_hermite_and_legendre=False)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save("../../data/mixed_method_2_aw_hermite_legendre/weak_landau/sol_runtime_NvH_" + str(
        setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T),
            np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/mixed_method_2_aw_hermite_legendre/weak_landau/sol_u_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/mixed_method_2_aw_hermite_legendre/weak_landau/sol_t_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

    # save time varying alpha and u (for the bulk Hermite)
    np.save("../../data/mixed_method_2_aw_hermite_legendre/weak_landau/alpha_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e1)

    np.save("../../data/mixed_method_2_aw_hermite_legendre/weak_landau/u_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e1)
