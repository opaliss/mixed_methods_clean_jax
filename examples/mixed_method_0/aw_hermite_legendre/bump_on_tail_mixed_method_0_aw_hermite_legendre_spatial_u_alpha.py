"""Module to run mixed method #0 (static) bump-on-tail testcase

Author: Opal Issan
Last updates: Oct 22nd, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_0.mixed_method_0_operators import charge_density_two_stream_mixed_method_0
from operators.legendre.legendre_operators import nonlinear_legendre, xi_legendre
from operators.aw_hermite.aw_hermite_operators import nonlinear_aw_hermite
from operators.mixed_method_0.setup_mixed_method_0_two_stream import SimulationSetupMixedMethod0
from operators.implicit_midpoint_adaptive_in_space_two_stream import \
    implicit_midpoint_solver_adaptive_in_space_two_stream
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import scipy


def rhs(y):
    # charge density computed
    rho = charge_density_two_stream_mixed_method_0(q_e=setup.q_e,
                                                   alpha_e=setup.alpha_e1[-1],
                                                   v_a=setup.v_a, v_b=setup.v_b,
                                                   C0_e_hermite=y[:setup.Nx],
                                                   C0_e_legendre=y[setup.Nv_e1 * setup.Nx: (setup.Nv_e1 + 1) * setup.Nx])

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    dydt_ = np.zeros(len(y))

    # evolving bulk aw_hermite
    L_term = setup.A_eH_diag @ (np.tile(setup.u_e1[-1], reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx]) \
             + setup.A_eH_off @ (np.tile(setup.alpha_e1[-1], reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx]) \
             + setup.nu_H * (setup.A_eH_col @ y[:setup.Nv_e1 * setup.Nx])

    N_term = nonlinear_aw_hermite(E=E, psi=y[:setup.Nv_e1 * setup.Nx],
                                  q=setup.q_e,
                                  m=setup.m_e,
                                  alpha=setup.alpha_e1[-1],
                                  Nv=setup.Nv_e1,
                                  Nx=setup.Nx)

    dudx = setup.D @ setup.u_e1[-1]
    dalphadx = setup.D @ setup.alpha_e1[-1]

    M_term_u = setup.M1_du_dx @ (np.tile(dudx, reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx]) \
                    + setup.M2_du_dx @ (np.tile(dudx * setup.u_e1[-1] / setup.alpha_e1[-1], reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx])

    M_term_alpha = setup.M1_dalpha_dx @ (np.tile(dalphadx, reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx])\
                    + setup.M2_dalpha_dx @ (np.tile(dalphadx * setup.u_e1[-1] / setup.alpha_e1[-1], reps=setup.Nv_e1) * y[:setup.Nv_e1 * setup.Nx])

    dydt_[:setup.Nv_e1 * setup.Nx] = L_term + N_term + M_term_u + M_term_alpha

    dydt_[setup.Nv_e1 * setup.Nx:] = setup.A_e_L @ y[setup.Nv_e1 * setup.Nx:] + nonlinear_legendre(E=E, psi=y[
                                                                                                            setup.Nv_e1 * setup.Nx:],
                                                                                                   Nv=setup.Nv_e2,
                                                                                                   Nx=setup.Nx,
                                                                                                   B_mat=setup.B_e_L,
                                                                                                   q=setup.q_e,
                                                                                                   m=setup.m_e,
                                                                                                   gamma=setup.gamma,
                                                                                                   v_a=setup.v_a,
                                                                                                   v_b=setup.v_b,
                                                                                                   xi_v_a=setup.xi_v_a,
                                                                                                   xi_v_b=setup.xi_v_b)

    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod0(Nx=51,
                                        Nv_e1=50,
                                        Nv_e2=50,
                                        epsilon=1e-2,
                                        v_a=-10,
                                        v_b=10,
                                        alpha_e1=np.sqrt(2),
                                        u_e1=0,
                                        u_e2=4.5,
                                        alpha_e2=1/np.sqrt(2),
                                        L=20*np.pi/3,
                                        dt=1e-2,
                                        T0=0,
                                        T=30,
                                        k0=1,
                                        nu_L=1,
                                        nu_H=10,
                                        gamma=0.5,
                                        u_tol=2,
                                        alpha_tol=1,
                                        n0_e1=0.9,
                                        n0_e2=0.1,
                                        adaptive_in_space=True,
                                        construct_integrals=False)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
    # bulk electrons => aw_hermite (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = setup.n0_e1 * (1 + setup.epsilon * np.cos(x_ * setup.k0 / setup.L * 2 * np.pi)) / setup.alpha_e1[-1]
    # beam electrons => legendre (unperturbed)
    v_ = np.linspace(setup.v_a, setup.v_b, setup.Nv_int, endpoint=True)
    x_component = 1 / (setup.v_b - setup.v_a) / np.sqrt(np.pi)
    for nn in range(setup.Nv_e2):
        xi = xi_legendre(n=nn, v=v_, v_a=setup.v_a, v_b=setup.v_b)
        exp_ = setup.n0_e2 * np.exp(-((v_ - setup.u_e2[-1]) ** 2) / (setup.alpha_e2[-1] ** 2)) / setup.alpha_e2[-1]
        v_component = scipy.integrate.trapezoid(xi * exp_, x=v_, dx=np.abs(v_[1] - v_[0]))
        y0[setup.Nx * setup.Nv_e1 + nn * setup.Nx: setup.Nx * setup.Nv_e1 + (
                nn + 1) * setup.Nx] = x_component * v_component

    setup.alpha_e1[-1] = setup.alpha_e1[-1] * np.ones(setup.Nx)
    setup.u_e1[-1] = setup.u_e1[-1] * np.ones(setup.Nx)
    if setup.construct_integrals:
        setup.update_J()
    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_adaptive_in_space_two_stream(y_0=y0,
                                                                                  right_hand_side=rhs,
                                                                                  a_tol=1e-10,
                                                                                  r_tol=1e-10,
                                                                                  max_iter=100,
                                                                                  param=setup,
                                                                                  adaptive_u_and_alpha=True,
                                                                                  bulk_hermite_adapt=True,
                                                                                  bump_hermite_adapt=False,
                                                                                  MM0=True,
                                                                                  adaptive_between_hermite_and_legendre=False)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)


    # save the runtime
    np.save("../../../data/mixed_method_0_aw_hermite_legendre/bump_on_tail/sol_runtime_NvH_" + str(
        setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T),
            np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../../data/mixed_method_0_aw_hermite_legendre/bump_on_tail/sol_u_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../../data/mixed_method_0_aw_hermite_legendre/bump_on_tail/sol_t_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

    # save time varying alpha and u
    np.save("../../../data/mixed_method_0_aw_hermite_legendre/bump_on_tail/alpha_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e1)

    np.save("../../../data/mixed_method_0_aw_hermite_legendre/bump_on_tail/u_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e1)
