"""Module to run the bump-on-tail instability Hermite testcase

Author: Opal Issan (oissan@ucsd.edu)
Last modified: Feb 12th, 2026
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
from operators.aw_hermite.aw_hermite_operators import nonlinear_aw_hermite, charge_density_two_stream_aw_hermite
from operators.implicit_midpoint_adaptive_two_stream import implicit_midpoint_solver_adaptive_two_stream
from operators.aw_hermite.setup_aw_hermite_two_stream import SimulationSetupTwoStreamHermite
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed for poisson's equation
    rho = charge_density_two_stream_aw_hermite(C0_e1=y[:setup.Nx],
                                               C0_e2=y[setup.Nx * setup.Nv_e1: setup.Nx * (setup.Nv_e1 + 1)],
                                               C0_i=np.ones(setup.Nx) / setup.alpha_i[-1],
                                               alpha_e1=setup.alpha_e1[-1],
                                               alpha_e2=setup.alpha_e2[-1],
                                               alpha_i=setup.alpha_i[-1],
                                               q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i)

    # electric field computed
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # initialize the rhs dydt
    dydt_ = np.zeros(len(y))
    # evolving electrons
    # electron species (1) => bulk
    dydt_[:setup.Nv_e1 * setup.Nx] = setup.A_e1 @ y[:setup.Nv_e1 * setup.Nx] \
                                     + nonlinear_aw_hermite(E=E,
                                                            psi=y[:setup.Nv_e1 * setup.Nx],
                                                            B=setup.B_e1,
                                                            alpha=setup.alpha_e1[-1],
                                                            Nx=setup.Nx,
                                                            Nv=setup.Nv_e1, q=setup.q_e1, m=setup.m_e1)

    # electron species (2) => bump
    dydt_[setup.Nv_e1 * setup.Nx:] = setup.A_e2 @ y[setup.Nv_e1 * setup.Nx:] \
                                     + nonlinear_aw_hermite(E=E,
                                                            psi=y[setup.Nv_e1 * setup.Nx:],
                                                            B=setup.B_e2,
                                                            alpha=setup.alpha_e2[-1],
                                                            Nx=setup.Nx,
                                                            Nv=setup.Nv_e2, q=setup.q_e1, m=setup.m_e1)
    return dydt_


if __name__ == "__main__":
    for Nv in 2**np.array([5]):
        if Nv == int(2 ** 5):
            nu = 1
        if Nv == int(2 ** 6):
            nu = 10
        elif Nv == int(2 ** 7):
            nu = 20
        elif Nv == int(2 ** 8):
            nu = 20
        setup = SimulationSetupTwoStreamHermite(Nx=101,
                                                Nv_e1=int(16),
                                                Nv_e2=int(Nv - 16),
                                                epsilon=1e-4,
                                                alpha_e1=np.sqrt(2),
                                                alpha_e2=np.sqrt(2),
                                                alpha_i=np.sqrt(2 / 1836),
                                                u_e1=0,
                                                u_e2=10,
                                                u_i=0,
                                                L=20 * np.pi,
                                                dt=1e-2,
                                                T0=0,
                                                T=120,
                                                k0=0.1,
                                                nu_e1=10,
                                                nu_e2=nu,
                                                n0_e1=0.99,
                                                n0_e2=0.01,
                                                alpha_tol=np.nan,
                                                u_tol=np.nan)

        # initial condition: read in result from previous simulation
        # ions (unperturbed + static)
        y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
        # first electron 1 species (perturbed)
        x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
        # first electron species ==> "bulk" (perturbed)
        y0[:setup.Nx] = setup.n0_e1 * (1 + setup.epsilon * np.cos(x_ * setup.k0)) / setup.alpha_e1[-1]
        # second electron species ==> "bump" (perturbed)
        y0[setup.Nv_e1 * setup.Nx: setup.Nv_e1 * setup.Nx + setup.Nx] = setup.n0_e2 / setup.alpha_e2[-1]

        # start timer
        start_time_cpu = time.process_time()
        start_time_wall = time.time()

        # integrate (implicit midpoint)
        sol_midpoint_u, setup = implicit_midpoint_solver_adaptive_two_stream(y_0=y0,
                                                                             right_hand_side=rhs,
                                                                             r_tol=1e-10,
                                                                             a_tol=1e-10,
                                                                             max_iter=100,
                                                                             bump_hermite_adapt=False,
                                                                             bulk_hermite_adapt=False,
                                                                             adaptive_u_and_alpha=False,
                                                                             adaptive_between_hermite_and_legendre=False,
                                                                             param=setup)

        end_time_cpu = time.process_time() - start_time_cpu
        end_time_wall = time.time() - start_time_wall

        print("runtime cpu = ", end_time_cpu)
        print("runtime wall = ", end_time_wall)

        # save runtime
        np.save("../../data/aw_hermite/bump_on_tail/sol_runtime_Nve1_" + str(setup.Nv_e1)
                + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0)
                + "_" + str(setup.T) + ".npy", np.array([end_time_cpu, end_time_wall]))

        # save results
        np.save("../../data/aw_hermite/bump_on_tail/sol_u_Nve1_" + str(setup.Nv_e1)
                + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", sol_midpoint_u)

        np.save("../../data/aw_hermite/bump_on_tail/sol_t_Nve1_" + str(setup.Nv_e1)
                + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.t_vec)

        # # save time varying alpha and u
        # np.save("../../data/aw_hermite/bump_on_tail/alpha_e1_Nve1_" + str(setup.Nv_e1)
        #         + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
        #         + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e1)
        #
        # np.save("../../data/aw_hermite/bump_on_tail/alpha_e2_Nve1_" + str(setup.Nv_e1)
        #         + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
        #         + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e2)
        #
        # np.save("../../data/aw_hermite/bump_on_tail/u_e1_Nve1_" + str(setup.Nv_e1)
        #         + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
        #         + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e1)
        #
        # np.save("../../data/aw_hermite/bump_on_tail/u_e2_Nve1_" + str(setup.Nv_e1)
        #         + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
        #         + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e2)
