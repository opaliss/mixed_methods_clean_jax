"""Module to run Legendre advection [d/dt f + v df/dx = 0]

Author: Opal Issan
Date: Nov 17th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.legendre.legendre_operators import xi_legendre
from operators.legendre.setup_legendre import SimulationSetupLegendre
from operators.implicit_midpoint_adaptive_single_stream import implicit_midpoint_solver_adaptive_single_stream
import time
import numpy as np
import scipy


def rhs(y):
    # evolving only electrons
    return setup.A_e @ y


if __name__ == "__main__":
    for Nv in [20, 50, 100, 200, 400]:
        # setup simulation parameters
        setup = SimulationSetupLegendre(Nx=101,
                                        Nv_e=Nv,
                                        epsilon=1,
                                        v_a=-5,
                                        v_b=5,
                                        gamma=0.5,
                                        L=2 * np.pi,
                                        dt=1e-2,
                                        T0=0,
                                        T=20,
                                        nu=0,
                                        alpha_e1=np.sqrt(2),
                                        alpha_e2=np.sqrt(2),
                                        u_e1=0,
                                        u_e2=0,
                                        n0_e1=1,
                                        n0_e2=0,
                                        k0=1)

        # initial condition: read in result from previous simulation
        y0 = np.zeros(setup.Nv_e * setup.Nx)
        # first electron 1 species (perturbed)
        x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
        v_ = np.linspace(setup.v_a, setup.v_b, 10000, endpoint=True)
        x_component = (1 + setup.epsilon * np.cos(x_)) / np.sqrt(np.pi) / (setup.v_b - setup.v_a) / setup.alpha_e1
        for nn in range(setup.Nv_e):
            xi = xi_legendre(n=nn, v=v_, v_a=setup.v_a, v_b=setup.v_b)
            v_component = scipy.integrate.trapezoid(xi * np.exp(-((v_ / setup.alpha_e1) ** 2)),
                                                    x=v_, dx=np.abs(v_[1] - v_[0]))
            y0[nn * setup.Nx: (nn + 1) * setup.Nx] = x_component * v_component

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
        np.save("../../data/legendre/advection/sol_runtime_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

        # save results
        np.save("../../data/legendre/advection/sol_u_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx) + "_"
                + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

        np.save("../../data/legendre/advection/sol_t_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
