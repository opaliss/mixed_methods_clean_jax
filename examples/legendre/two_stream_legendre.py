"""Module to run Legendre bump on tail testcase

Author: Opal Issan
Date: Dec 4th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.legendre.legendre_operators import nonlinear_legendre, charge_density_legendre, xi_legendre
from operators.legendre.setup_legendre import SimulationSetupLegendre
from operators.implicit_midpoint_adaptive_single_stream import implicit_midpoint_solver_adaptive_single_stream
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import scipy


def rhs(y):
    # charge density computed
    rho = charge_density_legendre(q_e=setup.q_e,
                                  q_i=setup.q_i,
                                  C0_e=y[:setup.Nx],
                                  C0_i=np.ones(setup.Nx) / (setup.v_b - setup.v_a),
                                  v_a=setup.v_a,
                                  v_b=setup.v_b)

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # evolving only electrons
    return setup.A_e @ y + nonlinear_legendre(E=E, psi=y, Nv=setup.Nv_e, Nx=setup.Nx, B_mat=setup.B_e,
                                              q=setup.q_e, m=setup.m_e, gamma=setup.gamma, v_a=setup.v_a, v_b=setup.v_b,
                                              xi_v_a=setup.xi_v_a, xi_v_b=setup.xi_v_b)


if __name__ == "__main__":
    setup = SimulationSetupLegendre(Nx=101,
                                    Nv_e=1024,
                                    epsilon=1e-2,
                                    v_a=-4,
                                    v_b=4,
                                    alpha_e1=np.sqrt(2),
                                    alpha_e2=np.sqrt(2),
                                    u_e1=0,
                                    u_e2=0,
                                    n0_e1=1,
                                    n0_e2=0,
                                    k0=0.5,
                                    L=4 * np.pi,
                                    dt=1e-2,
                                    Nv_int=5000,
                                    T0=0,
                                    T=35,
                                    nu=1,
                                    gamma=0.5)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.Nv_e * setup.Nx)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    v_ = np.linspace(setup.v_a, setup.v_b, setup.Nv_int, endpoint=True)
    x_component = (1 + setup.epsilon * np.cos(x_ * setup.k0)) / (setup.v_b - setup.v_a) / np.sqrt(np.pi)
    v_component1 = (v_ ** 2) * np.exp(-((v_ - setup.u_e1) ** 2) / (setup.alpha_e1 ** 2)) / setup.alpha_e1

    for nn in range(setup.Nv_e):
        xi = xi_legendre(n=nn, v=v_, v_a=setup.v_a, v_b=setup.v_b)
        v_component = scipy.integrate.trapezoid(xi * v_component1, x=v_, dx=np.abs(v_[1] - v_[0]))
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
    np.save("../../data/legendre/two_stream/sol_runtime_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/legendre/two_stream/sol_u_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx) + "_"
            + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/legendre/two_stream/sol_t_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)
