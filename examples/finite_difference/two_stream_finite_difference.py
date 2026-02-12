"""Module to simulate the two stream instability with 1D1V finite difference code

Last modified: Nov 26th, 2025
Author: Opal Issan (oissan@ucsd.edu)
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('../..')))

from operators.finite_difference import ddx_central
from operators.universal_functions import get_D_inv
from operators.poisson_solver import gmres_solver
from operators.implicit_midpoint_adaptive_single_stream_finite_difference import \
    implicit_midpoint_solver_adaptive_single_stream_finite_differencing
import time
import numpy as np


def rhs(y):
    # charge density computed
    rho = -np.sum(y, axis=1) * dv
    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=Dx, D_inv=Dx_inv, a_tol=1e-12, r_tol=1e-12)
    advection = -Dx @ (V * y)
    acceleration = (Dv @ (E[:, None] * y).T).T
    return advection + acceleration


if __name__ == "__main__":
    # simulation parameters
    Nx = 101
    Nv = 60000
    L = 4 * np.pi
    v_a = -4
    v_b = 4
    epsilon = 1e-2
    k0 = 0.5
    n0 = 1
    alpha = np.sqrt(2)
    nu = 0
    T = 35
    dt = 0.01
    t_vec = np.linspace(0, T, int(T/dt) + 1, endpoint=True)

    # x-v space
    x = np.linspace(0, L, Nx, endpoint=False)
    v = np.linspace(v_a, v_b, Nv, endpoint=True)
    V = np.outer(np.ones(Nx), v)
    dx = np.abs(x[1] - x[0])
    dv = np.abs(v[1] - v[0])

    # simulation operators
    Dx = ddx_central(Nx=Nx+1, dx=dx, periodic=True, order=2)
    Dx_inv = get_D_inv(Nx=Nx, D=Dx)
    Dv = ddx_central(Nx=Nv, dx=dv, periodic=False, order=2)

    # initial condition
    x_component = (1 + epsilon * np.cos(x * k0)) / np.sqrt(np.pi)
    v_component = v**2 * np.exp(-(v ** 2) / (alpha ** 2)) / alpha

    Y0 = np.outer(x_component, v_component)

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver_adaptive_single_stream_finite_differencing(Y0=Y0,
                                                                                         right_hand_side=rhs,
                                                                                         a_tol=1e-10,
                                                                                         r_tol=1e-10,
                                                                                         max_iter=100,
                                                                                         t_vec=t_vec,
                                                                                         skip_save=200)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save("sol_runtime_Nv_" + str(Nv) + "_Nx_" + str(Nx) + "_" + str(T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("sol_u_Nv_" + str(Nv) + "_Nx_" + str(Nx) + "_" + str(T), sol_midpoint_u)
    np.save("sol_t_Nv_" + str(Nv) + "_Nx_" + str(Nx) + "_" + str(T), t_vec)
