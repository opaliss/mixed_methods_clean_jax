"""Module to run Hermite linear advection testcase

Author: Opal Issan
Date: Nov 17th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import jax
from operators.aw_hermite.setup_aw_hermite import SimulationSetupHermite
from operators.implicit_midpoint_adaptive_single_stream import implicit_midpoint_solver_adaptive_single_stream
import time
#import numpy as np
import jax.numpy as jnp

def rhs(y):
    A_e = setup.alpha_e * setup.A_off + setup.u_e * setup.A_diag + setup.nu * setup.A_col
    # evolving only electrons
    return A_e @ y


if __name__ == "__main__":
    for Nv in [20]:
        setup = SimulationSetupHermite(Nx=101,
                                       Nv=Nv,
                                       epsilon=1,
                                       alpha_e=jnp.sqrt(2),
                                       alpha_i=jnp.sqrt(2 / 1836),
                                       u_e=0,
                                       u_i=0,
                                       L=2 * jnp.pi,
                                       dt=1e-2,
                                       T0=0,
                                       T=20,
                                       nu=0,
                                       u_tol=None,
                                       alpha_tol=None)

        # initial condition: read in result from previous simulation
        # first electron 1 species (perturbed)
        x_ = jnp.linspace(0, setup.L, setup.Nx, endpoint=False)
        y0 = jnp.concatenate([(1 + setup.epsilon * jnp.cos(x_)) / setup.alpha_e,
                              jnp.zeros((setup.Nv - 1) * setup.Nx)])

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
        jnp.save("../../data/aw_hermite/advection/sol_runtime_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), jnp.array([end_time_cpu, end_time_wall]))

        # save results
        jnp.save("../../data/aw_hermite/advection/sol_u_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx) + "_"
                + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

        jnp.save("../../data/aw_hermite/advection/sol_t_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

        jnp.save("../../data/aw_hermite/advection/alpha_e_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), setup.alpha_e)

        jnp.save("../../data/aw_hermite/advection/u_e_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
                + "_" + str(setup.T0) + "_" + str(setup.T), setup.u_e)