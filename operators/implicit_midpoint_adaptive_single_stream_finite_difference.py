"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: Oct 20th, 2025
"""
import numpy as np
import scipy
from operators.adaptive_aw_hermite import check_if_update_needed, updated_u, updated_alpha, get_projection_matrix
from operators.implicit_midpoint_adaptive_two_stream import implicit_nonlinear_equation


def implicit_midpoint_solver_adaptive_single_stream_finite_differencing(Y0, right_hand_side, t_vec,
                                                                        r_tol=1e-8, a_tol=1e-15, max_iter=100,
                                                                        skip_save=200):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param param: object of SimulationSetup with all the simulation setup parameters
    :param max_iter: maximum iterations of nonlinear solver, default is 100
    :param a_tol: absolute tolerance nonlinear solver, default is 1e-15
    :param r_tol: relative tolerance nonlinear solver, default is 1e-8
    :param y_0: initial condition
    :param adaptive: boolean
    :param right_hand_side: function of the right-hand-side, i.e. dy/dt = rhs(y, t)

    Returns
    -------
    u: (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # initialize the solution matrix
    Nx, Nv = np.shape(Y0)
    dt = np.abs(t_vec[1] - t_vec[0])

    Y = np.zeros((Nx, Nv, int(len(t_vec) / skip_save) + 1))
    Y[:, :, 0] = Y0
    Y_curr = Y0

    # for-loop each time-step
    for tt in range(1, len(t_vec)):
        # print out the current time stamp
        print("\n time = ", t_vec[tt])

        Y_curr = scipy.optimize.newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                                      y_old=Y_curr,
                                                                                      right_hand_side=right_hand_side,
                                                                                      dt=dt),
                                              xin=Y_curr,
                                              maxiter=max_iter,
                                              method='gmres',
                                              f_tol=a_tol,
                                              f_rtol=r_tol,
                                              verbose=True)
        if tt % skip_save == 0:
            Y[:, :, int(tt/skip_save)] = Y_curr
            np.save("Y3.npy", Y)
    return Y
