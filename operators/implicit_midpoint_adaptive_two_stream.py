"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: Oct 20th, 2025
"""
import numpy as np
import scipy
from operators.reprojection_between_hermite_and_legendre import reprojection_aw_hermite_and_legendre, reprojection_aw_hermite_and_legendre_old
from operators.adaptive_aw_hermite import P_case_i, check_if_update_needed, updated_u, updated_alpha, \
    get_projection_matrix


def implicit_nonlinear_equation(y_new, y_old, dt, right_hand_side):
    """return the nonlinear equation for implicit midpoint to optimize.

    :param y_new: 1d array, y_{n+1}
    :param y_old: 1d array, y_{n}
    :param dt: float, time step t_{n+1} - t_{n}
    :param right_hand_side: function, a function of the rhs of the dynamical system dy/dt = rhs(y, t)
    :return: y_{n+1} - y_{n} -dt*rhs(y=(y_{n}+y_{n+1})/2, t=t_{n} + dt/2)
    """
    return y_new - y_old - dt * right_hand_side(y=0.5 * (y_old + y_new))


def implicit_midpoint_solver_adaptive_two_stream(y_0, right_hand_side, param, r_tol=1e-8, a_tol=1e-15, max_iter=100,
                                                 bump_hermite_adapt=True, bulk_hermite_adapt=True, MM1=False, MM2=False,
                                                 adaptive_u_and_alpha=True, adaptive_between_hermite_and_legendre=True):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param bulk_hermite_adapt: boolean, default is True
    :param bump_hermite_adapt: boolean, default is True
    :param adaptive_between_hermite_and_legendre: boolean, default is True
    :param param: object of SimulationSetup with all the simulation setup parameters
    :param max_iter: maximum iterations of nonlinear solver, default is 100
    :param a_tol: absolute tolerance nonlinear solver, default is 1e-15
    :param r_tol: relative tolerance nonlinear solver, default is 1e-8
    :param y_0: initial condition
    :param adaptive_u_and_alpha: boolean, default is True
    :param right_hand_side: function of the right-hand-side, i.e. dy/dt = rhs(y, t)
    :param MM1: boolean if mixed method #1, default is False
    :param MM2: boolean if mixed method #2, default is False

    Returns
    -------
    u: (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # initialize the solution matrix
    y_sol = np.zeros((len(y_0), len(param.t_vec)))
    y_sol[:, 0] = y_0

    # for-loop each time-step
    for tt in range(1, len(param.t_vec)):
        # print out the current time stamp
        print("\n time = ", param.t_vec[tt])
        if adaptive_between_hermite_and_legendre:
            if np.max(y_sol[:, tt - 1][param.Nv_e1 * param.Nx - param.Nx:param.Nv_e1 * param.Nx]) \
                    >= param.threshold_last_hermite:
                print("re-projection is happening between hermite and Legendre formulations!")
                # we update the simulation
                y_sol[:, tt - 1] = reprojection_aw_hermite_and_legendre(cutoff=param.cutoff,
                                                                        Nx=param.Nx,
                                                                        Nv_e1=param.Nv_e1,
                                                                        Nv_e2=param.Nv_e2,
                                                                        y_curr=y_sol[:, tt - 1],
                                                                        v_a=param.v_a,
                                                                        v_b=param.v_b,
                                                                        alpha=param.alpha_e1[-1],
                                                                        u=param.u_e1[-1],
                                                                        Nv_int=param.Nv_int)
                # y_sol[:, tt-1] = reprojection_aw_hermite_and_legendre_old(cutoff=param.cutoff,
                #                                                           Nx=param.Nx,
                #                                                           Nv_e1=param.Nv_e1,
                #                                                           Nv_e2=param.Nv_e2,
                #                                                           y_curr=y_sol[:, tt - 1],
                #                                                           v_a=param.v_a,
                #                                                           v_b=param.v_b,
                #                                                           J=param.J_int)
        if adaptive_u_and_alpha:
            if bulk_hermite_adapt:
                # updated u (electron 1) parameter
                u_e1_curr = updated_u(u_prev=param.u_e1[-1],
                                      alpha_prev=param.alpha_e1[-1],
                                      C00=np.mean(y_sol[:, tt - 1][:param.Nx]),
                                      C10=np.mean(y_sol[:, tt - 1][param.Nx:2 * param.Nx]))

                # updated alpha (electron 1) parameter
                alpha_e1_curr = updated_alpha(alpha_prev=param.alpha_e1[-1],
                                              C20=np.mean(y_sol[:, tt - 1][2 * param.Nx: 3 * param.Nx]),
                                              C10=np.mean(y_sol[:, tt - 1][param.Nx: 2 * param.Nx]),
                                              C00=np.mean(y_sol[:, tt - 1][:param.Nx]))

                # electron 1 check mark
                if check_if_update_needed(u_s_curr=u_e1_curr, u_s=param.u_e1[-1], u_s_tol=param.u_tol,
                                          alpha_s_curr=alpha_e1_curr, alpha_s=param.alpha_e1[-1],
                                          alpha_s_tol=param.alpha_tol):
                    print("updating u or alpha (electron 1)")
                    print("[new] ue1 = ", u_e1_curr)
                    print("[new] alpha_e1 = ", alpha_e1_curr)
                    print("[curr] ue1 = ", param.u_e1[-1])
                    print("[curr] alpha_e1 = ", param.alpha_e1[-1])
                    # get Hermite projection matrix
                    P, case = get_projection_matrix(u_s_curr=u_e1_curr, u_s=param.u_e1[-1], alpha_s_curr=alpha_e1_curr,
                                                    alpha_s=param.alpha_e1[-1], Nx_total=param.Nx, Nv=param.Nv_e1,
                                                    alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)
                    if case == 1:
                        print("(e1) tolerance met for u and alpha")
                        # update parameters
                        param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)
                        param.replace_u_e1(u_e1_curr=u_e1_curr)
                        if MM1 or MM2:
                            param.update_IJ()
                            if MM2:
                                param.update_psi_dual_va_vb()

                    elif case == 2:
                        print("(e1) tolerance met for u")
                        param.replace_u_e1(u_e1_curr=u_e1_curr)
                        if MM1 or MM2:
                            param.update_IJ()
                            if MM2:
                                param.update_psi_dual_va_vb()

                    elif case == 3:
                        print("(e1) tolerance met for alpha")
                        param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)
                        if MM1 or MM2:
                            param.update_IJ()
                            if MM2:
                                param.update_psi_dual_va_vb()

                    # project the previous timestamp results
                    y_sol[:, tt - 1][:param.Nv_e1 * param.Nx] = P @ y_sol[:, tt - 1][:param.Nv_e1 * param.Nx]

                # update parameters electron 1
                param.add_alpha_e1(alpha_e1_curr=param.alpha_e1[-1])
                param.add_u_e1(u_e1_curr=param.u_e1[-1])

            if bump_hermite_adapt:
                # update u (electron 2) parameter
                u_e2_curr = updated_u(u_prev=param.u_e2[-1],
                                      alpha_prev=param.alpha_e2[-1],
                                      C00=np.mean(
                                          y_sol[:, tt - 1][param.Nv_e1 * param.Nx: param.Nv_e1 * param.Nx + param.Nx]),
                                      C10=np.mean(y_sol[:, tt - 1][
                                                  param.Nv_e1 * param.Nx + param.Nx: param.Nv_e1 * param.Nx + 2 * param.Nx]))

                # update alpha (electron 2) parameter
                alpha_e2_curr = updated_alpha(alpha_prev=param.alpha_e2[-1],
                                              C20=np.mean(y_sol[:, tt - 1][
                                                          param.Nv_e1 * param.Nx + 2 * param.Nx: param.Nv_e1 * param.Nx + 3 * param.Nx]),
                                              C10=np.mean(y_sol[:, tt - 1][
                                                          param.Nv_e1 * param.Nx + param.Nx: param.Nv_e1 * param.Nx + 2 * param.Nx]),
                                              C00=np.mean(y_sol[:, tt - 1][
                                                          param.Nv_e1 * param.Nx: param.Nv_e1 * param.Nx + param.Nx]))

                # electron 2 check mark
                if check_if_update_needed(u_s_curr=u_e2_curr, u_s=param.u_e2[-1], u_s_tol=param.u_tol,
                                          alpha_s_curr=alpha_e2_curr, alpha_s=param.alpha_e2[-1],
                                          alpha_s_tol=param.alpha_tol):
                    print("updating u or alpha (electron 2)")
                    print("ue2 = ", u_e2_curr)
                    print("alpha_e2 = ", alpha_e2_curr)
                    # get Hermite projection matrix
                    P, case = get_projection_matrix(u_s_curr=u_e2_curr, u_s=param.u_e2[-1], alpha_s_curr=alpha_e2_curr,
                                                    alpha_s=param.alpha_e2[-1], Nx_total=param.Nx, Nv=param.Nv_e2,
                                                    alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)
                    if case == 1:
                        print("(e2) tolerance met for u and alpha")
                        # update parameters
                        param.replace_alpha_e2(alpha_e2_curr=alpha_e2_curr)
                        param.replace_u_e2(u_e2_curr=u_e2_curr)

                    elif case == 2:
                        print("(e2) tolerance met for u")
                        param.replace_u_e2(u_e2_curr=u_e2_curr)

                    elif case == 3:
                        print("(e2) tolerance met for alpha")
                        param.replace_alpha_e2(alpha_e2_curr=alpha_e2_curr)

                    # project the previous timestamp results
                    y_sol[:, tt - 1][param.Nv_e1 * param.Nx: (param.Nv_e1 + param.Nv_e2) * param.Nx] = \
                        P @ y_sol[:, tt - 1][param.Nv_e1 * param.Nx: (param.Nv_e1 + param.Nv_e2) * param.Nx]

                # update parameters electron 2
                param.add_alpha_e2(alpha_e2_curr=param.alpha_e2[-1])
                param.add_u_e2(u_e2_curr=param.u_e2[-1])



        y_sol[:, tt] = scipy.optimize.newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                                            y_old=y_sol[:, tt - 1],
                                                                                            right_hand_side=right_hand_side,
                                                                                            dt=param.dt),
                                                    xin=y_sol[:, tt - 1],
                                                    maxiter=max_iter,
                                                    method='lgmres',
                                                    f_tol=a_tol,
                                                    f_rtol=r_tol,
                                                    verbose=True)
    return y_sol, param
