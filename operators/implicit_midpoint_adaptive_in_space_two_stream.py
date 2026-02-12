"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: Oct 20th, 2025
"""
import numpy as np
import scipy
from operators.reprojection_between_hermite_and_legendre import reprojection_adaptive_in_space_aw_hermite_and_legendre
from operators.adaptive_aw_hermite import check_if_update_needed, updated_u, updated_alpha, \
    get_projection_matrix
from operators.implicit_midpoint_adaptive_two_stream import implicit_nonlinear_equation


def implicit_midpoint_solver_adaptive_in_space_two_stream(y_0, right_hand_side, param, r_tol=1e-8,
                                                          a_tol=1e-15, max_iter=100,
                                                          bump_hermite_adapt=True,
                                                          bulk_hermite_adapt=True,
                                                          MM1=False,
                                                          MM2=False, MM0=False,
                                                          adaptive_u_and_alpha=True,
                                                          adaptive_between_hermite_and_legendre=True):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param bulk_hermite_adapt: boolean, default is True
    :param bump_hermite_adapt: boolean, default is True
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
        if adaptive_u_and_alpha:
            if bulk_hermite_adapt:
                # updated u (electron 1) parameter
                u_e1_curr = updated_u(u_prev=param.u_e1[-1],
                                      alpha_prev=param.alpha_e1[-1],
                                      C00=y_sol[:, tt - 1][:param.Nx],
                                      C10=y_sol[:, tt - 1][param.Nx:2 * param.Nx],
                                      sigma=param.Nx//10,
                                      method="max_f",
                                      C=y_sol[:, tt - 1][:param.Nx*param.Nv_e1],
                                      v_a=param.v_a, v_b=param.v_b, Nv_int=param.Nv_int)

                # updated alpha (electron 1) parameter
                alpha_e1_curr = updated_alpha(alpha_prev=param.alpha_e1[-1],
                                              C20=y_sol[:, tt - 1][2 * param.Nx: 3 * param.Nx],
                                              C10=y_sol[:, tt - 1][param.Nx: 2 * param.Nx],
                                              C00=y_sol[:, tt - 1][:param.Nx], sigma=param.Nx//10)

                # electron 1 check mark
                if check_if_update_needed(u_s_curr=u_e1_curr,
                                          u_s=param.u_e1[-1],
                                          u_s_tol=param.u_tol,
                                          alpha_s_curr=alpha_e1_curr,
                                          alpha_s=param.alpha_e1[-1],
                                          alpha_s_tol=param.alpha_tol):

                    print("updating u or alpha (electron 1)")
                    # get Hermite projection matrix
                    P, case, u_e1_curr, alpha_e1_curr = get_projection_matrix(u_s_curr=u_e1_curr,
                                                                              u_s=param.u_e1[-1],
                                                                              alpha_s_curr=alpha_e1_curr,
                                                                              alpha_s=param.alpha_e1[-1],
                                                                              Nx_total=param.Nx,
                                                                              Nv=param.Nv_e1,
                                                                              alpha_s_tol=param.alpha_tol,
                                                                              u_s_tol=param.u_tol, koshkarov=False)
                    y_next = P @ y_sol[:, tt - 1][:param.Nv_e1 * param.Nx]
                    sum_next = np.sum(np.abs(y_next[param.Nv_e1 * param.Nx - (param.Nv_e1 // 3) * param.Nx:
                                                    param.Nv_e1 * param.Nx]))
                    sum_prev = np.sum(np.abs(y_sol[:, tt - 1][param.Nv_e1 * param.Nx - (param.Nv_e1 // 3) * param.Nx:
                                                              param.Nv_e1 * param.Nx]))
                    print("sum prev = ", sum_prev)
                    print("sum next = ", sum_next)

                    if sum_next < sum_prev:
                        if case == 1:
                            print("(e1) case #1 tolerance met for u and alpha")
                            # update parameters
                            param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)
                            param.replace_u_e1(u_e1_curr=u_e1_curr)
                            if MM1 or MM2:
                                param.update_IJ()
                                if MM2:
                                    param.update_psi_dual_va_vb()

                        elif case == 2:
                            print("(e1) case #2 tolerance met for u")
                            param.replace_u_e1(u_e1_curr=u_e1_curr)
                            if MM1 or MM2:
                                param.update_IJ()
                                if MM2:
                                    param.update_psi_dual_va_vb()

                        elif case == 3:
                            print("(e1) case #3 tolerance met for alpha")
                            param.replace_alpha_e1(alpha_e1_curr=alpha_e1_curr)
                            if MM1 or MM2:
                                param.update_IJ()
                                if MM2:
                                    param.update_psi_dual_va_vb()

                        # project the previous timestamp results
                        y_sol[:, tt - 1][:param.Nv_e1 * param.Nx] = y_next
                    else:
                        print("projection is actually not favorable for the stability of Hermite! ")

                # update parameters electron 1
                param.add_alpha_e1(alpha_e1_curr=param.alpha_e1[-1])
                param.add_u_e1(u_e1_curr=param.u_e1[-1])

            if bump_hermite_adapt:
                # update u (electron 2) parameter
                u_e2_curr = updated_u(u_prev=param.u_e2[-1],
                                      alpha_prev=param.alpha_e2[-1],
                                      C00=y_sol[:, tt - 1][param.Nv_e1 * param.Nx: param.Nv_e1 * param.Nx + param.Nx],
                                      C10=y_sol[:, tt - 1][param.Nv_e1 * param.Nx + param.Nx: param.Nv_e1 * param.Nx + 2 * param.Nx],
                                      sigma=param.Nx//10)

                # update alpha (electron 2) parameter
                alpha_e2_curr = updated_alpha(alpha_prev=param.alpha_e2[-1],
                                              C20=y_sol[:, tt - 1][
                                                  param.Nv_e1 * param.Nx + 2 * param.Nx: param.Nv_e1 * param.Nx + 3 * param.Nx],
                                              C10=y_sol[:, tt - 1][
                                                  param.Nv_e1 * param.Nx + param.Nx: param.Nv_e1 * param.Nx + 2 * param.Nx],
                                              C00=y_sol[:, tt - 1][
                                                  param.Nv_e1 * param.Nx: param.Nv_e1 * param.Nx + param.Nx],
                                              sigma=param.Nx//10)

                # electron 2 check mark
                if check_if_update_needed(u_s_curr=u_e2_curr, u_s=param.u_e2[-1], u_s_tol=param.u_tol,
                                          alpha_s_curr=alpha_e2_curr, alpha_s=param.alpha_e2[-1],
                                          alpha_s_tol=param.alpha_tol):
                    print("updating u or alpha (electron 2)")
                    # print("ue2 = ", u_e2_curr)
                    # print("alpha_e2 = ", alpha_e2_curr)
                    # get Hermite projection matrix
                    P, case, u_e2_curr, alpha_e2_curr = get_projection_matrix(u_s_curr=u_e2_curr, u_s=param.u_e2[-1],
                                                                              alpha_s_curr=alpha_e2_curr,
                                                                              alpha_s=param.alpha_e2[-1],
                                                                              Nx_total=param.Nx, Nv=param.Nv_e2,
                                                                              alpha_s_tol=param.alpha_tol,
                                                                              u_s_tol=param.u_tol, koshkarov=False)
                    y_next = P @ y_sol[:, tt - 1][param.Nv_e1 * param.Nx:]
                    sum_next = np.sum(np.abs(y_next[param.Nv_e2 * param.Nx - (param.Nv_e2 // 3) * param.Nx: -1]))
                    sum_prev = np.sum(np.abs(y_sol[:, tt - 1][(param.Nv_e1+param.Nv_e2) * param.Nx - (param.Nv_e2 // 3) * param.Nx:-1]))
                    print("sum prev = ", sum_prev)
                    print("sum next = ", sum_next)

                    if sum_next < sum_prev:
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
                        y_sol[:, tt - 1][param.Nv_e1 * param.Nx: (param.Nv_e1 + param.Nv_e2) * param.Nx] = y_next
                    else:
                        print("projection is actually not favorable for the stability of Hermite! ")

                # update parameters electron 2
                param.add_alpha_e2(alpha_e2_curr=param.alpha_e2[-1])
                param.add_u_e2(u_e2_curr=param.u_e2[-1])

        if adaptive_between_hermite_and_legendre:
            if np.max(y_sol[:, tt - 1][param.Nv_e1 * param.Nx - param.Nx:param.Nv_e1 * param.Nx]) \
                    >= param.threshold_last_hermite:
                print("re-projection is happening between hermite and Legendre formulations!")
                if MM0:
                    param.update_J()
                # we update the simulation
                y_sol[:, tt - 1] = reprojection_adaptive_in_space_aw_hermite_and_legendre(cutoff=param.cutoff,
                                                                                          Nx=param.Nx,
                                                                                          Nv_e1=param.Nv_e1,
                                                                                          Nv_e2=param.Nv_e2,
                                                                                          y_curr=y_sol[:, tt - 1],
                                                                                          v_a=param.v_a,
                                                                                          v_b=param.v_b,
                                                                                          J=param.J_int)

        y_sol[:, tt] = scipy.optimize.newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                                            y_old=y_sol[:, tt - 1],
                                                                                            right_hand_side=right_hand_side,
                                                                                            dt=param.dt),
                                                    xin=y_sol[:, tt - 1],
                                                    maxiter=max_iter,
                                                    method='gmres',
                                                    f_tol=a_tol,
                                                    f_rtol=r_tol,
                                                    verbose=True)
    return y_sol, param
