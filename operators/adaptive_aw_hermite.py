"""update the AW Hermite projection based on alpha and u

Author: Opal Issan (oissan@ucsd.edu)
Date: Oct 24th, 2025
"""
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from operators.aw_hermite.aw_hermite_operators import aw_psi_hermite_vector


def updated_alpha(alpha_prev, C20, C10, C00, sigma=np.nan):
    """

    :param C00: float, the average zeroth moment
    :param C10: float, the average first moment
    :param C20: float, the average second moment
    :param alpha_prev: float, previous iterative alpha^{s}_{j-1} parameter
    :return: alpha at the updated iteration alpha^{s}_{j}
    """
    solution = alpha_prev * np.sqrt(1 + np.sqrt(2) * C20 / C00 - (C10 / C00) ** 2)
    if isinstance(solution, float) is False:
        if sigma != 0 and sigma != np.nan:
            solution = gaussian_filter1d(solution, sigma=sigma, mode="wrap")
    return solution


def updated_u(u_prev, alpha_prev, C00, C10, sigma=np.nan, method="physics", v_a=-6, v_b=6, Nv_int=5000, C=np.inf):
    """

    :param u_prev: float, previous iterative u^{s}_{j-1} parameter
    :param alpha_prev: float, previous iterative alpha^{s}_{j-1} parameter
    :param C00: float, the average zeroth moment
    :param C10: float, the average first moment
    :return: u at the updated iteration u^{s}_{j}
    """
    if method == "physics":
        u_proposed = u_prev + alpha_prev * C10 / C00 / np.sqrt(2)
        if isinstance(u_proposed, float) is False:
            if sigma != 0 and sigma != np.nan:
                u_proposed = gaussian_filter1d(u_proposed, sigma=sigma, mode="wrap")
        return u_proposed
    elif method == "max_f":
        v = np.linspace(v_a, v_b, Nv_int)
        Nx = int(len(C00))
        Nv = int(len(C)/Nx)
        sol = np.zeros((Nx, Nv_int))
        for jj in range(Nx):
            sol[jj, :] = C[jj::Nx] @ aw_psi_hermite_vector(n=Nv-1, alpha_s=alpha_prev[jj], u_s=u_prev[jj], v=v)
        u_proposed = np.max(sol, axis=1)
        if isinstance(u_proposed, float) is False:
            if sigma != 0 and sigma != np.nan:
                u_proposed = gaussian_filter1d(u_proposed, sigma=sigma, mode="wrap")
            return u_proposed


def a_constant(alpha_curr, alpha_prev):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j -1}
    :return: alpha^{s}_{j}/ alpha^{s}_{j-1}
    """
    return alpha_curr / alpha_prev


def b_constant(u_curr, u_prev, alpha_prev):
    """

    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j-1}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :return: (u^{s}_{j} - u^{s}_{j-1}) / alpha^{s}_{j-1}
    """
    return (u_curr - u_prev) / alpha_prev


def P_case_i(alpha_curr, alpha_prev, u_curr, u_prev, Nv):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j}
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # initialize P matrix
    P = np.zeros((Nv, Nv))

    # obtain a and b constants
    a = a_constant(alpha_curr=alpha_curr, alpha_prev=alpha_prev)
    a_square = a ** 2
    b = b_constant(u_curr=u_curr, u_prev=u_prev, alpha_prev=alpha_prev)

    """ Fill in the diagonal
    Inm = zeros(N); Jnm = zeros(N);
    Inm(1,1) = 1/a; Jnm(2,2) = 1/asq;
    M(1,1) = 1/a;
    """
    Inm, Jnm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))
    Inm[0, 0] = 1 / a
    Jnm[1, 1] = 1 / a_square
    P = np.diag(1. / (a ** np.arange(1, Nv + 1)))

    """
    for j=1:N-1
        M(j+1,j+1) = 1/a*M(j,j);
        n = j-1;
        Inm(n+2,1) = sqrt((n+1)/2)/(n+1)*(-2*b/a)*Inm(n+1,1);
        Jnm(n+3,2) = sqrt((n+2)/2)/(n+1)*(-2*b/a)*Jnm(n+2,2);
    end
    Jnm = Jnm(1:N,:);
    """

    for jj in range(1, Nv):
        n = jj - 1
        Inm[n + 1, 0] = np.sqrt((n + 1) / 2) / (n + 1) * (-2 * b / a) * Inm[n, 0]
        if jj < Nv - 1:
            Jnm[n + 2, 1] = np.sqrt((n + 2) / 2) / (n + 1) * (-2 * b / a) * Jnm[n + 1, 1]

    """ Fill in the 1st column
    for n=2:N-1
        if mod(n,2) == 0; ev = n; else; ev = n-1; end
        for k=0:2:ev
            if k<N-2
                Inm(n+1,k+3) = (n-k-1)*(n-k)/(k/2+1)*(-2*b/a)^(-2)*(1/asq-1)*Inm(n+1,k+1);
            end
        end
    end
    M(:,1) = sum(Inm,2);
    """
    for n in range(2, Nv):
        if n % 2 == 0:
            ev = n
        else:
            ev = n - 1
        for k in np.arange(0, ev + 1, 2):
            if k < Nv - 2:
                Inm[n, k + 2] = (n - k - 1) * (n - k) / (k / 2 + 1) * (-2 * b / a) ** (-2) * (1 / a_square - 1) * Inm[
                    n, k]

    P[:, 0] = np.sum(Inm, axis=1)

    """ Fill in the odd columns
    for m=0:2:N-3
        Inmp = zeros(N);
        for k=m+2:2:N-1
            Inmp(m+2+2:end,k+1) = 2/asq/sqrt((m+2)*(m+1))*(k-m)/2*(1/asq-1)^(-1)*Inm(m+2+2:end,k+1);
        end
        Inm = Inmp;
        II =  sum(Inm,2);
        M(m+2+2:end,m+2+1) = II(m+2+2:end);
    end"""
    for m in range(0, Nv - 2, 2):
        Inmp = np.zeros((Nv, Nv))
        for k in np.arange(m + 2, Nv, 2):
            Inmp[m + 3:, k] = 2 / a_square / np.sqrt((m + 2) * (m + 1)) * (k - m) / 2 * (1 / a_square - 1) ** (
                -1) * Inm[m + 3:, k]
        Inm = Inmp
        II = np.sum(Inm, axis=1)
        P[m + 3:, m + 2] = II[m + 3:]

    """ Fill in the 2nd column
    for n=3:N-1
        if mod(n,2) == 0; ev = n-1; else; ev = n; end
        for k=1:2:ev
            if k<N-2
                Jnm(n+1,k+3) = (n-k-1)*(n-k)/((k-1)/2+1)*(-2*b/a)^(-2)*(1/asq-1)*Jnm(n+1,k+1);
            end
        end
    end
    M(:,2) = sum(Jnm,2);
    """
    for n in range(3, Nv):
        if n % 2 == 0:
            ev = n - 1
        else:
            ev = n
        for k in np.arange(1, ev + 1, 2):
            if k < Nv - 2:
                Jnm[n, k + 2] = (n - k - 1) * (n - k) / ((k - 1) / 2 + 1) * (-2 * b / a) ** (-2) * (
                        1 / a_square - 1) * Jnm[n, k]

    P[:, 1] = np.sum(Jnm, axis=1)
    """ Fill in the even columns
    for m=1:2:N-3
        Jnmp = zeros(N);
        for k=m+2:2:N-1
            Jnmp(m+2+2:end,k+1) = 2/asq/sqrt((m+2)*(m+1))*(k-m)/2*(1/asq-1)^(-1)*Jnm(m+2+2:end,k+1);
        end
        Jnm = Jnmp;
        JJ =  sum(Jnm,2);
        M(m+2+2:end,m+2+1) = JJ(m+2+2:end);
    end
    """
    for m in np.arange(1, Nv - 2, 2):
        Jnmp = np.zeros((Nv, Nv))
        for k in np.arange(m + 2, Nv, 2):
            Jnmp[m + 3:, k] = 2 / a_square / np.sqrt((m + 2) * (m + 1)) * (k - m) / 2 * (1 / a_square - 1) ** (-1) \
                              * Jnm[m + 3:, k]
        Jnm = Jnmp
        JJ = np.sum(Jnm, axis=1)
        P[m + 3:, m + 2] = JJ[m + 3:]
    return np.tril(P)


def P_case_ii(alpha_prev, u_curr, u_prev, Nv):
    """

    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param u_curr: float, updated u^{s}_{j}
    :param u_prev: float, previous u^{s}_{j}
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # obtain b constant
    b = b_constant(u_curr=u_curr, u_prev=u_prev, alpha_prev=alpha_prev)

    """
    M = eye(N);
    for n=0:N-2 % fill in the 1st column
        M(n+2,1) = -sqrt(2)*b*sqrt(n+1)/(n+1)*M(n+1,1);
    end
    for m=2:N-1
        M(m+1:N,m) = -1/sqrt(2)/b/sqrt(m-1).*((m:N-1)-m+2).'.*M((m+1:N),m-1);
    end
    M = tril(M);
    """
    # initialize P
    P = np.eye(Nv)

    # loop over lower diagonal constants
    for n in range(0, Nv - 1):
        # fill in the 1st column
        P[n + 1, 0] = - np.sqrt(2) * b * np.sqrt(n + 1) / (n + 1) * P[n, 0]

    for m in range(2, Nv):
        P[m:, m - 1] = -1 / np.sqrt(2) / b / np.sqrt(m - 1) * (np.arange(m, Nv) - m + 2) * P[m:, m - 2]

    return np.tril(P)


def P_case_iii(alpha_curr, alpha_prev, Nv):
    """

    :param alpha_curr: float, updated alpha^{s}_{j}
    :param alpha_prev: float, previous alpha^{s}_{j-1}
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    # obtain a constant
    a = a_constant(alpha_curr=alpha_curr, alpha_prev=alpha_prev)
    a_squared = a ** 2
    """
    M = diag(1./a.^(1:N));
    for n=0:2:N-3 % fill in the first column
        M(n+3,1) = sqrt((n+1)/(n+2))*(1/asq-1)*M(n+1,1);
    end
    for m=1:N-3
        for n=3:N-1
            M(n+1,m+1) = m^(-1/2)*n^(1/2)/a*M(n,m);
        end
    end
    end
    """
    P = np.diag(1. / (a ** np.arange(1, Nv + 1)))
    for n in np.arange(0, Nv - 2, 2):
        # fill in the first column
        P[n + 2, 0] = np.sqrt((n + 1) / (n + 2)) * (1 / a_squared - 1) * P[n, 0]

    for m in range(1, Nv - 2):
        for n in range(3, Nv):
            P[n, m] = np.sqrt(n) / np.sqrt(m) / a * P[n - 1, m - 1]
    return np.tril(P)


def check_if_update_needed(u_s_curr, u_s, u_s_tol, alpha_s_curr, alpha_s, alpha_s_tol):
    """check if advection tolerance conditions are met

    :param u_s_curr: float, u^{s}_{j+1}
    :param u_s: float, u^{s}_{j}
    :param u_s_tol: float, u^{s}_{tol}
    :param alpha_s_curr: float, alpha^{s}_{j+1}
    :param alpha_s: float, alpha^{s}_{j}
    :param alpha_s_tol: float, alpha^{s}_{tol}
    :return: boolean (True/False)
    """
    if np.isscalar(u_s):
        if np.abs(u_s - u_s_curr) / alpha_s >= u_s_tol:
            return True
        elif np.abs((alpha_s - alpha_s_curr) / alpha_s) >= alpha_s_tol:
            return True
        else:
            return False
    else:
        u_diff_array = (u_s - u_s_curr)
        alpha_diff_array = (alpha_s - alpha_s_curr) / alpha_s
        if any(u_diff > u_s_tol for u_diff in u_diff_array):
            return True
        elif any(alpha_diff > alpha_s_tol for alpha_diff in alpha_diff_array):
            return True
        else:
            return False


def get_projection_matrix(u_s_curr, u_s, alpha_s_curr, alpha_s, Nx_total, Nv, alpha_s_tol, u_s_tol, koshkarov=True):
    """

    :param alpha_s_tol:
    :param u_s_tol:
    :param u_s_curr: float,  u^{s}_{j+1}
    :param u_s: float, U^{s}_{j}
    :param alpha_s_curr: float, alpha^{s}_{j+1}
    :param alpha_s: float, alpha^{s}_{j}
    :param Nx_total: int, total number of Fourier coefficients (2Nx+1 or Nx+1)
    :param Nv: int, total number of Hermite coefficients
    :return: projection matrix P (Nx_total Nv x Nx_total Nv)
    """
    if np.isscalar(u_s):
        # case (i)
        if np.abs(u_s_curr - u_s) >= u_s_tol and np.abs((alpha_s_curr - alpha_s) / alpha_s) >= alpha_s_tol:
            return scipy.sparse.kron(
                P_case_i(alpha_curr=alpha_s_curr, alpha_prev=alpha_s, u_curr=u_s_curr, u_prev=u_s, Nv=Nv),
                np.eye(N=Nx_total), format="bsr"), 1

        # case (ii)
        elif np.abs(u_s_curr - u_s) > u_s_tol and np.abs((alpha_s_curr - alpha_s) / alpha_s) <= alpha_s_tol:
            return scipy.sparse.kron(P_case_ii(alpha_prev=alpha_s, u_curr=u_s_curr, u_prev=u_s, Nv=Nv),
                                     np.eye(N=Nx_total), format="bsr"), 2

        # case (iii)
        elif np.abs(u_s_curr - u_s) < u_s_tol and np.abs((alpha_s_curr - alpha_s) / alpha_s) > alpha_s_tol:
            return scipy.sparse.kron(P_case_iii(alpha_curr=alpha_s_curr, alpha_prev=alpha_s, Nv=Nv), np.eye(N=Nx_total),
                                     format="bsr"), 3

        # no tolerance is met
        else:
            return np.eye(Nv * Nx_total), 0
    else:
        u_diff_array = (u_s - u_s_curr)
        alpha_diff_array = (alpha_s - alpha_s_curr) / alpha_s
        if any(u_diff > u_s_tol for u_diff in u_diff_array) and any(
                alpha_diff > alpha_s_tol for alpha_diff in alpha_diff_array):
            flag = 1
        elif any(u_diff > u_s_tol for u_diff in u_diff_array):
            flag = 2
        elif any(alpha_diff > alpha_s_tol for alpha_diff in alpha_diff_array):
            flag = 3
        else:
            flag = 0
        P_total = np.zeros((Nv * Nx_total, Nv * Nx_total))
        for ii in range(Nx_total):
            if koshkarov:
                if flag == 1:
                    a = a_constant(alpha_curr=alpha_s_curr[ii], alpha_prev=alpha_s[ii])
                    b = b_constant(alpha_prev=alpha_s[ii], u_prev=u_s[ii], u_curr=u_s_curr[ii])
                    P_total[ii::Nx_total, ii::Nx_total] = proj_1d_Koshkarov(a=a, b=b, Nn=Nv)
                elif flag == 2:
                    a = a_constant(alpha_curr=alpha_s[ii], alpha_prev=alpha_s[ii])
                    b = b_constant(alpha_prev=alpha_s[ii], u_prev=u_s[ii], u_curr=u_s_curr[ii])
                elif flag == 3:
                    a = a_constant(alpha_curr=alpha_s_curr[ii], alpha_prev=alpha_s[ii])
                    b = b_constant(alpha_prev=alpha_s[ii], u_prev=u_s[ii], u_curr=u_s[ii])
                P_total[ii::Nx_total, ii::Nx_total] = proj_1d_Koshkarov(a=a, b=b, Nn=Nv)
                if abs(b) < 1e-16 and abs(a - 1.0) < 1e-16:
                    print("a = ", a)
                    print("b = ", b)
                    u_s_curr[ii] = u_s_curr[ii]
                    alpha_s_curr[ii] = alpha_s[ii]

            else:
                # case (i)
                if flag == 1:
                    P_total[ii::Nx_total, ii::Nx_total] = P_case_i(alpha_curr=alpha_s_curr[ii],
                                                                   alpha_prev=alpha_s[ii], u_curr=u_s_curr[ii],
                                                                   u_prev=u_s[ii], Nv=Nv)
                # case (ii)
                elif flag == 2:

                    P_total[ii::Nx_total, ii::Nx_total] = P_case_ii(alpha_prev=alpha_s[ii], u_curr=u_s_curr[ii],
                                                                    u_prev=u_s[ii], Nv=Nv)

                # case (iii)
                elif flag == 3:
                    P_total[ii::Nx_total, ii::Nx_total] = P_case_iii(alpha_curr=alpha_s_curr[ii],
                                                                     alpha_prev=alpha_s[ii],
                                                                     Nv=Nv)

                # no tolerance is met == >
                # but this should never happen because this function is only called when a tol is met.
                elif flag == 0:
                    print("no tolerance is met for ii = ", ii)
                    u_s_curr[ii] = u_s[ii]
                    alpha_s_curr[ii] = alpha_s[ii]
        return P_total, flag, u_s_curr, alpha_s_curr


def proj_1d_Koshkarov(a, b, Nn):
    Pnn = np.zeros((Nn, Nn))
    for n in range(Nn):
        for m in range(Nn):
            if n == m:
                Pnn[n, m] = 1 / a ** (n + 1)

            elif n > m:  # lower diagonal

                if (abs(b) < 1e-16 and abs(a - 1.0) < 1e-16):
                    return np.eye(Nn)  # return 1 on diagonal?
                # elif (abs(b) < 1e-16):
                elif abs(b) <1e-16:

                    el = 1 / a
                    if ((n - m) % 2 != 0):
                        continue
                    for i in range(2, n - m + 1, 2):
                        el *= np.sqrt((i - 1.0) / (1.0 * i)) * (1.0 / (a * a) - 1.0)
                    for j in range(1, m + 1):
                        el *= np.sqrt((1.0 * (j + (n - m))) / (1.0 * j)) / a
                    Pnn[n, m] = el
                # elif (abs(a - 1.0) < 1e-16):
                elif abs(a - 1.0) < 1e-16:
                    el = 1.0

                    for i in range(1, n + 1):
                        el *= -np.sqrt(2.0 / i) * b

                    for j in range(1, m + 1):
                        el *= -1.0 / (b * np.sqrt(2 * j)) * (n + 1 - j)
                    Pnn[n, m] = el
                else:
                    alpha = -2.0 * b / a
                    beta = 1.0 / (a * a) - 1.0
                    chi = beta / (alpha * alpha)

                    temp = 1.0
                    for k in range(1, n - m + 1):
                        temp *= alpha / k

                    sum = temp
                    for k in range(1, int((1e-16 + 1.0 * (n - m)) / 2.0) + 1):
                        # for k in range(1, (n-m)/2+1):
                        temp *= (n - m - 2 * k + 2) * (n - m - 2 * k + 1) * chi / k
                        sum += temp

                    sum *= 1.0 / a ** (m + 1)
                    for k in range(m + 1, n + 1):
                        sum *= np.sqrt(k / 2.0)

                    Pnn[n, m] = sum
    return Pnn
