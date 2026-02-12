"""module with mixed method #1 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np


def extra_term_1_legendre(J_int, v_b, v_a, C_hermite_last, alpha, Nv_H, D, E):
    """

    :param J_int:
    :param v_b:
    :param v_a:
    :param C_hermite_last:
    :param alpha:
    :param Nv_H:
    :param D:
    :param E:
    :return:
    """
    A = -alpha / (v_b - v_a) * np.sqrt(Nv_H / 2) * (D @ C_hermite_last + 2 / (alpha ** 2) * E * C_hermite_last)
    # # todo: enforce conservation laws explicitly
    # J_int[0] = 0
    # J_int[1] = 0
    # J_int[2] = 0
    return np.kron(J_int, A)

