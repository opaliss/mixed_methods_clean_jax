"""module with mixed (static) method with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""


def charge_density_two_stream_mixed_method_0(q_e, alpha_e, v_a, v_b, C0_e_hermite, C0_e_legendre):
    """charge density (right hand side of Poisson equation)

    :param q_e: float, charge of electrons
    :param alpha_e: float, Hermite scaling parameter or thermal velocity of bulk electrons
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :param C0_e_legendre: 1d array, density of electrons (beam) described with Legendre basis
    :param C0_e_hermite: 1d array, density of electrons (bulk) described with Hermite basis
    :return: change density rho(x, t=t*)
    """
    return q_e * (alpha_e * C0_e_hermite + (v_b - v_a) * C0_e_legendre) + 1
