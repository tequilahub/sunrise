"""Variational quantum deflation (VQD) for excited states.

References
----------
O. Higgott, D. Wang and S. Brierley, Quantum 3, 156 (2019), doi:10.22331/q-2019-07-01-156.
"""

from numbers import Number

import tequila as tq


def vqd_projection_method(
    H: tq.QubitHamiltonian,
    U: tq.QCircuit,
    circuit_list: list[tq.QCircuit],
    constant_list: list[Number],
):
    r"""Return the variational quantum deflation (VQD) objective for the circuit ``U``.

    The energy is penalized by the overlap with each previously optimized state
    :math:`|\psi_k\rangle = V_k|0\rangle`,

    .. math::
        E = \langle\psi|H|\psi\rangle + \sum_k \beta_k |\langle\psi_k|\psi\rangle|^2,
        \qquad |\psi\rangle = U|0\rangle,

    where the overlap is measured as the probability of the all-zero state after
    :math:`V_k^\dagger U`. Minimizing it gives the next excited state.

    Parameters
    ----------
    H : tq.QubitHamiltonian
        Hamiltonian.
    U : tq.QCircuit
        Circuit of the state being optimized.
    circuit_list : list of tq.QCircuit
        Circuits :math:`V_k` of the previously optimized states.
    constant_list : list of float
        Penalty weights :math:`\beta_k`, one per previous state.

    Returns
    -------
    tq.Objective
        VQD objective, to be minimized.

    Raises
    ------
    ValueError
        If ``circuit_list`` and ``constant_list`` differ in length.

    Examples
    --------
    >>> E1 = sun.excited_states.vqd_projection_method(H, U1, [U0], [1.0])
    >>> result = tq.minimize(E1)
    """
    # TODO Adapt for sunrise functions
    E = tq.ExpectationValue(H=H, U=U)
    if len(circuit_list) != len(constant_list):
        raise ValueError(
            f"Circuit_list and constant_list have different lengths. len(circuit_list): '{len(circuit_list)}', len(constant_list): '{len(constant_list)}'"
        )
    U_list = []
    for i in range(len(circuit_list)):
        U_k = U + circuit_list[i].dagger()
        P_k = 1
        for k in U_k.qubits:
            P_k *= tq.paulis.Qp(k)
        E_k = tq.ExpectationValue(H=P_k, U=U_k)
        U_list.append(constant_list[i] * E_k)
    return E + sum(U_list)
