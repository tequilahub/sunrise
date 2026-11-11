
from numbers import Number

import tequila as tq


def vqd_projection_method(
    H: tq.QubitHamiltonian,
    U: tq.QCircuit,
    circuit_list: list[tq.QCircuit],
    constant_list: list[Number],
):
    """Return the Variational Quantum Deflation (VQD) objective with overlap penalties for the circuit ``U``.
    Parameters
    ----------
    H : QubitHamiltonian
        Hamiltonian.
    U : QCircuit
        Current circuit.
    circuit_list : list
        Circuits of the previously optimized states.
    constant_list : list of float
        Penalty weight of each previous state.
    Returns
    -------
    tq.Objective
    Raises
    ------
    ValueError
        If ``circuit_list`` and ``constant_list`` differ in length.
    """

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