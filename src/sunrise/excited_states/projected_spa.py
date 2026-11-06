from numbers import Number

import tequila as tq


def expectation_value_orthogonality_constraint(
    H: tq.QubitHamiltonian,
    U: tq.QCircuit,
    circuit_list: list[tq.QCircuit],
    constant_list: list[Number],
):
    E = tq.ExpectationValue(H=H, U=U)
    if len(circuit_list) != len(constant_list):
        raise ValueError(
            f"Circuit_list and constant_list have different lengths. len(circuit_list): '{len(circuit_list)}', len(constant_list): '{len(constant_list)}'"
        )
    list_length = len(circuit_list)
    for l in range(list_length):
        if circuit_list[l].extract_variables() is None:
            raise ValueError("Circuit_list contains unparametrized elements")
    U_list = []
    for i in range(list_length):
        U_k = U + circuit_list[i].dagger()
        P_k = 1
        for k in U_k.qubits:
            P_k *= tq.paulis.Qp(k)
        E_k = tq.ExpectationValue(H=P_k, U=U_k)
        U_list.append(constant_list[i] * E_k)
    return E + sum(U_list)
