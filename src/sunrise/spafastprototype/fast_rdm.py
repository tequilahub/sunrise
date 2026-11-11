"""Computation of the 1- and 2-RDM of separable circuits from small local simulations."""

import numpy
import tequila as tq

from sunrise.molecules.qubit_base import Molecule as QubitMolecule
from sunrise.molecules.qubit_base import NBodyTensor

from ._compute_rdms import _assemble_rdm1, _assemble_rdm2, compute_rdms
from .decompose import make_decomposed_circuits


def fast_rdm(U, mol, clusters, variables, test=False, backend="qulacs"):
    r"""Compute the spin-free 1- and 2-RDM of a hard-core boson circuit by exploiting its separability.

    Each RDM element is the expectation value of an operator O, decomposed into Pauli strings,
    ``<O>_U = sum_k c_k <P_k>_U``. If the circuit is a tensor product over qubit clusters,
    ``U = U_0 * U_1 * ...``, every global Pauli string is a tensor product of local ones,
    ``P_k = P_k,0 * P_k,1 * ...``, and the element factorizes::

        <O>_U = sum_k c_k <P_k,0>_U0 * <P_k,1>_U1 * ...

    So only the small local circuits of each cluster are simulated, and their expectation values
    are reassembled into the full RDM element. Local Pauli strings already evaluated are cached
    to avoid repeated simulations.

    Parameters
    ----------
    U : tq.QCircuit
        Circuit preparing the state. Must not couple different clusters.
    mol : QuantumChemistryBase
        Molecule providing the RDM operators (needs ``compute_rdms`` with
        ``evaluate=False`` returning the raw operators).
    clusters : list of list of int
        Qubits of each cluster. Gates spanning several clusters are dropped.
    variables : dict
        Values of the circuit variables.
    test : bool, optional
        If True, compare each element with a full simulation and print mismatches.
        Default is False.
    backend : str, optional
        Simulation backend. Default is ``"qulacs"``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(rdm1, rdm2)``, the latter in Dirac (physics) ordering.

    Examples
    --------
    >>> clusters = sun.SPAFP.make_decomposed_clusters(U, 12)
    >>> rdm1, rdm2 = sun.SPAFP.fast_rdm(mol=mol, U=U, clusters=clusters, variables=variables)
    """
    assert clusters is not None

    n_MOs = mol.n_orbitals
    circuits = make_decomposed_circuits(U, clusters=clusters)

    # needs hackend version that gives back the raw qops
    qops, _asd = compute_rdms(self=mol, U=U, use_hcb=True, evaluate=False)

    all_ps = [ps for op in qops for ps in op.paulistrings]

    all_ps_evaluated = {}
    all_psi_evaluated = {}
    for ps in all_ps:
        assert numpy.isclose(ps.coeff.imag, 0.0)
        # all ps are evaluated and stored with ps.coeff=1.0 to ensure:
        # - dictionary key consistency
        # - identical ps with different ps.coeff are only simulated once
        ps_key = str(tq.PauliString(ps._data, coeff=1.0))
        ps_evaluated = 1.0

        if len(ps.qubits) == 0:
            all_ps_evaluated[ps_key] = float(ps_evaluated)
            continue

        for i, u in enumerate(circuits):
            psi = tq.PauliString(
                {k: v for k, v in ps.items() if k in u.qubits}, coeff=1.0
            )
            if len(psi) == 0:
                continue

            psi_key = (str(psi), i)
            if psi_key not in all_psi_evaluated:
                Hi = tq.QubitHamiltonian.from_paulistrings(psi)
                psi_evaluated = tq.simulate(
                    tq.ExpectationValue(H=Hi, U=u), variables=variables, backend=backend
                )
                all_psi_evaluated[psi_key] = psi_evaluated
            else:
                psi_evaluated = all_psi_evaluated[psi_key]
            ps_evaluated *= psi_evaluated

        all_ps_evaluated[ps_key] = float(ps_evaluated)

    evaluated = []
    if test:
        testus = tq.simulate(
            tq.ExpectationValue(H=qops, U=U, shape=[len(qops)]),
            variables=variables,
            backend=backend,
        )

    for x, op in enumerate(qops):
        tmp = 0.0
        trace = {}
        for ps in op.paulistrings:
            ps_key = str(tq.PauliString(ps._data, coeff=1.0))
            tmp += float(ps.coeff.real) * all_ps_evaluated[ps_key]
            trace[str(ps)] = all_ps_evaluated[ps_key]
        evaluated.append(tmp)

        if test and not numpy.isclose(tmp, testus[x]):
            print(x, " ", op)
            print(f"{tmp} vs {testus[x]}")
            print(trace)

    len_1 = n_MOs * (n_MOs + 1) // 2
    evals = [float(x.real) for x in evaluated]
    evals_1, evals_2 = evals[:len_1], evals[len_1:]
    rdm1 = _assemble_rdm1(evals_1, n_MOs)
    rdm2 = _assemble_rdm2(evals_2, n_MOs)
    rdm2 = NBodyTensor(elems=rdm2, ordering="dirac", verify=False)
    rdm2 = rdm2.reorder(to="dirac").elems

    return rdm1, rdm2


if __name__ == "__main__":
    for n_atoms in [8, 12, 16, 20]:
        edges = [(2 * i, 2 * i + 1) for i in range(n_atoms // 2)]
        geometry = ""
        for i in range(n_atoms):
            geometry += f"H 0. 0. {i}.\n"

        mol = QubitMolecule(
            geometry=geometry,
            basis_set="sto-3g",
            backend="pyscf",
            transformation="ReorderedJordanWigner",
        ).use_native_orbitals()
        clusters = [
            [i for i in range(n_atoms // 2)],
            [i for i in range(n_atoms // 2, n_atoms)],
        ]
        if n_atoms == 16:
            clusters = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

        U = mol.make_ansatz(name="HCB-SPA", edges=edges)
        variables = {k: 1.0 for k in U.extract_variables()}
        rdm1, rdm2 = mol.compute_rdms(U=U, use_hcb=True, variables=variables)
        xrdm1, xrdm2 = fast_rdm(U=U, clusters=clusters, mol=mol, variables=variables)
        print(numpy.linalg.norm(rdm1 - xrdm1))
        print(numpy.linalg.norm(rdm2 - xrdm2))
