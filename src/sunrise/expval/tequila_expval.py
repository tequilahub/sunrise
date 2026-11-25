"""Fermionic backend ``"tequila"``: conversion of a fermionic braket into a tequila qubit Objective."""

from openfermion import FermionOperator
from tequila import BraKet, ExpectationValue, Objective
from tequila.hamiltonian.paulis import I, from_string

from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

from .fermionic_braket import FermBraketImpl


def TequilaBraket(braket: FermBraketImpl, *args, **kwargs) -> Objective:
    """Braket (overlap, expectation value or transition element) evaluated with the ``"tequila"`` backend, i.e. as a
    tequila qubit Objective.

    Circuits are mapped to qubit circuits with the molecule's transformation
    (an up-then-down orbital ordering is used) and the operator is transformed to
    a qubit operator.

    Parameters
    ----------
    braket : FermBraketImpl
        Fermionic braket holding ket, optional bra, operator and molecule.
    *args, **kwargs
        Passed on to tequila's ``ExpectationValue`` or ``BraKet``.

    Returns
    -------
    tq.Objective
        ExpectationValue if the braket is diagonal, otherwise a BraKet.
    """
    from sunrise.molecules.fermionic_base import FermionicBase

    mol = braket.molecule
    if isinstance(mol, FermionicBase):
        mol = QuantumChemistryBase(
            parameters=mol.parameters,
            transformation="REORDEREDJORDANWIGNER",
            integral_manager=mol.integral_manager,
        )
    bra = braket.bra
    if bra is not None:
        bra = bra.to_upthendown(mol.n_orbitals).to_qcircuit(mol)
    ket = braket.ket.to_upthendown(mol.n_orbitals).to_qcircuit(mol)
    operator = braket.operator
    if operator is None:
        operator = "H"
    if isinstance(operator, str):
        if operator == "H":
            operator = mol.make_hamiltonian()
        elif operator == "HCB":
            operator = mol.make_hardcore_boson_hamiltonian()
        elif operator == "I":
            operator = I([braket.qubits])
        else:
            operator = from_string(operator)
    elif isinstance(operator, FermionOperator):
        operator = mol.transformation(operator)
    if braket.is_diagonal:
        return ExpectationValue(*args, U=ket, H=operator, **kwargs)
    return BraKet(*args, ket=ket, bra=bra, operator=operator, **kwargs)
