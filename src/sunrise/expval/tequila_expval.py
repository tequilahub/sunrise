from tequila import BraKet, Objective
from tequila.hamiltonian.paulis import I, from_string
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from openfermion import FermionOperator
from .fermionic_braket import FermBraketImpl

def TequilaBraket(braket:"FermBraketImpl", *args, **kwargs) -> Objective:
    from sunrise.molecules.fermionic_base import FermionicBase
    mol = braket.molecule
    if isinstance(mol,FermionicBase):
        mol = QuantumChemistryBase(parameters=mol.parameters, transformation='REORDEREDJORDANWIGNER', integral_manager=mol.integral_manager)
    bra = braket.bra
    if bra is not None:
        bra = bra.to_upthendown(mol.n_orbitals).to_qcircuit(mol)
    ket = braket.ket.to_upthendown(mol.n_orbitals).to_qcircuit(mol)
    operator = braket.operator
    if operator is None:
        operator = "H"
    if isinstance(operator,str):
        if operator == 'H':
            operator = mol.make_hamiltonian()
        elif operator == "HCB":
            operator = mol.make_hardcore_boson_hamiltonian()
        elif operator == 'I':
            operator = I([braket.qubits])
        else:
            operator = from_string(operator)
    elif isinstance(operator,FermionOperator):
        operator = mol.transformation(operator)
    return BraKet(ket=ket, bra=bra, operator=operator, *args, **kwargs)
