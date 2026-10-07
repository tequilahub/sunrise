import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian,ExpectationValue
from sunrise.molecules.qubit_base.chemistry_tools import NBodyTensor
from tequila import TequilaException
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from tequila import TequilaException, simulate, Variable, Objective, grad
from sunrise.molecules.qubit_base import Molecule
from tequila.objective.objective import Variables
from numpy import argwhere
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import MoleculeFromPyscf
from ..fermionic_operations.circuit import FCircuit
from typing import Union,List
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
    if braket.is_diagonal:
        return ExpectationValue(U=ket, H=operator, *args, **kwargs)
    return BraKet(ket=ket, bra=bra, operator=operator, *args, **kwargs)
