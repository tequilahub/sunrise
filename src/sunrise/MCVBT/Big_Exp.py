from sunrise.molecules.qubit_base import QuantumChemistryBase
from sunrise.expval import Braket, Overlap
from tequila import BraKet as tqBraket
from tequila import Overlap as tqOverlap
from tequila import TequilaException
from typing import Union
from tequila import QCircuit
from sunrise import FCircuit
from tequila import TequilaException, Objective
from numpy import sign

def BigExpVal(circuits:list[Union[QCircuit,FCircuit]], coefficcents:list[float], mol:QuantumChemistryBase, **kwargs) -> Objective:
    n = len(circuits)
    SS = 0.
    EE = 0.
    ferm = all([isinstance(circuits[i],FCircuit) for i in range(n)])
    qub = all([isinstance(circuits[i],QCircuit) for i in range(n)])
    if not ferm and not qub:
        raise TequilaException("Mixture of Fermionic and Qubit Circuits provided, can't handle it.")
    for i in range(n):
        for j in range(n): #oder n ?
            if ferm:
                EE += (1*coefficcents[i])*(1*coefficcents[j])*Braket(ket=circuits[j], bra=circuits[i], mol=mol, operator="H")
                SS += (1*coefficcents[i])*(1*coefficcents[j])*Overlap(ket=circuits[j], bra=circuits[i], mol=mol)
            else:
                if "H" in kwargs:
                    H = kwargs["H"]
                    kwargs.pop("H")
                else:
                    H = mol.make_hamiltonian()
                EE += (1*coefficcents[i])*(1*coefficcents[j])*tqBraket(ket=circuits[i], bra=circuits[j], operator=H)
                SS += (1*coefficcents[i])*(1*coefficcents[j])*tqOverlap(ket=circuits[i], bra=circuits[j])
    f = lambda x:sign(x)*max(abs(x),1.e-6)
    SS = SS.apply(f)
    return EE/SS