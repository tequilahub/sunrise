from typing import Union,List,Optional
from openfermion.ops.operators.fermion_operator import FermionOperator
from tequila import TequilaException, Objective
from tequila.objective.objective import identity
from tequila.quantumchemistry.qc_base import QuantumChemistryBase #TODO modify when migrated
from sunrise.fermionic_operations.circuit import FCircuit
from .fermionic_braket import FermBraketImpl
from tequila import QTensor

SUPPORTED_FERMIONIC_BACKENDS = ["tequila", "fqe", "tcc", "spex"]
INSTALLED_FERMIONIC_BACKENDS = {}

try:
    from .tcc_expval import TCCBraket
    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass

try:
    from .fqe_expval import FQEBraKet
    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraKet
except ImportError:
    pass
try:
    from .spex_expval import SpexExpval
    INSTALLED_FERMIONIC_BACKENDS["spex"] = SpexExpval
except (ImportError, AttributeError):
    pass

from .tequila_expval import TequilaBraket
INSTALLED_FERMIONIC_BACKENDS["tequila"] = TequilaBraket

def show_available_modules():
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS.keys():
        print(k)


def show_supported_modules():
    print(SUPPORTED_FERMIONIC_BACKENDS)

def Fidelity(ket: FCircuit, bra: FCircuit, mol : Optional[QuantumChemistryBase] = None, *args, **kwargs) -> Objective:
    """

    Convenience initialization of an tq.Objective that corresponds to the fidelity |<bra|ket>|^2 between two quantum states
    initialized by the circuits bra and ket

    Notes
    -----

    note, that the fidelity is symmetric: F(a,b) = F(b,a)
    so it does not matter what is bra and what is ket

    Parameters:
    ----------
    bra: FCircuit
    ket: FCircuit
    mol: Sunrise Molecule

    Returns:
    ----------
    A tq.Objective that evaluated to the fidelity between the two states

    """
    return Objective(args=[Braket(ket=ket, bra=bra, molecule=mol, operator=None, *args, **kwargs).args[0]], transformation=lambda x: abs(x)**2)

def Overlap(ket : FCircuit, bra : FCircuit, mol : Optional[QuantumChemistryBase] = None, *args, **kwargs) -> Objective:
    """

    Function that calculates the overlap between two quantum states.

    Parameters:
    ----------
    bra: FCircuit
    ket: FCircuit
    mol: Sunrise Molecule

    Returns:
    ----------
    A tq.Objective that evaluated to the overlap between the two states

    """
    return Braket(ket=ket, bra=bra, operator=None, mol=mol, *args, **kwargs)

def Braket(
        ket : FCircuit,
        bra : Optional[FCircuit] = None,
        operator : Optional[Union[str,List[FermionOperator],FermionOperator]] =  None,
        mol: Optional[QuantumChemistryBase] = None,
        *args,
        **kwargs) -> Objective:
    """Function that allows to calculate different quantities
       depending on the passed parameters:
       1) If only ket is passed, returns the overlap with itself (1).
       2) If ket and bra are passed, returns the overlap between the two states.
       3) If ket and operator are passed, returns the expectation value of the operator for the given state.
       4) If ket, bra and operator are passed, returns the transition element of the operator.

       returns an instance of tq.Objective

    Args:
        ket (FCircuit): FCircuit corresponding to a state.
        bra (FCircuit, optional): FCircuit corresponding to a second state.
                                  Defaults to None.
        operator (FermionOperator, optional): Operator of which we want to
                                               calculate the transition element.
                                               Defaults to None unless molecule provided, then the fermionic 
                                               Hamiltonian is taken as default.
        mol: (QuantumChemistryBase, Optional) Sunrise Molecule to get the information (n electrons, integrals...)
                                                if not enough data provided

    Returns:
        tq.Objective representing the BraKet
    """
    if 'molecule' in kwargs:
        if mol is not None:
            raise TequilaException("Two molecules provided?")
        else:
            mol = kwargs["molecule"]
            kwargs.pop("molecule")
    if mol is not None and operator is None:
        operator = "H"
    if bra is not None and ket is None:
        ket = bra
        bra = None
    if isinstance(operator,list):
        return [Braket(ket=ket, bra=bra, operator=op, mol=mol, *args, **kwargs) for op in operator]
        return QTensor(objective_list=arglist, shape=len(operator)) # TODO: create multidimensional
    if bra is None and operator is None:
        return Objective() + 1.0
    return Objective(args=[FermBraketImpl(ket=ket, bra=bra, operator=operator, molecule=mol, *args, **kwargs)], transformation=identity)

def ExpectationValue(U : FCircuit, H : Union[str,List[FermionOperator],FermionOperator] = 'H', mol : Optional[QuantumChemistryBase] = None, *args, **kwargs) -> Objective:
    """
        Function that calculates the Expectation Value of an Hamiltonian operator for a given quantum states.
    
        Parameters
        ----------
        U : FCircuit sunrise object, corresponding to the quantum state.
    
        H : QubitHamiltonian tequila object
    
        Returns
        -------
        Tequila objectives to be simulated or compiled.
    
    """
    return Braket(ket=U,operator=H, mol=mol, *args, **kwargs)

def RealBraKet(ket : FCircuit, bra : Optional[FCircuit]=None, operator: Union[str,List[FermionOperator],FermionOperator] = None, mol : Optional[QuantumChemistryBase] = None, *args, **kwargs):
    """
    Function that calculates the Reak part of a given Expectation Value of an Hamiltonian operator for a given quantum states.

    Parameters
    ----------
    U : FCircuit sunrise object, corresponding to the quantum state.

    H : QubitHamiltonian tequila object

    Returns
    -------
    Tequila objectives to be simulated or compiled.
        
    """
    return Braket(bra=bra, ket=ket, operator=operator, mol=mol, *args, **kwargs) 

def ImagBraKet(*args, **kwargs):
    """
    Function that calculates the Imaginary part of a given Expectation Value of an Hamiltonian operator for a given quantum states.
    Defined just for consistency with Tequila, since chemistry molecules should be always real 

    Parameters
    ----------
    U : FCircuit sunrise object, corresponding to the quantum state.

    H : QubitHamiltonian tequila object

    Returns
    -------
    Tequila objectives to be simulated or compiled.

    """
    return Objective()
