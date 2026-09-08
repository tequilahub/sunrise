from .tequila_expval import TequilaBraket
from typing import Union,List
from openfermion import FermionOperator
from tequila import TequilaException

SUPPORTED_FERMIONIC_BACKENDS = ["tequila", "fqe", "tcc"]
INSTALLED_FERMIONIC_BACKENDS = {}

try:
    from sunrise.expval.tcc_expval import TCCBraket
    INSTALLED_FERMIONIC_BACKENDS["tcc"] = TCCBraket
except ImportError:
    pass
try:
    from sunrise.expval.fqe_expval import FQEBraKet
    INSTALLED_FERMIONIC_BACKENDS["fqe"] = FQEBraKet
except ImportError:
    pass

INSTALLED_FERMIONIC_BACKENDS["tequila"] = TequilaBraket

def show_available_modules():
    print("Available Fermionic Modules:")
    for k in INSTALLED_FERMIONIC_BACKENDS.keys():
        print(k)


def show_supported_modules():
    print(SUPPORTED_FERMIONIC_BACKENDS)

def Braket(backend:str='tequila',operator:Union[str,List[FermionOperator],FermionOperator]=None,*args,**kwargs)->TequilaBraket:
    '''
    Interface for Fermionic Braket <U_bra|H|U_ket>. Since it allows U1 != U2, and U1=U2, some keywords can be presented as
    keywordX -> X \in [_bra,_ket,None] (i.e. init_state,init_state_ket,init_state_bra). If X=None, bra and ket are setted. 
    If X=ket, but not X=bra provided, bra is setted to ket.
    Parameters
    ----------
    backend
        fermionic backend to compute the Braket. See sun.show_supported_modules() or show_available_modules().
    mol/molecule
        tequila or pyscf molecule to get the molecular info and the electronic integrals
    integral_mager + parameters:
        alternative to molecule, tequila objects providing required molecular information
    h/int1e/one_body_integrals + g/int2e/two_body_integrals + (Optional)c/e_core/constant_term
    + (Optional) s/ovlp/overlap_integrals + n_elec
        electronic integral information
    (Optional)n_qubits
        number of circuit's qubits. If not provided, assumed from molecular information
    bra/ket/circuit: FCircuit
        Fermionic Circuit containing the excitations, variables and initial state.
    (Optional) backend_kwargs:dict
        tcc
            engine
                see tcc -> UCC engine
            dtype
                see tcc.set_dtype
            backend
                see tcc.set_backend
            else: kwargs provided to the UCC object initialization
        tequila: everything provided in tq.ExpVal/Braket furthen than the circuit and operators
    '''
    if 'H' in kwargs:
            H = kwargs['H']
            kwargs.pop('H')
            if operator is not None:
                raise TequilaException('Two operators provided?')
            else:
                operator = H
    if isinstance(operator,list):
        return [INSTALLED_FERMIONIC_BACKENDS[backend.lower()](operator=op,*args,**kwargs) for op in operator]
    #any kwargs and circuit form should be managed inside each class
    return INSTALLED_FERMIONIC_BACKENDS[backend.lower()](operator=operator,*args,**kwargs) 
