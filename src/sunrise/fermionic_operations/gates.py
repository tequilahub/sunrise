import typing
import numbers
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from .fgateimpl import *
from .givens_rotations import get_givens_circuit
from numpy import ndarray

def FermionicExcitation(indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

def UR(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None)->FCircuit:
    return FCircuit.wrap_gate(URImpl(i,j,variables))

def UC(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None)->FCircuit:
    return FCircuit.wrap_gate(UCImpl(i,j,variables))

def UX(indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

def Phase(tarjet:int,variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    return FCircuit.wrap_gate(PhaseImpl(tarjet,variables,reordered))

def Givens(unitary:ndarray, tol:float=1e-12, ordering:typing.Union[list,tuple,str]='OPTIMIZED_ORDERING')->FCircuit:
    return get_givens_circuit(unitary,tol,ordering)