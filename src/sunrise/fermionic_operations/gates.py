import typing
import numbers
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from .fgateimpl import *
from .givens_rotations import get_givens_circuit
from numpy import ndarray

def FermionicExcitation(indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    '''
    Generic n-excitation gate defined by the indices [(0,2),(1,3),...].
    Reordered means for whether the indices provided are in up-down-up-down-... (False) or up-up-...-down-down-... convention 
    '''
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

def UR(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None)->FCircuit:
    '''
    Orbital Rotation gate, defined by a paired single excitation gate from Spatial orbital i to j (UR=exct([(2*i,2*j)],\theta) + exct([(2*i+1,2*j+1)],\theta))
    '''
    return FCircuit.wrap_gate(URImpl(i,j,variables))

def UC(i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None)->FCircuit:
    '''
    Orbital Correlator gate, defined by a paired double excitation gate from Spatial orbital i to j (UC=exct([(2*i,2*j),(2*i+1,2*j+1)],\theta))
    '''
    return FCircuit.wrap_gate(UCImpl(i,j,variables))

def UX(indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    '''
    Shortcut for generic n-excitation gate defined by the indices [(0,2),(1,3),...].
    Reordered means for whether the indices provided are in up-down-up-down-... (False) or up-up-...-down-down-... convention 
    '''
    return FCircuit.wrap_gate(FermionicExcitationImpl(indices,variables,reordered))

def Phase(tarjet:int,variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False)->FCircuit:
    '''
    Phase gate on Spin-Orbital i. Care with angles, some simulators only accept real wvf.
    '''
    return FCircuit.wrap_gate(PhaseImpl(tarjet,variables,reordered))

def Givens(unitary:ndarray, tol:float=1e-12, ordering:typing.Union[list,tuple,str]='OPTIMIZED_ORDERING')->FCircuit:
    '''
    Givens rotation generated from the given unitary, decomposed into UR + Phase gates.
    Customized matrix decomposition can be specified by: ordering
    '''
    return get_givens_circuit(unitary,tol,ordering)
