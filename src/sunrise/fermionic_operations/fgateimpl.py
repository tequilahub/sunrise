import numbers
import typing
from tequila.objective.objective import FixedVariable,Variable
from .circuit import FCircuit
from tequila import TequilaException
from tequila import assign_variable
from copy import deepcopy

class FGateImpl:
    def __init__(self,indices:typing.Union[list,tuple],variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None,reordered:bool=False):
        self.reordered:bool=reordered
        self._indices:list = indices
        self.variables=variables
        self._name:str = 'GenericFermionic'
        self.verify()
        return FCircuit.wrap_gate(gate=self)

    @property
    def name(self):
        return self._name
    
    def is_controlled(self):
        return False #TODO: at some point we should

    def to_upthendown(self,norb:int):
        if not self.reordered:
            self._indices = [[(idx[0]//2+(idx[0]%2)*norb,idx[1]//2+(idx[1]%2)*norb) for idx in gate] for gate in self._indices]
        return self
    
    def to_udud(self,norb:int):
        if self.reordered:
            self._indices = [[(2*(idx[0]%norb)+(idx[0]>=norb),2*(idx[1]%norb)+(idx[1]>=norb)) for idx in gate] for gate in self._indices]
        return self
    
    def __str__(self):
        return f'{self.name}(indices = {self.indices} ,variables = {repr(self.variables)})'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if self._name != other._name:
            return False
        elif self._indices != other._indices:
            return False
        elif self.extract_variables() != other.extract_variables():
            return False
        return True
            
    def is_parameterized(self)->bool:
        return not isinstance(self.variables,(FixedVariable,numbers.Number))

    @property
    def indices(self):
        return self._indices
     
    @indices.setter
    def indices(self,indices):
        self._indices = indices
    
    def extract_variables(self)->list[Variable]:
        if self.is_parameterized() and hasattr(self.variables, "extract_variables"):
            return self.variables.extract_variables()
        else:
            return []

    @property
    def variables(self)->Variable:
        return self._variables
    
    @variables.setter
    def variables(self,variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]):
        self._variables:Variable = assign_variable(variable=variables)

    def verify(self):
        assert isinstance(self._variables,(typing.Hashable, numbers.Real, Variable, FixedVariable))
        if isinstance(self._indices[0],numbers.Number): #[1,3]
            self._indices = [[tuple(self._indices),],] #->[[(1,3)]]
        elif isinstance(self._indices[0][0],numbers.Number): #[(0,2),(1,2)]
            self._indices = [self._indices,] #->[[(0,2),(1,3)],]
        elif not isinstance(self._indices[0][0][0],numbers.Number): #[[(0,2),(1,2)]]
            raise TequilaException(f'Indices formating not recognized, received {self._indices}')
        if isinstance(self._variables,FixedVariable):
            assert 1 == len(self._indices) 
        else:
            assert len(self._variables) == len(self._indices)

    @property
    def qubits(self):
        if self._indices is not None:
            q = []
            for gate in self._indices:
                for exct in gate:
                    q.extend(exct)
            return sorted(list(set(q)))
        else: return 0

    @property
    def n_qubits(self)->int:
        return len(self.qubits)
    
    def map_qubits(self,qubit_map:dict={}):
        self._indices =[[(qubit_map[idx[0]],qubit_map[idx[1]]) for idx in gate] for gate in self._indices]
        return self

    def map_variables(self,var_map:dict={}):
        if self.is_parameterized() and hasattr(self.variables, "extract_variables"):
            self.variables = self.variables.map_variables(var_map)
        return self 

    def dagger(self):
        indinces = []
        cir = deepcopy(self)
        for gate in reversed(cir._indices):
            indinces.extend([[tuple([*idx][::-1]) for idx in gate]])
        cir.indices = indinces
        cir.variables = -1*self.variables
        return cir

    @property
    def max_qubit(self):
        if self.qubits:
            return self.qubits[-1]
        else: return 0

    def __eq__(self, other) -> bool:
        if self.name != other.name:
            return False
        if self.reordered != other.reordered:
            return False
        if self.variables != other.variables:
            return False
        if self.indices != other.indices:
            return False
        return True

class FermionicExcitationImpl(FGateImpl):
    def __init__(self, indices:typing.Union[list,tuple], variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable,None]=None, reordered:bool=False):
        super().__init__(indices, variables, reordered)
        self._name = 'FermionicExcitation'

class URImpl(FGateImpl):
    def __init__(self, i:int,j:int, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable,None]=None):
        super().__init__([[(2*i,2*j)],[(2*i+1,2*j+1)]], variables, False)
        self._name = 'UR'
    def verify(self):
        assert isinstance(self._variables,(typing.Hashable, numbers.Real, Variable, FixedVariable))
        if isinstance(self._indices[0],numbers.Number): #[1,3]
            self._indices = [[tuple(self._indices),],] #->[[(1,3)]]
        elif isinstance(self._indices[0][0],numbers.Number): #[(0,2),(1,2)]
            self._indices = [self._indices,] #->[[(0,2),(1,3)],]
        elif not isinstance(self._indices[0][0][0],numbers.Number): #[[(0,2),(1,2)]]
            raise TequilaException(f'Indices formating not recognized, received {self._indices}')
        if isinstance(self._variables,FixedVariable):
            assert 2 == len(self._indices) 
        else:
            assert 2*len(self._variables) == len(self._indices)

class UCImpl(FGateImpl):
    def __init__(self,i,j, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None):
        super().__init__([[(2*i,2*j),(2*i+1,2*j+1)]], variables, False)
        self._name = 'UC'
    
class PhaseImpl(FGateImpl):
    def __init__(self,i, variables:typing.Union[typing.Hashable, numbers.Real, Variable, FixedVariable]=None, reordered:bool=False):
        super().__init__([[(i,i)]], variables, reordered)
        self._name = 'Ph'

    def __str__(self):
        return f'{self.name}(target = {(self.indices[0][0][0],)}, variable = {repr(self.variables)})'
