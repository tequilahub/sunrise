import tequila as tq
from tequila import BraKet,QCircuit,QubitHamiltonian,ExpectationValue
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila import TequilaException
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila import TequilaException,Molecule,simulate,Variable,Objective,grad
from tequila.objective.objective import Variables
from numpy import argwhere
from pyscf.gto import Mole
from sunrise.expval.pyscf_molecule import MoleculeFromPyscf
from ..fermionic_operations.circuit import FCircuit
from typing import Union,List
from openfermion import FermionOperator

class TequilaBraket:
    def __init__(self,bra:Union[FCircuit,None]=None,ket:Union[FCircuit,None]=None,operator:Union[str,QubitHamiltonian,FermionOperator,List[FermionOperator]]=None,backend_kwargs:dict={},*args,**kwargs):
        if 'circuit' in kwargs:
            circuit = kwargs['circuit']
            kwargs.pop('circuit')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = circuit

        if 'U' in kwargs:
            U = kwargs['U']
            kwargs.pop('U')
            if ket is not None:
                raise TequilaException('Two circuits provided?')
            else:
                ket = U
        if 'H' in kwargs:
                H = kwargs['H']
                kwargs.pop('H')
                if operator is not None:
                    raise TequilaException('Two operators provided?')
                else:
                    operator = H
        self.backend_kwargs = backend_kwargs
        self.molecule = None
        self._bra = self._ket = None
        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs['molecule']
            kwargs.pop('molecule')
            if isinstance(molecule,QuantumChemistryBase):
                self.molecule = molecule
            elif isinstance(molecule,Mole):
                self.molecule = MoleculeFromPyscf(molecule=molecule)
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
            self.molecule = Molecule(parameters=params,integral_manager=integral)
        else:
            int1e = None
            int2e = None
            e_core = None
            if "int1e"  in kwargs:
                int1e = kwargs['int1e']
                kwargs.pop('int1e')
            elif "one_body_integrals"  in kwargs:
                int1e = kwargs['one_body_integrals']
                kwargs.pop('one_body_integrals')
            elif "h"  in kwargs:
                int1e = kwargs['h']
                kwargs.pop('h')
            if 'int2e' in kwargs:
                int2e = kwargs['int2e']
                kwargs.pop('int2e')
            elif 'two_body_integrals' in kwargs:
                int2e = kwargs['two_body_integrals']
                kwargs.pop('two_body_integrals')
            elif 'g' in kwargs:
                int2e = kwargs['g']
                kwargs.pop('g')
            if isinstance(int2e,NBodyTensor):
                int2e = int2e.elems
            if 'e_core' in kwargs:
                e_core = kwargs['e_core']
                kwargs.pop('e_core')
            elif 'constant_term' in kwargs:
                e_core = kwargs['constant_term']
                kwargs.pop('constant_term')
            elif 'constant' in kwargs:
                e_core = kwargs['constant']
                kwargs.pop('constant')
            elif 'c' in kwargs:
                e_core = kwargs['c']
                kwargs.pop('c')    
            else: e_core = 0.
            if 'n_elec' in kwargs:
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif 'n_electrons' in kwargs: 
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif ket is not None and ket.initial_state is not None:
                if isinstance(ket.initial_state._state,dict):
                    n_elec = bin([*ket.initial_state._state.keys()][0])[2:].count('1')
                else:
                    n_elec = bin(argwhere(ket.initial_state._state>1.e-6)[0][0])[2:].count('1')
            else:
                raise TequilaException("No manner of defining the amount of electrons provided")
            if all([i is not None for i in[int2e,int1e]]):
                if isinstance(int2e,NBodyTensor):
                    int2e = int2e.reorder('of').elems
                if 'transformation' in kwargs:
                    trans = kwargs['transformation']
                else: trans = 'reordered-jordan-wigner'
                if 'molecule_arguments' in kwargs:
                    molecule_arguments = kwargs['molecule_arguments']
                else: molecule_arguments = {}
                self.molecule = Molecule(one_body_integrals=int1e,two_body_integrals=int2e,constant_term=e_core,n_electrons=n_elec,transformation=trans,**molecule_arguments)
            else:
                raise TequilaException('Not enough molecular data provided')
        self.molecule.integral_manager.upthendown=True
        if ket is not None:
            ket = ket.to_upthendown(molecule.n_orbitals)
            self._ket = ket
        if bra is not None:
            bra = bra.to_upthendown(molecule.n_orbitals)
            self._bra = bra 
        self.operator = None
        if operator is None:
            operator = "H"
        if isinstance(operator,str):
            if operator == 'H':
                self.operator = self.molecule.make_hamiltonian()
            elif operator == "HCB":
                self.operator = self.molecule.make_hardcore_boson_hamiltonian()
            elif operator == 'I':
                qubits = []
                if ket is not None:
                    qubits.extend(ket.qubits)
                if bra is not None:
                    qubits.extend(bra.qubits)
                qubits = list(set(qubits))
                operator = tq.gates.I([qubits])
            else:
                self.operator = tq.paulis.from_string(operator)
        elif isinstance(operator,FermionOperator):
            self.operator = self.molecule.transformation(operator)
        if 'name' in kwargs:
            self._name = kwargs['name']
        else: self._name = 'Expectation Value' if self.is_diagonal else "Transition Value"
        
    
    def __call__(self, variables:Union[list,dict]={}, *args, **kwargs)->float:
        return simulate(self.build(),variables=variables,*args, **kwargs)

    def grad(self,variable:Variable = None)->Objective:
        return grad(self.build(),variable=variable)

    def extract_variables(self) -> list:
        return self.build().extract_variables() 
    
    @property
    def bra(self)->QCircuit:
        if self._bra is not None:
            return self._bra.to_qcircuit(molecule=self.molecule)
        else: return None

    @bra.setter
    def bra(self, bra:FCircuit):
        self._bra = bra

    @property
    def ket(self)->QCircuit:
        if self._ket is not None:
            return self._ket.to_qcircuit(molecule=self.molecule)
        else: return None

    @ket.setter
    def ket(self, ket:FCircuit):
        self._ket = ket

    @property
    def variables(self) -> Variables:
        return self.variables_bra,self.variables_ket

    @property
    def variables_bra(self) -> Variables:
        if self._bra is None:
            return None
        return self._bra.variables

    @property
    def variables_ket(self) -> Variables:
        if self._ket is None:
            return None
        return self._ket.variables

    
    def __str__(self):
        res = ''
        if self.is_diagonal:
            res += f"{self._name} with indices: {self._ket.extract_indices()} with variables {self.variables_ket}"
        else:
            res += f"{self._name} with Bra= {self._bra.extract_indices()} with variables {self.variables_bra}\n"
            res += f"{len(self._name)*' '} with Ket= {self._ket.extract_indices()} with variables {self.variables_ket}"
        return res

    def __repr__(self):
        return self.__str__()
    
    @property
    def U(self):
        'Dummy function to work with tequila Objectives'
        if self.is_diagonal:
            return self._ket.extract_indices() 
        else:
            return (self._bra.extract_indices,self._ket.extract_indices())
        
    def count_measurements(self)->int:
        return self.build().count_measurements()
    
    def build(self,only_real:bool=True)->Objective:
        if self.is_diagonal:
            return ExpectationValue(U=self.ket,H=self.operator,**self.backend_kwargs)
        else:
            ov = BraKet(bra=self.bra,ket=self.ket,operator=self.operator,**self.backend_kwargs)
            if only_real:
                return ov[0]
            else:
                return ov
        
    @property
    def is_diagonal(self)->bool:
        if self._bra is None and self._ket is None:
            return True
        elif self._bra is not None and self._ket is not None:
            return self._bra == self._ket
        if self._bra is None or self._ket is None:
            return True