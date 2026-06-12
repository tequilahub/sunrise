from ..fermionic_operations.circuit import FCircuit
from tequila import TequilaException,Molecule,QubitWaveFunction,simulate,Variable,Objective,assign_variable,QubitHamiltonian
from tequila import grad as tq_grad
from tequila.objective.objective import Variables,FixedVariable
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString, BitNumbering
from numbers import Number
from numpy import ceil,argwhere,pi,prod,eye,zeros,isclose,allclose,ndarray,array,complex128,real,vdot
from pyscf.gto import Mole
from pyscf.scf import RHF
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
from typing import Union,List,Tuple,Callable
from collections import defaultdict
import ffsim
from ffsim import FermionOperator
from scipy.sparse.linalg import LinearOperator

class FFS_EXPVAL:
    def __init__(self,bra:Union[FCircuit,None]=None,ket:Union[FCircuit,None]=None,operator:Union[str,FermionOperator,List[FermionOperator]]=None,backend_kwargs:dict={},*args,**kwargs):
        self.operator = None
        self.molecule:ffsim.MolecularData = None
        self._init_state_bra:ndarray = None
        self._init_state_ket:ndarray = None
        self._ket: FCircuit = None
        self._bra: FCircuit = None
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
        if 'mol' in kwargs:
            if 'molecule' in kwargs and kwargs['molecule']:
                raise TequilaException("Two molecules provided?")
            kwargs['molecule'] = kwargs['mol']
            kwargs.pop('mol')

        run_hf = (bra is None or bra.initial_state is None) and (ket is None or ket.initial_state is None)   
        if 'molecule' in kwargs and kwargs['molecule']:
            molecule = kwargs['molecule']
            kwargs.pop('molecule')
            if isinstance(molecule,qc_base.QuantumChemistryBase):
                mo_coeff = molecule.integral_manager.orbital_coefficients 
                c,h,g = molecule.get_integrals()
                g = g.reorder('chem').elems
                spin = (molecule.parameters.multiplicity-1)/2
                n_alpha = int((molecule.n_electrons + spin)//2)
                n_beta  = int((molecule.n_electrons - spin)//2)
                point_group = None
                if hasattr(molecule,'point_group'):
                    point_group = molecule.point_group
                
                self.molecule = ffsim.MolecularData(
                    core_energy=c,
                    one_body_integrals=h,
                    two_body_integrals=g,
                    norb=molecule.n_orbitals,
                    nelec=(n_alpha, n_beta),
                    atom=molecule.parameters.get_geometry(),
                    basis=molecule.parameters.basis_set,
                    spin=spin,
                    symmetry=point_group,
                    mo_coeff=molecule.integral_manager.orbital_coefficients,
                    active_space=[i.idx_total for i in molecule.integral_manager.active_orbitals],
                )
            elif isinstance(molecule,Mole):
                mf = RHF(mol=molecule)
                self.molecule = ffsim.MolecularData.from_scf(mf)
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
            mo_coeff = integral.orbital_coefficients 
            c,h,g = integral.get_integrals()
            g = g.reorder('chem').elems
            spin = (params.multiplicity-1)/2
            n_alpha = int((molecule.n_electrons + spin)//2)
            n_beta  = int((molecule.n_electrons - spin)//2)
            point_group = None
            if 'point_group' in kwargs:
                point_group = kwargs['point_group']
                kwargs.pop('point_group')
            elif 'symmetry' in kwargs:
                point_group = kwargs['symmetry']
                kwargs.pop('symmetry')
            
            self.molecule = ffsim.MolecularData(
                core_energy=c,
                one_body_integrals=h,
                two_body_integrals=g,
                norb=len(integral.active_orbitals),
                nelec=(n_alpha, n_beta),
                atom=params.get_geometry(),
                basis=params.basis_set,
                spin=spin,
                symmetry=point_group,
                mo_coeff=integral.orbital_coefficients,
                active_space=[i.idx_total for i in integral.active_orbitals],
            )
        else:
            int1e = None
            int2e = None
            e_core = None
            mo_coeff = None
            n_elec = None
            n_alpha = None
            n_beta = None
            spin = 0
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
                int2e = int2e.reorder('chem').elems
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
            if 'mo_coeff' in kwargs:
                mo_coeff = kwargs['mo_coeff']
                kwargs.pop('mo_coeff')
            elif 'orbital_coefficients' in kwargs:
                mo_coeff = kwargs['orbital_coefficients']
                kwargs.pop('orbital_coefficients')
            if 'spin' in kwargs:
                spin = kwargs['spin']
                kwargs.pop('spin')
            elif 'multiplicity' in kwargs:
                spin = (kwargs['multiplicity']-1)/2
                kwargs.pop('multiplicity')
            mo_coeff = eye(len(int1e))
            if 'n_elec' in kwargs:
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
                if spin is not None:
                    n_alpha = int((molecule.n_electrons + spin)//2)
                    n_beta  = int((molecule.n_electrons - spin)//2)
            elif 'n_electrons' in kwargs: 
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
                if spin is not None:
                    n_alpha = int((molecule.n_electrons + spin)//2)
                    n_beta  = int((molecule.n_electrons - spin)//2)
            elif 'n_alpha' in kwargs and 'n_beta' in kwargs:
                n_alpha = kwargs['n_alpha']
                n_beta = kwargs['n_beta']
                kwargs.pop('n_alpha')
                kwargs.pop('n_beta')
                n_elec = n_alpha + n_beta
            elif ket is not None and ket.init_state is not None:
                if isinstance(ket.initial_state._state,dict):
                    n_elec = bin([*ket.initial_state._state.keys()][0])[2:].count('1')
                else:
                    n_elec = bin(argwhere(ket.init_state._state>1.e-6)[0][0])[2:].count('1')
                if spin is not None:
                    n_alpha = int((molecule.n_electrons + spin)//2)
                    n_beta  = int((molecule.n_electrons - spin)//2)
            else:
                raise TequilaException("No manner of defining the amount of electrons provided")
            if n_alpha is None and n_elec is not None: #both should be provided
                n_alpha = int(n_elec // 2)
                n_beta = int(n_elec - n_alpha)
            if all([i is not None for i in[int2e,int1e,mo_coeff,n_alpha,n_beta]]):
                if isinstance(int2e,NBodyTensor):
                    int2e = int2e.reorder('chem').elems
                self.molecule = ffsim.MolecularData(
                    core_energy=e_core,
                    one_body_integrals=int1e,
                    two_body_integrals=int2e,
                    norb=len(int1e),
                    nelec=(n_alpha, n_beta),
                    spin=spin,
                    symmetry=point_group,
                    mo_coeff=mo_coeff,
                    active_space=[*range(len(int1e))],
                )
            else:
                raise TequilaException('Not enough molecular data provided')

        self._cistrings = ffsim.addresses_to_strings(range(ffsim.dim(self.molecule.norb, self.molecule.nelec)), norb=self.molecule.norb, nelec=self.molecule.nelec, concatenate=True, bitstring_type=ffsim.BitstringType.INT)
        
        if run_hf: 
            self.init_state = ffsim.hartree_fock_state(self.molecule.norb, self.molecule.nelec)
        
        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra 
        if 'name' in kwargs:
            self._name = kwargs['name']
        else: self._name = 'Expectation Value' if self.is_diagonal else "Transition Value"
        if isinstance(operator,str) and operator == 'I':
            self._name = 'Transition Element'
        if operator is not None:
            self.operator = self.build_operator(operator)
        
    def __call__(self, variables:Union[list,dict,Variables] = {}, *args, **kwargs) -> float:
        return self.simulate(variables = variables)

    def simulate(self,variables:Union[list,dict]=None)->float:  
        check_variables = {k: k in variables for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException(
                "Objective did not receive all variables:\n"
                "You gave\n"
                " {}\n"
                " but the objective depends on\n"
                " {}\n"
                " missing values for\n"
                " {}".format(variables, self.extract_variables(), [k for k, v in check_variables.items() if not v])
            )
        if isinstance(variables,Variables):
            variables = variables.store
        if isinstance(variables,dict):
            v: dict = deepcopy(self.variables)
            if v is None:
                v = variables
            else:
                v.update(variables)
            tvars: list = deepcopy(self.BK.total_variables)
            variables:list = [map_variables(x,v) for x in tvars]
        if variables is None:
            if self.is_diagonal:
                return real(vdot(self.init_state_ket, self.operator @ self.init_state_ket))
            else:
                return real(vdot(self.init_state_bra, self.operator @ self.init_state_ket))
        

        return real(vdot())

    def extract_variables(self) -> list[Variable]:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables_bra = []
        variables_ket = []
        uniques_bra = []
        if self.bra is not None:
            variables_bra = self.bra.extract_variables()
        if self.ket is not None:
            variables_ket = self.ket.extract_variables()
        for v in variables_bra:
            if v not in variables_ket:
                uniques_bra.append(v)
        return uniques_bra + variables_ket

    # def grad(self,variable:Variable = None)->Objective:
    #     def apply_phase(braket: FFS_EXPVAL,exct:List[Tuple[int]],idx:int,variable,ket:bool=True,p0sign:bool=True)->Objective: 
    #         '''
    #         braket: FFS object to modify
    #         exct: Excitation indices on the tequila format [(0,2),(1,3),...] to which 
    #               apply the phase shift.
    #         ket: If true it will be applied the phase on the ket side, bra otherwise
    #         posing: it True: +pi, False: -Pi, correspond to the U0(\pm) not the actual sign implementation
    #         '''
    #         p0 = []
    #         s = {True:+1,False:-1} 
    #         for ind in exct:
    #             p0.append((ind[0],ind[0]))
    #             p0.append((ind[1],ind[1]))
    #         braket._name = 'Gradient'
    #         if ket:
    #             if braket.is_diagonal: 
    #                 k = deepcopy(braket.ket)
    #                 v = deepcopy(braket.params_ket)
    #                 braket.bra = deepcopy(k)
    #                 braket.variables_bra = deepcopy(v)
    #                 ph = tq_grad(v[idx],variable) if not isinstance(v[idx],(FixedVariable,Number,Variable)) else 1
    #                 v[idx] +=  s[ket]*pi/2 
    #                 for p in reversed(p0): #https://qiskit-community.github.io/ffsim/api/stubs/ffsim.apply_num_op_prod_interaction.html#ffsim.apply_num_op_prod_interaction 
    #                     k.insert(idx,[p])
    #                     v.insert(idx,assign_variable(s[not p0sign]*pi))
    #                 braket.ket = k
    #                 braket.variables_ket = v
    #             else:
    #                 k = deepcopy(braket.ket)
    #                 v = deepcopy(braket.params_ket)
    #                 if exct not in k:
    #                     return 0.
    #                 ph = tq_grad(1*v[idx],variable) if not isinstance(v[idx],(FixedVariable,Number,Variable)) else 1
    #                 v[idx] +=  s[ket]*pi/2
    #                 for p in reversed(p0):
    #                     k.insert(idx,[p])
    #                     v.insert(idx,assign_variable(s[not p0sign]*pi)) 
    #                 braket.ket = k
    #                 braket.variables_ket = v
    #         else: 
    #             if braket.is_diagonal: 
    #                 k = deepcopy(braket.ket)
    #                 v = deepcopy(braket.params_ket)
    #                 ph = tq_grad(v[idx],variable) if not isinstance(v[idx],(FixedVariable,Number,Variable)) else 1
    #                 v[idx] +=  s[ket]*pi/2 
    #                 for p in reversed(p0):
    #                     k.insert(idx,[p])
    #                     v.insert(idx,assign_variable(s[not p0sign]*pi)) 
    #                 braket.bra = k
    #                 braket.variables_bra = v
    #             else:
    #                 k = deepcopy(braket.bra)
    #                 v = deepcopy(braket.params_bra)
    #                 if exct not in k:
    #                     return 0.
    #                 ph = tq_grad(v[idx],variable) if not isinstance(v[idx],(FixedVariable,Number,Variable)) else 1
    #                 v[idx] +=  s[ket]*pi/2
    #                 for p in reversed(p0):
    #                     k.insert(idx,[p])
    #                     v.insert(idx,assign_variable(s[not p0sign]*pi)) 
    #                 braket.bra = k
    #                 braket.variables_bra = v
    #         return s[ket]*s[(len(p0)//2)%2]*s[p0sign]*ph*Objective([braket])
    #     if variable is None:
    #         # None means that all components are created
    #         variables = self.extract_variables()
    #         result = {}

    #         if len(variables) == 0:
    #             raise TequilaException("Error in gradient: Objective has no variables")

    #         for k in variables:
    #             assert k is not None
    #             result[k] = self.grad(k)
    #         return result
    #     else:
    #         variable = assign_variable(variable)
    #     if variable not in self.extract_variables():
    #         return 0.
    #     pos = [variable in v.extract_variables() for v in self.params] #FIX self.params?
    #     g = 0
    #     for idx in range(len(pos)):
    #         if not pos[idx]:
    #             continue
    #         if self.is_diagonal:
    #             exct = self.ket[idx]
    #         else:
    #             p = self.bra + self.ket
    #             exct = p[idx]
    #         g +=apply_phase(braket=deepcopy(self),exct=exct,idx=idx,variable=variable,ket=True,p0sign=True) 
    #         # g +=apply_phase(braket=deepcopy(self),exct=exct,idx=idx,variable=variable,ket=True,p0sign=False) #wfn always real for tcc #FIX Same for fss?
    #         g +=apply_phase(braket=deepcopy(self),exct=exct,idx=idx,variable=variable,ket=False,p0sign=True)
    #         # g +=apply_phase(braket=deepcopy(self),exct=exct,idx=idx,variable=variable,ket=False,p0sign=False) #wfn always real for tcc
    # #     return 0.5*g

    @property
    def bra(self) -> FCircuit:
        """
        Excitation operators applied to the bra.
        """
        return self._bra
    
    @bra.setter
    def bra(self, bra:FCircuit):
        '''
        Expected FCircuit.
        '''
        assert isinstance(bra,FCircuit), f"FCircuit expected, received {type(bra).__name__}"
        if bra.initial_state is not None:
            self.init_state_bra = bra.initial_state
        bra = bra.to_upthendown(self.molecule.norb)
        self._bra = bra
    
    @property
    def ket(self) -> FCircuit:
        """
        Excitation operators applied to the ket.
        """
        return self._ket
    
    @ket.setter
    def ket(self, ket:FCircuit):
        '''
        Expected FCircuit
        '''

        assert isinstance(ket,FCircuit), f"FCircuit expected, received {type(ket).__name__}"
        if ket.initial_state is not None:
            self.init_state_ket = ket.initial_state
        ket = ket.to_upthendown(self.molecule.norb)
        self._ket = ket

    @property
    def variables_bra(self) -> List[Variable]:
        """Tequila Circuit Bra variables."""
        return self.bra.variables

    @property
    def variables_ket(self) -> List[Variable]:
        """Tequila circuit Ket parameters."""
        return self.ket.variables

    @property
    def variables(self) -> List[Variable]:
        """Tequila circuit variables."""
        return self.variables_bra+self.variables_ket

    @property
    def init_state_bra(self) -> QubitWaveFunction:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.__civect_to_qwvf(self._init_state_bra)
    
    @property
    def init_state_ket(self) -> QubitWaveFunction:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        
        return self.__civect_to_qwvf(self._init_state_ket)
    
    @property
    def init_state(self) -> Tuple[QubitWaveFunction,QubitWaveFunction]:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.init_state_bra, self.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra:QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state_bra)
    
    @init_state_ket.setter
    def init_state_ket(self, init_state_ket:QubitWaveFunction):
        self._init_state_ket  = self.__qwvf_to_civect(init_state_ket)

    @init_state.setter
    def init_state(self, init_state:QubitWaveFunction):
        self.init_state_bra  = init_state
        self.init_state_ket  = init_state

    def __str__(self):
        res = ''
        if self.is_diagonal:
            res += f"{self._name} with indices: {self.ket.extract_indices()} with variables {self.params_ket}"
        else:
            res += f"{self._name} with Bra= {self.bra.extract_indices()} with variables {self.params_bra}\n"
            res += f"{len(self._name)*' '} with Ket= {self.ket.extract_indices()} with variables {self.params_ket}"
        return res

    def __repr__(self):
        return self.__str__()
    
    @property
    def is_diagonal(self): 
        if self.bra is None:
            return True
        if self.ket is None:
            return True
        if self.bra is not None and self.bra == self.ket:
            return True
        return False

    @property
    def U(self):
        'Dummy function to work with tequila Objectives'
        if self.is_diagonal:
            return self.ket 
        else:
            return [self.bra,self.ket]
    
    def count_measurements(self)->int:
        mes = 0
        if self.molecule.one_body_integrals is not None:
            mes += prod(self.molecule.one_body_integrals)
        if self.BK.int2e is not None:
            mes += prod(self.molecule.two_body_integrals)
        if mes:
            return mes
        else:
            return len(ffsim.hartree_fock_state(self.molecule.norb,self.molecule.nelec) )
    
    def __civect_to_qwvf(self, civect:ndarray) -> QubitWaveFunction:
        wvf = QubitWaveFunction(n_qubits=2*self.molecule.norb).to_array()
        for i,coeff in enumerate(self._cistrings):
            wvf[i] = coeff
        return QubitWaveFunction.from_array(array=wvf,numbering=BitNumbering.LSB)
        
    def __qwvf_to_civect(self, wvf:QubitWaveFunction) -> ndarray:
        wvf.n_qubits = 2*self.molecule.norb
        wvf = wvf.to_array(out_numbering=BitNumbering.LSB)
        return array([wvf[i] for i in self._cistrings])

    def build_operator(self,operator:Union[str,FermionOperator,QubitHamiltonian]=None)->Union[None,Callable,ffsim.FermionOperator]:
        '''
        Build the expectation value operator. 
        Even if it is accepted a QubitHamiltonian, we disencorage its use here for fermionic states limitations.
        It will be applied to the ci_vector and kept these results which doesn't leave the ci_vector.
        '''

        def from_string(operator:str) -> Union[ffsim.FermionOperator,LinearOperator]:
            if operator.upper()=="I":
                return LinearOperator(shape=(len(self._cistrings),len(self._cistrings)), matvec=lambda x:x, rmatvec=lambda x:x, dtype=complex128)
            elif operator.upper() == "H":
                op = ffsim.fermion_operator(self.molecule.hamiltonian)
                op.simplify()
                return op
            else:
                raise TequilaException(f"No operator str {operator} supported on FFSim BraKet")
        
        if isinstance(operator,str):
            operator = from_string(operator)
        elif isinstance(operator,FermionOperator):
            pass
        elif isinstance(operator,QubitHamiltonian):
            def f(v):
                wvf = self.__civect_to_qwvf(v)
                wvf.apply_qubitoperator(operator)
                return self.__qwvf_to_civect(wvf)
            if operator.is_hermitian():
                operator = LinearOperator(shape=(len(self._cistrings),len(self._cistrings)),matvec=f,rmatvec=f,n_qubits=2*self.molecule.norb)
            else:
                operator = LinearOperator(shape=(len(self._cistrings),len(self._cistrings)),matvec=f,n_qubits=2*self.molecule.norb)
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")
        
        return operator
        

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x
