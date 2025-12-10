import tencirchem as tcc
from tencirchem.static.ci_utils import get_ci_strings
from sunrise.expval.tcc_engine.braket import EXPVAL
from ..fermionic_operations.circuit import FCircuit
from tequila import TequilaException,Molecule,QubitWaveFunction,simulate,Variable,Objective,assign_variable,QubitHamiltonian
from tequila import grad as tq_grad
from tequila.objective.objective import Variables,FixedVariable
from tequila.quantumchemistry.chemistry_tools import NBodyTensor
from tequila.quantumchemistry import qc_base
from tequila.utils.bitstrings import BitString, BitNumbering
from numbers import Number
from numpy import ceil,argwhere,pi,prod,eye,zeros,isclose,allclose
from pyscf.gto import Mole
from pyscf.scf import RHF
from sunrise.expval.pyscf_molecule import from_tequila
from copy import deepcopy
from typing import Union,List,Tuple,Callable
from collections import defaultdict
from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.transforms.opconversions.term_reordering import reorder
from openfermion.utils.indexing import up_then_down

class TCCBraket:
    def __init__(self,bra:Union[FCircuit,None]=None,ket:Union[FCircuit,None]=None,operator:Union[str,FermionOperator,List[FermionOperator]]=None,backend_kwargs:dict={},*args,**kwargs):
        self.operator = None
        if 'engine' in backend_kwargs:
            engine = backend_kwargs['engine']
            backend_kwargs.pop('engine')
        else: engine = 'pyscf'
        if 'backend' in  backend_kwargs:
            tcc.set_backend(backend_kwargs['backend'])
            backend_kwargs.pop('backend')
        else: tcc.set_backend('numpy')
        if 'dtype' in backend_kwargs:
            tcc.set_dtype(backend_kwargs['dtype'])
            backend_kwargs.pop('dtype')
        
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
                aslst = [i.idx_total for i in molecule.integral_manager.active_orbitals]
                active_space = (molecule.n_electrons,molecule.n_orbitals)
                if allclose(mo_coeff,eye(len(mo_coeff))): #idea when initialized the molecule from integrals, the mo_coeff are setted to identity and the provided integral are saved on the place of the Atomic Integrals
                    e_core,int1e,int2e = molecule.get_integrals()
                    int2e = int2e.reorder('c').elems
                    self.BK:EXPVAL = EXPVAL.from_integral(int1e=int1e, int2e=int2e,n_elec=molecule.n_electrons, e_core=e_core,mo_coeff=mo_coeff,init_method="zeros",engine=engine,run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,**backend_kwargs)
                else:
                    molecule = from_tequila(molecule)
                    self.BK:EXPVAL = EXPVAL(mol=molecule,run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",aslst=aslst,active_space=active_space,engine=engine,mo_coeff=mo_coeff,**backend_kwargs)
            elif isinstance(molecule,Mole):
                mf = RHF(mol=molecule)
                mo_coeff =mf.mo_coeff
                aslst = [*range(molecule.nao_nr_range)]
                active_space = (molecule.nelectron,molecule.nao_nr)
                self.BK:EXPVAL = EXPVAL(mol=molecule,run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",aslst=aslst,active_space=active_space,engine=engine,mo_coeff=mo_coeff,**backend_kwargs)
        elif 'integral_manager' in kwargs and 'parameters' in kwargs:
            integral = kwargs['integral_manager']
            params = kwargs['parameters']
            kwargs.pop('integral_manager')
            kwargs.pop('parameters')
            mo_coeff = integral.orbital_coefficients
            molecule = Molecule(parameters=params,integral_manager=integral)
            aslst = [i.idx_total for i in integral.active_orbitals]
            active_space = (molecule.n_electrons,molecule.n_orbitals)
            molecule = from_tequila(molecule)
            self.BK:EXPVAL = EXPVAL(mol=molecule,run_hf= run_hf, aslst=aslst,active_space=active_space,run_mp2= False, run_ccsd= False, run_fci= False,init_method="zeros",engine=engine,mo_coeff=mo_coeff,**backend_kwargs)
        else:
            int1e = None
            int2e = None
            e_core = None
            mo_coeff = None
            ovlp = None
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
            # if 'mo_coeff' in kwargs:
            #     mo_coeff = kwargs['mo_coeff']
            #     kwargs.pop('mo_coeff')
            # elif 'orbital_coefficients' in kwargs:
            #     mo_coeff = kwargs['orbital_coefficients']
            #     kwargs.pop('orbital_coefficients')
            mo_coeff = eye(len(int1e)) #IDEA The intengrals refer to the MO, so the active space stuff is kept out of the braket 
            if 'ovlp' in kwargs:
                ovlp = kwargs['ovlp']
                kwargs.pop('ovlp')
            elif 'overlap_integrals' in kwargs:
                ovlp = kwargs['overlap_integrals']
                kwargs.pop('overlap_integrals')
            elif 's' in kwargs:
                ovlp = kwargs['s']
                kwargs.pop('s')
            if 'n_elec' in kwargs:
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif 'n_electrons' in kwargs: 
                n_elec=kwargs['n_elec']
                kwargs.pop('n_elec')
            elif ket is not None and ket.init_state is not None:
                if isinstance(ket.initial_state._state,dict):
                    n_elec = bin([*ket.initial_state._state.keys()][0])[2:].count('1')
                else:
                    n_elec = bin(argwhere(ket.init_state._state>1.e-6)[0][0])[2:].count('1')
            else:
                raise TequilaException("No manner of defining the amount of electrons provided")
            if all([i is not None for i in[int2e,int1e,mo_coeff]]):
                if isinstance(int2e,NBodyTensor):
                    int2e = int2e.reorder('chem').elems
                self.BK:EXPVAL = EXPVAL.from_integral(int1e=int1e, int2e=int2e,n_elec= n_elec, e_core=e_core,ovlp=ovlp,mo_coeff=mo_coeff,init_method="zeros",engine=engine,run_hf= run_hf, run_mp2= False, run_ccsd= False, run_fci= False,**backend_kwargs)
            else:
                raise TequilaException('Not enough molecular data provided')
        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra 
        self.opt_res = defaultdict(None)
        if 'name' in kwargs:
            self._name = kwargs['name']
        else: self._name = 'Expectation Value' if self.is_diagonal else "Transition Value"
        if isinstance(operator,str) and operator == 'I':
            self._name = 'Transition Element'
        if operator is not None:
            self.operator = self.build_operator(operator)
        
    def minimize(self,**kwargs)->float:
        if 'init_guess_bra' in kwargs:
            self.init_guess_bra = kwargs['init_guess_bra']
        if "init_guess_ket" in kwargs:
            self.init_guess_ket = kwargs['init_guess_ket']
        if "init_guess" in kwargs:
            self.init_state = kwargs["init_guess"]
        e = self.BK.kernel()
        if self.BK.opt_res is not None:
            self.opt_res = deepcopy(self.BK.opt_res)
            self.opt_res.x = [-2*i for i in self.opt_res.x] #translating to tq
            return self.BK.opt_res.e
        else: 
            self.opt_res['e'] = e
            return e

    def __call__(self, variables:Union[list,dict]={}, *args, **kwargs) -> float:
        return self.simulate(variables=variables)

    def simulate(self,variables:Union[list,dict]=None)->float:
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
            return self.BK.expval(hamiltonian=self.operator)
        return self.BK.expval(angles=[-0.5*i for i in variables],hamiltonian=self.operator)

    def extract_variables(self) -> list:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        v = []
        for d in self.params:
            if isinstance(d,FixedVariable):
                continue
            elif hasattr(d,'extract_variables'):
                v.extend(d.extract_variables())
            else:
                v.extend(d)
        unique = []
        for i in v:
            if i not in unique:
                unique.append(i)
        return unique

    def grad(self,variable:Variable = None)->Objective:
        def apply_phase(braket: TCCBraket,exct:List[Tuple[int]],variable,ket:bool=True,p0sign:bool=True)->Objective:
            '''
            braket: TCC object to modify
            exct: Excitation indices on the tequila format [(0,2),(1,3),...] to which 
                  apply the phase shift.
            ket: If true it will be applied the phase on the ket side, bra otherwise
            posing: it True: +pi, False: -Pi, correspond to the U0(\pm) not the actual sign implementation
            '''
            p0 = []
            s = {True:+1,False:-1} 

            for idx in exct:
                p0.append((idx[0],idx[0]))
                p0.append((idx[1],idx[1]))
            braket._name = 'Gradient'
            if ket:
                if braket.is_diagonal: 
                    k = deepcopy(braket.ket)
                    v = deepcopy(braket.params_ket)
                    braket.bra = deepcopy(k)
                    braket.variables_bra = deepcopy(v)
                    idx = k.index(exct)
                    ph = tq_grad(v[idx],variable) if not isinstance(v[idx],Union[FixedVariable,Number,Variable]) else 1
                    v[idx] +=  s[ket]*pi/2 
                    for p in reversed(p0):
                        k.insert(idx,[p])
                        v.insert(idx,assign_variable(s[not p0sign]*pi))
                    braket.ket = k
                    braket.variables_ket = v
                else:
                    k = deepcopy(braket.ket)
                    v = deepcopy(braket.params_ket)
                    if exct not in k:
                        return 0.
                    idx = k.index(exct)
                    ph = tq_grad(1*v[idx],variable) if not isinstance(v[idx],Union[FixedVariable,Number,Variable]) else 1
                    v[idx] +=  s[ket]*pi/2
                    for p in reversed(p0):
                        k.insert(idx,[p])
                        v.insert(idx,assign_variable(s[not p0sign]*pi)) 
                    braket.ket = k
                    braket.variables_ket = v
            else: 
                if braket.is_diagonal: 
                    k = deepcopy(braket.ket)
                    v = deepcopy(braket.params_ket)
                    idx = k.index(exct)
                    ph = tq_grad(v[idx],variable) if not isinstance(v[idx],Union[FixedVariable,Number,Variable]) else 1
                    v[idx] +=  s[ket]*pi/2 
                    for p in reversed(p0):
                        k.insert(idx,[p])
                        v.insert(idx,assign_variable(s[not p0sign]*pi)) 
                    braket.bra = k
                    braket.variables_bra = v
                else:
                    k = deepcopy(braket.bra)
                    v = deepcopy(braket.params_bra)
                    if exct not in k:
                        return 0.
                    idx = k.index(exct)
                    ph = tq_grad(v[idx],variable) if not isinstance(v[idx],Union[FixedVariable,Number,Variable]) else 1
                    v[idx] +=  s[ket]*pi/2
                    for p in reversed(p0):
                        k.insert(idx,[p])
                        v.insert(idx,assign_variable(s[not p0sign]*pi)) 
                    braket.bra = k
                    braket.variables_bra = v
            return s[ket]*s[(len(p0)//2)%2]*s[p0sign]*ph*Objective([braket]) #TODO: Check this correct
        if variable is None:
            # None means that all components are created
            variables = self.extract_variables()
            result = {}

            if len(variables) == 0:
                raise TequilaException("Error in gradient: Objective has no variables")

            for k in variables:
                assert k is not None
                result[k] = self.grad(k)
            return result
        else:
            variable = assign_variable(variable)
        if variable not in self.extract_variables():
            return 0.
        if 'civector' in self.BK.engine:
            try:
                _,grad = self.BK.expval_and_grad(angles=variable)
                return grad
            except: raise TequilaException("For civector engine it doesn't work out P0 approach for gradient, try better the TCCBraket.minimize() or change to other engine")
        ex_ops = self.param_to_ex_ops[variable]
        g = 0
        for exct in ex_ops:
            g +=apply_phase(braket=deepcopy(self),exct=exct,variable=variable,ket=True,p0sign=True) 
            # g +=apply_phase(braket=deepcopy(self),exct=exct,variable=variable,ket=True,p0sign=False) #wfn always real for tcc 
            g +=apply_phase(braket=deepcopy(self),exct=exct,variable=variable,ket=False,p0sign=True)
            # g +=apply_phase(braket=deepcopy(self),exct=exct,variable=variable,ket=False,p0sign=False) #wfn always real for tcc
        return 0.5*g

    @property
    def energy(self)->float:
        if isinstance(self.opt_res,dict):
            return self.opt_res['e']
        else:
            return self.opt_res.e

    @property
    def bra(self):
        """
        Excitation operators applied to the bra.
        """
        return indices_tcc_to_tq(self.BK.ex_ops_bra)  
    
    @bra.setter
    def bra(self, bra:Union[List[List[Tuple[int]]],FCircuit]):
        '''
        Expected FCircuit or indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        if isinstance(bra,FCircuit):
            if bra.initial_state is not None:
                ini_bra = init_state_from_wavefunction(bra.initial_state)
                for i in ini_bra:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in ini_bra:
                    init[i[0]] = i[1]
                self.init_state_bra = init
            self.variables_bra = bra.variables
            bra = bra.to_upthendown(len(self.BK.aslst))
            self.BK.ex_ops_bra,_,_ =indices_tq_to_tcc(bra.extract_indices())
        else:
            bra,params_bra,_ = indices_tq_to_tcc(bra)
            if self.variables_bra is None:
                self.variables_bra = params_bra
            self.BK.ex_ops_bra = bra
    
    @property
    def ket(self):
        """
        Excitation operators applied to the ket.
        """
        return indices_tcc_to_tq(self.BK.ex_ops_ket)
    
    @ket.setter
    def ket(self, ket:Union[List[List[Tuple[int]]],FCircuit]):
        '''
        Expected FCircuit or indices in [[(0,2),(1,3),...],[(a,b),(c,d),...],...] Format (Upthendown order)
        '''
        if isinstance(ket,FCircuit):
            if ket.initial_state is not None:
                ini_ket = init_state_from_wavefunction(ket.initial_state)
                for i in ini_ket:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in ini_ket:
                    init[i[0]] = i[1]
                self.init_state_ket = init
            self.variables_ket = ket.variables
            ket = ket.to_upthendown(len(self.BK.aslst))
            self.BK.ex_ops_ket,_,_ =indices_tq_to_tcc(ket.extract_indices())
        else:
            ket,params_ket,_ = indices_tq_to_tcc(ket)
            if self.variables_ket is None:
                self.variables_ket = params_ket
            self.BK.ex_ops_ket = ket

    @property
    def variables_bra(self) -> dict:
        """Tequila Circuit Bra variables."""
        bar:dict = deepcopy(self.BK.var_to_param_bra)
        if bar is not None:
            for i in bar.keys(): #Here to tequila
                if isinstance(bar[i],(Number,FixedVariable)):
                    bar[i] = assign_variable(-2*(bar[i]))
        return bar 
    
    @property
    def params_bra(self):
        """TCC Circuit Bra parameters (values after minimization or variables name)."""
        if self.variables_bra is None: return []
        d = {v: k for k, v in self.variables_bra.items()}
        return [assign_variable(-2*v) if isinstance(v,(Number,FixedVariable)) else d[assign_variable(v)] for v in self.BK.params_bra]
    
    @variables_bra.setter
    def variables_bra(self, variables_bra):
        '''
        See TCC variables
        '''
        for idx,i in enumerate(variables_bra):
            if isinstance(assign_variable(i),(Number, FixedVariable)):
                variables_bra[idx]=assign_variable(-0.5*(i%(2*pi)))
        self.BK.params_bra = variables_bra

    @property
    def variables_ket(self):
        """Tequila circuit Ket parameters."""
        bar = deepcopy(self.BK.var_to_param_ket)
        if bar is not None:
            for i in bar.keys(): #to tequila
                if isinstance(bar[i],(Number,FixedVariable)):
                    bar[i] = assign_variable(-2*(bar[i]))
        return bar
    
    @property
    def params_ket(self):
        """TCC Circuit Ket parameters (values after minimization or variables name)."""
        d = {v: k for k, v in self.variables_ket.items()}
        return [assign_variable(-2*v) if isinstance(v,(Number,FixedVariable)) else d[assign_variable(v)] for v in self.BK.params_ket]
    
    @variables_ket.setter
    def variables_ket(self, variables_ket):
        '''
        See TCC variables
        '''
        for idx,i in enumerate(variables_ket):
            if isinstance(assign_variable(i),(Number, FixedVariable)):
                variables_ket[idx]=assign_variable(-0.5*(i%(2*pi)))
        self.BK.params_ket = variables_ket
    
    @property
    def variables(self) -> dict:
        """Tequila circuit variables."""
        bar:dict = deepcopy(self.BK.var_to_param)
        if bar is not None:
            for i in bar.keys(): #Here to tequila 
                if isinstance(bar[i],(Number, FixedVariable)):
                    bar[i] = assign_variable(-2*bar[i])
        return bar 
    
    @property
    def params(self):
        """TCC parameters (values after minimization or variables name)."""
        k = self.params_ket
        if self.BK.params_bra is None:
            return k
        return self.params_bra+k
    
    @variables.setter
    def variables(self, variables):
        """Tequila circuit variables."""
        for idx,i in enumerate(variables):
            if isinstance(assign_variable(i),(Number, FixedVariable)):
                variables[idx]=assign_variable(-0.5*(i%(2*pi)))
        self.BK.params = variables

    @property
    def init_state_bra(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_bra
    
    @property
    def init_state_ket(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_ket
    
    @property
    def init_state(self):
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        BK.get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self.BK.init_state_bra ,self.BK.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self.BK.init_state_bra = init_state_bra
    
    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        self.BK.init_state_ket  = init_state_ket

    @init_state.setter
    def init_state(self, init_state):
        self.BK.init_state_bra  = init_state
        self.BK.init_state_ket  = init_state

    @property
    def init_guess(self):
        """
        Initial Angle Value for minimization, default all 0.
        """
        return [-2*i for i in self.BK.init_guess]
    
    @init_guess.setter
    def init_guess(self, init_guess):
        self.BK.init_guess = [-i/2 for i in init_guess]

    @property 
    def param_to_ex_ops_bra(self):
        if self.params_bra is None: return {}
        d = defaultdict(list)
        for i, j in enumerate(self.params_bra):
            if hasattr(j,'extract_variables'):
                for v in j.extract_variables():
                    d[v].append(self.bra[i])
            else:
                d[j].append(self.bra[i])
        return d
    
    
    @property
    def param_to_ex_ops_ket(self):
        d = defaultdict(list)
        for i, j in enumerate(self.params_ket):
            if hasattr(j,'extract_variables'):
                for v in j.extract_variables():
                    d[v].append(self.ket[i])
            else:
                d[j].append(self.ket[i])
        return d
    
    @property
    def param_to_ex_ops(self):
        db = self.param_to_ex_ops_bra if self.params_bra is not None else {}
        dk = self.param_to_ex_ops_ket
        db.update(dk)
        return db

    def __str__(self):
        res = ''
        if self.is_diagonal:
            res += f"{self._name} with indices: {self.ket} with variables {self.params_ket}"
        else:
            res += f"{self._name} with Bra= {self.bra} with variables {self.params_bra}\n"
            res += f"{len(self._name)*' '} with Ket= {self.ket} with variables {self.params_ket}"
        return res

    def __repr__(self):
        return self.__str__()
    
    @property
    def is_diagonal(self):
        return self.BK.is_diagonal()

    @property
    def U(self):
        'Dummy function to work with tequila Objectives'
        if self.is_diagonal:
            return self.ket 
        else:
            return [self.bra,self.ket]
    
    def count_measurements(self)->int:
        mes = 0
        if self.BK.int1e is not None:
            mes += prod(self.BK.int1e.shape)
        if self.BK.int2e is not None:
            mes += prod(self.BK.int2e.shape)
        if mes:
            return mes
        else:
            return len(self.BK.civector(params=[.0 for _ in range(self.BK.n_variables_ket)]))

    def build_operator(self,operator:Union[str,FermionOperator,QubitHamiltonian]=None)->Union[None,Callable]:
        '''
        Build the expectation value operator. 
        Even if it is accepted a QubitHamiltonian, we disencorage its use here since TCC works on fermionic states.
        It will be applied to the ci_vector and kept these results which doesn't leave the ci_vector space with READ Coeff.
        '''
        def from_string(operator:str):
            if operator.upper()=="I":
                nmo = len(self.BK.aslst)
                self.BK.hamiltonian = None
                self.BK.int1e = zeros((nmo,nmo))
                int2e = zeros((nmo,nmo,nmo,nmo))
                for i in range(nmo):
                    int2e[i,i,i,i] = 1#0.5
                self.BK.int2e = int2e
                self.BK.e_core = 0.
                self.BK.hamiltonian_lib = {}
            elif operator.upper() == "H":
                pass
            else:
                raise TequilaException(f"No operator str {operator} supported on TCC BraKet")
        
        if operator is None:
            return None
        if isinstance(operator,str):
            from_string(operator)
            return None
        elif isinstance(operator,FermionOperator):
            operator = reorder(operator=operator,order_function=up_then_down,num_modes=2*len(self.BK.aslst))
            operator = jordan_wigner(operator)
            operator.compress()
            operator = QubitHamiltonian.from_openfermion(operator)
            self.BK.e_core = 0
        elif isinstance(operator,QubitHamiltonian):
            self.BK.e_core = 0
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")
        ci_vec = get_ci_strings(n_elec_s=self.BK.n_elec,n_qubits=2*len(self.BK.aslst),mode='fermion')
        self.BK.e_core = 0
        def callable_operator(ci_vect:List[int],operator:QubitHamiltonian,n_qubits:int, ket:List[float])->List[float]:
            hket = zeros(len(ci_vect))
            wfv = QubitWaveFunction(n_qubits=n_qubits,numbering=BitNumbering.LSB,dense=True)
            for idx,val in enumerate(ket):
                wfv._state[ci_vect[idx]] = val
            hket = wfv.apply_qubitoperator(operator)
            hket = hket.to_array(out_numbering=BitNumbering.LSB)
            return [hket[i].real for i in ci_vect]
        return lambda ket: callable_operator(ci_vect=ci_vec,operator=operator,n_qubits=2*len(self.BK.aslst), ket=ket)

def init_state_from_wavefunction(wvf:QubitWaveFunction):
    if not isinstance(wvf._state,dict):
        return init_state_from_array(wvf=wvf)
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:]
        if len(vec) < wvf.n_qubits:
            vec = '0'*(wvf.n_qubits-len(vec))+vec
        init_state.append([vec,wvf._state[i].real])#tcc automatically does this, but with an anoying message everytime
    return init_state

def init_state_from_array(wvf:QubitWaveFunction,tol=1e-6):
    '''
    Expected Initial State in UptheDown
    '''
    if isinstance(wvf._state,dict):
        return init_state_from_wavefunction(wvf)
    init_state = []
    nq = wvf.n_qubits
    nq = int(2* (ceil(nq/2))) #if HCB may be odd amount of qubits
    for i,idx in enumerate(wvf._state):
        vec = BitString.from_int(i)
        vec.nbits = nq
        vec = vec.binary
        if abs(idx) > tol:
            init_state.append([vec,idx.real]) #tcc automatically does this, but with an anoying message everytime
    return init_state

def indices_tq_to_tcc(indices:List[List[Tuple[int]]]=None):
    '''
    Expected indices like [[(0,1),(n_mo+0,n_mo+1),...],[(a,b),(c,d),...],...] (in upthendown)
    Returned [(...,n_mo+1,1,0,n_mo+0,...),(...,d,b,a,c,...)]
    '''
    if indices is None:
        return None,None,None
    if not len(indices):
        return [],[],[]
    assert isinstance(indices,(list,tuple))
    if isinstance(indices[0],Number):
        indices = [indices]
    ex_ops = []
    params = []
    param_ids = []
    for exct in indices:
        exc = []
        params.append(str(exct))
        param_ids.append(len(param_ids))
        for idx in exct:
            exc.append(idx[0])
            exc.insert(0,idx[1])
        ex_ops.append(tuple(exc))
    return ex_ops,params,param_ids

def indices_tcc_to_tq(indices:List[List[Tuple[int]]]=None)->List[List[Tuple[int]]]:
    '''
    Expected indices like [(...,n_mo+1,1,0,n_mo+0,...),(...,d,b,a,c,...)]
    Returned [[(0,1),(n_mo+0,n_mo+1),...],[(a,b),(c,d),...],...] (in upthendown)
    '''
    if indices is None:
        return None
    assert isinstance(indices,(list,tuple))
    if not len(indices):
        return []
    if isinstance(indices[0],Number):
        indices = [indices]
    ex_ops = []
    for exct in indices:
        exc = []
        for i in range(len(exct)//2):
            exc.append((exct[-i-1],exct[i]))
        ex_ops.append(exc[::-1])
    return ex_ops

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x