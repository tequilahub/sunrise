from __future__ import annotations
from tequila.circuit._gates_impl import assign_variable
from tequila.circuit.gates import QubitExcitationImpl,X,Phase
from tequila import Variable,BitString
from tequila.quantumchemistry.chemistry_tools import FermionicGateImpl
from tequila.quantumchemistry.qc_base import QuantumChemistryBase
from tequila.utils.exceptions import TequilaException, TequilaWarning
from tequila import assign_variable,QCircuit,QubitWaveFunction
from typing import List,Union,Iterable,Optional,Callable
import copy
from collections import defaultdict
from numpy import ndarray,where,isclose,pi,array
import warnings
import numbers
from copy import deepcopy
import sunrise
from tequila.circuit.gates import X

class FCircuit:
    """
    Fundamental class representing an abstract circuit.

    Attributes
    ----------
    depth:
        returns the gate depth of the circuit.
    gates:
        returns the gates in the circuit, as a list.
    moments:
        returns the circuit as a list of Moment objects.
    n_qubits:
        the number of qubits on which the circuit operates.
    numbering:
        returns the numbering convention use by tequila circuits.
    qubits:
        returns a list of qubits acted upon by the circuit.
    initial_state:
        returns QubitWaveFunction which defines initial state. It must be on UPTHENDOWN
        

    Methods
    -------
    make_parameter_map:


    """

    def __init__(self, gates:list=None, initial_state:Union[QCircuit,QubitWaveFunction,str,int]=None, parameter_map=None):
        """
        init
        Parameters
        ----------
        gates:
            (Default value = None)
            the gates to include in the circuit.
        parameter_map:
            (Default value = None)
            mapping to indicate where in the circuit certain parameters appear.
        """
        self._n_qubits = None
        self._min_n_qubits = 0
        self._initial_state = None
        if gates is None:
            self._gates = []
        else:
            self._gates = list(gates)

        if parameter_map is None:
            self._parameter_map = self.make_parameter_map()
        else:
            self._parameter_map = parameter_map
        self.initial_state = initial_state 
        self.verify()

    def export_to(self, filename: str,*args, **kwargs):
        """
        Export to png, pdf, qpic, tex with qpic backend
        Parameters
        """
        U = deepcopy(self)
        if 'n_orb' in kwargs:
            n_orb = kwargs['n_orb']
        else: n_orb = self.n_qubits//2
        U.to_udud(n_orb)
        if 'n_qubits_is_double' in kwargs:
            n_qubits_is_double = kwargs['n_qubits_is_double']
            kwargs.pop('n_qubits_is_double')
        else: n_qubits_is_double = False
        gU = sunrise.graphical.GraphicalCircuit.from_circuit(U=U, n_qubits_is_double=n_qubits_is_double)
        gU.export_to(filename=filename,*args,**kwargs)
        

    @property
    def initial_state(self)->QubitWaveFunction:
        return self._initial_state

    @initial_state.setter
    def initial_state(self,initial_state=None):
        '''
        IT MUST BE UPTHENDOWN
        '''
        if initial_state is None or isinstance(initial_state,QubitWaveFunction):
            pass
        elif isinstance(initial_state,Union[QCircuit,FCircuit]):
            initial_state = sunrise.simulate(initial_state,variables={})
        elif isinstance(initial_state,str):
            initial_state = QubitWaveFunction.from_string(initial_state)
        elif isinstance(initial_state,(list,ndarray)):
            initial_state = QubitWaveFunction.from_array(initial_state)
        else:
            try:
                initial_state = QubitWaveFunction.convert_from(val=initial_state,n_qubits=self.n_qubits)
            except:
                raise TequilaException(f'Init_state format not recognized, provided {type(initial_state).__name__}')
        if initial_state is not None:
            self.n_qubits = max(self.n_qubits,initial_state.n_qubits)
            initial_state._n_qubits = self.n_qubits
            self._initial_state = initial_state
            self._verify_state()
    @property
    def depth(self):
        """
        gate depth of the abstract circuit.
        Returns
        -------
        int: the depth.

        """
        return len(self.gates)

    @property
    def gates(self)->List[sunrise.fermionic_excitation.fgateimpl.FGateImpl]:
        if self._gates is None:
            return []
        else:
            return self._gates

    @gates.setter
    def gates(self, other):
        self._gates = list(other)

    @property
    def qubits(self):
        accumulate = []
        for g in self.gates:
            accumulate += list(g.qubits)
        if self._initial_state is not None:
            accumulate.extend([*range(self._initial_state.n_qubits)])
        return sorted(list(set(accumulate)))

    @property
    def n_qubits(self):
        idx = max(self.max_qubit() + 1, self._min_n_qubits)
        wvf = self._initial_state.n_qubits if self._initial_state is not None else 0
        return max(idx,wvf)

    @n_qubits.setter
    def n_qubits(self, other):
        self._min_n_qubits = other
        if other < self.max_qubit() + 1:
            raise TequilaException(
                "You are trying to set n_qubits to "
                + str(other)
                + " but your circuit needs at least: "
                + str(self.max_qubit() + 1)
            )
        return self

    @property
    def variables(self)->list:
        v = []
        for gate in self.gates:
            if gate._name == 'UR':
                v.append(gate.variables)
            v.append(gate.variables)
        return v

    def make_parameter_map(self) -> dict:
        """
        Returns
        -------
            ParameterMap of the circuit: A dictionary with
            keys: variables in the circuit
            values: list of all gates and their positions in the circuit
            e.g. result[Variable("a")] = [(3, Rx), (5, Ry), ...]
        """
        parameter_map = defaultdict(list)
        for idx, gate in enumerate(self.gates):
            if gate.is_parameterized():
                variables = gate.extract_variables()
                for variable in variables:
                    parameter_map[variable] += [(idx, gate)]
        return parameter_map

    def is_primitive(self):
        """
        Check if this is a single gate wrapped in this structure
        :return: True if the circuit is just a single gate
        """
        return len(self.gates) == 1

    def replace_gates(self, positions: list, circuits: list, replace: list = None):
        """
        Notes
        ----------
        Replace or insert gates at specific positions into the circuit
        at different positions (faster than multiple calls to replace_gate)

        Parameters
        ----------
        positions: list of int:
            the positions at which the gates should be added. Always refer to the positions in the original circuit
        circuits: list or QCircuit:
            the gates to add at the corresponding positions
        replace: list of bool: (Default value: None)
            Default is None which corresponds to all true
            decide if gates shall be replaced or if the new parts shall be inserted without replacement
            if replace[i] = true: gate at position [i] will be replaced by gates[i]
            if replace[i] = false: gates[i] circuit will be inserted at position [i] (beaming before gate previously at position [i])
        Returns
        -------
            new circuit with inserted gates
        """

        assert len(circuits) == len(positions)
        if replace is None:
            replace = [True] * len(circuits)
        else:
            assert len(circuits) == len(replace)

        dataset = zip(positions, circuits, replace)
        dataset = sorted(dataset, key=lambda x: x[0])

        new_gatelist = []
        last_idx = -1

        for idx, circuit, do_replace in dataset:
            # failsafe
            if hasattr(circuit, "gates"):
                gatelist = circuit.gates
            elif isinstance(circuit, Iterable):
                gatelist = circuit
            else:
                gatelist = [circuit]

            new_gatelist += self.gates[last_idx + 1 : idx]
            new_gatelist += gatelist
            if not do_replace:
                new_gatelist.append(self.gates[idx])

            last_idx = idx

        new_gatelist += self.gates[last_idx + 1 :]

        result = QCircuit(gates=new_gatelist)
        result.n_qubits = max(result.n_qubits, self.n_qubits)
        return result

    def insert_gates(self, positions, gates):
        """
        See replace_gates
        """
        return self.replace_gates(positions=positions, circuits=gates, replace=[False] * len(gates))

    def dagger(self):
        """
        Returns
        ------
        QCircuit:
            The adjoint of the circuit
        """
        result = FCircuit()
        for g in reversed(self.gates):
            result += self.wrap_gate(g.dagger())
        return result

    def extract_variables(self) -> list:
        """
        return a list containing all the variable objects contained in any of the gates within the unitary
        including those nested within gates themselves.

        Returns
        -------
        list:
            the variables of the circuit
        """
        return list(self._parameter_map.keys())

    def is_fully_parametrized(self):
        """
        Returns
        -------
        bool:
            whether or not all gates in the circuit are paremtrized
        """
        for gate in self.gates:
            if not gate.is_parameterized():
                return False
            else:
                if hasattr(gate, "parameter"):
                    if not hasattr(gate.parameter, "wrap"):
                        return False
                    else:
                        continue
                else:
                    continue
        return True

    def is_fully_unparametrized(self):
        """
        Returns
        -------
        bool:
            whether or not all gates in the circuit are unparametrized
        """
        for gate in self.gates:
            if not gate.is_parameterized():
                continue
            else:
                if hasattr(gate, "parameter"):
                    if not hasattr(gate.parameter, "wrap"):
                        continue
                    else:
                        return False
                else:
                    return False
        return True

    def is_mixed(self):
        return not (self.is_fully_parametrized() or self.is_fully_unparametrized())

    def extract_indices(self) -> list:
        l = []
        for gate in self.gates:
            l.extend(gate.indices)
        return l

    def max_qubit(self):
        """
        Returns:
        int:
             Highest index of qubits in the circuit
        """
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit)
        if self.initial_state is not None:
            qmax = max(qmax,self.initial_state.n_qubits-1)
        return qmax

    def __iadd__(self, other):
        other = self.wrap_gate(gate=other.gates)

        offset = len(self.gates)
        for k, v in other._parameter_map.items():
            self._parameter_map[k] += [(x[0] + offset, x[1]) for x in v]

        self._gates += other.gates
        self._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        if self._initial_state is None:
            self._initial_state = other._initial_state
        elif other._initial_state is not None and other._initial_state != self._initial_state:
            raise TequilaException(f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}")
        return self

    def __add__(self, other):
        other = self.wrap_gate(other.gates)
        gates = [deepcopy(g) for g in (self.gates + other.gates)]
        result = FCircuit(gates=gates)
        result._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        initial_state = self.initial_state
        if self._initial_state is None:
                initial_state = other._initial_state
        elif other._initial_state is not None and other._initial_state != self._initial_state:
            raise TequilaException(f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}")
        result.initial_state = initial_state
        return result

    def __str__(self):
        result = "Fermionic circuit: \n"
        if self.initial_state is not None:
            result += 'with Initial State: \n'
            result += str(self.initial_state) +'\n'
        if len(self.gates):
            result += 'with gates:\n'
            for g in self.gates:
                result += str(g) + "\n"
        return result

    def __eq__(self, other) -> bool:
        if len(self.gates) != len(other.gates):
            return False
        for i,gate in enumerate(self.gates):
            if gate != other.gates[i]:
                return False
        if len(self.initial_state.to_array()) != len(other.initial_state.to_array()):
            return False
        if self.initial_state.numbering != other.initial_state.numbering:
            return False
        return isclose(self.initial_state.inner(other=other.initial_state),1.,atol=1.e-4)

    def __repr__(self):
        return self.__str__()

    def to_upthendown(self,norb)->FCircuit:
        '''
        Initial State can't be reordered, it must always be on upthendown
        '''
        u = []
        for gate in self.gates:
            g = deepcopy(gate).to_upthendown(norb)
            g.reordered = True
            u.append(g)
        return FCircuit(gates=u,parameter_map=self._parameter_map,initial_state=self.initial_state)

    def to_udud(self,norb)->FCircuit:
        '''
        Initial State can't be reordered, it must always be on upthendown
        '''
        u = []
        for gate in self.gates:
            g = deepcopy(gate).to_udud(norb)
            g.reordered = False
            u.append(g)
        return FCircuit(gates=u,parameter_map=self._parameter_map,initial_state=self.initial_state)

    @classmethod
    def from_Qcircuit(cls,circuit:QCircuit,**kwargs):
        operations = FCircuit()
        reference = QCircuit()
        if 'reordered' in kwargs:
            reordered = kwargs['reordered']
        else: reordered = True 
        begining = True
        if 'variables' in kwargs:
            circuit = circuit.map_variables(kwargs['variables'])
            kwargs.pop("variables")
        for gate in circuit.gates:
            if begining and not hasattr(gate,'_parameter') or isinstance(gate._parameter,numbers.Number):
                reference += gate
            elif isinstance(gate,QubitExcitationImpl): #maybe we can consider other gates but basic implementation for the moment
                if isinstance(gate._parameter,numbers.Number):
                    reference += gate
                elif isinstance(gate,FermionicGateImpl):
                    begining = False 
                    operations += sunrise.gates.FermionicExcitation(indices=gate.indices,variables=gate.parameter,reordered=reordered)
                else:
                    temp = []
                    for i in range(len(gate._target)//2):
                        temp.append((gate._target[2*i],gate._target[2*i+1]))
                    operations += sunrise.gates.FermionicExcitation(indices=temp,variables=gate.parameter,reordered=reordered)
            else:
                raise TequilaException(f'Gate {gate._name}({gate._parameter}) not allowed')
        if not reordered and len(reference.gates):
            raise TequilaException('reordered=False only allowed if any initial_state provided due to backend restrictions')
        return cls(gates=operations._gates,initial_state=reference)

    @classmethod
    def from_edges(cls,edges:Union[list,tuple],label=None,n_orb:int=0, use_units_of_pi=False,ladder=True):
        operations = FCircuit()
        if n_orb is not None:
            include_reference = True
            reference = QCircuit()
        else: 
            include_reference = False
            reference = None
        for edge in edges:
            if include_reference:
                reference += X([edge[0],edge[0]+n_orb])
            previous = edge[0]
            for i in edge[1:]:
                v = Variable(((previous,i),'D',label))
                if use_units_of_pi:
                    angle = angle * pi
                operations += sunrise.gates.UC(i=previous,j=i,variables=v)
                if ladder:
                    previous = i
        return cls(gates=operations._gates,initial_state=reference)

    @staticmethod
    def wrap_gate(gate):
        """
        take a gate and return a qcircuit containing only that gate.
        Parameters
        ----------
        gate: QGateImpl
            the gate to wrap in a circuit.

        Returns
        -------
        QCircuit:
            a one gate circuit.
        """
        if isinstance(gate, FCircuit):
            return gate
        if isinstance(gate, list):
            return FCircuit(gates=gate)
        else:
            return FCircuit(gates=[gate])

    def verify(self) -> bool:
        """
        make sure self is built properly and of the correct types.
        Returns
        -------
        bool:
            whether or not the circuit is properly constructed.

        """
        if not len(self._gates) is None and self.initial_state is None:
            return True
        test = []
        for (k,v,) in self._parameter_map.items():
            test = [self.gates[x[0]] == x[1] for x in v]
            test += [k in self._gates[x[0]].extract_variables() for x in v]
        return all(test)

    def map_qubits(self, qubit_map):
        """

        E.G.  Rx(1)Ry(2) --> Rx(3)Ry(1) with qubit_map = {1:3, 2:1}

        Parameters
        ----------
        qubit_map
            a dictionary which maps old to new qubits

        Returns
        -------
        A new circuit with mapped qubits
        """

        new_gates = [gate.map_qubits(qubit_map) for gate in self.gates]
        # could speed up by applying qubit_map to parameter_map here
        # currently its recreated in the init function
        return FCircuit(gates=new_gates)

    def map_variables(self, variables: dict, *args, **kwargs):
        """

        Parameters
        ----------
        variables
            dictionary with old variable names as keys and new variable names or values as values
        Returns
        -------
        Circuit with changed variables

        """

        variables = {assign_variable(k): assign_variable(v) for k, v in variables.items()}

        # failsafe
        my_variables = self.extract_variables()
        for k, v in variables.items():
            if k not in my_variables:
                warnings.warn(
                    "map_variables: variable {} is not part of circuit with variables {}".format(k, my_variables),
                    TequilaWarning,
                )

        new_gates = [copy.deepcopy(gate).map_variables(variables) for gate in self.gates]

        return FCircuit(gates=new_gates,initial_state=self.initial_state)

    def to_matrix(self, variables=None): #TODO:  Should we?
        pass 

    def add_controls(self, control, inpl: Optional[bool] = False) -> Optional[FCircuit]: #TODO
        pass 
      
    def to_qcircuit(self,molecule:QuantumChemistryBase=None,transformation:Callable=None) -> QCircuit: 
        '''
        Transform from FCircuit to Qcircuit
        :param molecule Any kind of Tequila/Sunrise Molecule with molecule.make_excitation_gate function
        :param transformation Callable function such as transformation(FCircuit) -> QCircuit 
        '''
        U = deepcopy(self)
        if molecule is not None:
            assert molecule.transformation is not None
            res = QCircuit()
            U = U.to_udud(molecule.n_orbitals)
            molecule.transformation.upthendown = True
            if self.initial_state is not None:
                idx = where(array(self._initial_state.to_array())>1.e-6)[0]
                if not len(idx):
                    pass
                elif len(idx)>1:
                    warnings.warn("Don't now how to prepare the circuit initial state, skyped for safety",TequilaWarning)
                else:
                    res += X(target=[i for i in range(len(bin(idx[0])[2:])) if bin(idx[0])[2:][i]=='1'])
            for gate in U.gates:
                if gate.name == 'UR':
                    res += molecule.make_excitation_gate(indices=gate.indices[0],angle=gate.variables)#TODO: Control
                    res += molecule.make_excitation_gate(indices=gate.indices[1],angle=gate.variables)#TODO: Control
                elif gate.name in ['UC', 'FermionicExcitation', 'GenericFermionic']:
                    res += molecule.make_excitation_gate(indices=gate.indices[0],angle=gate.variables)#TODO: Control
                elif gate.name == 'Ph':
                    res +=  Phase(target=(gate.indices[0][0][0]//2 + (gate.indices[0][0][0]%2)*molecule.n_orbitals),angle=gate.variables)  #TODO: Control
                else:
                    raise TequilaException(f'Gate {gate} not idenified')
            return res
        elif transformation is not None:
            return transformation(U)
        else:
            raise TequilaException("No Possible to transform from FCircuit to QCircuit without further information")
    
    def _verify_state(self):
        state = self._initial_state.to_array()
        indices = where(array(state)>1.e-6)[0]
        nozero = [bin(i)[2:] for i in indices]
        ne = nozero[0].count('1')
        if not all([st.count('1')==ne for st in nozero]):
            raise TequilaException('The Initial State is not Particle-conserving')
        if not all([isclose(state[i].imag,0.,atol=1.e-6) for i in indices]):
            raise TequilaException('No Imaginary states for Chemistry States')
    

if __name__ == '__main__':
    import tequila as tq
    import sunrise as sun
    
    U = sun.gates.FermionicExcitation([(0,2),(1,3)],"a")
    U.initial_state = tq.gates.X([0,4])
    U += sun.gates.UC(0,1,"b")
    U = U + sun.gates.UR(1,2,'c')
    U += sun.gates.FermionicExcitation([(1,2)],'d',reordered=True)
    print(U)
    # print(U.make_parameter_map())
    # print(U.max_qubit())
    A = U.to_upthendown(4)
    print(A)
    # print(A.extract_indices())
    print(A.variables)
    # print(A.initial_state)
    print(A.extract_variables())
    # print(A.make_parameter_map())
    E = A.to_udud(4)
    print(E)
    B = FCircuit.from_edges([(0,1),(2,3)])
    print(B.extract_variables())
    print(B.extract_indices())
    print(B)
    C = FCircuit.from_edges([(0,1),(2,3)],n_orb=4)
    print(C)
