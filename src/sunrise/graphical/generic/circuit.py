from typing import List,Union
from copy import deepcopy
from ..core import Gate
from sunrise.graphical import generic
from ..quantum_chemistry import PairCorrelatorGate
from ..quantum_chemistry import SingleExcitation
from ..quantum_chemistry import GenericExcitation
from ..generic.gate import GenericGate
from ..quantum_chemistry import DoubleExcitation
from ..quantum_chemistry import OrbitalRotatorGate
from ...fermionic_operations.circuit import FCircuit
from tequila import QCircuit
from tequila.circuit.gates import X
from typing import Union

class GraphicalCircuit(Gate):
    """
    Basic wrapper class for multiple gates.
    Gates can be added in a similar as with tequila's QCircuit: Circuit += gate
    """

    gates: List[Gate]

    def __init__(self, gates: List[Gate] = []):
        if gates is not None: self.gates = gates

    def __add__(self, other: Gate):
        self.gates.append(other)
        return self
    def __str__(self):
        res = "GraphicalCircuit:\n"
        for gate in self.gates:
            res += gate.__str__() +'\n'
        return res
    def dagger(self) -> "GraphicalCircuit":
        return GraphicalCircuit([gate.dagger() for gate in reversed(self.gates)])

    def _render(self, state, style) -> str:
        output = ""
        sim = self._simplify_GraphicalCircuit()
        for gate in sim.gates:
            output += gate.render(state, style) + " \n"
        return output

    def used_wires(self) -> List[int]:
        wires = set()
        for gate in self.gates:
            wires.update(gate.used_wires())
        return list(wires)

    def construct_circuit(self) -> QCircuit:
        U = QCircuit()
        for gate in self.gates:
            U += gate.construct_circuit()
        return U

    def map_variables(self, variables) -> "GraphicalCircuit":
        gates = []
        for gate in self.gates:
            gates.append(gate.map_variables(variables))
        return GraphicalCircuit(gates)
    def _simplify_GraphicalCircuit(self) -> "GraphicalCircuit":
        cp = deepcopy(self)
        for g in range(len(self.gates)-1)[::-1]:
            if isinstance(self.gates[g],generic.GenericGate) and isinstance(self.gates[g+1],generic.GenericGate):
                cp.gates[g].U += cp.gates[g+1].U
                cp.gates[g].qubits = [*set(cp.gates[g].qubits+cp.gates[g+1].qubits)]
                cp.gates.pop(g+1)
        return cp
        
    @classmethod
    def from_circuit(cls,U:Union[QCircuit,FCircuit,'GraphicalCircuit'], n_qubits_is_double: bool = False, *args, **kwargs) -> 'GraphicalCircuit':
        if isinstance(U,QCircuit):
            return cls.from_qcircuit(U=U,n_qubits_is_double=n_qubits_is_double,*args,**kwargs)
        elif isinstance(U,FCircuit):
            return cls.from_fcircuit(U=U,n_qubits_is_double=n_qubits_is_double,*args,**kwargs)
        elif isinstance(U,'GraphicalCircuit'):
            return cls(gates=U.gates)
        else: raise Exception(f'Type {type(U).__name__} not recognized, expected tq QCircuit or sun FCircuit')
            

    @classmethod
    def from_qcircuit(cls,U:QCircuit, n_qubits_is_double: bool = False, *args, **kwargs) -> 'GraphicalCircuit':
        """
        Converts a tequila QCircuit into a GraphicalCircuit object, which is renderable with `export_qpic`.
        """
        circuit = U._gates
        res = []
        was_ur = False
        for i, gate in enumerate(circuit):
            if gate._name == 'FermionicExcitation':
                index = ()
                for pair in gate.indices:
                    for so in pair:
                        index += (so,)
                if len(index) == 2:
                    if was_ur:
                        was_ur = False
                        continue
                    elif gate != circuit[-1] and circuit[i + 1]._name == 'FermionicExcitation' and len(circuit[i + 1].indices[0]) == 2 and circuit[i + 1].indices[0][0] // 2 == index[0] // 2 and circuit[i + 1].indices[0][1] // 2 == index[1] // 2 and not n_qubits_is_double:  # hope you enjoy this conditional
                        res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                        was_ur = True
                    else:
                        res.append(SingleExcitation(index[0], index[1], angle=gate._parameter,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
                elif len(index) ==4:
                    if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[3] // 2 and not n_qubits_is_double:  ## TODO: Maybe generalized for further excitations
                        res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                    else:
                        res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
                else: res.append(GenericExcitation(indices=index,angle=gate._parameter, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
            elif gate._name == 'QubitExcitation':
                index = gate._target
                if len(index) == 2:
                    if was_ur:
                        was_ur = False
                        continue
                    elif gate != circuit[-1] and circuit[i + 1]._name == 'QubitExcitation' and len(circuit[i + 1]._target) == 2 and circuit[i + 1]._target[0] // 2 == index[0] // 2 and circuit[i + 1]._target[1] // 2 == index[1] // 2 and not n_qubits_is_double:  # hope you enjoy this conditional
                        res.append(OrbitalRotatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter, *args, **kwargs))
                        was_ur = False
                    else:
                        res.append(SingleExcitation(index[0], index[1], angle=gate._parameter,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
                else:
                    if index[0] // 2 == index[2] // 2 and index[1] // 2 == index[3] // 2 and not n_qubits_is_double:  ## TODO: Maybe generalized for further excitations
                        res.append(PairCorrelatorGate(index[0] // 2, index[1] // 2, angle=gate._parameter,*args, **kwargs))
                    else:res.append(DoubleExcitation(index[0], index[1], index[2], index[3], angle=gate._parameter, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
            else:
                res.append(GenericGate(U=QCircuit(gates=[gate]), name="simple", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
        return cls(gates=res) 
    
    @classmethod
    def from_fcircuit(cls,U:FCircuit, n_qubits_is_double: bool = False, *args, **kwargs) -> 'GraphicalCircuit':
        """
        Converts a tequila FCircuit into a GraphicalCircuit object, which is renderable with `export_qpic`.
        """

        circuit = U._gates
        res = []
        if U.initial_state is not None:
            E = X(target=U.qubits)
            # E.qubits = [*range(U.initial_state.n_qubits)]
            res.append(GenericGate(U=E, name="initialstate", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
        for gate in circuit:
            index = ()
            for pair in gate.indices:
                for so in pair:
                    index += (so,)
            if gate._name == 'FermionicExcitation':
                if len(index) == 1:
                    res.append(SingleExcitation(index[0][0], index[0][1], angle=gate.variables,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
                elif len(index) == 2:
                    res.append(DoubleExcitation(index[0][0], index[0][1], index[1][0], index[1][1], angle=gate.variables, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
                else: res.append(GenericExcitation(indices=index,angle=gate.variables, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
            elif gate._name == 'UR':
                if not n_qubits_is_double:
                    res.append(OrbitalRotatorGate(index[0][0] // 2, index[0][1] // 2, angle=gate.variables, *args, **kwargs))
                else: 
                    res.append(SingleExcitation(index[0][0], index[0][1], angle=gate.variables,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
                    res.append(SingleExcitation(index[1][0], index[1][1], angle=gate.variables,n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
            elif gate._name == 'UC':
                if not n_qubits_is_double:
                    res.append(PairCorrelatorGate(index[0][0] // 2, index[0][1] // 2, angle=gate.variables,*args, **kwargs))
                else: 
                    res.append(DoubleExcitation(index[0][0], index[0][1], index[0][0], index[1], angle=gate.variables, n_qubits_is_double=n_qubits_is_double, *args,**kwargs))
            else:
                res.append(GenericGate(U=gate, name="simple", n_qubits_is_double=n_qubits_is_double, *args, **kwargs))
        return cls(gates=res)