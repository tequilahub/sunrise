from __future__ import annotations
from typing import Any, Union, Optional, Callable
from openfermion.ops.operators.fermion_operator import FermionOperator
from tequila import TequilaException, Objective, assign_variable
from tequila import grad as tqgrad
from tequila.objective.objective import Variable, identity, FixedVariable
from tequila.quantumchemistry.qc_base import QuantumChemistryBase #TODO modify when migrated
from sunrise.fermionic_operations.circuit import FCircuit
from sunrise.fermionic_operations.fgateimpl import FGateImpl
from sunrise.fermionic_operations.gates import FermionicExcitation
from copy import deepcopy
from math import pi
from tequila.objective.quantum_arg import QuantumArg


class FermBraketImpl(QuantumArg):
    def __init__(
        self,
        ket:FCircuit,
        bra:Optional[FCircuit] = None,
        operator:Optional[Union[str,FermionOperator]] = None,
        molecule:Optional[QuantumChemistryBase] = None,
        contraction: Optional[Callable] = None,
        shape: Optional[tuple] = None,
        samples: Optional[int] = None,
        *args,
        **kwargs,
    ):

        if ket is None:
            raise TequilaException("BraKet requires a ket circuit")

        self._contraction:Optional[Callable] = contraction
        self._shape:Optional[tuple] = shape
        self.samples:Optional[int] = samples
        self.molecule:Optional[QuantumChemistryBase] = molecule
        self.ket = ket
        self.bra = bra
        self.operator:Optional[Union[str,FermionOperator]] = operator
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise TequilaException(
            "FermBraketImpl is symbolic and should not be evaluated directly. "
            "Use sm.simulate(FermBraketImpl(...)) for backend execution." # TODO or BraKet(...)[0/1] for symbolic objectives. this is from tequila, should we?
        )

    def compile(self) -> tuple:
        return Objective(args=[self],transformation=identity)

    def grad(self, variable: Variable = None, *args, **kwargs) -> Objective:
        return _grad_FermBraket(objective=self, variable=variable, *args, **kwargs)

    def extract_variables(self) -> list:
        var = list(self.ket.extract_variables())
        if self.bra is not None:
            var.extend(self.bra.extract_variables())

        seen, unique = set(), []
        for v in var:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def count_measurements(self) -> int:
        if self.molecule is not None:
            return self.molecule.n_orbitals**4
        else:
            return 1

    def map_variables(self, variables: dict, *args, **kwargs) -> FermBraketImpl:
        return FermBraketImpl(
            ket=self.ket.map_variables(variables, *args, **kwargs),
            bra=self.bra.map_variables(variables, *args, **kwargs) if self.bra is not None else None,
            operator=self.operator,
            contraction=self._contraction,
            shape=self._shape,
            samples=self.samples,
            *self._args,
            **self._kwargs,
        )

    def map_qubits(self, qubit_map: dict) -> FermBraketImpl:
        op = None
        if hasattr(self.operator,'map_qubits'):
            op = self.operator.map_qubits(qubit_map) 
        elif isinstance(self.operator,str):
            op = self.operator
        
        return FermBraketImpl(
            ket=self.ket.map_qubits(qubit_map),
            bra=self.bra.map_qubits(qubit_map) if self.bra is not None else None,
            operator=op,
            contraction=self._contraction,
            shape=self._shape,
            samples=self.samples,
            *self._args,
            **self._kwargs,
        )

    @property
    def U(self):
        return self.ket

    @property
    def H(self) -> tuple:
        if self.operator is None:
            return tuple()
        return (self.operator,)

    @property
    def is_overlap(self) -> bool:
        return self.operator is None

    @property
    def is_self_overlap(self) -> bool:
        return self.is_overlap and self.is_diagonal

    @property
    def is_diagonal(self) -> bool:
        return self.bra is None or self.bra == self.ket

    @property
    def qubits(self) -> list[int]:
        q = set(self.ket.qubits)
        if self.bra is not None:
            q = set(self.bra.qubits) | q
        if self.operator is not None and hasattr(self.operator,'paulistrings'):
            for ps in self.operator.paulistrings:
                q |= set(ps.qubits)
        return sorted(q)

    @property
    def is_compiled(self) -> bool:
        return False

    @property
    def bra(self) -> FCircuit:
        """
        Excitation operators applied to the bra.
        """
        if self._bra is None:
            return self.ket
        return self._bra
    
    @bra.setter
    def bra(self, bra:FCircuit):
        '''
        Expected FCircuit.
        '''
        assert bra is None or isinstance(bra,FCircuit), f"FCircuit expected, received {type(bra).__name__}"
        if self.molecule is not None and bra is not None:
            bra = bra.to_upthendown(self.molecule.n_orbitals)
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
        if self.molecule is not None:
            ket = ket.to_upthendown(self.molecule.n_orbitals)
        self._ket = ket

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, operator):
        self._operator = operator

    @property
    def is_compiled(self):
        return False

    def __repr__(self) -> str:
        if self.operator:
            return f"FermBraKetImpl(<bra|H|ket>, qubits={self.qubits})"
        return f"BraKetImpl(<bra|ket>, qubits={self.qubits})"

    def __str__(self) -> str:
        return self.__repr__()

def _grad_FermBraket(objective: "FermBraketImpl", variable:Variable = None) -> Objective:
        if variable is None:
            # None means that all components are created
            variables = objective.extract_variables()
            result = {}

            if len(variables) == 0:
                raise TequilaException("Error in gradient: Objective has no variables")

            for k in variables:
                assert k is not None
                result[k] = grad_FermBraket(braket,k)
            return result
        else:
            variable = assign_variable(variable)

        if variable not in objective.extract_variables():
            return 0.

        bra = Objective()
        if not objective.is_diagonal:
            for i,gate in enumerate(objective.bra.gates):
                if variable in gate.extract_variables():
                    g = deepcopy(gate)
                    bra += _grad_shift_rule(g=g, i=i, variable=variable, expval=objective, bra=True)
                    bra += _grad_shift_rule(g=g, i=i, variable=variable, expval=objective, bra=False)
        ket = Objective()
        for i,gate in enumerate(objective.ket.gates):
            if variable in gate.extract_variables():
                g = deepcopy(gate)
                ket += _grad_shift_rule(g=g, i=i, variable=variable, expval=objective, bra=True)
                ket += _grad_shift_rule(g=g, i=i, variable=variable, expval=objective, bra=False)
                    
        return bra + ket

def _grad_shift_rule(g:FGateImpl, i:int, variable:Variable, expval: FermBraketImpl, bra:bool) -> Objective:
    """
    function for getting the gradients of directly differentiable gates. Expects precompiled circuits.
    :param unitary: FCircuit: the FCircuit object containing the gate to be differentiated
    :param g: a parametrized: the gate being differentiated
    :param i: int: the position in unitary at which g appears
    :param variable: Variable or String: the variable with respect to which gate g is being differentiated
    :param hamiltonian: the hamiltonian with respect to which unitary is to be measured, in the case that unitary
        is contained within an ExpectationValue
    :return: an Objective, whose calculation yields the gradient of g w.r.t variable
    """
    gradient = 0.
    expval = deepcopy(expval)
    if type(g).__name__ == 'URImpl':
        gprima = FCircuit()
        for gate in g.indices:
            gprima += FermionicExcitation(indices=[gate],variables=g.variables,reordered=g.reordered)
        if bra:
            expval.bra = expval.bra.replace_gates([i],[gprima])
            gradient += _grad_shift_rule(g=expval.bra.gates[i], i=i, variable=variable, expval=expval, bra=bra)
            gradient += _grad_shift_rule(g=expval.bra.gates[i+1], i=i+1, variable=variable, expval=expval, bra=bra)
        else:
            expval.ket = expval.ket.replace_gates([i],[gprima])
            gradient += _grad_shift_rule(g=expval.ket.gates[i], i=i, variable=variable, expval=expval, bra=bra)
            gradient += _grad_shift_rule(g=expval.ket.gates[i+1], i=i+1, variable=variable, expval=expval, bra=bra)
        return gradient
    s = {False:+1, True:-1}
    inner_grad = _grad_phase(g.variables, variable)
    U0 = FCircuit()
    p0 = []
    for gate in g.indices:
        p0sing = s[len(gate)%2]
        for (idx,jdx) in gate:
            p0.extend([(idx,idx),(jdx,jdx)])
        U0 += FermionicExcitation(indices=p0,variables=p0sing*pi/2,reordered=g.reordered)
    if bra:
        fbra = deepcopy(expval.bra)
        fbra.gates[i].variables = fbra.gates[i].variables + s[bra]*pi
        fbra = fbra.insert_gates(positions=[i],gates=[U0])
        expval.bra = fbra
        gradient += s[not bra]*inner_grad*Objective(args=[expval,],transformation=identity)
    else:
        fket = deepcopy(expval.ket)
        fket.gates[i].variables = fket.gates[i].variables + s[bra]*pi
        fket = fket.insert_gates(positions=[i],gates=[U0])
        if expval.is_diagonal:
            expval.bra = fket
        else:
            expval.ket = fket
        gradient += s[not bra]*inner_grad*Objective(args=[expval,],transformation=identity)
    return  -0.5*gradient

def _grad_phase(arg, variable):
    """
    a modified loop over __grad_objective, which gets derivatives
     all the way down to variables, return 1 or 0 when a variable is (isnt) identical to var.
    :param arg: a transform or variable object, to be differentiated
    :param variable: the Variable with respect to which par should be differentiated.
    :ivar var: the string representation of variable
    """

    assert isinstance(variable, Variable)
    if isinstance(arg, Variable):
        if arg == variable:
            return 1.0
        else:
            return 0.0
    elif isinstance(arg, FixedVariable):
        return 0.0
    elif isinstance(arg, FermBraketImpl):
        return _grad_FermBraket(arg, variable=variable)
    elif hasattr(arg, "grad"):
        return arg.grad(variable)
    else:
        return tqgrad(objective=arg, variable=variable)