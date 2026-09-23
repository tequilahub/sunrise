import spex_tequila as spex

from sunrise.fermionic_operations.circuit import FCircuit

from tequila import TequilaException, QubitWaveFunction, Variable, QubitHamiltonian
from tequila.objective.objective import Variables, Objective
from tequila.utils.bitstrings import BitNumbering, reverse_int_bits
from numpy import ndarray, array, complex128, real
from openfermion import FermionOperator
from numbers import Number
from typing import Union, List, Callable
from .fermionic_braket import FermBraketImpl
from tequila.quantumchemistry.qc_base import QuantumChemistryBase # TODO change when migrated
_ZERO_TOL = 1e-12


class SpexExpval:
    def __init__(
        self,
        braket:"FermBraketImpl",
        *args,
        **kwargs,
    ):
        self.operator = None
        self.hamiltonian: List[spex.FermionTerm] = []
        self._init_state_bra: dict = None
        self._init_state_ket: dict = None
        self._ket = None
        self._bra = None
        self._name = None
        self._molecule = None
        # Molecule Data
        self.norb: int = None
        self.n_alpha: int = None
        self.n_beta: int = None
        self.core_energy: float = 0.0
        self.one_body_integrals: ndarray = None
        self.two_body_integrals: ndarray = None
        self.mo_coeff: ndarray = None
        self.spin: float = 0
        self.symmetry = None
        self.atom = None
        self.basis = None
        self.active_space = None

        bra = braket.bra
        ket = braket.ket
        self.molecule = braket.molecule
        operator = braket.operator
        run_hf = (bra is None or bra.initial_state is None) and (ket is None or ket.initial_state is None)
        self.hamiltonian = self._build_hamiltonian()

        if run_hf:
            occupied = [i for i in range(self.n_alpha)] + [self.norb + i for i in range(self.n_beta)]
            hf_state = {sum(1 << o for o in occupied): 1.0}
            self._init_state_bra = hf_state
            self._init_state_ket = hf_state

        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        if 'name' in kwargs:
            self._name = kwargs.pop('name')
        else:
            self._name = 'Expectation Value' if self.is_diagonal else 'Transition Value'
        if operator is None:
            operator = 'H'
        if isinstance(operator, str) and operator == 'I':
            self._name = 'Transition Element'
        self.operator = self.build_operator(operator)

    def _build_hamiltonian(self) -> List[spex.FermionTerm]:
        # Integrals in chemists notation g[p,q,r,s]=(pq|rs); spatial orbital p maps to
        # spin orbitals p (alpha) and p+norb (beta) in up-then-down ordering.
        norb = self.norb
        terms: List[spex.FermionTerm] = []
        if self.core_energy is not None and abs(self.core_energy) > _ZERO_TOL:
            terms.append(spex.FermionTerm([], [], complex(self.core_energy)))
        if self.one_body_integrals is not None:
            h = array(self.one_body_integrals, dtype=complex128)
            for p in range(norb):
                for q in range(norb):
                    if abs(h[p, q]) < _ZERO_TOL:
                        continue
                    terms.append(spex.FermionTerm([p], [q], h[p, q]))
                    terms.append(spex.FermionTerm([p + norb], [q + norb], h[p, q]))
        if self.two_body_integrals is not None:
            g = array(self.two_body_integrals, dtype=complex128)
            for p in range(norb):
                for q in range(norb):
                    for r in range(norb):
                        for s in range(norb):
                            w = 0.5 * g[p, r, q, s]
                            if abs(w) < _ZERO_TOL:
                                continue
                            terms.append(spex.FermionTerm([q, p], [r, s], w))
                            terms.append(spex.FermionTerm([q + norb, p + norb], [r + norb, s + norb], w))
                            terms.append(spex.FermionTerm([q + norb, p], [r, s + norb], w))
                            terms.append(spex.FermionTerm([q, p + norb], [r + norb, s], w))
        return terms

    @property
    def bra(self) -> 'FCircuit':
        return self._bra

    @bra.setter
    def bra(self, bra):
        if not isinstance(bra, FCircuit):
            raise TypeError(f"FCircuit expected, received {type(bra).__name__}")
        if bra.initial_state is not None:
            self._init_state_bra = self.__qwvf_to_civect(bra.initial_state)
        bra = bra.to_upthendown(self.norb)
        self._bra = bra

    @property
    def ket(self) -> 'FCircuit':
        return self._ket

    @ket.setter
    def ket(self, ket):
        if not isinstance(ket, FCircuit):
            raise TypeError(f"FCircuit expected, received {type(ket).__name__}")
        if ket.initial_state is not None:
            self._init_state_ket = self.__qwvf_to_civect(ket.initial_state)
        ket = ket.to_upthendown(self.norb)
        self._ket = ket

    @property
    def variables(self) -> List[Variable]:
        bra = self.bra.variables if self.bra is not None else []
        ket = self.ket.variables if self.ket is not None else []
        return bra + ket

    @property
    def init_state(self) -> tuple:
        return self.__civect_to_qwvf(self._init_state_bra), self.__civect_to_qwvf(self._init_state_ket)

    @init_state.setter
    def init_state(self, init_state: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state)
        self._init_state_ket = self.__qwvf_to_civect(init_state)

    @property
    def is_diagonal(self):
        return self.bra is None or self.ket is None or self.bra == self.ket

    @property
    def U(self):
        return self.ket if self.is_diagonal else [self.bra, self.ket]

    def __civect_to_qwvf(self, state_dict: dict) -> QubitWaveFunction:
        return _civect_to_qwvf(state_dict, 2 * self.norb)

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> dict:
        return _qwvf_to_civect(wvf, 2 * self.norb)

    def build_operator(self, operator: Union[str, FermionOperator, QubitHamiltonian] = None) -> Union[None, Callable, List[spex.FermionTerm]]:
        # Supported: "I"/"H", openfermion FermionOperator, tequila QubitHamiltonian (callable on state dict)

        def from_string(operator: str) -> Union[Callable, List[spex.FermionTerm]]:
            if operator.upper() == "I":
                return lambda x: x
            elif operator.upper() == "H":
                return self.hamiltonian
            else:
                raise TequilaException(f"No operator str {operator} supported on Spex BraKet")

        if isinstance(operator, str):
            operator = from_string(operator)
        elif isinstance(operator, Number):
            operator = [spex.FermionTerm([], [], complex(operator))]
        elif isinstance(operator, FermionOperator):
            # openfermion's terms dict already merges identical terms
            terms: List[spex.FermionTerm] = []
            for term, weight in operator.terms.items():
                if abs(weight) < _ZERO_TOL:
                    continue
                if len(term) == 0:
                    terms.append(spex.FermionTerm([], [], weight))
                    continue
                # openfermion terms act right-to-left; spex applies annihilation then creation
                # in vector order, so both lists are the reversed tuple order.
                creation = [i for i, a in term if a == 1]
                annihilation = [i for i, a in term if a == 0]
                # openfermion numbers spin-orbitals interleaved (even=alpha, odd=beta);
                # spex works internally in up-then-down ordering, remap the indices.
                creation = [i // 2 + (i % 2) * self.norb for i in creation]
                annihilation = [i // 2 + (i % 2) * self.norb for i in annihilation]
                terms.append(spex.FermionTerm(list(reversed(creation)), list(reversed(annihilation)), weight))
            operator = terms
        elif isinstance(operator, QubitHamiltonian):
            # spex evaluates <phi|H|psi> for a Pauli hamiltonian in one call, so
            # hand it the terms directly rather than building H|psi> in python.
            # Its Pauli routines index qubits MSB-first while its fermionic
            # states are LSB-indexed, so the indices are flipped here, once.
            n_qubits = 2 * self.norb
            terms = []
            for pauli_string in operator.paulistrings:
                weight = complex(pauli_string.coeff)
                if abs(weight) < _ZERO_TOL:
                    continue
                term = spex.ExpPauliTerm()
                term.pauli_map = {n_qubits - 1 - q: p.upper() for q, p in pauli_string.items()}
                terms.append((term, weight))
            # an operator that vanished entirely still has to evaluate to zero
            operator = terms if terms else [spex.FermionTerm([], [], 0.0)]
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")

        return operator

    def __call__(self, variables: Union[list, dict, Variables] = {}, *args, **kwargs) -> float:
        return self.simulate(variables=variables)

    def simulate(self, variables: Union[list, dict, Variables] = None) -> float:
        variables = {} if variables is None else variables
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
        if isinstance(variables, Variables):
            variables = variables.store
        if not isinstance(variables, dict):
            raise TequilaException(f'variables must be a dict or tequila Variables, received {type(variables).__name__}')

        # Normalize keys: Variable objects and their names are both accepted.
        dvars = {}
        for k, val in variables.items():
            name = getattr(k, 'name', k)
            dvars[name] = val
            if isinstance(k, Variable):
                dvars[k] = val

        check_variables = {k: (k in dvars or getattr(k, 'name', None) in dvars) for k in self.extract_variables()}
        if not all(check_variables.values()):
            missing = [k for k, v in check_variables.items() if not v]
            raise TequilaException(f'Objective did not receive all variables: {variables} given, missing {missing}')

        ket_state = self._apply_circuit(self._init_state_ket, self.ket, dvars)
        if self.is_diagonal:
            bra_state = ket_state
        else:
            bra_state = self._apply_circuit(self._init_state_bra, self.bra, dvars)

        if callable(self.operator):
            ket_state = self.operator(ket_state)
        if isinstance(self.operator, list) and self.operator and isinstance(self.operator[0], tuple):
            # list of (ExpPauliTerm, weight): a qubit hamiltonian, evaluated by spex
            if not bra_state or not ket_state:
                return 0.0
            result = spex.expectation_value(bra_state, ket_state, self.operator, 2 * self.norb)
        elif isinstance(self.operator, list):
            result = spex.expectation_value_fermionic(bra_state, ket_state, self.operator)
        else:
            result = spex.expectation_value_fermionic(bra_state, ket_state, [spex.FermionTerm([], [], 1.0)])
        return float(real(result))

    def extract_variables(self) -> List[Variable]:
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

    def _apply_circuit(self, state: dict, circuit, dvars: dict) -> dict:
        return _apply_circuit_to_state(state, circuit, dvars)

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

    def count_measurements(self)->int:
        return len(self.hamiltonian)

    @property
    def molecule(self) -> QuantumChemistryBase:
        return self._molecule

    @molecule.setter
    def molecule(self, molecule:QuantumChemistryBase):
        self._molecule = molecule
        self.mo_coeff = molecule.integral_manager.orbital_coefficients
        c, h, g = molecule.get_integrals()
        g = g.reorder('chem').elems
        self.core_energy = c
        self.one_body_integrals = h
        self.two_body_integrals = g
        self.norb = molecule.n_orbitals
        self.spin = (molecule.parameters.multiplicity - 1) / 2
        self.n_alpha = int((molecule.n_electrons + 2 * self.spin) // 2)
        self.n_beta = int((molecule.n_electrons - 2 * self.spin) // 2)
        self.symmetry = getattr(molecule, 'point_group', None)
        self.atom = molecule.parameters.get_geometry()
        self.basis = molecule.parameters.basis_set
        self.active_space = [i.idx_total for i in molecule.integral_manager.active_orbitals]

def _qwvf_to_civect(wvf: QubitWaveFunction, n_qubits: int) -> dict:
    wvf.n_qubits = n_qubits
    result = {}
    for bs, coeff in wvf.items():
        key = int(bs)
        if wvf.numbering != BitNumbering.LSB:
            key = reverse_int_bits(key, n_qubits)
        if abs(coeff) > _ZERO_TOL:
            result[key] = complex(coeff)
    return result


def _civect_to_qwvf(state_dict: dict, n_qubits: int) -> QubitWaveFunction:
    wvf = QubitWaveFunction(n_qubits=n_qubits, numbering=BitNumbering.LSB)
    wvf._state = {
        int(k): complex(v) for k, v in (state_dict or {}).items() if abs(v) > _ZERO_TOL
    }
    return wvf


def _apply_circuit_to_state(state: dict, circuit, dvars: dict) -> dict:
    from sunrise  import simulate
    # each gate maps to FermionTerm(creation_idx=[from...], annihilation_idx=[to...], 1j)
    state = {} if state is None else state
    result = {int(k): complex(v) for k, v in state.items() if abs(v) > _ZERO_TOL}
    if circuit is None:
        return result
    for gate in circuit.gates:
        indices = gate.indices
        parameter = gate.variables
        if not indices:
            continue

        if isinstance(parameter, Variable):
            theta = float(parameter.map_variables(dvars))
        elif isinstance(parameter, Objective):
            theta = float(simulate(parameter, dvars))
        else:
            theta = parameter
        for term_pairs in indices:
            creation = [p[0] for p in term_pairs]
            annihilation = [p[1] for p in term_pairs]
            n_pairs = len(term_pairs)
            weight = 1.0j * ((-1) ** (n_pairs * (n_pairs - 1) // 2))
            term = spex.FermionTerm(creation, annihilation, weight)
            result = spex.apply_fermion_excitation(result, term, theta)
    return result


def spex_circuit_simulator(U, variables, n_orb, **backend_kwargs) -> QubitWaveFunction:
    n_qubits = 2 * n_orb

    # Capture the initial state before to_upthendown, which mutates its n_qubits.
    initial_state = U.initial_state
    state = _qwvf_to_civect(initial_state, n_qubits) if initial_state is not None else {}

    U = U.to_upthendown(n_orb)

    dvars = {}
    if variables is not None:
        for k, val in variables.items():
            name = getattr(k, 'name', k)
            dvars[name] = val
            if isinstance(k, Variable):
                dvars[k] = val

    state = _apply_circuit_to_state(state, U, dvars)
    return _civect_to_qwvf(state, n_qubits)
