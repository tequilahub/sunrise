"""
ffsim backend: the states are dense ffsim CI vectors of one (n_alpha, n_beta) sector, on which
every FCircuit gate exp(i*theta/2*G) acts through ffsim.linear_operator(G).

Limits compared to spex, which keeps only the determinants it reaches:
- gates and measured FermionOperators must conserve the particle number and S_z (ffsim's
  linear_operator only works within one sector): spin-flip excitations like a+_alpha a_beta, or
  operators changing N, raise a TequilaException. Spin-mixed doubles that keep S_z are fine.
- an initial_state mixing different (n_alpha, n_beta) sectors raises a TequilaException.
- the CI vector is dense, binom(norb, n_alpha) * binom(norb, n_beta) * 16 bytes
  (about 38 GB for H18 in STO-3G).
- ffsim has no Pauli operators: a tequila QubitHamiltonian (e.g. the hardcore-boson Hamiltonian)
  is applied in python on the up-then-down QubitWaveFunction of the ket and projected back on
  the CI sector, which is exact for <bra|H|ket> but slow.
"""

from __future__ import annotations

from numbers import Number
from typing import Callable

import ffsim
from ffsim import FermionOperator
from numpy import ceil, complex128, cos, exp, ndarray, real, sin, vdot, zeros
from openfermion import FermionOperator as OFFermionOperator
from pyscf.gto import Mole
from pyscf.scf import RHF
from scipy.sparse.linalg import LinearOperator
from tequila import (
    Objective,
    QubitHamiltonian,
    QubitWaveFunction,
    TequilaException,
    Variable,
    simulate,
)
from tequila.objective.objective import Variables

from sunrise.molecules.qubit_base import qc_base

from ...fermionic_operations.circuit import FCircuit
from ..fermionic_braket import FermBraketImpl
from ..fqe.fermionic_utils import make_excitation_generator_op
from ..spex.spex_expval import _civect_to_qwvf, _qwvf_to_civect


class FFS_EXPVAL:
    def __init__(
        self,
        braket: FermBraketImpl,
        *args,
        **kwargs,
    ):

        self.operator = None
        self.molecule: ffsim.MolecularData = None
        self._init_state_bra: ndarray = None
        self._init_state_ket: ndarray = None
        self._ket: FCircuit = None
        self._bra: FCircuit = None
        self._ket_generators: list = None
        self._bra_generators: list = None
        # (variable values, bra, ket) of the last states call, see states
        self._states_cache: tuple = None

        bra = braket.bra
        ket = braket.ket
        molecule = braket.molecule
        operator = braket.operator

        run_hf = (bra is None or bra.initial_state is None) and (
            ket is None or ket.initial_state is None
        )
        if isinstance(molecule, qc_base.QuantumChemistryBase):
            c, h, g = molecule.get_integrals()
            g = g.reorder("chem").elems
            # ffsim (as pyscf) stores the spin as 2S = n_alpha - n_beta
            spin = molecule.parameters.multiplicity - 1
            n_alpha = int((molecule.n_electrons + spin) // 2)
            n_beta = int((molecule.n_electrons - spin) // 2)
            point_group = None
            if hasattr(molecule, "point_group"):
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
                active_space=[
                    i.idx_total for i in molecule.integral_manager.active_orbitals
                ],
            )
        elif isinstance(molecule, Mole):
            mf = RHF(mol=molecule)
            self.molecule = ffsim.MolecularData.from_scf(mf)

        self._cistrings = ffsim.addresses_to_strings(
            range(ffsim.dim(self.molecule.norb, self.molecule.nelec)),
            norb=self.molecule.norb,
            nelec=self.molecule.nelec,
            concatenate=True,
            bitstring_type=ffsim.BitstringType.INT,
        )

        if run_hf:
            hf_state = ffsim.hartree_fock_state(self.molecule.norb, self.molecule.nelec)
        else:
            # a side without initial state stays zero, as in spex
            hf_state = zeros(len(self._cistrings), dtype=complex128)
        self._init_state_bra = hf_state
        self._init_state_ket = hf_state

        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = "Expectation Value" if self.is_diagonal else "Transition Value"
        if isinstance(operator, str) and operator == "I":
            self._name = "Transition Element"
        if operator is not None:
            self.operator = self.build_operator(operator)

    def __call__(
        self, variables: list | dict | Variables = None, *args, **kwargs
    ) -> float:
        if variables is None:
            variables = {}
        return self.simulate(variables=variables)

    def simulate(self, variables: list | dict | None = None) -> float:
        bra, ket = self.states(variables)
        # no operator: overlap <bra|ket>
        if self.operator is not None:
            ket = self.operator @ ket
        return float(real(vdot(bra, ket)))

    def states(self, variables: list | dict | None = None) -> tuple[ndarray, ndarray]:
        """
        The CI vectors of bra and ket after applying their circuits.
        """
        if variables is None:
            variables = {}
        if isinstance(variables, Variables):
            variables = variables.store
        # Variable objects and their names are both accepted
        v = {}
        for k, val in variables.items():
            v[getattr(k, "name", k)] = val
            if isinstance(k, Variable):
                v[k] = val
        check_variables = {
            k: (k in v or getattr(k, "name", None) in v)
            for k in self.extract_variables()
        }
        if not all(list(check_variables.values())):
            raise TequilaException(
                "Objective did not receive all variables:\n"
                "You gave\n"
                f" {variables}\n"
                " but the objective depends on\n"
                f" {self.extract_variables()}\n"
                " missing values for\n"
                f" {[k for k, c in check_variables.items() if not c]}"
            )

        # the states are kept for the next call with the same values,
        # e.g. the operators of compute_rdms
        key = tuple(float(map_variables(k, v)) for k in self.extract_variables())
        if self._states_cache and self._states_cache[0] == key:
            return self._states_cache[1:]
        ket = _apply_circuit_to_state(
            self._init_state_ket, self.ket, self._ket_generators, v
        )
        if self.is_diagonal:
            bra = ket
        else:
            bra = _apply_circuit_to_state(
                self._init_state_bra, self.bra, self._bra_generators, v
            )
        self._states_cache = (key, bra, ket)
        return bra, ket

    def extract_variables(self) -> list[Variable]:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
        variables_bra = []
        variables_ket = []
        if self.bra is not None:
            variables_bra = self.bra.extract_variables()
        if self.ket is not None:
            variables_ket = self.ket.extract_variables()
        # a set, comparing Variables one by one is slow for large circuits
        seen = set(variables_ket)
        uniques_bra = [v for v in variables_bra if v not in seen]
        return uniques_bra + variables_ket

    @property
    def bra(self) -> FCircuit:
        """
        Excitation operators applied to the bra.
        """
        return self._bra

    @bra.setter
    def bra(self, bra: FCircuit):
        """
        Expected FCircuit.
        """
        assert isinstance(bra, FCircuit), (
            f"FCircuit expected, received {type(bra).__name__}"
        )
        if bra.initial_state is not None:
            self.init_state_bra = bra.initial_state
        bra = bra.to_upthendown(self.molecule.norb)
        self._bra = bra
        self._bra_generators = _build_generators(
            bra, self.molecule.norb, self.molecule.nelec
        )
        self._states_cache = None

    @property
    def ket(self) -> FCircuit:
        """
        Excitation operators applied to the ket.
        """
        return self._ket

    @ket.setter
    def ket(self, ket: FCircuit):
        """
        Expected FCircuit
        """

        assert isinstance(ket, FCircuit), (
            f"FCircuit expected, received {type(ket).__name__}"
        )
        if ket.initial_state is not None:
            self.init_state_ket = ket.initial_state
        ket = ket.to_upthendown(self.molecule.norb)
        self._ket = ket
        self._ket_generators = _build_generators(
            ket, self.molecule.norb, self.molecule.nelec
        )
        self._states_cache = None

    @property
    def variables_bra(self) -> list[Variable]:
        """Tequila Circuit Bra variables."""
        return self.bra.variables

    @property
    def variables_ket(self) -> list[Variable]:
        """Tequila circuit Ket parameters."""
        return self.ket.variables

    @property
    def variables(self) -> list[Variable]:
        """Tequila circuit variables."""
        return self.variables_bra + self.variables_ket

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
    def init_state(self) -> tuple[QubitWaveFunction, QubitWaveFunction]:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.
        """
        return self.init_state_bra, self.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state_bra)
        self._states_cache = None

    @init_state_ket.setter
    def init_state_ket(self, init_state_ket: QubitWaveFunction):
        self._init_state_ket = self.__qwvf_to_civect(init_state_ket)
        self._states_cache = None

    @init_state.setter
    def init_state(self, init_state: QubitWaveFunction):
        self.init_state_bra = init_state
        self.init_state_ket = init_state

    def __str__(self):
        res = ""
        if self.is_diagonal:
            res += f"{self._name} with indices: {self.ket.extract_indices()} with variables {self.variables_ket}"
        else:
            res += f"{self._name} with Bra= {self.bra.extract_indices()} with variables {self.variables_bra}\n"
            res += f"{len(self._name) * ' '} with Ket= {self.ket.extract_indices()} with variables {self.variables_ket}"
        return res

    def __repr__(self):
        return self.__str__()

    @property
    def is_diagonal(self):
        if self.bra is None:
            return True
        if self.ket is None:
            return True
        return bool(self.bra is not None and self.bra == self.ket)

    @property
    def U(self):
        "Dummy function to work with tequila Objectives"
        if self.is_diagonal:
            return self.ket
        else:
            return [self.bra, self.ket]

    def count_measurements(self) -> int:
        mes = 0
        if self.molecule.one_body_integrals is not None:
            mes += self.molecule.one_body_integrals.size
        if self.molecule.two_body_integrals is not None:
            mes += self.molecule.two_body_integrals.size
        if mes:
            return mes
        else:
            return len(self._cistrings)

    def __civect_to_qwvf(self, civect: ndarray) -> QubitWaveFunction:
        # ffsim's strings are the LSB integers of the up-then-down qubit wavefunction
        return _civect_to_qwvf(
            dict(zip(self._cistrings, civect)), 2 * self.molecule.norb
        )

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> ndarray:
        return _qwvf_to_ffsim(wvf, self.molecule.norb, self.molecule.nelec)

    def build_operator(
        self, operator: str | FermionOperator | QubitHamiltonian = None
    ) -> None | Callable | LinearOperator:
        """
        Build the expectation value operator.
        Even if it is accepted a QubitHamiltonian, we disencorage its use here for fermionic states limitations.
        It will be applied to the ci_vector and kept these results which doesn't leave the ci_vector.
        """
        norb, nelec = self.molecule.norb, self.molecule.nelec

        def from_string(operator: str) -> LinearOperator:
            if operator.upper() == "I":
                return LinearOperator(
                    shape=(len(self._cistrings), len(self._cistrings)),
                    matvec=lambda x: x,
                    rmatvec=lambda x: x,
                    dtype=complex128,
                )
            elif operator.upper() == "H":
                return ffsim.linear_operator(self.molecule.hamiltonian, norb, nelec)
            else:
                raise TequilaException(
                    f"No operator str {operator} supported on FFSim BraKet"
                )

        if isinstance(operator, str):
            operator = from_string(operator)
        elif isinstance(operator, Number):
            operator = _linear_operator(
                FermionOperator({(): complex(operator)}), norb, nelec
            )
        elif isinstance(operator, OFFermionOperator):
            # openfermion numbers spin-orbitals interleaved (even=alpha, odd=beta)
            operator = FermionOperator(
                {
                    tuple((a == 1, bool(p % 2), p // 2) for p, a in term): weight
                    for term, weight in operator.terms.items()
                }
            )
            operator = _linear_operator(operator, norb, nelec)
        elif isinstance(operator, FermionOperator):
            operator = _linear_operator(operator, norb, nelec)
        elif isinstance(operator, QubitHamiltonian):

            def f(v):
                wvf = self.__civect_to_qwvf(v).apply_qubitoperator(H)
                return self.__qwvf_to_civect(wvf)

            H = operator
            operator = LinearOperator(
                shape=(len(self._cistrings), len(self._cistrings)),
                matvec=f,
                dtype=complex128,
            )
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")

        return operator


def map_variables(x: list[Variable, Objective], dvariables: dict):
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x


def _qwvf_to_ffsim(wvf: QubitWaveFunction, norb: int, nelec: tuple) -> ndarray:
    # spin orbital p of the up-then-down ordering is orbital p % norb, beta if p >= norb:
    # the bit order of ffsim's strings, so the LSB integer keys are ffsim strings
    state = _qwvf_to_civect(wvf, 2 * norb)
    for k in state:
        if (bin(k & ((1 << norb) - 1)).count("1"), bin(k >> norb).count("1")) != nelec:
            raise TequilaException(
                f"FFSim BraKet: state outside the (n_alpha, n_beta) = {nelec} sector"
            )
    vec = zeros(ffsim.dim(norb, nelec), dtype=complex128)
    if state:
        vec[ffsim.strings_to_addresses(list(state.keys()), norb, nelec)] = list(
            state.values()
        )
    return vec


def _linear_operator(op: FermionOperator, norb: int, nelec: tuple) -> LinearOperator:
    if not (op.conserves_particle_number() and op.conserves_spin_z()):
        raise TequilaException(
            f"FFSim BraKet: ffsim only supports operators conserving N and S_z, received {op}"
        )
    return ffsim.linear_operator(op, norb, nelec)


def _build_generators(circuit: FCircuit, norb: int, nelec: tuple) -> list:
    # per gate, per term of gate.indices: (is_number_operator, ffsim linear operator of its generator)
    generators = []
    for gate in circuit.gates:
        terms = []
        for term_pairs in gate.indices:
            op = make_excitation_generator_op(term_pairs)
            op = FermionOperator(
                {
                    tuple((a == 1, p >= norb, p % norb) for p, a in t): w
                    for t, w in op.terms.items()
                }
            )
            is_number = all(p == q for p, q in term_pairs)
            terms.append((is_number, _linear_operator(op, norb, nelec)))
        generators.append(terms)
    return generators


def _apply_circuit_to_state(
    state: ndarray, circuit: FCircuit, generators: list, dvariables: dict
) -> ndarray:
    result = state.astype(complex128)
    if circuit is None:
        return result
    for gate, terms in zip(circuit.gates, generators):
        if not terms:
            continue
        theta = float(map_variables(gate.variables, dvariables))
        for is_number, G in terms:
            if is_number:
                # number operators (Phase gates) as spex: exp(-i*theta/2*N), N a projector
                result = result + (exp(-0.5j * theta) - 1) * (G @ result)
                continue
            # G^3 = G: exp(i*theta/2*G) = 1 + i*sin(theta/2)*G + (cos(theta/2) - 1)*G^2
            Gv = G @ result
            result = result + 1j * sin(theta / 2) * Gv + (cos(theta / 2) - 1) * (G @ Gv)
    return result


def ffsim_circuit_simulator(
    U: FCircuit, variables, n_orb: int | None = None, **backend_kwargs
) -> QubitWaveFunction:
    if n_orb is None:
        n_orb = int(ceil((U.max_qubit() + 1) / 2))
    n_qubits = 2 * n_orb

    # Capture the initial state before to_upthendown, which mutates its n_qubits.
    initial_state = U.initial_state
    if initial_state is None:
        return QubitWaveFunction(n_qubits=n_qubits)
    # the CI sector (n_alpha, n_beta) is read off the initial state
    key = next(iter(_qwvf_to_civect(initial_state, n_qubits)))
    nelec = (bin(key & ((1 << n_orb) - 1)).count("1"), bin(key >> n_orb).count("1"))
    state = _qwvf_to_ffsim(initial_state, n_orb, nelec)

    U = U.to_upthendown(n_orb)

    v = {}
    if variables is not None:
        for k, val in variables.items():
            v[getattr(k, "name", k)] = val
            if isinstance(k, Variable):
                v[k] = val

    state = _apply_circuit_to_state(state, U, _build_generators(U, n_orb, nelec), v)
    strings = ffsim.addresses_to_strings(
        range(len(state)), n_orb, nelec, bitstring_type=ffsim.BitstringType.INT
    )
    return _civect_to_qwvf(dict(zip(strings, state)), n_qubits)
