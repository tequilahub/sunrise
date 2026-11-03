r"""Fermionic backend ``"ffsim"``: dense CI vectors simulated with ffsim.

The states are dense ffsim CI vectors of one :math:`(n_\alpha, n_\beta)` sector, in up-then-down
spin orbital ordering. Every FCircuit gate :math:`e^{i \frac{\theta}{2} G}` acts through
``ffsim.linear_operator(G)``, with ``G`` from ``make_excitation_generator_op`` (minus the generator
of ``FermionicExcitation``, so it is the same gate).

Limits compared to spex, which keeps only the determinants it reaches:

- Gates and measured FermionOperators must conserve the particle number and :math:`S_z`
  (ffsim's ``linear_operator`` only works within one sector): spin-flip excitations like
  :math:`a^\dagger_\alpha a_\beta`, or operators changing N, raise a TequilaException.
  Spin-mixed doubles that keep :math:`S_z` are fine.
- An initial state mixing different :math:`(n_\alpha, n_\beta)` sectors raises a
  TequilaException.
- The CI vector is dense, :math:`\binom{n_{orb}}{n_\alpha} \binom{n_{orb}}{n_\beta} \cdot 16`
  bytes (about 38 GB for H18 in STO-3G).
- ffsim has no Pauli operators: a tequila QubitHamiltonian (e.g. the hard-core boson
  Hamiltonian) is applied in Python on the up-then-down QubitWaveFunction of the ket and
  projected back on the CI sector, which is exact for :math:`\langle bra|H|ket \rangle` but slow.
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
    """Expectation value or transition element evaluated with the ``"ffsim"`` backend.

    The molecule is converted to ``ffsim.MolecularData``. Circuits are converted once to ffsim
    linear operators of their generators and applied to the initial state (Hartree-Fock unless a
    circuit defines one; if only one side defines it, the other side starts from zero, as in spex).

    Parameters
    ----------
    braket : FermBraketImpl
        Braket to evaluate. Its molecule (sunrise molecule or PySCF ``Mole``) is required.
    name : str, optional
        Label used when printing. Passed as keyword argument.
    *args, **kwargs
        Ignored apart from ``name``.
    """

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
        """Evaluate the braket, see ``simulate``."""
        if variables is None:
            variables = {}
        return self.simulate(variables=variables)

    def simulate(self, variables: list | dict | None = None) -> float:
        """Evaluate the braket for the given variable values.

        Parameters
        ----------
        variables : dict or Variables, optional
            Values of the circuit variables, keyed by Variable or name.

        Returns
        -------
        float
            Real part of the braket.

        Raises
        ------
        TequilaException
            If a variable value is missing.
        """
        bra, ket = self.states(variables)
        # no operator: overlap <bra|ket>
        if self.operator is not None:
            ket = self.operator @ ket
        return float(real(vdot(bra, ket)))

    def states(self, variables: list | dict | None = None) -> tuple[ndarray, ndarray]:
        """Return the CI vectors of bra and ket after applying their circuits.

        Parameters
        ----------
        variables : dict or Variables, optional
            Values of the circuit variables, keyed by Variable or name.

        Returns
        -------
        bra : numpy.ndarray
            CI vector of the bra (the ket if the braket is diagonal).
        ket : numpy.ndarray
            CI vector of the ket.

        Raises
        ------
        TequilaException
            If a variable value is missing.
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

        ket = _apply_circuit_to_state(
            self._init_state_ket, self.ket, self._ket_generators, v
        )
        if self.is_diagonal:
            return ket, ket
        bra = _apply_circuit_to_state(
            self._init_state_bra, self.bra, self._bra_generators, v
        )
        return bra, ket

    def extract_variables(self) -> list[Variable]:
        """Return the unique variables of bra and ket."""
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

    @property
    def bra(self) -> FCircuit:
        """FCircuit or None : circuit preparing the bra state, in up-then-down ordering.

        Setting it requires an FCircuit: its initial state, if any, becomes the bra initial state, and
        the ffsim generators of its gates are built.
        """
        return self._bra

    @bra.setter
    def bra(self, bra: FCircuit):
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

    @property
    def ket(self) -> FCircuit:
        """FCircuit or None : circuit preparing the ket state, in up-then-down ordering.

        Setting it requires an FCircuit: its initial state, if any, becomes the ket initial state, and
        the ffsim generators of its gates are built.
        """
        return self._ket

    @ket.setter
    def ket(self, ket: FCircuit):

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

    @property
    def variables_bra(self) -> list[Variable]:
        """list of Variable : variables of the bra circuit."""
        return self.bra.variables

    @property
    def variables_ket(self) -> list[Variable]:
        """list of Variable : variables of the ket circuit."""
        return self.ket.variables

    @property
    def variables(self) -> list[Variable]:
        """list of Variable : variables of bra followed by ket, possibly with repetitions."""
        return self.variables_bra + self.variables_ket

    @property
    def init_state_bra(self) -> QubitWaveFunction:
        """QubitWaveFunction : initial state of the bra before the excitations, usually Hartree-Fock."""
        return self.__civect_to_qwvf(self._init_state_bra)

    @property
    def init_state_ket(self) -> QubitWaveFunction:
        """QubitWaveFunction : initial state of the ket before the excitations, usually Hartree-Fock."""

        return self.__civect_to_qwvf(self._init_state_ket)

    @property
    def init_state(self) -> tuple[QubitWaveFunction, QubitWaveFunction]:
        """tuple of QubitWaveFunction : initial states of bra and ket. Setting it sets both."""
        return self.init_state_bra, self.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra: QubitWaveFunction):
        self._init_state_bra = self.__qwvf_to_civect(init_state_bra)

    @init_state_ket.setter
    def init_state_ket(self, init_state_ket: QubitWaveFunction):
        self._init_state_ket = self.__qwvf_to_civect(init_state_ket)

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
        """bool : True if bra and ket are the same (or one is missing), i.e. the braket is an expectation value."""
        if self.bra is None:
            return True
        if self.ket is None:
            return True
        return bool(self.bra is not None and self.bra == self.ket)

    @property
    def U(self):
        """FCircuit or list of FCircuit : the ket if diagonal, otherwise ``[bra, ket]``. Needed by
        tequila Objectives.
        """
        if self.is_diagonal:
            return self.ket
        else:
            return [self.bra, self.ket]

    def count_measurements(self) -> int:
        """Return a cost measure of the braket: the number of integrals (or the CI-vector length)."""
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
        """Convert an operator into an ffsim linear operator on the CI vector.

        Parameters
        ----------
        operator : str or number or FermionOperator or QubitHamiltonian
            "I" (identity), "H" (molecular Hamiltonian), a constant, an ffsim FermionOperator, an
            openfermion FermionOperator (interleaved spin orbital ordering) or a tequila
            QubitHamiltonian. "HCB" is not supported, use the spex backend.

        Returns
        -------
        scipy.sparse.linalg.LinearOperator

        Raises
        ------
        TequilaException
            If the operator is not supported or does not conserve N and :math:`S_z`.

        Notes
        -----
        A QubitHamiltonian is accepted but discouraged: it is applied to the CI vector through a
        QubitWaveFunction, keeping only the part inside the CI sector, which is slow.
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
    """Return the value of ``x``, evaluating it with ``dvariables`` if it is a Variable or Objective."""
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x


def _qwvf_to_ffsim(wvf: QubitWaveFunction, norb: int, nelec: tuple) -> ndarray:
    # spin orbital p of the up-then-down ordering is orbital p % norb, beta if p >= norb:
    # the bit order of ffsim's strings, so the LSB integer keys are ffsim strings
    """Convert an up-then-down QubitWaveFunction to a dense ffsim CI vector of the sector ``nelec``.

    Raises a TequilaException if the state has components outside the sector.
    """
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
    """Return ``ffsim.linear_operator(op)``, raising a TequilaException if ``op`` does not conserve N and S_z."""
    if not (op.conserves_particle_number() and op.conserves_spin_z()):
        raise TequilaException(
            f"FFSim BraKet: ffsim only supports operators conserving N and S_z, received {op}"
        )
    return ffsim.linear_operator(op, norb, nelec)


def _build_generators(circuit: FCircuit, norb: int, nelec: tuple) -> list:
    """Return, per gate and per term of ``gate.indices``, ``(is_number_operator, generator)``.

    The generator is the ffsim linear operator of the excitation generator of the term.
    """
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
    r"""Apply the gates of ``circuit`` to the CI vector ``state`` with the values in ``dvariables``.

    Since :math:`G^3 = G`, each generator acts as

    .. math::
        e^{i \frac{\theta}{2} G} = 1 + i \sin\frac{\theta}{2} G + \left(\cos\frac{\theta}{2} - 1\right) G^2,

    and number operators (``Phase`` gates) as :math:`1 + (e^{-i \theta / 2} - 1) N`, as in spex.
    """
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
    r"""Simulate an FCircuit with ffsim and return the final wavefunction.

    Parameters
    ----------
    U : FCircuit
        Circuit to simulate. Its initial state fixes the :math:`(n_\alpha, n_\beta)` sector;
        without one an empty wavefunction is returned.
    variables : dict, optional
        Values of the circuit variables.
    n_orb : int, optional
        Number of spatial orbitals. Default is inferred from the circuit.
    **backend_kwargs
        Ignored.

    Returns
    -------
    QubitWaveFunction
        Final state.

    Raises
    ------
    TequilaException
        If a gate does not conserve N and :math:`S_z`.
    """
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
