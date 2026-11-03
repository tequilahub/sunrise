"""Abstract fermionic circuit (``FCircuit``): an ordered list of fermionic gates acting on spin orbitals, independent of any qubit encoding."""

from __future__ import annotations

import copy
import numbers
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable

from numpy import argwhere, array, isclose, ndarray, pi, where
from tequila import (
    BitNumbering,
    BitString,
    QCircuit,
    QubitWaveFunction,
    Variable,
    assign_variable,
)
from tequila.circuit.gates import Phase, QubitExcitationImpl, X
from tequila.utils.exceptions import TequilaException, TequilaWarning

import sunrise
from sunrise.molecules.qubit_base.chemistry_tools import FermionicGateImpl
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase


class FCircuit:
    """Abstract circuit made of fermionic gates (excitations, orbital rotations, phases).

    The circuit is not tied to a qubit mapping. It can be simulated directly or
    converted to a ``tq.QCircuit`` with ``to_qcircuit``. Circuits are
    concatenated with ``+`` / ``+=``.

    Parameters
    ----------
    gates : list of FGateImpl, optional
        Gates of the circuit.
    initial_state : tq.QCircuit or FCircuit or tq.QubitWaveFunction or str or int or list or numpy.ndarray, optional
        Reference state (e.g. the Hartree-Fock determinant). It must be given in
        the up-then-down qubit ordering and must conserve the particle number.
    parameter_map : dict, optional
        Map ``{Variable: [(position, gate), ...]}``. Built from ``gates`` if not given.

    Attributes
    ----------
    depth : int
        Number of gates.
    gates : list of FGateImpl
        Gates of the circuit.
    n_qubits : int
        Number of qubits (spin orbitals) the circuit acts on.
    n_electrons : int or None
        Number of electrons of the initial state (None if there is none).
    qubits : list of int
        Sorted qubits acted upon by the circuit.
    initial_state : tq.QubitWaveFunction
        Reference state, always in up-then-down ordering.

    Examples
    --------
    >>> U = sun.gates.UC(0, 1, "a") + sun.gates.UR(1, 2, "b")
    >>> U = sun.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=4)
    """

    def __init__(
        self,
        gates: list | None = None,
        initial_state: QCircuit | QubitWaveFunction | str | int | None = None,
        parameter_map=None,
    ):
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

    @property
    def initial_state(self) -> QubitWaveFunction:
        """QubitWaveFunction or None : reference state of the circuit, in up-then-down ordering."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state=None):
        if initial_state is None or isinstance(initial_state, QubitWaveFunction):
            pass
        elif isinstance(initial_state, (QCircuit, FCircuit)):
            initial_state = sunrise.simulate(initial_state, variables={})
        elif isinstance(initial_state, str):
            initial_state = QubitWaveFunction.from_string(initial_state)
        elif isinstance(initial_state, (list, ndarray)):
            initial_state = QubitWaveFunction.from_array(initial_state)
        else:
            try:
                initial_state = QubitWaveFunction.convert_from(
                    val=initial_state, n_qubits=self.n_qubits
                )
            except Exception:  # noqa: BLE001
                raise TequilaException(
                    f"Init_state format not recognized, provided {type(initial_state).__name__}"
                )
        if initial_state is not None:
            self.n_qubits = max(self.n_qubits, initial_state.n_qubits)
            initial_state._n_qubits = self.n_qubits
            self._initial_state = initial_state
            self._verify_state()

    @property
    def depth(self):
        """int : number of gates."""
        return len(self.gates)

    @property
    def gates(self) -> list[sunrise.fermionic_excitation.fgateimpl.FGateImpl]:
        """list of FGateImpl : gates of the circuit."""
        if self._gates is None:
            return []
        else:
            return self._gates

    @gates.setter
    def gates(self, other):
        self._gates = list(other)

    @property
    def qubits(self):
        """list of int : sorted qubits acted upon by the gates or covered by the initial state."""
        accumulate = []
        for g in self.gates:
            accumulate += list(g.qubits)
        if self._initial_state is not None:
            accumulate.extend([*range(self._initial_state.n_qubits)])
        return sorted(set(accumulate))

    @property
    def n_qubits(self):
        """int : number of qubits of the circuit. Setting it raises if the value is smaller than the highest qubit used."""
        idx = max(self.max_qubit() + 1, self._min_n_qubits)
        wvf = self._initial_state.n_qubits if self._initial_state is not None else 0
        return max(idx, wvf)

    @property
    def n_electrons(self):
        """int or None : number of electrons in the initial state, None if there is no initial state."""
        if self.initial_state is None:
            return None
        elif isinstance(self.initial_state._state, dict):
            return bin(next(iter(self.initial_state._state.keys())))[2:].count("1")
        else:
            return bin(argwhere(self.initial_state._state > 1.0e-6)[0][0])[2:].count(
                "1"
            )

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
    def variables(self) -> list:
        """list : variables of each gate, one entry per gate (two for ``UR`` gates)."""
        v = []
        for gate in self.gates:
            if gate._name == "UR":
                v.append(gate.variables)
            v.append(gate.variables)
        return v

    def make_parameter_map(self) -> dict:
        """Build the map between variables and the gates that use them.

        Returns
        -------
        dict
            ``{Variable: [(position, gate), ...]}``, e.g.
            ``result[Variable("a")] = [(3, Rx), (5, Ry)]``.
        """
        parameter_map = defaultdict(list)
        for idx, gate in enumerate(self.gates):
            if gate.is_parameterized():
                variables = gate.extract_variables()
                for variable in variables:
                    parameter_map[variable] += [(idx, gate)]
        return parameter_map

    def is_primitive(self):
        """Return True if the circuit consists of a single gate."""
        return len(self.gates) == 1

    def replace_gates(
        self, positions: list, circuits: list, replace: list | None = None
    ):
        """Replace or insert gates at the given positions and return a new circuit.

        Parameters
        ----------
        positions : list of int
            Positions in the original circuit.
        circuits : list of FCircuit or list of FGateImpl
            Gates (or circuits) to place at the corresponding positions.
        replace : list of bool, optional
            If ``replace[i]`` is True the gate at ``positions[i]`` is replaced, if False
            ``circuits[i]`` is inserted before it. Default is True for all.

        Returns
        -------
        FCircuit
            New circuit; the initial state is copied.
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

        result = FCircuit(gates=new_gatelist)
        result.n_qubits = max(result.n_qubits, self.n_qubits)
        result.initial_state = deepcopy(self.initial_state)
        return result

    def insert_gates(self, positions, gates):
        """Insert gates before the given positions and return a new circuit.

        Parameters
        ----------
        positions : list of int
            Positions in the original circuit.
        gates : list of FCircuit or list of FGateImpl
            Gates (or circuits) inserted before the corresponding positions.

        Returns
        -------
        FCircuit

        See Also
        --------
        replace_gates : Replace or insert gates.
        """
        return self.replace_gates(
            positions=positions, circuits=gates, replace=[False] * len(gates)
        )

    def dagger(self):
        """Return the adjoint circuit (gates reversed and individually inverted).

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = UR + UC + UR.dagger()
        """
        result = FCircuit()
        for g in reversed(self.gates):
            result += self.wrap_gate(g.dagger())
        return result

    def extract_variables(self) -> list:
        """Return the variables appearing in the circuit.

        Returns
        -------
        list of tq.Variable
        """
        return list(self._parameter_map.keys())

    def is_fully_parametrized(self):
        """Return True if all gates depend on variables."""
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
        """Return True if no gate depends on variables."""
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
        """Return True if the circuit has both parametrized and fixed gates."""
        return not (self.is_fully_parametrized() or self.is_fully_unparametrized())

    def extract_indices(self) -> list:
        """Return the excitation indices of all gates as one flat list."""
        l = []
        for gate in self.gates:
            l.extend(gate.indices)
        return l

    def max_qubit(self):
        """Return the highest qubit index in the circuit, including the initial state (int)."""
        qmax = 0
        for g in self.gates:
            qmax = max(qmax, g.max_qubit)
        if self.initial_state is not None:
            qmax = max(qmax, self.initial_state.n_qubits - 1)
        return qmax

    def __iadd__(self, other):
        """Append the gates of ``other`` in place. Raises if both circuits have an initial state."""
        if hasattr(other, "initial_state"):
            w = other.initial_state
        else:
            w = None
        other = self.wrap_gate(gate=other.gates)
        other.initial_state = w
        offset = len(self.gates)
        for k, v in other._parameter_map.items():
            self._parameter_map[k] += [(x[0] + offset, x[1]) for x in v]

        self._gates += other.gates
        self._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        if self._initial_state is None:
            self._initial_state = other._initial_state
        elif (
            other._initial_state is not None
        ):  # and other._initial_state != self._initial_state:
            raise TequilaException(
                f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}"
            )
        return self

    def __add__(self, other):
        """Return a new circuit with the gates of ``other`` appended. Raises if both circuits have an initial state."""
        if hasattr(other, "initial_state"):
            w = other.initial_state
        else:
            w = None
        other = self.wrap_gate(gate=other.gates)
        other.initial_state = w
        gates = [deepcopy(g) for g in (self.gates + other.gates)]
        result = FCircuit(gates=gates)
        result._min_n_qubits = max(self._min_n_qubits, other._min_n_qubits)
        initial_state = self.initial_state
        if self._initial_state is None:
            initial_state = other._initial_state
        elif (
            other._initial_state is not None
        ):  #  and other._initial_state != self._initial_state:
            raise TequilaException(
                f"FermionicCircuit + FermionicCircuit with two different initial states:\n{self._initial_state}, {other._initial_state}"
            )
        result.initial_state = initial_state
        return result

    def __str__(self):
        result = "Fermionic circuit: \n"
        if self.initial_state is not None:
            result += "with Initial State: \n"
            result += str(self.initial_state) + "\n"
        if len(self.gates):
            result += "with gates:\n"
            for g in self.gates:
                result += str(g) + "\n"
        return result

    def __eq__(self, other) -> bool:
        if len(self.gates) != len(other.gates):
            return False
        for i, gate in enumerate(self.gates):
            if gate != other.gates[i]:
                return False
        return True  # IDEA avoiding initial state comparations to avoid
        if len(self.initial_state.to_array()) != len(other.initial_state.to_array()):
            return False
        if self.initial_state.numbering != other.initial_state.numbering:
            return False
        return isclose(
            self.initial_state.inner(other=other.initial_state), 1.0, atol=1.0e-4
        )

    def __repr__(self):
        return self.__str__()

    def to_upthendown(self, norb) -> FCircuit:
        """Return a copy whose gate indices use the up-then-down spin orbital ordering.

        The initial state is not reordered, it is always in up-then-down ordering.

        Parameters
        ----------
        norb : int
            Number of spatial orbitals.

        Returns
        -------
        FCircuit
        """
        u = []
        for gate in self.gates:
            g = deepcopy(gate)
            if hasattr(g, "to_upthendown"):
                g = g.to_upthendown(norb)
                g.reordered = True
            u.append(g)
        return FCircuit(
            gates=u, parameter_map=self._parameter_map, initial_state=self.initial_state
        )

    def to_udud(self, norb) -> FCircuit:
        """Return a copy whose gate indices use the interleaved (up, down, up, down, ...) ordering.

        The initial state is not reordered, it is always in up-then-down ordering.

        Parameters
        ----------
        norb : int
            Number of spatial orbitals.

        Returns
        -------
        FCircuit
        """
        u = []
        for gate in self.gates:
            g = deepcopy(gate)
            if hasattr(g, "to_udud"):
                g = g.to_udud(norb)
                g.reordered = False
            u.append(g)
        return FCircuit(
            gates=u, parameter_map=self._parameter_map, initial_state=self.initial_state
        )

    @classmethod
    def from_Qcircuit(cls, circuit: QCircuit, **kwargs):
        """Build an FCircuit from a ``tq.QCircuit`` made of fixed gates followed by (qubit) excitation gates.

        Leading fixed gates become the initial state, parametrized excitation gates
        become ``FermionicExcitation`` gates.

        Parameters
        ----------
        circuit : tq.QCircuit
            Circuit to convert.
        variables : dict, optional
            Passed through ``circuit.map_variables`` before the conversion.
        reordered : bool, optional
            Whether the indices of the circuit are in up-then-down ordering.
            Default is True. False requires a circuit without fixed gates.

        Returns
        -------
        FCircuit

        Raises
        ------
        TequilaException
            If the circuit contains unsupported gates.
        """
        operations = FCircuit()
        reference = QCircuit()
        reordered = kwargs.get("reordered", True)
        beginning = True
        if "variables" in kwargs:
            circuit = circuit.map_variables(kwargs["variables"])
            kwargs.pop("variables")
        for gate in circuit.gates:
            if (
                beginning
                and not hasattr(gate, "_parameter")
                or isinstance(gate._parameter, numbers.Number)
            ):
                reference += gate
            elif isinstance(
                gate, QubitExcitationImpl
            ):  # maybe we can consider other gates but basic implementation for the moment
                if isinstance(gate._parameter, numbers.Number):
                    reference += gate
                elif isinstance(gate, FermionicGateImpl):
                    beginning = False
                    operations += sunrise.gates.FermionicExcitation(
                        indices=gate.indices,
                        variables=gate.parameter,
                        reordered=reordered,
                    )
                else:
                    temp = []
                    for i in range(len(gate._target) // 2):
                        temp.append((gate._target[2 * i], gate._target[2 * i + 1]))
                    operations += sunrise.gates.FermionicExcitation(
                        indices=temp, variables=gate.parameter, reordered=reordered
                    )
            else:
                raise TequilaException(
                    f"Gate {gate._name}({gate._parameter}) not allowed"
                )
        if not reordered and len(reference.gates):
            raise TequilaException(
                "reordered=False only allowed if any initial_state provided due to backend restrictions"
            )
        return cls(gates=operations._gates, initial_state=reference)

    @classmethod
    def from_edges(
        cls,
        edges: list | tuple | None,
        label=None,
        n_orb: int = 0,
        use_units_of_pi=False,
        ladder=True,
    ):
        """Build a circuit of ``UC`` gates from a list of edges (paths of spatial orbitals).

        For each edge ``(a, b, c, ...)`` one ``UC`` gate is added per consecutive pair,
        with variable ``((i, j), "D", label)``.

        Parameters
        ----------
        edges : list of tuple of int
            Orbital paths, e.g. ``[(0, 1), (2, 3)]``.
        label : optional
            Label added to the variable names.
        n_orb : int, optional
            Number of spatial orbitals. If given, the initial state is the determinant
            with the first orbital of each edge doubly occupied.
        use_units_of_pi : bool, optional
            Multiply the angles by pi. Default is False.
        ladder : bool, optional
            If True (default) pairs are consecutive ``(a, b), (b, c), ...``, otherwise
            all pairs start from the first orbital ``(a, b), (a, c), ...``.

        Returns
        -------
        FCircuit

        References
        ----------
        J. S. Kottmann and A. Aspuru-Guzik, Phys. Rev. A 105, 032449 (2022), arXiv:2105.03836.

        Examples
        --------
        >>> U = sun.FCircuit.from_edges([(0, 1), (2, 3)], n_orb=4)
        """
        operations = FCircuit()
        if n_orb:
            include_reference = True
            n_state = [0] * 2 * n_orb
            reference = QubitWaveFunction(
                n_qubits=2 * n_orb, dense=False, numbering=BitNumbering.LSB
            )
        else:
            include_reference = False
            reference = None
        for edge in edges:
            if include_reference:
                n_state[edge[0]] = 1
                n_state[edge[0] + n_orb] = 1
            previous = edge[0]
            for i in edge[1:]:
                angle = Variable(((previous, i), "D", label))
                if use_units_of_pi:
                    angle = angle * pi
                operations += sunrise.gates.UC(i=previous, j=i, variables=angle)
                if ladder:
                    previous = i
        if include_reference:
            reference[BitString.from_array(n_state)] = 1.0
        return cls(gates=operations._gates, initial_state=reference)

    @staticmethod
    def wrap_gate(gate) -> FCircuit:
        """Wrap a gate (or list of gates) in an FCircuit; an FCircuit is returned unchanged.

        Parameters
        ----------
        gate : FGateImpl or list of FGateImpl or FCircuit

        Returns
        -------
        FCircuit
        """
        if isinstance(gate, FCircuit):
            return gate
        if isinstance(gate, list):
            return FCircuit(gates=gate)
        else:
            return FCircuit(gates=[gate])

    def verify(self) -> bool:
        """Check that the parameter map is consistent with the gates.

        Returns
        -------
        bool
        """
        if not len(self._gates) is None and self.initial_state is None:
            return True
        test = []
        for (
            k,
            v,
        ) in self._parameter_map.items():
            test = [self.gates[x[0]] == x[1] for x in v]
            test += [k in self._gates[x[0]].extract_variables() for x in v]
        return all(test)

    def map_qubits(self, qubit_map):
        """Return a new circuit with relabeled qubits.

        The initial state is not carried over.

        Parameters
        ----------
        qubit_map : dict
            Map from old to new qubit indices, e.g. ``{1: 3, 2: 1}``.

        Returns
        -------
        FCircuit
        """

        new_gates = [gate.map_qubits(qubit_map) for gate in self.gates]
        # could speed up by applying qubit_map to parameter_map here
        # currently its recreated in the init function
        return FCircuit(gates=new_gates)  # TODO: initial state?

    def map_variables(self, variables: dict, *args, **kwargs):
        """Return a new circuit with variables replaced.

        Parameters
        ----------
        variables : dict
            Map from old variables (or names) to new variables or numerical values.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U0 = U.map_variables({d: 0 for d in U.extract_variables()})  # all angles set to 0
        """

        variables = {
            assign_variable(k): assign_variable(v) for k, v in variables.items()
        }

        # failsafe
        my_variables = self.extract_variables()
        for k in variables:
            if k not in my_variables:
                warnings.warn(
                    f"map_variables: variable {k} is not part of circuit with variables {my_variables}",
                    TequilaWarning,
                )

        new_gates = [
            copy.deepcopy(gate).map_variables(variables) for gate in self.gates
        ]

        return FCircuit(gates=new_gates, initial_state=self.initial_state)

    def to_matrix(self, variables=None):  # TODO:  Should we?
        """Not implemented."""

    def add_controls(self, control, inpl: bool = False) -> FCircuit | None:  # TODO
        """Not implemented."""

    def to_qcircuit(
        self,
        molecule: QuantumChemistryBase | None = None,
        transformation: Callable | None = None,
    ) -> QCircuit:
        """Convert to a ``tq.QCircuit``.

        Parameters
        ----------
        molecule : QuantumChemistryBase, optional
            Molecule providing ``make_excitation_gate`` and the fermion-to-qubit transformation.
        transformation : callable, optional
            Function ``transformation(FCircuit) -> QCircuit``. Used if no molecule is given.

        Returns
        -------
        tq.QCircuit

        Raises
        ------
        TequilaException
            If neither argument is given or a gate type is not supported.

        Examples
        --------
        >>> Uq = U.to_qcircuit(molecule=mol)
        """
        U = deepcopy(self)
        if molecule is not None:
            res = QCircuit()
            U = U.to_udud(molecule.n_orbitals)
            molecule.transformation.upthendown = True
            if self.initial_state is not None:
                idx = where(array(self._initial_state.to_array()) > 1.0e-6)[
                    0
                ]  # TODO change to dict
                if not len(idx):
                    pass
                elif len(idx) > 1:
                    warnings.warn(
                        "Don't now how to prepare the circuit initial state, skyped for safety",
                        TequilaWarning,
                    )
                else:
                    res += X(
                        target=[
                            i
                            for i in range(len(bin(idx[0])[2:]))
                            if bin(idx[0])[2:][i] == "1"
                        ]
                    )
            for gate in U.gates:
                if gate.name == "UR":
                    res += molecule.make_excitation_gate(
                        indices=gate.indices[0], angle=gate.variables
                    )  # TODO: Control
                    res += molecule.make_excitation_gate(
                        indices=gate.indices[1], angle=gate.variables
                    )  # TODO: Control
                elif gate.name in ["UC", "FermionicExcitation", "GenericFermionic"]:
                    res += molecule.make_excitation_gate(
                        indices=gate.indices[0], angle=gate.variables
                    )  # TODO: Control
                elif gate.name == "Ph":
                    res += Phase(
                        target=(
                            gate.indices[0][0][0] // 2
                            + (gate.indices[0][0][0] % 2) * molecule.n_orbitals
                        ),
                        angle=-0.5 * gate.variables,
                        # Ph(v) = exp(-i*v/2*n) while tq Phase(a) = exp(i*a*n)
                    )  # TODO: Control
                else:
                    raise TequilaException(f"Gate {gate} not idenified")
            return res
        elif transformation is not None:
            return transformation(U)
        else:
            raise TequilaException(
                "No Possible to transform from FCircuit to QCircuit without further information"
            )

    def _verify_state(self):
        """Check that the initial state conserves the particle number and is real."""
        if isinstance(self.initial_state._state, dict):
            nozero = [bin(i)[2:] for i in self._initial_state._state]
            ne = nozero[0].count("1")
            if not all(st.count("1") == ne for st in nozero):
                raise TequilaException("The Initial State is not Particle-conserving")
            if not all(
                isclose(i.imag, 0.0, atol=1.0e-6)
                for i in self._initial_state._state.values()
            ):
                raise TequilaException("No Imaginary states for Chemistry States")
        else:
            return True  # Since dense QubitWavefunction scales 2^N, it becomes quickly intractable

    def _is_upthendown(self):
        """True if the gates currently carry upthendown indices."""
        return any(getattr(g, "reordered", False) for g in self.gates)

    def _is_mixed_ordering(self):
        """True if the circuit contains both reordered and non-reordered fermionic gates."""
        flags = {
            getattr(g, "reordered", False) for g in self.gates if hasattr(g, "_render")
        }
        return len(flags) > 1

    def render(self, state, style, show_spatial_orbitals=True):
        """Return the qpic text for the circuit (used by ``export_qpic``)."""
        # Inline initial wires logic
        iw = set()
        if self.initial_state is not None:
            from numpy import abs as nabs
            from numpy import argmax, where

            arr = nabs(self.initial_state.to_array())
            idx = where(arr > 1e-6)[0]
            if len(idx) != 1:
                idx = [int(argmax(arr))]
            n_state = self.initial_state.n_qubits
            n = max(n_state, self.n_qubits)
            bits = format(int(idx[0]), f"0{n_state}b").ljust(n, "0")
            norb = n // 2
            occ = [q for q, b in enumerate(bits) if b == "1"]

            is_upthendown = any(
                getattr(g, "reordered", False)
                for g in self.gates
                if hasattr(g, "_render")
            )
            if show_spatial_orbitals:
                iw = {q if q < norb else q - norb for q in occ}
            elif is_upthendown:
                iw = set(occ)
            else:
                iw = {2 * (q % norb) + (q >= norb) for q in occ}

        chunks = []
        if iw:
            chunks.append(" ".join(f"a{q}" for q in sorted(iw)) + " G I ")

        pending = []
        for gate in self.gates:
            if hasattr(gate, "_render"):
                if pending:
                    chunks.append(
                        " ".join(f"a{q}" for q in sorted(set(pending))) + " G G "
                    )
                    pending = []
                chunks.append(gate._render(state, style, show_spatial_orbitals))
            else:
                pending += list(gate.qubits)

        if pending:
            chunks.append(" ".join(f"a{q}" for q in sorted(set(pending))) + " G G ")

        out = "\n".join(chunks) + "\n"
        if style.group_together:
            out += "\n" + state.all_names() + " TOUCH\n"
        return out

    def export_qpic(
        self,
        filename,
        filepath=None,
        style=None,
        show_spatial_orbitals=True,
        labels=None,
        colors=None,
        wire_colors=None,
        select=None,
        **kwargs,
    ):
        """Write the circuit as a qpic file ``<filename>.qpic``.

        Parameters
        ----------
        filename : str
            File name without extension.
        filepath : str, optional
            Output directory.
        style : dict, optional
            Drawing style options.
        show_spatial_orbitals : bool, optional
            Draw one wire per spatial orbital instead of per spin orbital. Default is True.
        labels, colors, wire_colors : dict, optional
            Custom gate labels, colors and wire colors.
        select : dict, optional
            Map ``{wire: "B"}``; wires marked ``"B"`` are drawn red.
        """
        from os import path

        from sunrise.graphical.core.color import DefaultColors
        from sunrise.graphical.core.state import CircuitState
        from sunrise.graphical.core.style import parse_circuit_style

        if style is None:
            style = {}
        if labels is None:
            labels = {}
        if colors is None:
            colors = {}
        if wire_colors is None:
            wire_colors = {}
        if not len(wire_colors) and select is not None:
            wire_colors = dict(wire_colors)
            for s in select:
                if select[s] == "B":
                    wire_colors[s] = "red"

        result = ""
        style_obj, colors = parse_circuit_style(style, colors)

        for color, rgb in DefaultColors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"
        for color, rgb in colors.items():
            result += f"COLOR {color} {rgb.r} {rgb.g} {rgb.b}\n"

        # Inline _used_wires and _initial_wires logic
        norb = self.n_qubits // 2
        wires = set()
        if self.initial_state is not None:
            from numpy import abs as nabs
            from numpy import argmax, where

            arr = nabs(self.initial_state.to_array())
            idx = where(arr > 1e-6)[0]
            if len(idx) != 1:
                idx = [int(argmax(arr))]
            n_state = self.initial_state.n_qubits
            n = max(n_state, self.n_qubits)
            bits = format(int(idx[0]), f"0{n_state}b").ljust(n, "0")
            occ = [q for q, b in enumerate(bits) if b == "1"]
            is_upthendown = any(
                getattr(g, "reordered", False)
                for g in self.gates
                if hasattr(g, "_render")
            )
            if show_spatial_orbitals:
                wires.update(q if q < norb else q - norb for q in occ)
            elif is_upthendown:
                wires.update(occ)
            else:
                wires.update(2 * (q % norb) + (q >= norb) for q in occ)

        for g in self.gates:
            if hasattr(g, "_render"):
                if not show_spatial_orbitals:
                    wires.update(g.qubits)
                elif getattr(g, "reordered", False):
                    wires.update(q if q < norb else q - norb for q in g.qubits)
                else:
                    wires.update(q // 2 for q in g.qubits)
            else:
                wires.update(g.qubits)

        if not wires:
            wires = {0}
        max_wire = max(wires)

        for wire in range(max_wire + 1):
            label = labels.get(wire, str(wire))
            color = wire_colors.get(wire, "black")
            result += f"color={color} a{wire} W {label}\n"
        for wire in range(max_wire + 1):
            result += f"a{wire} /\n"

        state_obj = CircuitState(self.n_qubits // 2)
        result += self.render(state_obj, style_obj, show_spatial_orbitals) + "\n"

        ext = "" if filename.endswith(".qpic") else ".qpic"
        final_path = path.join(filepath, filename + ext) if filepath else filename + ext
        with open(final_path, "w") as file:
            file.write(result)

    def export_to(self, filename: str, *args, **kwargs):
        """Draw the circuit to a file.

        Parameters
        ----------
        filename : str
            Output name. The extension selects the format: ``qpic``, ``pdf`` (default) or ``png``.
        n_orb : int, optional
            Number of spatial orbitals. Default is ``n_qubits // 2``.
        show_spatial_orbitals : bool, optional
            One wire per spatial orbital. Default is True.
        **kwargs
            Passed to ``export_qpic``.
            ``filepath`` is the directory of the output file; ``n_qubits_is_double`` is an old
            alias (True shows spin orbitals, i.e. ``show_spatial_orbitals=False``).

        Raises
        ------
        TequilaException
            If the extension is not supported.

        Examples
        --------
        >>> U.export_to("circuit.pdf", style={"single": "red", "generic": "#AC61AC"})  # doctest: +SKIP
        """
        from copy import deepcopy

        from tequila import TequilaException

        from sunrise.graphical.qpic_visualization import qpic_to_pdf, qpic_to_png

        U = deepcopy(self)
        n_orb = kwargs.pop("n_orb", self.n_qubits // 2)
        if "n_qubits_is_double" in kwargs:
            show_spatial_orbitals = not kwargs.pop("n_qubits_is_double")
        else:
            show_spatial_orbitals = kwargs.pop("show_spatial_orbitals", True)
        filepath = kwargs.get("filepath", None)

        # Inline _is_mixed_ordering check
        flags = {
            getattr(g, "reordered", False) for g in U.gates if hasattr(g, "_render")
        }
        if len(flags) > 1:
            try:
                U = U.to_upthendown(n_orb)
            except AttributeError:
                pass

        if show_spatial_orbitals:
            try:
                U = U.to_udud(n_orb)
            except AttributeError:
                pass

        filename_tmp = filename.split(".")
        if len(filename_tmp) == 1:
            ftype, fname = "pdf", filename
        else:
            ftype, fname = filename_tmp[-1], ".".join(filename_tmp[:-1])

        if ftype == "qpic":
            U.export_qpic(
                fname, *args, show_spatial_orbitals=show_spatial_orbitals, **kwargs
            )
        elif ftype == "pdf":
            U.export_qpic(
                fname, *args, show_spatial_orbitals=show_spatial_orbitals, **kwargs
            )
            qpic_to_pdf(filename=fname, filepath=filepath)
        elif ftype == "png":
            U.export_qpic(
                fname, *args, show_spatial_orbitals=show_spatial_orbitals, **kwargs
            )
            qpic_to_png(filename=fname, filepath=filepath)
        else:
            raise TequilaException(f"Extension {ftype} not supported directly.")


if __name__ == "__main__":
    import tequila as tq

    import sunrise as sun

    U = sun.gates.FermionicExcitation([(0, 2), (1, 3)], "a")
    U.initial_state = tq.gates.X([0, 4])
    U += sun.gates.UC(0, 1, "b")
    U = U + sun.gates.UR(1, 2, "c")
    U += sun.gates.FermionicExcitation([(1, 2)], "d", reordered=True)
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
    B = FCircuit.from_edges([(0, 1), (2, 3)])
    print(B.extract_variables())
    print(B.extract_indices())
    print(B)
    C = FCircuit.from_edges([(0, 1), (2, 3)], n_orb=4)
    print(C)
