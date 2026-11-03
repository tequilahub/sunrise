"""Fermion-to-qubit encodings (Jordan-Wigner, Bravyi-Kitaev, tapered variants, ...).

Most are interfaces to OpenFermion. ``known_encodings`` maps names to classes.
"""

import abc

import numpy
import openfermion
from tequila import TequilaException
from tequila.circuit.circuit import QCircuit
from tequila.circuit.gates import CNOT, X
from tequila.hamiltonian.qubit_hamiltonian import QubitHamiltonian


def known_encodings():
    # convenience for testing and I/O
    """Return the available encodings as dict from normalized name to class or factory.

    Names are upper case without "_" and "-", e.g. "JORDANWIGNER" or "REORDEREDJORDANWIGNER"
    (the "Reordered" aliases use ``up_then_down=True``).

    Examples
    --------
    >>> sorted(sun.chemistry.encodings.known_encodings())
    """
    encodings = {
        "JordanWigner": JordanWigner,
        "BravyiKitaev": BravyiKitaev,
        "BravyiKitaevFast": BravyiKitaevFast,
        "BravyiKitaevTree": BravyiKitaevTree,
        "TaperedBravyiKitaev": TaperedBravyiKitaev,
        "TaperedBinary": TaperedBinary,
    }
    # aliases
    encodings = {
        **encodings,
        "ReorderedJordanWigner": lambda **kwargs: JordanWigner(
            up_then_down=True, **kwargs
        ),
        "ReorderedBravyiKitaev": lambda **kwargs: BravyiKitaev(
            up_then_down=True, **kwargs
        ),
        "ReorderedBravyiKitaevTree": lambda **kwargs: BravyiKitaevTree(
            up_then_down=True, **kwargs
        ),
        "ReorderedTaperedBinary": lambda **kwargs: TaperedBinary(
            up_then_down=True, **kwargs
        ),
    }
    return {
        k.replace("_", "").replace("-", "").upper(): v for k, v in encodings.items()
    }


class EncodingBase(metaclass=abc.ABCMeta):
    # true if the encoding is fully integrated
    # false: can only do special things (like building the Hamiltionian)
    # but is not consistent with UCC gate generation
    """Base class of fermion-to-qubit encodings.

    Calling an instance with an openfermion FermionOperator returns the QubitHamiltonian.
    Spin orbitals are ordered (0_up, 0_down, 1_up, 1_down, ...), or with ``up_then_down``
    all up orbitals come first in the qubit register.

    Parameters
    ----------
    n_electrons : int
        Number of (active) electrons.
    n_orbitals : int
        Number of (active) spatial orbitals.
    up_then_down : bool, optional
        Order the qubits as all up then all down spin orbitals. Default is False.
    """

    _ucc_support = False

    @property
    def supports_ucc(self):
        """bool : True if the encoding is consistent with general UCC gates, False if it only supports special things like building the Hamiltonian."""
        return self._ucc_support

    @property
    def name(self):
        """str : name of the encoding, prefixed with "Reordered" if ``up_then_down``."""
        prefix = ""
        if self.up_then_down:
            prefix = "Reordered"
        if hasattr(self, "_name"):
            return prefix + self._name
        else:
            return prefix + type(self).__name__

    def __init__(self, n_electrons, n_orbitals, up_then_down=False, *args, **kwargs):
        """See the class docstring."""
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals
        self.up_then_down = up_then_down

    def __call__(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> QubitHamiltonian:
        """Return the qubit operator of a fermionic operator in this encoding.

        Parameters
        ----------
        fermion_operator : openfermion.FermionOperator

        Returns
        -------
        QubitHamiltonian
        """
        if self.up_then_down:
            op = openfermion.reorder(
                operator=fermion_operator,
                order_function=openfermion.up_then_down,
                num_modes=2 * self.n_orbitals,
            )
        else:
            op = fermion_operator

        fop = self.do_transform(*args, fermion_operator=op, **kwargs)
        fop.compress()
        return self.post_processing(QubitHamiltonian.from_openfermion(fop))

    def post_processing(self, op, *args, **kwargs):
        """Return the qubit operator after the transformation (identity by default, override if needed)."""
        return op

    def up(self, i):
        """Return the qubit index of the up spin orbital of spatial orbital ``i``.

        Examples
        --------
        >>> q = mol.transformation.up(0)  # qubit of the up spin orbital of orbital 0
        """
        if self.up_then_down:
            return i
        else:
            return 2 * i

    def down(self, i):
        """Return the qubit index of the down spin orbital of spatial orbital ``i``.

        Examples
        --------
        >>> q = mol.transformation.down(0)  # qubit of the down spin orbital of orbital 0
        """
        if self.up_then_down:
            return i + self.n_orbitals
        else:
            return 2 * i + 1

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        """Return the openfermion QubitOperator of a FermionOperator, implemented by the encodings."""
        raise NotImplementedError(
            f"{type(self).__name__}::do_transform: called base class"
        )

    def map_state(self, state: list, *args, **kwargs) -> list:
        """Return the qubit basis state corresponding to a spin orbital occupation vector.

        Parameters
        ----------
        state : list of int
            Occupation numbers of the spin orbitals ``[0_up, 0_down, 1_up, 1_down, ...]``.

        Returns
        -------
        list of int
            Basis state in this encoding.
        """
        # default is a lazy workaround, but it workds
        n_qubits = 2 * self.n_orbitals

        spin_orbitals = sorted([i for i, x in enumerate(state) if int(x) == 1])

        string = "1.0 ["
        for i in spin_orbitals:
            string += str(i) + "^ "
        string += "]"

        fop = openfermion.FermionOperator(string, 1.0)
        op = self(fop)
        from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction

        wfn = QubitWaveFunction.from_basis_state(n_qubits, 0)
        wfn = wfn.apply_qubitoperator(operator=op)
        assert wfn.length() == 1
        key = next(iter(wfn.keys())).array
        return key

    @abc.abstractmethod
    def me_to_jw(self) -> QCircuit:
        """Return the circuit from this encoding to Jordan-Wigner (needed for the default conversions)."""

    # independent conversion methods, these are used for default conversions
    # arXiv:1808.10402 IV. B. 2, Eq. 57
    # original: https://doi.org/10.1063/1.4768229
    def _jw_to_bk(self) -> QCircuit:
        """Return the circuit from Jordan-Wigner to Bravyi-Kitaev."""
        U = QCircuit()  # Constructs empty circuit

        flipper = False
        for i in range(self.n_orbitals * 2):
            # even qubits only hold their own value
            if i % 2 == 0:
                continue

            # sum always includes the last qubit
            U += CNOT(control=i - 1, target=i)

            # every second odd qubit ties together with the last odd qubit
            if flipper:
                U += CNOT(control=i - 2, target=i)

            flipper = not flipper

            # we have now created the 4x4 blocks on the diagonal of this operators matrix

            # every power of 2 connects to the last power of 2
            # this corresponds to the last row in the recursive definitions being all 1s
            x = numpy.log2(i + 1)
            if x.is_integer() and x >= 3:
                x = int(x)
                U += CNOT(control=2 ** (x - 1) - 1, target=i)

        return U

    def _hcb_to_jw(self):
        """Return the circuit from hard-core boson to Jordan-Wigner (CNOTs from up to down qubits)."""
        U = QCircuit()
        for i in range(self.n_orbitals):
            U += X(target=self.down(i), control=self.up(i))
        return U

    # Convenience Methods
    def jw_to_me(self) -> QCircuit:
        """Return the circuit from Jordan-Wigner to this encoding."""
        return self.me_to_jw().dagger()

    def me_to_bk(self) -> QCircuit:
        """Return the circuit from this encoding to Bravyi-Kitaev."""
        return self.me_to_jw() + self._jw_to_bk()

    def bk_to_me(self) -> QCircuit:
        """Return the circuit from Bravyi-Kitaev to this encoding."""
        return self.me_to_bk().dagger()

    def hcb_to_me(self) -> QCircuit:
        """Return the circuit from the hard-core boson encoding to this encoding.

        Examples
        --------
        >>> U = mol.transformation.hcb_to_me()
        """
        return self._hcb_to_jw() + self.jw_to_me()

    def __str__(self):
        return type(self).__name__


class JordanWigner(EncodingBase):
    """Jordan-Wigner encoding (``openfermion.jordan_wigner``)."""

    _ucc_support = True

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        return openfermion.jordan_wigner(fermion_operator, *args, **kwargs)

    def map_state(self, state: list, *args, **kwargs):
        state = state + [0] * (2 * self.n_orbitals - len(state))
        if self.up_then_down:
            return [state[2 * i] for i in range(self.n_orbitals)] + [
                state[2 * i + 1] for i in range(self.n_orbitals)
            ]
        else:
            return state

    def hcb_to_me(self, *args, **kwargs):
        U = QCircuit()
        for i in range(self.n_orbitals):
            U += X(target=self.down(i), control=self.up(i))
        return U

    def me_to_jw(self) -> QCircuit:
        return QCircuit()

    def jw_to_me(self) -> QCircuit:
        return QCircuit()


class BravyiKitaev(EncodingBase):
    """Bravyi-Kitaev encoding (``openfermion.bravyi_kitaev``)."""

    _ucc_support = True

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        return openfermion.bravyi_kitaev(fermion_operator, n_qubits=self.n_orbitals * 2)

    def me_to_jw(self) -> QCircuit:
        return self._jw_to_bk().dagger()

    def jw_to_me(self) -> QCircuit:
        return self._jw_to_bk()

    def bk_to_me(self) -> QCircuit:
        return QCircuit()

    def me_to_bk(self) -> QCircuit:
        return QCircuit()


class BravyiKitaevTree(EncodingBase):
    """Bravyi-Kitaev tree encoding (``openfermion.bravyi_kitaev_tree``)."""

    _ucc_support = True

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        return openfermion.bravyi_kitaev_tree(
            fermion_operator, n_qubits=self.n_orbitals * 2
        )

    def me_to_jw(self) -> QCircuit:
        raise TequilaException(f"{type(self).__name__}::me_to_jw: unimplemented")


class BravyiKitaevFast(EncodingBase):
    """Bravyi-Kitaev superfast encoding (``openfermion.bravyi_kitaev_fast``).

    Only works for full Hamiltonians, not for UCC generators.

    References
    ----------
    S. B. Bravyi and A. Y. Kitaev, Ann. Phys. 298, 210 (2002).
    K. Setia and J. D. Whitfield, J. Chem. Phys. 148, 164104 (2018).
    """

    _ucc_support = False

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        n_qubits = openfermion.count_qubits(fermion_operator)
        if n_qubits != self.n_orbitals * 2:
            raise NotImplementedError(
                f"BravyiKitaevFast transformation currently only possible for full Hamiltonians (no UCC generators).\nfermion_operator was {fermion_operator}"
            )
        op = openfermion.get_interaction_operator(fermion_operator)
        return openfermion.bravyi_kitaev_fast(op)

    def me_to_jw(self) -> QCircuit:
        raise TequilaException(f"{type(self).__name__}::me_to_jw: unimplemented")


class TaperedBravyiKitaev(EncodingBase):
    """Tapered Bravyi-Kitaev encoding (``openfermion.symmetry_conserving_bravyi_kitaev``).

    Reduces the Hamiltonian by 2 qubits, ``up_then_down`` cannot be changed. Does not
    work for UCC generators yet. See the OpenFermion documentation for details.

    Parameters
    ----------
    n_electrons, n_orbitals : int
        Number of electrons and spatial orbitals.
    active_fermions : int, optional
        Default is ``n_electrons``.
    active_orbitals : int, optional
        Number of active spin orbitals. Default is ``2 * n_orbitals``.

    References
    ----------
    S. Bravyi, J. M. Gambetta, A. Mezzacapo and K. Temme, arXiv:1701.08213 (2017) (tapering).
    V. Havlíček, M. Troyer and J. D. Whitfield, Phys. Rev. A 95, 032332 (2017), arXiv:1701.07072
    (Bravyi-Kitaev tree encoding).
    """

    _ucc_support = False

    def __init__(
        self,
        n_electrons,
        n_orbitals,
        active_fermions=None,
        active_orbitals=None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        if active_fermions is None:
            self.active_fermions = n_electrons
        else:
            self.active_fermions = active_fermions

        if active_orbitals is None:
            self.active_orbitals = (
                n_orbitals * 2
            )  # in openfermion those are spin-orbitals
        else:
            self.active_orbitals = active_orbitals

        if "up_then_down" in kwargs:
            raise TequilaException(
                f"Don't pass up_then_down argument to {type(self).__name__}, it can't be changed"
            )
        super().__init__(
            *args,
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            up_then_down=False,
            **kwargs,
        )

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        if openfermion.count_qubits(fermion_operator) != self.n_orbitals * 2:
            raise NotImplementedError(
                "TaperedBravyiKitaev not ready for UCC generators yet"
            )
        return openfermion.symmetry_conserving_bravyi_kitaev(
            fermion_operator,
            active_orbitals=self.active_orbitals,
            active_fermions=self.active_fermions,
        )

    def map_state(self, state: list, *args, **kwargs):
        non_tapered_trafo = BravyiKitaevTree(
            up_then_down=True, n_electrons=self.n_electrons, n_orbitals=self.n_orbitals
        )
        key = non_tapered_trafo.map_state(*args, state=state, **kwargs)
        n_qubits = self.n_orbitals * 2
        active_qubits = [
            i for i in range(n_qubits) if i not in [n_qubits - 1, n_qubits // 2 - 1]
        ]
        key = [key[i] for i in active_qubits]
        return key

    def me_to_jw(self) -> QCircuit:
        raise TequilaException(f"{type(self).__name__}::me_to_jw: unimplemented")


class TaperedBinary(EncodingBase):
    """Tapered binary encoding (``openfermion.binary_code_transform`` with a checksum code).

    Reduces the Hamiltonian by 2 qubits. See the OpenFermion documentation for details.

    Parameters
    ----------
    n_electrons, n_orbitals : int
        Number of electrons and spatial orbitals.
    active_fermions : int, optional
        Default is ``n_electrons``.
    active_orbitals : int, optional
        Number of active spin orbitals. Default is ``2 * n_orbitals``.
    up_then_down : bool, optional
        Order the qubits as all up then all down spin orbitals. Default is False.

    References
    ----------
    M. Steudtner and S. Wehner, New J. Phys. 20, 063010 (2018), arXiv:1712.07067.
    """

    _ucc_support = True

    def __init__(
        self,
        n_electrons,
        n_orbitals,
        active_fermions=None,
        active_orbitals=None,
        up_then_down=False,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        if active_fermions is None:
            self.active_fermions = n_electrons  # it maybe changed at some point to support different number of alpha and beta electr
        else:
            self.active_fermions = active_fermions
        if active_orbitals is None:
            self.active_orbitals = (
                n_orbitals * 2
            )  # in openfermion those are spin-orbitals
        else:
            self.active_orbitals = active_orbitals
        super().__init__(
            *args,
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            up_then_down=up_then_down,
            **kwargs,
        )
        self.spin_parity = bool(
            (self.n_electrons / 2) % 2
        )  # total parity of all \sigma\in\{\alpha\beta\} SOs

        self.modes = self.active_orbitals  # asumed n alpha = n beta

    def __call__(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> QubitHamiltonian:
        """Return the qubit operator of a fermionic operator (no reordering is done here, see ``post_processing``)."""
        fop = self.do_transform(*args, fermion_operator=fermion_operator, **kwargs)
        fop.compress()
        return self.post_processing(QubitHamiltonian.from_openfermion(fop))

    def do_transform(
        self, fermion_operator: openfermion.FermionOperator, *args, **kwargs
    ) -> openfermion.QubitOperator:
        mul = openfermion.interleaved_code(self.modes) * (
            2 * openfermion.transforms.checksum_code(self.modes // 2, self.spin_parity)
        )
        return openfermion.transforms.binary_code_transform(fermion_operator, mul)

    def post_processing(self, op: QubitHamiltonian, *args, **kwargs):
        """Return the qubit operator with the qubits mapped to the (up, down, ...) order unless ``up_then_down``."""
        if not self.up_then_down:
            da = {i: 2 * i for i in range(self.n_orbitals - 1)}
            db = {
                i + self.n_orbitals - 1: 2 * i + 1 for i in range(self.n_orbitals - 1)
            }
            da.update(db)
            op = op.map_qubits(da)
        return op

    def map_state(self, state: list, *args, **kwargs):
        if len(state) < (2 * self.n_orbitals - 2):
            state = state + [0] * (2 * self.n_orbitals - len(state) - 2)
        if self.up_then_down:
            return [state[2 * i] for i in range(self.n_orbitals - 1)] + [
                state[2 * i + 1] for i in range(self.n_orbitals - 1)
            ]
        else:
            return state

    def me_to_jw(self) -> QCircuit:
        """Return the circuit to Jordan-Wigner.

        With ``up_then_down`` the circuit expects qubit ``n_orbitals`` to be free, otherwise
        map it properly.
        """
        U = QCircuit()
        if self.spin_parity:  # since we know the spin parity
            U += X(2 * self.n_orbitals - 2) + X(2 * self.n_orbitals - 1)
        for i in range(
            self.n_orbitals - 1
        ):  # we can taper only one electron per \sigma, we need only to find the difference
            U += CNOT(2 * i, 2 * self.n_orbitals - 2)
            U += CNOT(2 * i + 1, 2 * self.n_orbitals - 1)
        if self.up_then_down:
            da = {2 * i: i for i in range(self.n_orbitals)}
            db = {2 * i + 1: self.n_orbitals + i - 1 for i in range(self.n_orbitals)}
            db.update(da)
            U = U.map_qubits(db)
        return U

    def hcb_to_me(self) -> QCircuit:
        """Return the circuit from the hard-core boson encoding to this encoding.

        The hard-core boson term of the tapered orbital is not projected out, but it does not
        affect expectation values since it is not in the Hamiltonian.
        """
        U = QCircuit()
        for i in range(self.n_orbitals - 1):
            U += CNOT(
                control=i + i * (not self.up_then_down),
                target=i
                + self.up_then_down * (self.n_orbitals - 1)
                + (not self.up_then_down) * (i + 1),
            )
        U.n_qubits = 2 * self.n_orbitals - 2
        return U
