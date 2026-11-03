"""Molecule class with a mixed encoding, where each spatial orbital is either fermionic or a hard-core boson."""

from __future__ import annotations

import copy
import typing
import warnings
from itertools import product

import numpy
from openfermion import FermionOperator
from tequila import BitString, TequilaException, TequilaWarning
from tequila.circuit import QCircuit, gates
from tequila.hamiltonian import QubitHamiltonian
from tequila.hamiltonian.paulis import Zero
from tequila.objective.objective import ExpectationValue, Variable, Variables
from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float

from sunrise.molecules.hybrid_base.encodings import known_encodings
from sunrise.molecules.hybrid_base.fermionicgateimpl import FermionicGateImpl
from sunrise.molecules.qubit_base.chemistry_tools import (
    NBodyTensor,
    ParametersQC,
    prepare_product_state,
)
from sunrise.molecules.qubit_base.encodings import EncodingBase
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase as qc_base


class HybridBase(qc_base):
    r"""Base class of hybrid molecules: each spatial orbital is encoded either as fermionic ("F") or
    as hard-core boson ("B").

    A fermionic orbital uses two qubits (alpha and beta spin orbital). A bosonic orbital
    represents a spin-paired electron pair with a single qubit, which saves qubits but only
    allows spin-paired excitations. Without ``select`` all orbitals are bosonic. The class
    extends ``QuantumChemistryBase`` (see there for the inherited methods) with the Hamiltonian,
    RDMs and circuits for this encoding.

    Parameters
    ----------
    parameters : ParametersQC
        Quantum chemistry parameters (geometry, basis set, charge, frozen core, ...).
    select : str, dict, list or tuple
        Encoding of each spatial orbital, "F" (fermionic, two qubits) or "B" (hard-core boson,
        one qubit for a spin-paired electron pair). As str or list e.g. ``"FBB"``, as dict
        ``{0: "F", 1: "B"}``. A list of int is read as the indices of the "F" orbitals.
        Missing orbitals are set to "B".
    transformation : str or callable, optional
        Fermion-to-qubit encoding of the fermionic orbitals: "JordanWigner" (default),
        "BravyiKitaev", "BravyiKitaevTree", "TaperedBinary" or the reordered ones (see
        ``known_encodings``), or an already initialized encoding. The bosonic orbitals are
        always hard-core bosons.
    active_orbitals : list of int, optional
        Active orbitals, the others are frozen. With N electrons, the first N//2 orbitals
        are considered occupied when the active space is built.
    frozen_orbitals : list of int, optional
        Orbitals removed from the active orbitals.
    orbital_type : str, optional
        Type of the orbitals, e.g. "hf" or "native".
    reference_orbitals : list of int, optional
        Orbitals considered occupied when creating an active space, as total indices. Default
        are the first ``n_electrons // 2``.
    orbitals : list, optional
        Orbital information (``OrbitalData``, can be a dict).
    **kwargs
        ``condense`` (bool, default True) drops the unused beta qubit of bosonic orbitals,
        ``two_qubit`` (bool, default False) uses two qubits for bosonic orbitals (implies
        ``condense=False``), ``integral_tresh`` (float, default 1e-6) is the threshold for
        simplifying the Hamiltonian, ``n_electrons`` overrides the number of electrons,
        ``transformation__<option>`` is passed to the encoding. The rest (e.g.
        ``one_body_integrals``, ``two_body_integrals``, ``nuclear_repulsion``) is passed to
        the integral manager.
        Also ``point_group`` and ``integral_manager`` (an existing integral manager to reuse).

    References
    ----------
    F. J. del Arco Santos et al., Quantum Sci. Technol. (2025), doi:10.1088/2058-9565/adbdee.

    Examples
    --------
    >>> mol = sun.Molecule(geometry="H 0. 0. 0.\nH 0. 0. 1.6", basis_set="sto-3g",
    ...                   select="BF", backend="pyscf", nature="hybrid")
    >>> H = mol.make_hamiltonian()
    >>> U = mol.make_ansatz("SPA", edges=mol.get_spa_edges())
    """

    def __init__(
        self,
        parameters: ParametersQC,
        select: str | dict | None = None,
        transformation: str | typing.Callable | None = None,
        active_orbitals: list | None = None,
        frozen_orbitals: list | None = None,
        orbital_type: str | None = None,
        reference_orbitals: list | None = None,
        orbitals: list | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the molecule, see the class docstring for the parameters."""
        if select is None:
            select = {}
        self._select = {}
        self._rdm1 = None
        self._rdm2 = None
        # given with the point_group keyword, otherwise detected when it is first needed
        self.point_group = kwargs.get("point_group")
        if "condense" in kwargs:
            self.condense = kwargs["condense"]
            kwargs.pop("condense")
        else:
            self.condense = True
        if "two_qubit" in kwargs:
            self.two_qubit = kwargs["two_qubit"]
            kwargs.pop("two_qubit")
            if self.two_qubit:
                self.condense = False
        else:
            self.two_qubit = False
        if "integral_tresh" in kwargs:
            self.integral_tresh = kwargs["integral_tresh"]
            kwargs.pop("integral_tresh")
        else:
            self.integral_tresh = 1.0e-6
        self.parameters = parameters
        n_electrons = parameters.total_n_electrons
        if "n_electrons" in kwargs:
            n_electrons = kwargs["n_electrons"]

        if reference_orbitals is None:
            reference_orbitals = [i for i in range(n_electrons // 2)]
        self._reference_orbitals = reference_orbitals

        if orbital_type is None:
            orbital_type = "unknown"
        # no frozen core with native orbitals (i.e. atomics)
        overriding_freeze_instruction = (
            orbital_type is not None and orbital_type.lower() == "native"
        )
        # determine frozen core automatically if set
        # only if molecule is computed from scratch and not passed down from above
        overriding_freeze_instruction = (
            overriding_freeze_instruction or n_electrons != parameters.total_n_electrons
        )
        overriding_freeze_instruction = (
            overriding_freeze_instruction or frozen_orbitals is not None
        )
        if not overriding_freeze_instruction and self.parameters.frozen_core:
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            if frozen_orbitals is None:
                frozen_orbitals = [i for i in range(n_core_electrons // 2)]

        # initialize integral manager
        if "integral_manager" in kwargs:
            self.integral_manager = kwargs["integral_manager"]
        else:
            self.integral_manager = self.initialize_integral_manager(
                *args,
                active_orbitals=active_orbitals,
                reference_orbitals=reference_orbitals,
                orbitals=orbitals,
                frozen_orbitals=frozen_orbitals,
                orbital_type=orbital_type,
                basis_name=self.parameters.basis_set,
                condense=self.condense,
                two_qubit=self.two_qubit,
                **kwargs,
            )

        if orbital_type is not None and orbital_type.lower() == "native":
            self.integral_manager.transform_to_native_orbitals()
        # tq init overwriten bcs I need the number of orbitals for select->transformation
        if not isinstance(transformation, typing.Callable):
            self.transformation: EncodingBase = self._initialize_transformation(
                *args, transformation=transformation, **kwargs
            )  # here select set only to full-HCB. changed afterwards
        else:
            self.transformation: EncodingBase = transformation
        self.update_select(select, n_orb=self.n_orbitals)
        self.up_then_down = self.transformation.up_then_down

    # Select Related Functions
    def update_select(
        self, select: str | dict | list | tuple, n_orb: int | None = None
    ):
        """Set the F/B encoding of the orbitals.

        Parameters
        ----------
        select : str, dict, list or tuple
            Encoding of each spatial orbital, "F" (fermionic, two qubits) or "B" (hard-core boson,
            one qubit for a spin-paired electron pair). As str or list e.g. ``"FBB"``, as dict
            ``{0: "F", 1: "B"}``. A list of int is read as the indices of the "F" orbitals.
            Missing orbitals are set to "B".
        n_orb : int, optional
            Number of orbitals. Default is ``self.n_orbitals``.

        Raises
        ------
        ValueError
            If a string, list or dict contains characters other than "F" and "B".
        TequilaException
            If ``select`` has an unsupported type.

        Examples
        --------
        >>> mol.update_select([2, 3, 10, 11])  # these orbitals become fermionic
        """
        if n_orb is None:
            n_orb = self.n_orbitals

        def verify_selection_str(select: str, n_orb: int) -> dict:
            """Return the selection as dict, padded with "B" or cut to ``n_orb`` orbitals."""
            sel = {}
            if len(select) >= n_orb:
                select = select[:n_orb]
            else:
                select += (n_orb - len(select)) * "B"
            for i in range(len(select)):
                if select[i] in ["F", "B"]:
                    sel.update({i: select[i]})
                else:
                    raise ValueError(
                        f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic)."
                    )
            return sel

        def verify_selection_dict(select: dict, n_orb: int) -> dict:
            """Return the selection as complete dict, missing orbitals are "B"."""
            sel = {}
            for i in range(n_orb):
                if i in [*select.keys()]:
                    if select[i] in ["F", "B"]:
                        sel.update({i: select[i]})
                    else:
                        raise ValueError(
                            f"Warning, encoding character not recognised on entry { ({i: sel[i]}) }.\n Please choose between F (Fermionic) and B (Bosonic)."
                        )
                else:
                    sel.update({i: "B"})
            return sel

        def verify_selection_list(select: list | tuple, n_orb: int) -> dict:
            """Return the selection as dict from a list of "F"/"B" or of the "F" orbital indices."""
            select = [*select]
            sel = {}
            if len(select) and isinstance(select[0], int):
                for i in range(self.n_orbitals):
                    if i in select:
                        sel[i] = "F"
                    else:
                        sel[i] = "B"
            else:
                if len(select) >= n_orb:
                    select = select[:n_orb]
                else:
                    select = select + (n_orb - len(select)) * ["B"]
                for i in range(len(select)):
                    if select[i] in ["F", "B"]:
                        sel.update({i: select[i]})
                    else:
                        raise ValueError(
                            f"Warning, encoding character not recognised on position {i}: {select[i]}.\n Please choose between F (Fermionic) and B (Bosonic)."
                        )
            return sel

        def select_to_list(select: dict):
            """Return the bosonic orbitals, the fermionic orbitals and the fermionic spin orbitals."""

            hcb = 0
            jworb = 0
            BOS_MO = []
            FER_MO = []
            FER_SO = []
            for i, enc in select.items():
                if enc == "B":
                    BOS_MO.append(i)
                    hcb += 1
                elif enc == "F":
                    FER_MO.append(i)
                    FER_SO.append(2 * i)
                    FER_SO.append(2 * i + 1)
                    jworb += 1
                else:
                    print(
                        "Warning, codification not recognized: ,",
                        i,
                        " returning void lists",
                    )
                    return [], [], []
            self.bos_orb = hcb
            self.fer_orb = jworb
            return BOS_MO, FER_MO, FER_SO

        if type(select) is dict:
            select = verify_selection_dict(select=select, n_orb=n_orb)
        elif type(select) is str:
            select = verify_selection_str(select=select, n_orb=n_orb)
        elif type(select) is list or type(select) is tuple:
            select = verify_selection_list(select=select, n_orb=n_orb)
        else:
            try:
                select = verify_selection_list(select=select, n_orb=n_orb)
            except Exception:  # noqa: BLE001
                raise TequilaException(
                    f"Warning, encoding format not recognised: {type(select)}.\n Please choose either a Str, Dict, List or Tuple."
                )
        self.BOS_MO, self.FER_MO, self.FER_SO = select_to_list(select)
        self._select = select
        self.transformation.update_select(select)

    @classmethod
    def from_tequila(cls, molecule=None, transformation=None, *args, **kwargs):
        """Create a hybrid molecule from another molecule, reusing its integrals and orbitals.

        Parameters
        ----------
        molecule : QuantumChemistryBase
            Molecule to copy.
        transformation : EncodingBase, optional
            Encoding. Default is the one of ``molecule``, converted to the hybrid version.
        **kwargs
            ``active_orbitals`` (default: those of ``molecule``), and options of the constructor
            such as ``select``.

        Returns
        -------
        HybridBase
        """
        c = molecule.integral_manager.constant_term
        h1 = molecule.integral_manager.one_body_integrals
        h2 = molecule.integral_manager.two_body_integrals
        S = molecule.integral_manager.overlap_integrals
        if "active_orbitals" not in kwargs:
            active_orbitals = [
                o.idx_total for o in molecule.integral_manager.active_orbitals
            ]
        else:
            active_orbitals = kwargs["active_orbitals"]
            kwargs.pop("active_orbitals")
        if transformation is None:
            transformation = molecule.transformation
        if not hasattr(transformation, "select"):
            transformation = cls._new_transformation(
                self=cls,
                transformation=type(transformation).__name__,
                n_electrons=molecule.n_electrons,
                n_orbitals=molecule.n_orbitals,
            )
        parameters = molecule.parameters

        return cls(
            *args,
            nuclear_repulsion=c,
            one_body_integrals=h1,
            two_body_integrals=h2,
            overlap_integrals=S,
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            active_orbitals=active_orbitals,
            transformation=transformation,
            orbital_type=molecule.integral_manager._orbital_type,
            parameters=parameters,
            reference_orbitals=molecule.integral_manager.active_space.reference_orbitals,
            **kwargs,
        )

    def use_native_orbitals(
        self, inplace=False, core: list | None = None, *args, **kwargs
    ):
        """Return the molecule in the native (orthonormalized atomic) orbital basis.

        For standard basis sets these are the orthonormalized Gaussian basis functions. Core and
        active orbitals are chosen by overlap with the current ones if not given. The F/B selection
        is carried over to the new orbitals, so check that it is still what you want.

        Parameters
        ----------
        inplace : bool, optional
            Change this molecule instead of returning a new one.
        core : list of int, optional
            Core orbitals (frozen) of the current orbitals, as total indices. Default are the frozen
            orbitals of the integral manager. If given without ``active`` the active orbitals are
            those with the lowest overlap with the core.
        **kwargs
            ``active`` (list of int) are the active orbitals in the native basis. Default are those
            with the lowest overlap with the core orbitals. Needed e.g. for H-He-H with PySCF at large
            distances, where the first HF orbital equals a native orbital,
            ``|phi_0> = 1 |chi_1> = |varphi_1>``.
            ``reference_orbitals`` (list of int) are the doubly occupied orbitals.

        Returns
        -------
        HybridBase
            The molecule in native orbitals (``self`` if ``inplace``).

        Examples
        --------
        >>> mol = sun.Molecule(geometry=geometry, basis_set="sto-3g").use_native_orbitals()
        """
        from sunrise.utils.orbital_transformation import (
            get_active,
            get_core,
            orthogonalize_active_space,
        )

        c = self.integral_manager.orbital_coefficients.copy()
        s = self.integral_manager.overlap_integrals.copy()
        d = self.integral_manager.get_orthonormalized_orbital_coefficients().copy()

        active = None
        if not self.integral_manager.active_space_is_trivial() and core is None:
            core = [
                i.idx_total for i in self.integral_manager.orbitals if i.idx is None
            ]
        if "active" in kwargs:
            active = kwargs["active"]
            kwargs.pop("active")
            if core is None:
                core = get_core(c, d, s, active)
        else:
            if active is None:
                if core is None:
                    core = []
                    active = [i for i in range(len(self.integral_manager.orbitals))]
                else:
                    if isinstance(core, int):
                        core = [core]
                    active = get_active(
                        c,
                        d,
                        s,
                        [
                            i
                            for i in range(len(self.integral_manager.orbitals))
                            if i not in core
                        ],
                    )
        assert len(active) + len(core) == len(self.integral_manager.orbitals)
        if "reference_orbitals" in kwargs:
            reference_orbitals = kwargs["reference_orbitals"]
            kwargs.pop("reference_orbitals")
            assert len(reference_orbitals) == self.parameters.total_n_electrons // 2, (
                f"Number of  provided reference_orbitals incorrect. Expected {self.parameters.total_n_electrons // 2}, received {len(reference_orbitals)}"
            )
        else:
            reference_orbitals = [
                i.idx_total for i in self.integral_manager.reference_orbitals
            ]
        to_active = [
            i for i in range(len(self.integral_manager.orbitals)) if i not in core
        ]
        to_active = {active[i]: to_active[i] for i in range(len(active))}
        if len(core):
            # keep the core orbitals at their own indices (`core` is passed as
            # frozen_idx below) and only replace the active columns
            c_combined = c.copy()
            for act_idx in active:
                c_combined[:, to_active[act_idx]] = d[:, act_idx]
            coeff = orthogonalize_active_space(
                c_combined, s, core, [*to_active.values()]
            )
            if (
                not all(i == to_active[i] for i in to_active)
                and len(self.BOS_MO)
                and len(self.FER_MO)
            ):
                print("Orbital may be reordered, please double check F/B selection")
            if len(active) == len(self.select):
                new_select = {i: self.select[i] for i in range(len(active))}
            else:
                # do not shadow `s` (the overlap integrals) - still needed below
                remaining = {i: self.select[i] for i in self.select if i not in core}
                new_select = {
                    i: remaining[[*remaining.keys()][i]] for i in range(len(remaining))
                }
            if inplace:
                self.integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term,
                    active_orbitals=[*to_active.values()],
                    reference_orbitals=reference_orbitals,
                    frozen_orbitals=core,
                    orbital_coefficients=coeff,
                    overlap_integrals=s,
                    orbital_type=f"orthonormalized-{self.integral_manager._basis_name}-basis",
                )
                self.update_select(new_select)
                return self
            else:
                integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term,
                    active_orbitals=[*to_active.values()],
                    reference_orbitals=reference_orbitals,
                    frozen_orbitals=core,
                    orbital_coefficients=coeff,
                    overlap_integrals=s,
                    orbital_type=f"orthonormalized-{self.integral_manager._basis_name}-basis",
                )
                parameters = copy.deepcopy(self.parameters)
                result = HybridBase(
                    parameters=parameters,
                    integral_manager=integral_manager,
                    transformation=self.transformation,
                    select=new_select,
                    two_qubit=self.two_qubit,
                    condense=self.condense,
                    integral_tresh=self.integral_tresh,
                    active_orbitals=[*to_active.values()],
                )
                return result
        # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
        if inplace:
            self.integral_manager.transform_to_native_orbitals()
            return self
        else:
            integral_manager = copy.deepcopy(self.integral_manager)
            integral_manager.transform_to_native_orbitals()
            parameters = copy.deepcopy(self.parameters)
            result = HybridBase(
                parameters=parameters,
                integral_manager=integral_manager,
                transformation=self.transformation,
                select=self.select,
                two_qubit=self.two_qubit,
                condense=self.condense,
                integral_tresh=self.integral_tresh,
            )
            return result

    @property
    def select(self):
        """dict : encoding of each orbital, ``{orbital: "F" or "B"}``."""
        if self._select is not None:
            return self._select
        else:
            return {}

    @select.setter
    def select(self, select):
        self.update_select(select=select)

    @property
    def condense(self):
        """bool : whether the unused beta qubit of bosonic orbitals is dropped."""
        if self._condense is not None:
            return self._condense
        else:
            return True

    @condense.setter
    def condense(self, condense):
        self._condense = condense

    @property
    def two_qubit(self):
        """bool : whether bosonic orbitals use two qubits."""
        if self._two_qubit is not None:
            return self._two_qubit
        else:
            return False

    @two_qubit.setter
    def two_qubit(self, two_qubit):
        self._two_qubit = two_qubit

    # Tranformation Related Function
    def _initialize_transformation(
        self, transformation=None, *args, **kwargs
    ) -> EncodingBase:
        """Create the encoding of this molecule from its name (default "JordanWigner") or return it as is."""

        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {
            k.split("__")[1]: v
            for k, v in kwargs.items()
            if (hasattr(k, "lower") and "transformation__" in k.lower())
        }

        trafo_args["n_electrons"] = self.n_electrons
        trafo_args["n_orbitals"] = self.n_orbitals
        trafo_args["select"] = self.select
        trafo_args["condense"] = self.condense
        trafo_args["two_qubit"] = self.two_qubit
        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise ValueError(
                    f"Unkown Fermion-to-Qubit encoding {transformation}. Try something like: {list(encodings.keys())}"
                )
        return transformation

    def _new_transformation(
        self,
        transformation=None,
        n_electrons: int | None = None,
        n_orbitals: int | None = None,
        select: str | dict | list | tuple | None = None,
        condense: bool = True,
        two_qubit: bool = False,
        *args,
        **kwargs,
    ) -> EncodingBase:
        """Like ``_initialize_transformation``, but with all properties of the encoding given as arguments."""
        if select is None:
            select = {}
        if transformation is None:
            transformation = "JordanWigner"

        # filter out arguments to the transformation
        trafo_args = {
            k.split("__")[1]: v
            for k, v in kwargs.items()
            if (hasattr(k, "lower") and "transformation__" in k.lower())
        }

        trafo_args["n_electrons"] = n_electrons
        trafo_args["n_orbitals"] = n_orbitals
        trafo_args["select"] = select
        trafo_args["condense"] = condense
        trafo_args["two_qubit"] = two_qubit
        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise ValueError(
                    f"Unkown Fermion-to-Qubit encoding {transformation}. Try something like: {list(encodings.keys())}"
                )
        return transformation

    # Hamiltonian Related Funcions
    def make_hamiltonian(self) -> QubitHamiltonian:
        """Return the molecular Hamiltonian as qubit operator, in the encoding of this molecule.

        It is the sum of the fermionic, the bosonic and the interaction part, plus the nuclear
        repulsion. Terms below ``integral_tresh`` are removed.

        Returns
        -------
        QubitHamiltonian

        Examples
        --------
        >>> H = mol.make_hamiltonian()
        """

        def make_fermionic_hamiltonian() -> QubitHamiltonian:
            """Return the part of the Hamiltonian acting only on fermionic orbitals."""
            H_FER = []
            for i in self.FER_SO:
                for j in self.FER_SO:
                    if (i % 2) == (j % 2):
                        H_FER.append((((j, 1), (i, 0)), self.h1[j // 2][i // 2]))
                    for k in self.FER_SO:
                        for l in self.FER_SO:
                            if (i % 2) == (l % 2) and (k % 2) == (j % 2):
                                H_FER.append(
                                    (
                                        ((l, 1), (k, 1), (j, 0), (i, 0)),
                                        0.5
                                        * self._h2.elems[l // 2][k // 2][j // 2][
                                            i // 2
                                        ],
                                    )
                                )
            return self.transformation(H_FER)

        def make_bosonic_hamiltonian() -> QubitHamiltonian:
            """Return the part of the Hamiltonian acting only on bosonic orbitals."""
            H_BOS = []
            for i in self.BOS_MO:
                for j in self.BOS_MO:
                    e1 = self._h2.elems[j][j][i][i]
                    if i == j:
                        e1 += 2 * self.h1[i][i]
                    if not self.two_qubit:
                        H_BOS.append((((2 * j, 1), (2 * i, 0)), e1))
                    else:
                        H_BOS.append(
                            (
                                (
                                    (2 * j, 1),
                                    (2 * j + 1, 1),
                                    (2 * i + 1, 0),
                                    (2 * i, 0),
                                ),
                                e1,
                            )
                        )
                    if i != j:
                        e2 = 2 * self._h2.elems[j][i][i][j] - self._h2.elems[j][i][j][i]
                        if not self.two_qubit:
                            H_BOS.append(
                                (((2 * i, 1), (2 * i, 0), (2 * j, 1), (2 * j, 0)), e2)
                            )
                        else:
                            H_BOS.append(
                                (
                                    (
                                        (2 * i, 1),
                                        (2 * i + 1, 1),
                                        (2 * i, 0),
                                        (2 * i + 1, 0),
                                        (2 * j, 1),
                                        (2 * j + 1, 1),
                                        (2 * j, 0),
                                        (2 * j + 1, 0),
                                    ),
                                    e2,
                                )
                            )
            return self.transformation(H_BOS)

        def make_interaction_hamiltonian() -> QubitHamiltonian:
            """Return the part of the Hamiltonian coupling bosonic and fermionic orbitals."""
            H_INT = []
            h2 = self._h2.elems
            for j in self.BOS_MO:
                for k in self.FER_SO:
                    for l in self.FER_SO:
                        if (k % 2) != (l % 2):  # hhjj
                            if not self.two_qubit:
                                H_INT.append(
                                    (
                                        ((2 * j, 1), (l, 0), (k, 0)),
                                        ((l % 2) - (k % 2))
                                        * 0.5
                                        * h2[j][j][l // 2][k // 2],
                                    )
                                )
                            else:
                                H_INT.append(
                                    (
                                        (
                                            (2 * j + k % 2, 1),
                                            (2 * j + l % 2, 1),
                                            (l, 0),
                                            (k, 0),
                                        ),
                                        0.5 * h2[j][j][l // 2][k // 2],
                                    )
                                )
                        if (k % 2) != (l % 2):  # jjhh
                            if not self.two_qubit:
                                H_INT.append(
                                    (
                                        ((l, 1), (k, 1), (2 * j, 0)),
                                        ((k % 2) - (l % 2))
                                        * 0.5
                                        * h2[l // 2][k // 2][j][j],
                                    )
                                )
                            else:
                                H_INT.append(
                                    (
                                        (
                                            (l, 1),
                                            (k, 1),
                                            (2 * j + k % 2, 0),
                                            (2 * j + l % 2, 0),
                                        ),
                                        0.5 * h2[l // 2][k // 2][j][j],
                                    )
                                )
                        if (k % 2) == (l % 2):
                            if not self.two_qubit:
                                e2 = (
                                    2 * h2[k // 2][j][j][l // 2]
                                    + 2 * h2[j][k // 2][l // 2][j]
                                    - h2[k // 2][j][l // 2][j]
                                    - h2[j][k // 2][j][l // 2]
                                )
                                H_INT.append(
                                    (((2 * j, 1), (2 * j, 0), (k, 1), (l, 0)), 0.5 * e2)
                                )
                            else:
                                e2 = h2[j][k // 2][l // 2][j] + h2[k // 2][j][j][l // 2]
                                H_INT.append(
                                    (((j * 2, 1), (j * 2, 0), (k, 1), (l, 0)), 0.5 * e2)
                                )
                                H_INT.append(
                                    (
                                        (
                                            (j * 2 + 1, 1),
                                            (j * 2 + 1, 0),
                                            (k, 1),
                                            (l, 0),
                                        ),
                                        0.5 * e2,
                                    )
                                )
                                H_INT.append(
                                    (
                                        (
                                            (k, 1),
                                            (l, 0),
                                            (j * 2 + k % 2, 1),
                                            (j * 2 + k % 2, 0),
                                        ),
                                        -0.5
                                        * (
                                            h2[k // 2][j][l // 2][j]
                                            + h2[j][k // 2][j][l // 2]
                                        ),
                                    )
                                )

            return self.transformation(H_INT)

        self.C, self.h1, self._h2 = self.get_integrals(ordering="openfermion")
        H = (
            make_fermionic_hamiltonian()
            + make_bosonic_hamiltonian()
            + make_interaction_hamiltonian()
            + self.C
        )
        return H.simplify(self.integral_tresh)

    def make_hardcore_boson_hamiltonian(self) -> QubitHamiltonian:
        """Return the Hamiltonian, same as ``make_hamiltonian`` (prints a warning if some orbitals
        are fermionic).

        Examples
        --------
        >>> H = mol.make_hardcore_boson_hamiltonian()
        """
        if len(self.FER_MO):
            print(
                "Warning HCB Hamiltonian called but the encoding is not full HCB, with encoding: ",
                self.encoding,
            )
        return self.make_hamiltonian()

    def compute_rdms(
        self,
        U: QCircuit = None,
        variables: Variables = None,
        spin_free: bool = True,
        get_rdm1: bool = True,
        get_rdm2: bool = True,
        ordering="dirac",
        use_hcb: bool = False,
        rdm_trafo: QubitHamiltonian = None,
        evaluate=True,
    ):
        r"""Compute the one- and two-particle reduced density matrices (RDMs) of the state ``U|0>``.

        Only real orbitals and real RDMs are supported.

        .. math::
            \gamma^p_q = \langle \psi | a^p a_q | \psi \rangle, \quad
            \gamma^{pq}_{rs} = \langle \psi | a^p a^q a_s a_r | \psi \rangle, \quad
            \psi = U|0\rangle

        By default the 2-RDM is returned in the physics (Dirac) ordering,
        ``rdm2[p, q, r, s]`` :math:`= \gamma^{pq}_{rs}`, see ``ordering``.

        The RDMs are stored in ``rdm1`` and ``rdm2``. They are expectation values of the encoded
        operators, so they depend on the encoding; for Jordan-Wigner they are the usual
        second-quantized RDMs in the occupation picture.

        Parameters
        ----------
        U : tq.QCircuit
            Circuit preparing the state.
        variables : dict or tq.Variables, optional
            Values of the circuit parameters, if ``U`` is parametrized.
        spin_free : bool, optional
            Unused, the RDMs are always spin-free (summed over spin, size ``n_orbitals``).
        get_rdm1, get_rdm2 : bool, optional
            Which RDMs to compute. Computing both at once is more efficient.
        ordering : str {"dirac", "phys", "chem", "openfermion"}, optional
            Index ordering of the 2-RDM. Default is "dirac", the physics ordering
            ``rdm2[p, q, r, s] = <psi|a^p a^q a_s a_r|psi>``; "chem" (Mulliken) and "openfermion" are
            also possible.
        use_hcb : bool, optional
            Unused.
        rdm_trafo : QubitHamiltonian, optional
            Transformation applied to the operators, ``a^p a_q -> T^dagger a^p a_q T``. Default is the identity.
        evaluate : bool, optional
            If True evaluate the expectation values with ``tq.simulate``. If False return them as
            ``tq.QTensor``.

        Returns
        -------
        tuple or numpy.ndarray
            ``(rdm1, rdm2)`` if both are requested, otherwise the requested one.

        Raises
        ------
        TequilaException
            If ``U`` is None or the encoding is Bravyi-Kitaev-Superfast.

        Examples
        --------
        >>> rdm1, rdm2 = mol.compute_rdms(U, variables=variables)
        """
        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException("Need to specify a Quantum Circuit.")
        # Check whether transformation is BKSF.
        # Issue here: when a single operator acts only on a subset of qubits, BKSF might not yield the correct
        # transformation, because it computes the number of qubits incorrectly in this case.
        # A hotfix such as for symmetry_conserving_bravyi_kitaev would require deeper changes, thus omitted for now
        if type(self.transformation).__name__ == "BravyiKitaevFast":
            raise TequilaException(
                "The Bravyi-Kitaev-Superfast transformation does not support general FermionOperators yet."
            )

        def _build_1bdy_operators_mix() -> list:
            """Return the one-body operators (fermionic orbital pairs and bosonic diagonal terms) as symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(self.n_orbitals):
                for q in range(p + 1):
                    if self.select[p] == "F" and self.select[q] == "F":
                        op_tuple = [(((2 * p, 1), (2 * q, 0)), 0.5)]
                        op = self.transformation(op_tuple)
                        op += op.dagger()
                        op_tuple = [(((2 * p + 1, 1), (2 * q + 1, 0)), 0.5)]
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                    elif p == q and self.select[p] == "B":
                        if not self.two_qubit:
                            op_tuple = [(((2 * p, 1), (2 * q, 0)), 1)]
                            op = self.transformation(op_tuple)
                            op += op.dagger()
                        else:
                            op_tuple = [(((2 * p, 1), (2 * p, 0)), 0.5)]
                            op = self.transformation(op_tuple)
                            op += op.dagger()
                            op_tuple = [(((2 * p + 1, 1), (2 * p + 1, 0)), 0.5)]
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                    else:
                        op = Zero()
                    ops += [op]
            return ops

        def __case_2bdy(i: int, j: int, k: int, l: int) -> int:
            """Classify indices ``i, j, k, l``: 1 if all are fermionic, 2 if all bosonic, 3 if two of each, 0 otherwise."""
            list = [self.select[i], self.select[j], self.select[k], self.select[l]]
            b = list.count("B")
            f = list.count("F")
            if f == 4:
                return 1
            elif b == 4:
                return 2
            elif f == b:
                return 3
            else:
                return 0

        def _build_2bdy_operators_mix() -> list:
            """Return the two-body operators (fermionic, bosonic and mixed, see ``__case_2bdy``) as symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (
                    p >= q or r >= s
                ):
                    case = __case_2bdy(p, q, r, s)
                    if case == 1:  # case JJJJ
                        # Spin aaaa
                        op_tuple = (
                            [(((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)), 0.5)]
                            if (p != q and r != s)
                            else [((), 0)]
                        )
                        op = self.transformation(op_tuple)
                        op += op.dagger()
                        # Spin abba
                        op_tuple = (
                            [
                                (
                                    (
                                        (2 * p, 1),
                                        (2 * q + 1, 1),
                                        (2 * s + 1, 0),
                                        (2 * r, 0),
                                    ),
                                    0.5,
                                )
                            ]
                            if (2 * p != 2 * q + 1 and 2 * s + 1 != 2 * r)
                            else [((), 0)]
                        )
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        # Spin baab
                        op_tuple = (
                            [
                                (
                                    (
                                        (2 * p + 1, 1),
                                        (2 * q, 1),
                                        (2 * s, 0),
                                        (2 * r + 1, 0),
                                    ),
                                    0.5,
                                )
                            ]
                            if (2 * p + 1 != 2 * q and 2 * s != 2 * r + 1)
                            else [((), 0)]
                        )
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        # Spin bbbb
                        op_tuple = (
                            [
                                (
                                    (
                                        (2 * p + 1, 1),
                                        (2 * q + 1, 1),
                                        (2 * s + 1, 0),
                                        (2 * r + 1, 0),
                                    ),
                                    0.5,
                                )
                            ]
                            if (p != q and r != s)
                            else [((), 0)]
                        )
                        opa = self.transformation(op_tuple)
                        op += opa + opa.dagger()
                        ops += [op]
                    elif case == 2:  # case HHHH
                        if not self.two_qubit:
                            # Spin aaaa+bbbb dont allow p=q=r=s bcs self-interaction orb ijji
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * q, 0), (2 * p, 0)), -1)]
                                if (p != q and p == s and q == r)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            # Spin abba+ baab allow p=q=r=s orb iijj
                            op_tuple = (
                                [(((2 * p, 1), (2 * s, 0)), 1)]
                                if (p == q and s == r)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin abba+ baab dont allow p=q=r=s orb ijij
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * q, 0)), 2)]
                                if (p != q and p == r and s == q)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        else:
                            # Spin aaaa+ bbbb dont allow p=q=r=s  orb ijji
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q, 1),
                                            (2 * q, 0),
                                            (2 * p, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p != q and r != s and p == s and q == r)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q + 1, 1),
                                            (2 * q + 1, 0),
                                            (2 * p + 1, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p != q and r != s and p == s and q == r)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin abba+ baab allow p=q=r=s orb iijj
                            sign = numpy.sign(p - s)
                            if not sign:
                                sign = 1
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * p + 1, 1),
                                            (2 * s + 1, 0),
                                            (2 * s, 0),
                                        ),
                                        sign * 1,
                                    )
                                ]
                                if (p == q and s == r)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # Spin aaaa+ bbbb dont allow p=q=r=s orb ijij
                            sign = numpy.sign(q - p)
                            if not sign:
                                sign = 1
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * p, 0),
                                            (2 * q + 1, 0),
                                        ),
                                        2 * sign,
                                    )
                                ]
                                if (p != q and p == r and s == q)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q + 1, 1),
                                            (2 * p + 1, 0),
                                            (2 * q + 1, 0),
                                        ),
                                        2 * sign,
                                    )
                                ]
                                if (p != q and p == r and s == q)
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        ops += [op]
                    elif case == 3:  # case HJJH+JHHJ+HJHJ+JHJH+HHJJ+JJHH
                        if not self.two_qubit:
                            # uddu+duud hhjj
                            op_tuple = (
                                [(((2 * p, 1), (2 * r + 1, 0), (2 * s, 0)), 0.5)]
                                if (p == q and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = (
                                [(((2 * p, 1), (2 * r, 0), (2 * s + 1, 0)), -0.5)]
                                if (p == q and self.select[p] == "B")
                                else [((), 0)]
                            )  # -0.5 bcs sign(2r,2s+1)
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud jjhh
                            op_tuple = (
                                [(((2 * p, 1), (2 * q + 1, 1), (2 * r, 0)), 0.5)]
                                if r == s and self.select[r] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [(((2 * p + 1, 1), (2 * q, 1), (2 * r, 0)), -0.5)]
                                if r == s and self.select[r] == "B"
                                else [((), 0)]
                            )  # -0.5 bcs sign(2r,2s+1)
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd hjjh
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q, 1),
                                            (2 * r, 0),
                                            (2 * p, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p == s and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q + 1, 1),
                                            (2 * r + 1, 0),
                                            (2 * p, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p == s and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd jhhj
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q, 1),
                                            (2 * q, 0),
                                            (2 * s, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (r == q and self.select[r] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * q, 0),
                                            (2 * s + 1, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (r == q and self.select[r] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu hjhj
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * s, 0)), 1)]
                                if p == r and self.select[p] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q + 1, 1),
                                            (2 * p, 0),
                                            (2 * s + 1, 0),
                                        ),
                                        1,
                                    )
                                ]
                                if p == r and self.select[p] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu jhjh
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * q, 0)), 1)]
                                if (s == q and self.select[s] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * r + 1, 0),
                                            (2 * q, 0),
                                        ),
                                        1,
                                    )
                                ]
                                if (s == q and self.select[s] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        else:
                            # uddu+duud hhjj
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * p + 1, 1),
                                            (2 * r + 1, 0),
                                            (2 * s, 0),
                                        ),
                                        0.5,
                                    )
                                ]
                                if (p == q and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op = opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * p, 1),
                                            (2 * r, 0),
                                            (2 * s + 1, 0),
                                        ),
                                        0.5,
                                    )
                                ]
                                if (p == q and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud jjhh
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q + 1, 1),
                                            (2 * r + 1, 0),
                                            (2 * r, 0),
                                        ),
                                        0.5,
                                    )
                                ]
                                if r == s and self.select[r] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * r, 0),
                                            (2 * r + 1, 0),
                                        ),
                                        0.5,
                                    )
                                ]
                                if r == s and self.select[r] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd hjjh
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * r, 0),
                                            (2 * p + 1, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p == s and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q + 1, 1),
                                            (2 * r + 1, 0),
                                            (2 * p, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (p == s and self.select[p] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # uddu+duud+uuuu+dddd jhhj
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p, 1),
                                            (2 * q + 1, 1),
                                            (2 * q + 1, 0),
                                            (2 * s, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (r == q and self.select[r] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q, 1),
                                            (2 * q, 0),
                                            (2 * s + 1, 0),
                                        ),
                                        -0.5,
                                    )
                                ]
                                if (r == q and self.select[r] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu hjhj
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * p, 0), (2 * s, 0)), -1)]
                                if p == r and self.select[p] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q + 1, 1),
                                            (2 * p + 1, 0),
                                            (2 * s + 1, 0),
                                        ),
                                        -1,
                                    )
                                ]
                                if p == r and self.select[p] == "B"
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            # dddd+uuuu jhjh
                            op_tuple = (
                                [(((2 * p, 1), (2 * q, 1), (2 * r, 0), (2 * q, 0)), -1)]
                                if (s == q and self.select[s] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                            op_tuple = (
                                [
                                    (
                                        (
                                            (2 * p + 1, 1),
                                            (2 * q + 1, 1),
                                            (2 * r + 1, 0),
                                            (2 * q + 1, 0),
                                        ),
                                        -1,
                                    )
                                ]
                                if (s == q and self.select[s] == "B")
                                else [((), 0)]
                            )
                            opa = self.transformation(op_tuple)
                            op += opa + opa.dagger()
                        ops += [op]
                    else:
                        ops += [Zero()]
            return ops

        def _assemble_rdm1_mix(evals) -> numpy.ndarray:
            """Assemble the one-particle RDM from the expectation values using the symmetries."""
            N = self.n_orbitals
            rdm1 = numpy.zeros([N, N])
            ctr: int = 0
            for p in range(N):
                for q in range(p + 1):
                    rdm1[p, q] = evals[ctr]
                    # Symmetry pq = qp
                    rdm1[q, p] = rdm1[p, q]
                    ctr += 1
            return rdm1

        def _assemble_rdm2_mix(evals) -> numpy.ndarray:
            """Assemble the spin-free two-particle RDM from the expectation values using the symmetries."""
            ctr: int = 0
            rdm2 = numpy.zeros(
                [self.n_orbitals, self.n_orbitals, self.n_orbitals, self.n_orbitals]
            )
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p * self.n_orbitals + q >= r * self.n_orbitals + s and (
                    p >= q or r >= s
                ):
                    rdm2[p, q, r, s] = evals[ctr]
                    # Symmetry pqrs = rspq
                    rdm2[r, s, p, q] = rdm2[p, q, r, s]
                    ctr += 1
            # Further permutational symmetry: pqrs = qpsr
            for p, q, r, s in product(range(self.n_orbitals), repeat=4):
                if p >= q or r >= s:
                    rdm2[q, p, s, r] = rdm2[p, q, r, s]
            return rdm2

        # Build operator lists
        qops = []
        qops += _build_1bdy_operators_mix() if get_rdm1 else []
        qops += _build_2bdy_operators_mix() if get_rdm2 else []

        # Compute expected values
        rdm1 = None
        rdm2 = None
        from tequila import QTensor

        if evaluate:
            if rdm_trafo is None:
                evals = simulate(
                    ExpectationValue(H=qops, U=U, shape=[len(qops)]),
                    variables=variables,
                )
            else:
                qops = [
                    rdm_trafo.dagger() * qops[i] * rdm_trafo for i in range(len(qops))
                ]
                evals = simulate(
                    ExpectationValue(H=qops, U=U, shape=[len(qops)]),
                    variables=variables,
                )
        else:
            if rdm_trafo is None:
                evals = [ExpectationValue(H=x, U=U) for x in qops]
                N = self.n_orbitals  # if spin_free else 2*self.n_orbitals
                rdm1 = QTensor(shape=[N, N])
                rdm2 = QTensor(shape=[N, N, N, N])
            else:
                raise TequilaException(
                    "compute_rdms: rdm_trafo was set but evaluate flag is False (not supported)"
                )

        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = self.n_orbitals * (self.n_orbitals + 1) // 2
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        rdm1 = []
        rdm2 = []
        # Build matrices using the expectation values
        rdm1 = _assemble_rdm1_mix(evals_1) if get_rdm1 else rdm1
        rdm2 = _assemble_rdm2_mix(evals_2) if get_rdm2 else rdm2
        if get_rdm2:
            rdm2_ = NBodyTensor(elems=rdm2, ordering="dirac")
            rdm2_.reorder(to=ordering)
            rdm2 = rdm2_.elems
        if get_rdm1:
            if get_rdm2:
                self._rdm2 = rdm2
                self._rdm1 = rdm1
                return rdm1, rdm2
            else:
                self._rdm1 = rdm1
                return rdm1
        elif get_rdm2:
            self._rdm2 = rdm2
            return rdm2
        else:
            warnings.warn(
                "compute_rdms called with instruction to not compute?", TequilaWarning
            )

    def transform_orbitals(
        self,
        orbital_coefficients,
        ignore_active_space=False,
        name=None,
        *args,
        **kwargs,
    ):
        """Return a new molecule with transformed orbitals.

        Parameters
        ----------
        orbital_coefficients : numpy.ndarray
            Transformation matrix, first index is the old orbital (summed over), second index the new
            orbital. By default defined on the active space.
        ignore_active_space : bool, optional
            If True the matrix acts on all orbitals instead of the active space.
        name : str, optional
            Name of the new orbitals.

        Returns
        -------
        HybridBase
            The molecule keeps the encoding and ``select``.

        Examples
        --------
        >>> mol = mol.transform_orbitals(C, ignore_active_space=True)  # C: new orbital coefficients
        """

        U = numpy.eye(self.integral_manager.orbital_coefficients.shape[0])
        # mo_coeff by default only acts on the active space
        active_indices = [o.idx_total for o in self.integral_manager.active_orbitals]

        if ignore_active_space:
            U = orbital_coefficients
        else:
            for kk, k in enumerate(active_indices):
                for ll, l in enumerate(active_indices):
                    U[k][l] = orbital_coefficients[kk][ll]

        # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
        integral_manager = copy.deepcopy(self.integral_manager)
        integral_manager.transform_orbitals(U=U, name=name)
        result = HybridBase(
            parameters=self.parameters,
            integral_manager=integral_manager,
            transformation=self.transformation,
            select=self.select,
            two_qubit=self.two_qubit,
            condense=self.condense,
            integral_tresh=self.integral_tresh,
        )
        return result

    # Cicuit Related Functions
    def verify_excitation(
        self, indices: typing.Iterable[tuple[int, int]], warning: bool = True
    ) -> bool:
        """Check that an excitation respects the bosonic restrictions.

        A bosonic orbital can only be excited as a spin-paired pair (both spin orbitals together).

        Parameters
        ----------
        indices : list of tuple of int
            Excitation ``[(p0, q0), (p1, q1), ...]`` in spin orbitals (spatial orbital ``i`` has the
            alpha spin orbital ``2*i`` and the beta spin orbital ``2*i+1``), ``(p, q)`` destroys p and
            creates q. A flat list ``[p0, q0, p1, q1, ...]`` is also accepted.
        warning : bool, optional
            If True (default) raise an exception for forbidden excitations, otherwise return False.

        Returns
        -------
        bool
            True if the excitation is allowed.

        Raises
        ------
        TequilaException
            If the excitation is forbidden and ``warning`` is True.
        """
        sel = self.select
        if not isinstance(indices[0], typing.Iterable):
            converted = [
                (indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)
            ]
            indices = converted
        froml = []
        tol = []
        for op in indices:
            if sel[op[0] // 2] == "B":  # we dont care of any Fermionic possibility
                froml.append(op[0])
            if sel[op[1] // 2] == "B":
                tol.append(op[1])
        for t in froml:
            if t in tol:
                froml.remove(t)
                tol.remove(t)
            elif 2 * (t // 2) + (not t % 2) in froml:
                froml.remove(2 * (t // 2))
                froml.remove(2 * (t // 2) + 1)
        for t in tol:
            if 2 * (t // 2) + (not t % 2) in tol:
                tol.remove(2 * (t // 2))
                tol.remove(2 * (t // 2) + 1)
        if len(froml) or len(tol):
            if warning:
                raise TequilaException(
                    "Excitations not allowed for BOSONIC restrictions"
                )
            else:
                return False
        return True

    def UR(
        self,
        i,
        j,
        angle=None,
        label=None,
        control=None,
        assume_real=True,
        *args,
        **kwargs,
    ) -> QCircuit:
        """Return the orbital rotation circuit between spatial orbitals ``i`` and ``j`` (both must
        be fermionic).

        Both spin orbitals are rotated with the same angle, see arXiv:2207.12421 (Quantum 7, 1073 (2023)) Eq. 6.
        The default variable name is ``("R", i, j)``.

        Parameters
        ----------
        i, j : int
            Spatial orbitals, both need to be fermionic. A positive angle moves electrons from i to j.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is an automatically named variable (in units of pi).
        label : hashable, optional
            Added to the automatic variable name, ``("R", i, j, label)``. Useful for repeated gates.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real, which halves the gradient cost.

        Returns
        -------
        QCircuit

        Raises
        ------
        TequilaException
            If orbital i or j is bosonic (use ``UC`` instead).

        Examples
        --------
        >>> U = mol.UR(0, 1, angle="a") + mol.UR(2, 3, angle="a")
        """
        if not (self.select[i] == "F" and self.select[j] == "F"):
            raise TequilaException(f"Rotation{(i, j)} not allowed, try Correlator")
        if angle is None:
            if label is None:
                angle = Variable(name=("R", i, j)) * numpy.pi
            else:
                angle = Variable(name=("R", i, j, label)) * numpy.pi

        circuit = self.make_excitation_gate(
            *args,
            indices=[(2 * i, 2 * j)],
            angle=angle,
            assume_real=assume_real,
            control=control,
            **kwargs,
        )
        circuit += self.make_excitation_gate(
            *args,
            indices=[(2 * i + 1, 2 * j + 1)],
            angle=angle,
            assume_real=assume_real,
            control=control,
            **kwargs,
        )
        return circuit

    def UC(
        self,
        i,
        j,
        angle=None,
        label=None,
        control=None,
        assume_real=True,
        *args,
        **kwargs,
    ) -> QCircuit:
        """Return the orbital correlator circuit between spatial orbitals ``i`` and ``j``.

        It is a spin-paired double excitation from i to j, see arXiv:2207.12421 (Quantum 7, 1073 (2023)) Eq. 22.
        The default variable name is ``("C", i, j)``.

        Parameters
        ----------
        i, j : int
            Spatial orbitals. A positive angle moves electrons from i to j.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is an automatically named variable (in units of pi).
        label : hashable, optional
            Added to the automatic variable name, ``("C", i, j, label)``. Useful for repeated gates.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real, which halves the gradient cost.

        Returns
        -------
        QCircuit

        Examples
        --------
        >>> U = mol.UC(1, 2, angle="b")
        """
        if angle is None:
            if label is None:
                angle = Variable(name=("C", i, j)) * numpy.pi
            else:
                angle = Variable(name=("C", i, j, label)) * numpy.pi
        if (
            "jordanwigner" in self.transformation.name.lower()
            and not self.transformation.up_then_down
        ):
            # for JW we can use the optimized form shown in arXiv:2207.12421 Eq.22
            target = [(2 * i, 2 * j), (2 * i + 1, 2 * j + 1)]
            G = self.make_excitation_generator(indices=target)
            P0 = self.make_excitation_generator(indices=target, form="p0")
            return gates.GeneralizedRotation(
                angle=angle, generator=G, p0=P0, assume_real=assume_real, **kwargs
            )
        else:
            return self.make_excitation_gate(
                *args,
                indices=[(2 * i, 2 * j), (2 * i + 1, 2 * j + 1)],
                angle=angle,
                control=control,
                assume_real=assume_real,
                **kwargs,
            )

    def make_excitation_gate(
        self,
        indices: typing.Iterable[tuple[int, int]],
        angle,
        control=None,
        assume_real=True,
        **kwargs,
    ) -> QCircuit:
        """Return a fermionic excitation gate ``exp(-i a/2 G)``.

        The generator is ``G = i (prod_k a^dagger_{q_k} a_{p_k} - h.c.)`` for the pairs ``(p_k, q_k)``.

        Parameters
        ----------
        indices : list of tuple of int
            Excitation ``[(p0, q0), (p1, q1), ...]`` in spin orbitals (spatial orbital ``i`` has the
            alpha spin orbital ``2*i`` and the beta spin orbital ``2*i+1``), ``(p, q)`` destroys p and
            creates q. A flat list ``[p0, q0, p1, q1, ...]`` is also accepted.
        angle : float, hashable or tq.Objective
            Rotation angle.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real, which halves the gradient cost.
        **kwargs
            ``warning`` (bool) is passed to ``verify_excitation``; ``opt`` (bool) uses an optimized
            compilation, see ``FermionicGateImpl``.

        Returns
        -------
        QCircuit

        Raises
        ------
        TequilaException
            If the excitation violates the bosonic restrictions or the encoding does not support UCC.

        Examples
        --------
        >>> U = mol.make_excitation_gate(indices=[(0, 4), (1, 5)], angle="a")  # pair excitation
        """
        if "warning" in kwargs:
            warning = kwargs["warning"]
            kwargs.pop("warning")
            allowed = self.verify_excitation(indices, warning=warning)
        else:
            self.verify_excitation(indices)
            allowed = True  # if it was False, previous line exception
        if allowed:
            if "opt" in kwargs:
                opt = kwargs["opt"]
                kwargs.pop("opt")
            else:
                opt = False
            if not self.supports_ucc():
                raise TequilaException(
                    f"Molecule with transformation {self.transformation} does not support general UCC operations"
                )
            generator = self.make_excitation_generator(
                indices=indices, remove_constant_term=control is None
            )
            p0 = self.make_excitation_generator(
                indices=indices, form="P0", remove_constant_term=control is None
            )
            return QCircuit.wrap_gate(
                FermionicGateImpl(
                    angle=angle,
                    generator=generator,
                    p0=p0,
                    transformation=type(self.transformation).__name__.lower(),
                    indices=indices,
                    select=self.select,
                    condense=self.condense,
                    up_then_down=self.up_then_down,
                    two_qubit=self.two_qubit,
                    assume_real=assume_real,
                    opt=opt,
                    control=control,
                    **kwargs,
                )
            )

    def make_excitation_generator(
        self,
        indices: typing.Iterable[tuple[int, int]],
        form: str | None = None,
        remove_constant_term: bool = True,
    ) -> QubitHamiltonian:
        """Return the generator of a UCC-type excitation as qubit operator.

        It is ``1j * M(a^dagger_{a_0} a_{i_0} a^dagger_{a_1} a_{i_1} ... - h.c.)``, where ``M`` is
        the encoding of this molecule.

        Parameters
        ----------
        indices : iterable of tuple of int
            Excitation ``[(p0, q0), (p1, q1), ...]`` in spin orbitals (spatial orbital ``i`` has the
            alpha spin orbital ``2*i`` and the beta spin orbital ``2*i+1``), ``(p, q)`` destroys p and
            creates q. A flat list ``[p0, q0, p1, q1, ...]`` is also accepted.
        form : str {"involution", "projector"}, optional
            Manipulate the generator to an involution or a projector. Default is the standard
            fermionic excitation operator.
        remove_constant_term : bool, optional
            Remove the constant term, which has no effect on the unitary. Keep it if the unitary is
            controlled. Default is True.

        Returns
        -------
        QubitHamiltonian

        Raises
        ------
        TequilaException
            If the encoding does not support UCC or ``indices`` has a wrong format.

        Examples
        --------
        >>> G = mol.make_excitation_generator(indices=[(0, 2), (1, 3)])
        """
        if not self.supports_ucc():
            raise TequilaException(
                f"Molecule with transformation {self.transformation} does not support general UCC operations"
            )
        # check indices and convert to list of tuples if necessary
        if len(indices) == 0:
            raise TequilaException("make_excitation_operator: no indices given")
        elif not isinstance(indices[0], typing.Iterable):
            if len(indices) % 2 != 0:
                raise TequilaException(
                    "make_excitation_generator: unexpected input format of indices\n"
                    "use list of tuples as [(i_0, a_0),(i_1, a_1) ...]\n"
                    "or list as [i_0, a_0, i_1, a_1, ... ]\n"
                    f"you gave: {indices}"
                )
            converted = [
                (indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)
            ]
        else:
            converted = indices
        # convert everything to native python int
        # otherwise openfermion will complain
        converted = [(int(pair[0]), int(pair[1])) for pair in converted]
        # (p, q) destroys p and creates q: the operator below is a^dagger_q a_p
        converted = [(q, p) for p, q in converted]
        # convert to openfermion input format
        # the encoding drops the second spin-orbital of a HCB orbital, which is only correct if the two operators of
        # a HCB orbital are next to each other: write prod_k a^dagger_{c_k} a_{d_k} (the pairs commute, so sorted)
        # as a^dagger_{c_0} ... a^dagger_{c_n} a_{d_n} ... a_{d_0}, which is the same operator
        converted = sorted(converted)
        ofi = []
        dag = []
        for pair in converted:
            assert len(pair) == 2
            ofi += [(int(pair[0]), 1)]
            dag += [(int(pair[1]), 1)]
        for pair in reversed(converted):
            ofi += [(int(pair[1]), 0)]
            dag += [(int(pair[0]), 0)]
        op = FermionOperator(tuple(ofi), 1.0j)  # 1j makes it hermitian
        op += FermionOperator(tuple(dag), -1.0j)

        if isinstance(form, str) and form.lower() != "fermionic":
            # indices for all the Na operators
            Na = [x for pair in converted for x in [(pair[0], 1), (pair[0], 0)]]
            # indices for all the Ma operators (Ma = 1 - Na)
            Ma = [x for pair in converted for x in [(pair[0], 0), (pair[0], 1)]]
            # indices for all the Ni operators
            Ni = [x for pair in converted for x in [(pair[1], 1), (pair[1], 0)]]
            # indices for all the Mi operators
            Mi = [x for pair in converted for x in [(pair[1], 0), (pair[1], 1)]]

            # can gaussianize as projector or as involution (last is default)
            if form.lower() == "p+":
                op *= 0.5
                op += FermionOperator(Na + Mi, 0.5)
                op += FermionOperator(Ni + Ma, 0.5)
            elif form.lower() == "p-":
                op *= 0.5
                op += FermionOperator(Na + Mi, -0.5)
                op += FermionOperator(Ni + Ma, -0.5)
            elif form.lower() == "g+":
                op += FermionOperator(
                    [], 1.0
                )  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, -1.0)
                op += FermionOperator(Ni + Ma, -1.0)
            elif form.lower() == "g-":
                op += FermionOperator(
                    [], -1.0
                )  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, 1.0)
                op += FermionOperator(Ni + Ma, 1.0)
            elif form.lower() == "p0":
                # P0: we only construct P0 and don't keep the original generator
                op = FermionOperator(
                    [], 1.0
                )  # Just for clarity will be subtracted anyway
                op += FermionOperator(Na + Mi, -1.0)
                op += FermionOperator(Ni + Ma, -1.0)
            else:
                raise TequilaException(
                    f"Unknown generator form {form}, supported are G, P+, P-, G+, G- and P0"
                )
        qop = self.transformation(op)
        # remove constant terms
        # they have no effect in the unitary (if not controlled)
        if remove_constant_term:
            qop.qubit_operator.terms[()] = 0.0

        # check if the operator is hermitian and cast coefficients to floats
        # in order to avoid trouble with the simulation backends
        assert qop.is_hermitian()
        for k, v in qop.qubit_operator.terms.items():
            qop.qubit_operator.terms[k] = to_float(v)
        if len(qop) == 0:
            warnings.warn(
                "Excitation generator is a unit operator.\n"
                "Non-standard transformations might not work with general fermionic operators\n"
                "indices = " + str(indices),
                category=TequilaWarning,
            )
        return qop.simplify()

    # Ansatzs and Algorithm
    def prepare_reference(self, state=None, *args, **kwargs) -> QCircuit:
        """Return the circuit preparing the reference (Hartree-Fock) state in the encoding of this
        molecule.

        Parameters
        ----------
        state : list of int, optional
            Occupation of the spin orbitals. Default is the reference state of this molecule.

        Returns
        -------
        QCircuit

        Examples
        --------
        >>> U = mol.prepare_reference()  # Hartree-Fock state
        """
        if state is None:
            state = self._reference_state()
        reference_state = BitString.from_array(
            self.transformation.map_state(state=state)
        )
        U = prepare_product_state(reference_state)
        # prevent trace out in direct wfn simulation
        if not self.condense or self.two_qubit:
            U.n_qubits = self.n_orbitals * 2  # adapt when tapered transformations work
        else:
            U.n_qubits = len(self.FER_SO) + len(
                self.BOS_MO
            )  # adapt when tapered transformations work
        return U

    def prepare_hardcore_boson_reference(self) -> QCircuit:
        """Return the circuit preparing the reference state in the hard-core boson (HCB)
        approximation.

        Each qubit represents two spin-paired electrons. Use with ``make_hardcore_boson_hamiltonian``.

        Returns
        -------
        QCircuit
        """
        pos = self.transformation.pos
        U = gates.X(target=[pos[2 * i.idx] for i in self.reference_orbitals])
        U.n_qubits = self.n_orbitals
        return U

    def make_ansatz(self, name: str, *args, **kwargs) -> QCircuit:
        """Return the circuit of a named ansatz, calling the matching ``make_*_ansatz`` method.

        Several parts can be combined with "+", they use different variable labels and only the
        first starts from the reference.

        Parameters
        ----------
        name : str
            Name of the ansatz, e.g. "UpCCGSD", "UpCCD", "SPA", "UCCSD", "SPA+UpCCD", "SPA+GS".
            Names containing "SPA" call ``make_spa_ansatz``, "UCCSD" ``make_uccsd_ansatz``, and
            names containing "D" or "S" ``make_upccgsd_ansatz``.
        *args, **kwargs
            Passed to the method that builds the ansatz.
            For combined ansatze such as "SPA+UpCCD", ``label`` labels the variables of each part.

        Returns
        -------
        QCircuit

        Raises
        ------
        TequilaException
            If the name is unknown.

        See Also
        --------
        make_spa_ansatz : Separable Pair Ansatz (SPA).
        make_upccgsd_ansatz : (k-)UpCCGSD-type ansatz.

        Examples
        --------
        >>> U = mol.make_ansatz("SPA", edges=mol.get_spa_edges())
        """
        name = name.lower()
        if name.strip() == "":
            return QCircuit()
        if "+" in name:
            U = QCircuit()
            subparts = name.split("+")
            U = self.make_ansatz(*args, name=subparts[0], **kwargs)
            # making sure the there are is no undesired behaviour in layers after +
            # reference should not be included since we are not starting from |00...0> anymore
            if "include_reference" in kwargs:
                kwargs.pop("include_reference")
            # hcb optimization can also not be used (in almost all cases)
            if "hcb_optimization" in kwargs:
                kwargs.pop("hcb_optimization")
            # making sure that we have no repeating variable names
            label = None
            if "label" in kwargs:
                label = kwargs["label"]
                kwargs.pop("label")
            for i, subpart in enumerate(subparts[1:]):
                U += self.make_ansatz(
                    *args,
                    name=subpart,
                    label=(label, i),
                    include_reference=False,
                    hcb_optimization=False,
                    **kwargs,
                )
            return U

        if name == "uccsd":
            return self.make_uccsd_ansatz(*args, **kwargs)
        elif "spa" in name.lower():
            if "hcb" not in kwargs:
                hcb = False
                if "hcb" in name.lower():
                    hcb = True
                kwargs["hcb"] = hcb
            return self.make_spa_ansatz(*args, **kwargs)
        elif "d" in name or "s" in name:
            return self.make_upccgsd_ansatz(*args, name=name, **kwargs)
        else:
            raise TequilaException(f"unknown ansatz with name={name}")

    def make_spa_ansatz(
        self,
        edges=None,
        hcb=False,
        use_units_of_pi=False,
        label=None,
        optimize=None,
        ladder=True,
        check_edges=True,
    ) -> QCircuit:
        """Return the Separable Pair Ansatz (SPA) circuit.

        Each electron pair is correlated with its own group of orbitals (an "edge").
        The molecule needs an even number of electrons.

        Parameters
        ----------
        edges : list of tuple of int, optional
            Orbitals of each pair, e.g. ``[(0,), (1, 2, 3), (4, 5)]`` are three pairs with one, three
            and two orbitals. There must be ``n_electrons // 2`` edges without common orbitals.
            Default is ``self.get_spa_edges()``.
        hcb : bool, optional
            Build the circuit in the hard-core boson space, without mapping to the encoding of the
            molecule. Use with ``make_hardcore_boson_hamiltonian``.
        use_units_of_pi : bool, optional
            Give the circuit angles in units of pi.
        label : hashable, optional
            Label for the variables.
        optimize : bool, optional
            Build the shallow circuit of Ry and CNOT gates. Default is True if the encoding allows
            it (``hcb_to_me`` is available) or has no fermionic orbitals.
        ladder : bool, optional
            If True (default) excitations are local, e.g. for the edge (1, 2, 3) they are 1->2 and
            2->3. If False they are 1->2 and 1->3.
        check_edges : bool, optional
            Check that the edges are valid (number of edges, no shared orbitals). Default is True;
            only switch it off if you are sure about the edges.

        Returns
        -------
        QCircuit

        Raises
        ------
        TequilaException
            If the number of electrons is odd, or the edges are wrong in number or overlapping.

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        References
        ----------
        J. S. Kottmann and A. Aspuru-Guzik, Phys. Rev. A 105, 032449 (2022), arXiv:2105.03836.

        Examples
        --------
        >>> U = mol.make_spa_ansatz(edges=mol.get_spa_edges())
        """
        if edges is None:
            edges = self.get_spa_edges()
        # sanity checks
        # current SPA implementation needs even number of electrons
        if self.n_electrons % 2 != 0:
            raise TequilaException(
                f"need even number of electrons for SPA ansatz.\n{self.n_electrons} active electrons"
            )
        if check_edges:
            # making sure that enough edges are assigned
            if len(edges) != self.n_electrons // 2:
                raise TequilaException(
                    f"number of edges need to be equal to number of active electrons//2\n{len(edges)} edges given\n{self.n_electrons} active electrons\nfrozen core is {self.parameters.frozen_core}"
                )
            # making sure that orbitals are uniquely assigned to edges
            for edge_qubits in edges:
                for q1 in edge_qubits:
                    for edge2 in edges:
                        if edge2 == edge_qubits:
                            continue
                        elif q1 in edge2:
                            raise TequilaException(
                                f"make_spa_ansatz: faulty list of edges, orbitals are overlapping e.g. orbital {q1} is in edge {edge_qubits} and edge {edge2}"
                            )

        # auto assign if the circuit construction is optimized
        # depending on the current qubit encoding (if hcb_to_me is implemnented we can optimize)
        if optimize is None:
            try:
                have_hcb_to_me = self.hcb_to_me() is not None
            except Exception:  # noqa: BLE001
                have_hcb_to_me = False
            if have_hcb_to_me or not len(self.FER_SO):
                optimize = True
            else:
                optimize = False
        pos = self.transformation.pos
        U = QCircuit()
        # construction of the optimized circuit
        if optimize:
            # circuit in HCB representation
            # depends a bit on the ordering of the spin-orbitals in the encoding
            # here we transform it to the qubits representing the up-spins
            # the hcb_to_me sequence will then transfer to the actual encoding later
            for edge_orbitals in edges:
                edge_qubits = [pos[2 * i] for i in edge_orbitals]
                U += gates.X(edge_qubits[0])
                if len(edge_qubits) == 1:
                    continue
                for i in range(1, len(edge_qubits)):
                    q1 = edge_qubits[i]
                    c = edge_qubits[i - 1]
                    if not ladder:
                        c = edge_qubits[0]
                    angle = Variable(
                        name=((edge_orbitals[i - 1], edge_orbitals[i]), "D", label)
                    )
                    if use_units_of_pi:
                        angle = angle * numpy.pi
                    if i - 1 == 0:
                        U += gates.Ry(angle=angle, target=q1, control=None)
                    else:
                        U += gates.Ry(angle=angle, target=q1, control=c)
                    U += gates.CNOT(q1, c)

            if not hcb:
                U += self.transformation.hcb_to_me()
        else:
            orbs = [edge[0] for edge in edges]
            if hcb:
                U = gates.X([pos[2 * i] for i in orbs])
            else:
                state = [0] * 2 * self.n_orbitals
                for i in orbs:
                    state[2 * i] = 1
                    state[2 * i + 1] = 1
                U = self.prepare_reference(state)

            for edge_qubits in edges:
                previous = edge_qubits[0]
                if len(edge_qubits) > 1:
                    for q1 in edge_qubits[1:]:
                        c = previous
                        if not ladder:
                            c = edge_qubits[0]
                        angle = Variable(name=((c, q1), "D", label))
                        if use_units_of_pi:
                            angle = angle * numpy.pi
                        if hcb:
                            U += self.make_hardcore_boson_excitation_gate(
                                indices=[(c, q1)], angle=angle
                            )
                        else:
                            U += self.make_excitation_gate(
                                indices=[(2 * c, 2 * q1), (2 * c + 1, 2 * q1 + 1)],
                                angle=angle,
                            )
                        previous = q1
        if not self.condense:
            U.n_qubits = 2 * self.n_orbitals
        return U

    def make_upccgsd_ansatz(
        self,
        include_reference: bool = True,
        name: str = "UpCCGSD",
        indices: list[tuple[tuple[int]]] | None = None,
        label: str | None = None,
        order: int | None = None,
        assume_real: bool = True,
        hcb_optimization: bool | None = None,
        spin_adapt_singles: bool = True,
        neglect_z: bool = False,
        mix_sd: bool = False,
        first_double: bool = True,
        *args,
        **kwargs,
    ) -> QCircuit:
        """Return a (k-)UpCCGSD-type ansatz circuit, similar to Lee et al.

        The variant is chosen by ``name``, see ``make_upccgsd_indices``. A prefix ``k-`` sets the
        order, e.g. "2-UpCCGSD"; "UpCCD" has only doubles.

        Parameters
        ----------
        include_reference : bool, optional
            Start from the reference state. Default is True.
        name : str, optional
            Name of the ansatz, e.g. "UpCCGSD", "UpCCD", "2-UpCCGSD". Names containing "HCB" use the
            hard-core boson optimization.
        indices : list, optional
            Excitation indices of the layers (e.g. ``[((0, 3),), ((1, 4),)]``), replacing the ones
            given by ``name``. Default is ``make_upccgsd_indices(name)``.
        label : hashable, optional
            Label added to the variable names, ``(label, (x, (p, q)))`` instead of ``(x, (p, q))``
            with ``x`` the layer.
        order : int, optional
            Number of repetitions of the layer, each with independent parameters. Default is 1 or
            the ``k`` of the name.
        assume_real : bool, optional
            Assume a real wavefunction (true for a real reference), which reduces the gradient cost.
        hcb_optimization : bool, optional
            Use the hard-core boson compilation. Needs an encoding with ``hcb_to_me``. Default is True
            if available and the reference is included.
        spin_adapt_singles : bool, optional
            Use spin-adapted single excitations. Default is True.
        neglect_z : bool, optional
            Neglect the Z strings in the singles (needs Jordan-Wigner). Set automatically if the name
            contains "A" (e.g. "UpCCGASD"). Default is False.
        mix_sd : bool, optional
            Order the excitations as pairs (double, single) acting on the same orbitals instead of all
            doubles then all singles. Useful for strongly correlated systems with a large
            Trotterization error of the UCC ansatz.
        first_double : bool, optional
            If True (default) doubles come before singles, otherwise singles first.
        **kwargs
            ``indices`` can be passed to use custom excitations, as list of tuples of
            spin orbital pairs, e.g. ``[((2*p, 2*q), (2*p+1, 2*q+1)), ...]``.

        Returns
        -------
        QCircuit

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        References
        ----------
        J. Lee, W. J. Huggins, M. Head-Gordon and K. B. Whaley, J. Chem. Theory Comput. 15, 311 (2019), arXiv:1810.02327.

        Examples
        --------
        >>> U = mol.make_upccgsd_ansatz(name="UpCCGSD")
        """

        name = name.upper()

        if "A" in name:
            neglect_z = True

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except Exception:  # noqa: BLE001
                order = 1

        if indices is None:
            indices = self.make_upccgsd_indices(key=name)
        # check if the used qubit encoding has a hcb transformation
        have_hcb_trafo = self.transformation.hcb_to_me() is not None

        # consistency checks for optimization
        if have_hcb_trafo and hcb_optimization is None and include_reference:
            hcb_optimization = True
        if "HCB" in name:
            hcb_optimization = True
        if hcb_optimization and not have_hcb_trafo and "HCB" not in name:
            raise TequilaException(
                f"use_hcb={hcb_optimization} but transformation={self.transformation} has no 'hcb_to_me' function. Try transformation='ReorderedJordanWigner'"
            )
        if ("S" in name and "HCB" in name) and ("HCB" in name and "S" in name):
            raise ValueError(
                f"name={name}, Singles can't be realized without mapping back to the standard encoding leave S or HCB out of the name"
            )
        if hcb_optimization and mix_sd and (order == 1):
            raise TequilaException(
                "Mixed SD can not be employed together with HCB Optimization with only one UCC layer"
            )
        if hcb_optimization and not first_double and (order == 1):
            raise TequilaException(
                "First Doubles can not be employed together with HCB Optimization with only one UCC layer"
            )
        # convenience
        S = "S" in name.upper()
        D = "D" in name.upper()
        # first layer
        if not hcb_optimization:
            U = QCircuit()
            if include_reference:
                U = self.prepare_reference()

            U += self.make_upccgsd_layer(
                *args,
                include_singles=S,
                include_doubles=D,
                indices=indices,
                assume_real=assume_real,
                label=(label, 0),
                spin_adapt_singles=spin_adapt_singles,
                mix_sd=mix_sd,
                first_double=first_double,
                **kwargs,
            )
        else:
            U = QCircuit()
            if include_reference:
                U = self.prepare_hardcore_boson_reference()
            if D:
                U += self.make_hardcore_boson_upccgd_layer(
                    *args,
                    indices=indices,
                    assume_real=assume_real,
                    label=(label, 0),
                    **kwargs,
                )
            U += self.transformation.hcb_to_me()
            if S:
                U += self.make_upccgsd_singles(
                    *args,
                    indices=indices,
                    assume_real=assume_real,
                    label=(label, 0),
                    spin_adapt_singles=spin_adapt_singles,
                    neglect_z=neglect_z,
                    **kwargs,
                )

        for k in range(1, order):
            U += self.make_upccgsd_layer(
                include_singles=S,
                include_doubles=D,
                indices=indices,
                label=(label, k),
                spin_adapt_singles=spin_adapt_singles,
                neglect_z=neglect_z,
                mix_sd=mix_sd,
                first_double=first_double,
            )
        if not self.condense:
            U.n_qubits = 2 * self.n_orbitals
        return U

    def make_upccgsd_layer(
        self,
        indices,
        include_singles=True,
        include_doubles=True,
        assume_real=True,
        label=None,
        spin_adapt_singles: bool = True,
        angle_transform=None,
        mix_sd=False,
        neglect_z=False,
        first_double: bool = True,
        *args,
        **kwargs,
    ) -> QCircuit:
        """Return one layer of singles and/or doubles for the given orbital-pair indices, in the
        order set by ``mix_sd`` and ``first_double``.

        Parameters
        ----------
        indices : list
            Orbital pairs, each given as ``[(p, q)]`` (spatial orbitals), e.g. from ``make_upccgsd_indices``.
        include_singles, include_doubles : bool, optional
            Which excitation types the layer contains. Default is both.
        assume_real : bool, optional
            Assume a real wavefunction (halves gradient costs). Default is True.
        label : hashable, optional
            Label of the variables.
        spin_adapt_singles : bool, optional
            Use spin-adapted singles. Default is True.
        angle_transform : callable, optional
            Function applied to the angle names of the singles.
        mix_sd : bool, optional
            Alternate doubles and singles on the same orbitals instead of all doubles first.
        neglect_z : bool, optional
            Neglect the Z strings in the singles (needs Jordan-Wigner). Default is False.
        first_double : bool, optional
            If True (default) doubles come before singles, otherwise singles first.

        Returns
        -------
        tq.QCircuit
        """
        U = QCircuit()
        if include_singles and not mix_sd and not first_double:
            U += self.make_upccgsd_singles(
                indices=indices,
                assume_real=assume_real,
                label=label,
                spin_adapt_singles=spin_adapt_singles,
                angle_transform=angle_transform,
                neglect_z=neglect_z,
            )
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = ((idx,), "D", label)
            if include_singles and mix_sd and not first_double:
                U += self.make_upccgsd_singles(
                    indices=[(idx,)],
                    assume_real=assume_real,
                    label=label,
                    spin_adapt_singles=spin_adapt_singles,
                    angle_transform=angle_transform,
                    neglect_z=neglect_z,
                )
            if include_doubles:
                if (
                    "jordanwigner" in self.transformation.name.lower()
                    and not self.transformation.up_then_down
                ):
                    # we can optimize with qubit excitations for the JW representation
                    target = [
                        (2 * idx[0], 2 * idx[1]),
                        (2 * idx[0] + 1, 2 * idx[1] + 1),
                    ]
                    G = self.make_excitation_generator(indices=target)
                    P0 = self.make_excitation_generator(indices=target, form="p0")
                    U += gates.GeneralizedRotation(
                        angle=angle,
                        generator=G,
                        p0=P0,
                        assume_real=assume_real,
                        **kwargs,
                    )
                else:
                    U += self.make_excitation_gate(
                        angle=angle,
                        indices=(
                            (2 * idx[0], 2 * idx[1]),
                            (2 * idx[0] + 1, 2 * idx[1] + 1),
                        ),
                        assume_real=assume_real,
                        **kwargs,
                    )
            if include_singles and mix_sd and first_double:
                U += self.make_upccgsd_singles(
                    indices=[(idx,)],
                    assume_real=assume_real,
                    label=label,
                    spin_adapt_singles=spin_adapt_singles,
                    angle_transform=angle_transform,
                    neglect_z=neglect_z,
                )

        if include_singles and not mix_sd and first_double:
            U += self.make_upccgsd_singles(
                indices=indices,
                assume_real=assume_real,
                label=label,
                spin_adapt_singles=spin_adapt_singles,
                angle_transform=angle_transform,
                neglect_z=neglect_z,
            )
        return U

    def make_upccgsd_singles(
        self,
        indices="UpCCGSD",
        spin_adapt_singles=True,
        label=None,
        angle_transform=None,
        assume_real=True,
        neglect_z=False,
        *args,
        **kwargs,
    ) -> QCircuit:
        """Return the singles of a UpCCGSD-type layer for the given orbital-pair indices (or ansatz
        name).

        Parameters
        ----------
        indices : str or list, optional
            Name of an ansatz (e.g. ``"UpCCGSD"``) whose orbital pairs are used, or the pairs as ``[(p, q)]`` entries.
        spin_adapt_singles : bool, optional
            If True one angle per pair is shared by up and down spin, otherwise separate angles.
        label : hashable, optional
            Label of the variables.
        angle_transform : callable, optional
            Function applied to the angle names.
        assume_real : bool, optional
            Assume a real wavefunction (halves gradient costs). Default is True.
        neglect_z : bool, optional
            Neglect the Z strings in the singles (needs Jordan-Wigner). Default is False.

        Returns
        -------
        tq.QCircuit
        """
        if neglect_z and "jordanwigner" not in self.transformation.name.lower():
            raise TequilaException(
                "neglegt-z approximation in UpCCGSD singles needs the (Reversed)JordanWigner representation"
            )
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices)
        U = QCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            if self.select[idx[0]] == "F" and self.select[idx[1]] == "F":
                if spin_adapt_singles:
                    angle = (idx, "S", label)
                    if angle_transform is not None:
                        angle = angle_transform(angle)
                    if neglect_z:
                        targeta = [
                            self.transformation.pos[2 * idx[0]],
                            self.transformation.pos[2 * idx[1]],
                        ]
                        targetb = [
                            self.transformation.pos[2 * idx[0] + 1],
                            self.transformation.pos[2 * idx[1] + 1],
                        ]
                        U += gates.QubitExcitation(
                            angle=angle,
                            target=targeta,
                            assume_real=assume_real,
                            **kwargs,
                        )
                        U += gates.QubitExcitation(
                            angle=angle,
                            target=targetb,
                            assume_real=assume_real,
                            **kwargs,
                        )
                    else:
                        U += self.make_excitation_gate(
                            angle=angle,
                            indices=[(2 * idx[0], 2 * idx[1])],
                            assume_real=assume_real,
                            **kwargs,
                        )
                        U += self.make_excitation_gate(
                            angle=angle,
                            indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                            assume_real=assume_real,
                            **kwargs,
                        )
                else:
                    angle1 = (idx, "SU", label)
                    angle2 = (idx, "SD", label)
                    if angle_transform is not None:
                        angle1 = angle_transform(angle1)
                        angle2 = angle_transform(angle2)
                    if neglect_z:
                        targeta = [
                            self.transformation.up(idx[0]),
                            self.transformation.up(idx[1]),
                        ]
                        targetb = [
                            self.transformation.down(idx[0]),
                            self.transformation.down(idx[1]),
                        ]
                        U += gates.QubitExcitation(
                            angle=angle1,
                            target=targeta,
                            assume_real=assume_real,
                            **kwargs,
                        )
                        U += gates.QubitExcitation(
                            angle=angle2,
                            target=targetb,
                            assume_real=assume_real,
                            **kwargs,
                        )
                    else:
                        U += self.make_excitation_gate(
                            angle=angle1,
                            indices=[(2 * idx[0], 2 * idx[1])],
                            assume_real=assume_real,
                            **kwargs,
                        )
                        U += self.make_excitation_gate(
                            angle=angle2,
                            indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)],
                            assume_real=assume_real,
                            **kwargs,
                        )

        return U

    def make_hardcore_boson_excitation_gate(
        self, indices, angle, control=None, assume_real=True, compile_options="optimize"
    ) -> QCircuit:
        """Return an excitation gate in the hard-core boson (HCB) approximation (electrons forced
        into spin pairs).

        All electrons are forced into spin pairs. Use with ``make_hardcore_boson_hamiltonian``.

        Parameters
        ----------
        indices : list of tuple of int
            Excitations ``[(p, q), ...]`` in spatial orbitals.
        angle : float, hashable or tq.Objective
            Rotation angle.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real.
        compile_options : str, optional
            Compilation options of ``tq.gates.QubitExcitation``.

        Returns
        -------
        QCircuit

        Raises
        ------
        TequilaException
            If the indices are inconsistent with the encoding.
        """
        target = []
        for pair in indices:
            assert len(pair) == 2
            target += [
                self.transformation.pos[2 * pair[0]],
                self.transformation.pos[2 * pair[1]],
            ]
        if self.transformation.up_then_down:
            consistency = [x < self.n_orbitals for x in target]
        else:
            consistency = [True]
        if not all(consistency):
            raise TequilaException(
                f"make_hardcore_boson_excitation_gate: Inconsistencies in indices={indices} for encoding: {self.transformation}"
            )
        return gates.QubitExcitation(
            angle=angle,
            target=target,
            assume_real=assume_real,
            control=control,
            compile_options=compile_options,
        )

    def hcb_to_me(self, U: QCircuit | None = None, **kwargs) -> QCircuit:
        """Map a circuit from the hard-core boson (HCB) encoding to the encoding of this molecule.

        Parameters
        ----------
        U : tq.QCircuit, optional
            Circuit in the HCB picture, acting on the up qubits. The mapping circuit is appended to it.
            If None, only the mapping circuit is returned.
        **kwargs
            Passed to ``transformation.hcb_to_me``, e.g. ``all`` or ``bos``.

        Returns
        -------
        QCircuit

        Examples
        --------
        >>> U = mol.hcb_to_me(U=U_hcb)  # map a hard-core boson circuit to this encoding
        """
        UX = self.transformation.hcb_to_me(**kwargs)
        if U is None:
            return UX
        return U + UX

    def compute_energy(self, method: str, *args, **kwargs) -> float:
        """Compute the energy of a classical or quantum method.

        Parameters
        ----------
        method : str
            Classical methods from the backend (e.g. "hf", "mp2", "ccsd", "ccsd(t)", "fci"), or
            quantum ansatz names (e.g. "UpCCD", "UpCCGSD", "UCCSD", see ``make_upccgsd_ansatz``).
            A method with "hybrid" (e.g. "hybrid-fci") is computed with ``compute_restricted_energy``.
        **kwargs
            For quantum methods, arguments of the minimizer.

        Returns
        -------
        float
        """
        method = method.lower()
        if "hybrid" in method:
            method = method.replace("hybrid", "").strip()
            if method[0] == "-":
                method = method[1:]
            return self.compute_restricted_energy(method, *args, **kwargs)
        else:
            return super().compute_energy(method, *args, **kwargs)

    def compute_restricted_energy(self, method: str, *args, **kwargs) -> float:
        """Compute an energy with the interaction restricted according to ``select``.

        Unlike ``compute_energy``, only the integrals allowed by the bosonic/fermionic selection are
        kept, and the energy is computed with a PySCF molecule built from them.

        Parameters
        ----------
        method : str
            Method name, see ``compute_energy``.
        **kwargs
            For quantum methods, arguments of the minimizer.

        Returns
        -------
        float
        """
        from sunrise.molecules.qubit_base import Molecule as mol

        c, h, g = self.get_integrals()
        BOS_L = self.BOS_MO
        NBOS_L = self.FER_MO
        for i in BOS_L:
            for j in BOS_L:
                if i != j:
                    h[i][j] = 0.0
        for i in BOS_L:
            for j in NBOS_L:
                h[i][j] = 0.0
                h[j][i] = 0.0
        new_g = numpy.zeros(
            shape=(len(g.elems), len(g.elems), len(g.elems), len(g.elems))
        )
        for i in NBOS_L:
            for j in NBOS_L:
                for k in NBOS_L:
                    for l in NBOS_L:
                        new_g[i][j][k][l] = g.elems[i][j][k][l]
        for i in BOS_L:
            for j in BOS_L:
                new_g[i][i][j][j] = g.elems[i][i][j][j]
                new_g[i][j][j][i] = g.elems[i][j][j][i]
                new_g[i][j][i][j] = g.elems[i][j][i][j]
        for i in NBOS_L:
            for j in NBOS_L:
                for k in BOS_L:
                    new_g[i][j][k][k] = g.elems[i][j][k][k]
                    new_g[k][k][i][j] = g.elems[k][k][i][j]
                    new_g[i][k][k][j] = g.elems[i][k][k][j]
                    new_g[k][i][j][k] = g.elems[k][i][j][k]
                    new_g[i][k][j][k] = g.elems[i][k][j][k]
                    new_g[k][i][k][j] = g.elems[k][i][k][j]
        g.elems = new_g
        parameters = copy.deepcopy(self.parameters)
        return mol(
            parameters=parameters,
            one_body_integrals=h,
            two_body_integrals=g,
            nuclear_repulsion=c,
            backend="pyscf",
            transformation=self.transformation.name,
            n_electrons=self.n_electrons,
        ).compute_energy(*args, method=method, **kwargs)

    def get_restricted_integrals(
        self, *args, **kwargs
    ) -> tuple[float, numpy.ndarray, numpy.ndarray]:
        """Return the integrals with only the terms allowed by ``select``.

        Returns
        -------
        tuple
            ``(constant term, one-body integrals, two-body integrals as NBodyTensor)``.

        Examples
        --------
        >>> integrals = mol.get_restricted_integrals()
        """
        c, h, g = self.get_integrals()
        BOS_L = self.BOS_MO
        NBOS_L = self.FER_MO
        for i in BOS_L:
            for j in BOS_L:
                if i != j:
                    h[i][j] = 0.0
        for i in BOS_L:
            for j in NBOS_L:
                h[i][j] = 0.0
                h[j][i] = 0.0
        new_g = numpy.zeros(
            shape=(len(g.elems), len(g.elems), len(g.elems), len(g.elems))
        )
        for i in NBOS_L:
            for j in NBOS_L:
                for k in NBOS_L:
                    for l in NBOS_L:
                        new_g[i][j][k][l] = g.elems[i][j][k][l]
        for i in BOS_L:
            for j in BOS_L:
                new_g[i][i][j][j] = g.elems[i][i][j][j]
                new_g[i][j][j][i] = g.elems[i][j][j][i]
                new_g[i][j][i][j] = g.elems[i][j][i][j]
        for i in NBOS_L:
            for j in NBOS_L:
                for k in BOS_L:
                    new_g[i][j][k][k] = g.elems[i][j][k][k]
                    new_g[k][k][i][j] = g.elems[k][k][i][j]
                    new_g[i][k][k][j] = g.elems[i][k][k][j]
                    new_g[k][i][j][k] = g.elems[k][i][j][k]
                    new_g[i][k][j][k] = g.elems[i][k][j][k]
                    new_g[k][i][k][j] = g.elems[k][i][k][j]
        g.elems = new_g
        return c, h, g
