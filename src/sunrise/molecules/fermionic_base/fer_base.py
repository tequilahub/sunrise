"""Fermionic molecule base class: second-quantized operators, orbital transformations and ansatz circuits."""

from __future__ import annotations

import copy
import typing
import warnings
from itertools import product
from numbers import Number

import numpy
from tequila import (
    BitNumbering,
    BitString,
    QTensor,
    QubitWaveFunction,
    TequilaException,
    TequilaWarning,
)
from tequila.hamiltonian import QubitHamiltonian
from tequila.objective.objective import Objective, Variable, Variables

from sunrise.fermionic_operations import FCircuit, gates
from sunrise.fermionic_operations.givens_rotations import (
    get_givens_circuit as _get_givens_circuit,
)
from sunrise.fermionic_operations.givens_rotations import n_rotation as _n_rotation
from sunrise.fermionic_operations.givens_rotations import (
    reconstruct_matrix_from_circuit,
)
from sunrise.molecules.qubit_base.chemistry_tools import (
    Amplitudes,
    ClosedShellAmplitudes,
    NBodyTensor,
    ParametersQC,
)
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase

from ...expval import Braket

# from tequila.simulators.simulator_api import simulate
from ...expval.minimize import simulate

try:
    # if you are experiencing import errors you need to update openfermion
    # required is version >= 1.0
    # otherwise replace with from openfermion.hamiltonians import MolecularData
    import openfermion
    from openfermion.ops import FermionOperator as OFFermionOperator
    from openfermion.utils import hermitian_conjugated
except Exception:  # noqa: BLE001
    try:
        pass
    except Exception as E:
        raise TequilaException(
            f"{E!s}\nIssue with Tequila Chemistry: Please update openfermion"
        ) from E

OPTIMIZED_ORDERING = "Optimized"


class FermionicBase(QuantumChemistryBase):
    r"""Base class of fermionic molecules: circuits are ``FCircuit`` objects (fermionic gates)
    instead of qubit circuits.

    This is the class returned by ``sun.Molecule(..., nature="fermionic")``. It holds the integrals
    and orbitals (through the integral manager) and defines the main methods; the derived classes
    add backend specific parts (PySCF, Psi4, MADNESS, see ``<backend>_interface.py``).

    Parameters
    ----------
    parameters : ParametersQC
        Quantum chemistry parameters (geometry, basis set, charge, multiplicity, ...).
    active_orbitals : list of int, optional
        Active orbitals (total indices). The other orbitals are frozen. Default is all orbitals.
    frozen_orbitals : list of int, optional
        Convenience: orbitals removed from the active orbitals.
    orbital_type : str, optional
        Kind of the orbitals, e.g. ``"hf"``, ``"native"``. Default is ``"unknown"``.
        With ``"native"`` the orbitals are transformed to the (orthonormalized) native basis on init.
    reference_orbitals : list of int, optional
        Orbitals considered occupied when the active space is built (total indices).
        Default is the first ``n_electrons // 2``.
    orbitals : list, optional
        Orbital information (``OrbitalData`` or dictionaries).
    *args, **kwargs
        Passed to the integral manager, e.g. ``one_body_integrals``, ``two_body_integrals``,
        ``nuclear_repulsion``, ``overlap_integrals``, ``orbital_coefficients``, ``point_group``.
        Also ``fermionic_backend`` (default fermionic backend of the molecule, used e.g. by
        ``compute_energy``), ``n_electrons`` (overrides the number from ``parameters``) and
        ``integral_manager`` (an existing integral manager to reuse).

    Examples
    --------
    >>> import sunrise as sun
    >>> mol = sun.Molecule(geometry="H 0 0 0\nH 0 0 0.75", basis_set="sto-3g", nature="fermionic")
    >>> U = mol.make_ansatz("UpCCD")
    """

    def __init__(
        self,
        parameters: ParametersQC,
        active_orbitals: list | None = None,
        frozen_orbitals: list | None = None,
        orbital_type: str | None = None,
        reference_orbitals: list | None = None,
        orbitals: list | None = None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""
        self.transformation = None
        self.fermionic_backend = None
        if "fermionic_backend" in kwargs:
            self.fermionic_backend = kwargs["fermionic_backend"]
            kwargs.pop("fermionic_backend")
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
                **kwargs,
            )

        if orbital_type is not None and orbital_type.lower() == "native":
            self.integral_manager.transform_to_native_orbitals()

        self._rdm1 = None
        self._rdm2 = None
        # given with the point_group keyword, otherwise detected when it is first needed
        self.point_group = kwargs.get("point_group", None)

    @classmethod
    def from_tequila(cls, molecule, *args, **kwargs):
        """Create a fermionic molecule from a qubit molecule, reusing its integrals and orbitals.

        Parameters
        ----------
        molecule : QuantumChemistryBase
            Molecule whose integrals, orbitals and parameters are copied.
        **kwargs
            Passed to the constructor. ``active_orbitals`` overrides the active space of ``molecule``.

        Returns
        -------
        FermionicBase
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
        parameters = molecule.parameters
        return cls(
            *args,
            nuclear_repulsion=c,
            one_body_integrals=h1,
            two_body_integrals=h2,
            overlap_integrals=S,
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            active_orbitals=active_orbitals,
            orbital_type=molecule.integral_manager._orbital_type,
            parameters=parameters,
            reference_orbitals=molecule.integral_manager.active_space.reference_orbitals,
            **kwargs,
        )

    def supports_ucc(self):
        """Return True: fermionic molecules support UCC-type operations (e.g.
        ``make_excitation_gate``).
        """
        return True

    @classmethod
    def from_openfermion(
        cls,
        molecule: openfermion.MolecularData,
        *args,
        **kwargs,
    ):
        """Create a molecule from an openfermion ``MolecularData`` object.

        Parameters
        ----------
        molecule : openfermion.MolecularData
            The openfermion molecule (geometry, basis, charge and multiplicity are taken from it).
        *args, **kwargs
            Passed to the constructor.

        Returns
        -------
        FermionicBase
        """
        geometry = molecule.geometry
        if not isinstance(geometry, str):
            # openfermion usually stores the geometry as a list of (atom, (x, y, z)), ParametersQC expects a string
            geometry = "\n".join(
                "{} {} {} {}".format(atom, *coordinates)
                for atom, coordinates in geometry
            )
        parameters = ParametersQC(
            basis_set=molecule.basis,
            geometry=geometry,
            units="angstrom",
            description=molecule.description,
            multiplicity=molecule.multiplicity,
            charge=molecule.charge,
        )
        return cls(*args, parameters=parameters, molecule=molecule, **kwargs)

    def make_excitation_generator(
        self,
        indices: typing.Iterable[tuple[int, int]],
        form: str | None = None,
    ) -> openfermion.FermionOperator:
        """Return the generator of a UCC-type excitation as fermion operator.

        The generator is ``i (a^dagger_{a_0} a_{i_0} a^dagger_{a_1} a_{i_1} ... - h.c.)``, optionally
        manipulated into an involution or projector.

        Parameters
        ----------
        indices : iterable of tuple of int
            Pairs ``[(i_0, a_0), (i_1, a_1), ...]``, each pair destroys ``i`` and creates ``a``
            (spin orbital indices: even alpha, odd beta). A flat list ``[i_0, a_0, i_1, a_1, ...]`` is also accepted.
        form : str, optional
            ``"fermionic"`` or None for the standard excitation operator (default), otherwise one of
            ``"p+"``, ``"p-"``, ``"p0"`` (projector forms) or ``"g+"``, ``"g-"`` (involution forms).

        Returns
        -------
        openfermion.FermionOperator

        Raises
        ------
        TequilaException
            If no indices are given, their format is wrong or ``form`` is unknown.

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
        ofi = []
        dag = []
        for pair in converted:
            assert len(pair) == 2
            ofi += [
                (int(pair[0]), 1),
                (int(pair[1]), 0),
            ]  # openfermion does not take other types of integers like numpy.int64
            dag += [(int(pair[0]), 0), (int(pair[1]), 1)]

        op = openfermion.FermionOperator(tuple(ofi), 1.0j)  # 1j makes it hermitian
        op += openfermion.FermionOperator(tuple(reversed(dag)), -1.0j)

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
                op += openfermion.FermionOperator(Na + Mi, 0.5)
                op += openfermion.FermionOperator(Ni + Ma, 0.5)
            elif form.lower() == "p-":
                op *= 0.5
                op += openfermion.FermionOperator(Na + Mi, -0.5)
                op += openfermion.FermionOperator(Ni + Ma, -0.5)

            elif form.lower() == "g+":
                op += openfermion.FermionOperator(
                    [], 1.0
                )  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            elif form.lower() == "g-":
                op += openfermion.FermionOperator(
                    [], -1.0
                )  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, 1.0)
                op += openfermion.FermionOperator(Ni + Ma, 1.0)
            elif form.lower() == "p0":
                # P0: we only construct P0 and don't keep the original generator
                op = openfermion.FermionOperator(
                    [], 1.0
                )  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            else:
                raise TequilaException(
                    f"Unknown generator form {form}, supported are G, P+, P-, G+, G- and P0"
                )
        return op

    def make_hardcore_boson_excitation_gate(
        self, indices, angle, control=None, assume_real=True, compile_options="optimize"
    ) -> FCircuit:
        """Return the pair excitation gate, moving both electrons of a spatial-orbital pair.

        Fermionic molecules have no hard-core boson qubit reduction: the gate is a ``FermionicExcitation`` of the spin orbitals
        (2p -> 2q) and (2p+1 -> 2q+1). Use it together with ``make_hardcore_boson_hamiltonian``.

        Parameters
        ----------
        indices : list of tuple of int
            Pairs ``(p, q)`` of spatial orbitals, as in ``format_excitation_indices``.
        angle : float, hashable or tq.Objective
            Angle (variables) of the gate.
        control : int or list of int, optional
            Not supported: raises NotImplementedError if given.
        assume_real : bool, optional
            Not used.
        compile_options : str, optional
            Not used.

        Returns
        -------
        FCircuit

        Raises
        ------
        NotImplementedError
            If ``control`` is given.
        """
        if control is not None:
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        spin_indices = []
        for p, q in indices:
            spin_indices += [(2 * p, 2 * q), (2 * p + 1, 2 * q + 1)]
        return gates.FermionicExcitation(indices=spin_indices, variables=angle)

    def UR(
        self, i, j, angle=None, label=None, control=None, *args, **kwargs
    ) -> FCircuit:
        """Return the orbital rotation circuit between spatial orbitals ``i`` and ``j``.

        See arXiv:2207.12421 (Quantum 7, 1073 (2023)) Eq. 6 for ``UR(0, 1)``.

        Parameters
        ----------
        i, j : int
            Spatial orbital indices. A positive angle moves electrons from ``i`` to ``j``.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is the variable ``("R", i, j)`` (times pi).
        label : hashable, optional
            Used instead of ``angle`` to name the variable ``("R", i, j, label)``, e.g. to repeat the gate with independent variables.
        control : int or list of int, optional
            Not supported: raises NotImplementedError if given.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = mol.UR(0, 1, angle="a") + mol.UR(2, 3, angle="a")
        """
        if angle is None:
            if label is None:
                angle = Variable(name=("R", i, j)) * numpy.pi
            else:
                angle = Variable(name=("R", i, j, label)) * numpy.pi
        if control is not None:
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        return gates.UR(i, j, angle)

    def UC(
        self, i, j, angle=None, label=None, control=None, *args, **kwargs
    ) -> FCircuit:
        """Return the orbital correlator circuit between spatial orbitals ``i`` and ``j``.

        It is a spin-paired double excitation, see arXiv:2207.12421 (Quantum 7, 1073 (2023)) Eq. 22 for ``UC(1, 2)``.

        Parameters
        ----------
        i, j : int
            Spatial orbital indices. A positive angle moves electrons from ``i`` to ``j``.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is the variable ``("C", i, j)`` (times pi).
        label : hashable, optional
            Used instead of ``angle`` to name the variable ``("C", i, j, label)``.
        control : int or list of int, optional
            Not supported: raises NotImplementedError if given.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = mol.UC(1, 2, angle="b")
        """
        if angle is None:
            if label is None:
                angle = Variable(name=("C", i, j)) * numpy.pi
            else:
                angle = Variable(name=("C", i, j, label)) * numpy.pi
        if control is not None:
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        return gates.UC(i, j, angle)

    def make_excitation_gate(self, indices, angle, control=None, **kwargs) -> FCircuit:
        r"""Return a fermionic excitation gate ``exp(-i a/2 G)``.

        The generator is :math:`G = i(\prod_k a^\dagger_{q_k} a_{p_k} - \mathrm{h.c.})` for the given pairs.

        Parameters
        ----------
        indices : list of tuple of int
            Pairs ``[(p_0, q_0), (p_1, q_1), ...]`` in spin orbital indices, each pair destroys ``p`` and creates ``q``.
        angle : float, hashable or tq.Objective
            Rotation angle ``a``.
        control : int or list of int, optional
            Not supported: raises NotImplementedError if given.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = mol.make_excitation_gate(indices=[(0, 4), (1, 5)], angle="a")  # pair excitation
        """
        if control is not None:
            raise NotImplementedError(
                "Not implemented yet for Fermionic Backends"
            )  # TODO
        return gates.FermionicExcitation(indices=indices, variables=angle)

    def transform_orbitals(
        self,
        orbital_coefficients: FCircuit | numpy.array,
        ignore_active_space=False,
        name=None,
        *args,
        **kwargs,
    ):
        """Return a new molecule with transformed orbitals.

        Parameters
        ----------
        orbital_coefficients : FCircuit or numpy.ndarray
            Transformation matrix, second index is the new orbital and first index the old one (summed over).
            A circuit is converted to the matrix it implements. By default it acts on the active space only.
        ignore_active_space : bool, optional
            If True the matrix is taken as given for all orbitals, not only the active ones.
        name : str, optional
            Name of the new orbitals.

        Returns
        -------
        FermionicBase
            New molecule (a plain ``FermionicBase``, not a backend class, so that classical backend methods stay consistent).

        Examples
        --------
        >>> mol = mol.transform_orbitals(C, ignore_active_space=True)  # C: new orbital coefficients
        """
        if isinstance(orbital_coefficients, FCircuit):
            orbital_coefficients = numpy.array(
                reconstruct_matrix_from_circuit(
                    orbital_coefficients, self.n_orbitals, tol=1.0e-6
                )
            ).T
        U = numpy.eye(self.integral_manager.orbital_coefficients.shape[0])
        # mo_coeff by default only acts on the active space
        active_indices = [o.idx_total for o in self.integral_manager.active_orbitals]

        if ignore_active_space:
            U = orbital_coefficients
        else:
            for kk, k in enumerate(active_indices):
                for ll, l in enumerate(active_indices):
                    if isinstance(orbital_coefficients[kk][ll], Objective):
                        U[k][l] = simulate(orbital_coefficients[kk][ll])
                    else:
                        U[k][l] = orbital_coefficients[kk][ll]

        # can not be an instance of a specific backend (otherwise we get inconsistencies with classical methods in the backend)
        integral_manager = copy.deepcopy(self.integral_manager)
        integral_manager.transform_orbitals(U=U, name=name)
        result = FermionicBase(
            parameters=self.parameters, integral_manager=integral_manager
        )
        return result

    def use_native_orbitals(
        self, inplace=False, core: list | None = None, *args, **kwargs
    ):
        """Return the molecule in the native (orthonormalized atomic) orbital basis.

        For standard basis sets these are the orthonormalized Gaussian basis functions.

        Parameters
        ----------
        inplace : bool, optional
            If True update this molecule, otherwise return a new one.
        core : list of int, optional
            Orbitals of the current basis that are frozen and doubly occupied. The active native orbitals are
            then chosen as the ones with the smallest overlap with the core orbitals. Default is inherited from the current active space.
        active : list of int, optional
            (keyword) Active native orbitals. Default is all, or those chosen through ``core``.
            Example: for H, He, H ``active=[0, 1, 2]`` are the 1s orbitals of left H, He, right H;
            with ``core=[0]`` and no ``active``, ``active=[0, 2]`` is selected (He 1s overlaps most with the lowest HF orbital).
        reference_orbitals : list of int, optional
            (keyword) Occupied orbitals, ``n_electrons // 2`` of them, which set the number of
            electrons in the active space. Default is the current ones.

        Returns
        -------
        FermionicBase
            The molecule in the native basis (``self`` if ``inplace``).

        Examples
        --------
        >>> mol = sun.Molecule(geometry=geom, basis_set="sto-3g", nature="fermionic").use_native_orbitals()
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
                return self
            else:
                integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term,
                    active_orbitals=[*to_active.values()],
                    reference_orbitals=[
                        i.idx_total for i in self.integral_manager.reference_orbitals
                    ],
                    frozen_orbitals=core,
                    orbital_coefficients=coeff,
                    overlap_integrals=s,
                    orbital_type=f"orthonormalized-{self.integral_manager._basis_name}-basis",
                )
                parameters = copy.deepcopy(self.parameters)
                result = FermionicBase(
                    parameters=parameters,
                    integral_manager=integral_manager,
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
            result = FermionicBase(
                parameters=self.parameters,
                integral_manager=integral_manager,
                fermionic_backend=self.fermionic_backend,
            )
            return result

    def make_annihilation_op(self, orbital, coefficient=1.0) -> OFFermionOperator:
        """Return the annihilation operator of a spin orbital as fermion operator.

        Spin orbitals are labeled interleaved, ``2i`` (up) and ``2i+1`` (down) for spatial orbital ``i``.
        No fermion-to-qubit encoding is involved, so the operator is the same for every fermionic backend.

        Parameters
        ----------
        orbital : int
            Index of the spin orbital.
        coefficient : float, optional
            Prefactor of the operator. Default is 1.0.

        Returns
        -------
        openfermion.FermionOperator
        """
        assert orbital <= self.n_orbitals * 2
        return OFFermionOperator(f"{orbital}", coefficient)

    def make_creation_op(self, orbital, coefficient=1.0) -> OFFermionOperator:
        """Return the creation operator of a spin orbital as fermion operator.

        Spin orbitals are labeled interleaved, ``2i`` (up) and ``2i+1`` (down) for spatial orbital ``i``.
        No fermion-to-qubit encoding is involved, so the operator is the same for every fermionic backend.

        Parameters
        ----------
        orbital : int
            Index of the spin orbital.
        coefficient : float, optional
            Prefactor of the operator. Default is 1.0.

        Returns
        -------
        openfermion.FermionOperator
        """
        assert orbital <= self.n_orbitals * 2
        return OFFermionOperator(f"{orbital}^", coefficient)

    def make_number_op(self, orbital) -> OFFermionOperator:
        """Return the number operator of a spin orbital as fermion operator.

        Spin orbitals are labeled interleaved, ``2i`` (up) and ``2i+1`` (down) for spatial orbital ``i``.
        No fermion-to-qubit encoding is involved, so the operator is the same for every fermionic backend.

        Parameters
        ----------
        orbital : int
            Index of the spin orbital.

        Returns
        -------
        openfermion.FermionOperator
        """
        return self.make_creation_op(orbital) * self.make_annihilation_op(orbital)

    def make_sz_op(self) -> OFFermionOperator:
        """Return the total :math:`S_z` operator of the molecule as fermion operator.

        Same as OpenFermion's ``sz_operator`` (interleaved spin orbitals), independent of any encoding.
        """
        sz = OFFermionOperator()
        for i in range(0, self.n_orbitals * 2, 2):
            one = 0.5 * self.make_creation_op(i) * self.make_annihilation_op(i)
            two = 0.5 * self.make_creation_op(i + 1) * self.make_annihilation_op(i + 1)
            sz += one - two
        return sz

    def make_sp_op(self) -> OFFermionOperator:
        """Return the total :math:`S_+` operator of the molecule as fermion operator.

        Same as OpenFermion's ``s_plus_operator`` (interleaved spin orbitals), independent of any encoding.
        """
        sp = OFFermionOperator()
        for i in range(self.n_orbitals):
            sp += self.make_creation_op(i * 2) * self.make_annihilation_op(i * 2 + 1)
        return sp

    def make_sm_op(self) -> OFFermionOperator:
        """Return the total :math:`S_-` operator of the molecule as fermion operator.

        Same as the conjugate of OpenFermion's ``s_plus_operator`` (interleaved spin orbitals), independent of any encoding.
        """
        sm = OFFermionOperator()
        for i in range(self.n_orbitals):
            sm += self.make_creation_op(i * 2 + 1) * self.make_annihilation_op(i * 2)
        return sm

    def make_s2_op(self) -> OFFermionOperator:
        """Return the total :math:`S^2` operator of the molecule as fermion operator.

        Same as OpenFermion's ``s_squared_operator`` (interleaved spin orbitals), independent of any encoding.
        """
        s2_op = self.make_sm_op() * self.make_sp_op() + self.make_sz_op() * (
            self.make_sz_op() + 1
        )
        return s2_op

    def make_hamiltonian(self, *args, **kwargs) -> OFFermionOperator:
        """Return the molecular Hamiltonian in the active space as fermion operator.

        Parameters
        ----------
        *args, **kwargs
            Ignored. The active space is fixed by the molecule; ``active_indices`` only triggers a warning.

        Returns
        -------
        openfermion.FermionOperator

        Examples
        --------
        >>> H = mol.make_hamiltonian()
        """

        # warnings for backward comp
        if "active_indices" in kwargs:
            warnings.warn(
                "active space can't be changed in molecule. Will ignore active_orbitals passed to make_hamiltonian",
                TequilaWarning,
            )

        of_molecule = self.make_molecule()
        fop = of_molecule.get_molecular_hamiltonian()
        fop = openfermion.transforms.get_fermion_operator(fop)
        return fop

    def make_hardcore_boson_hamiltonian(self) -> OFFermionOperator | str:
        """Return the string ``"H"``; only a placeholder for the HCB optimization.

        In the hard-core boson orbital optimization it is faster to use the fermionic Hamiltonian
        with pseudo hard-core boson states and circuits.
        """
        return "H"

    def prepare_reference(self, state=None, *args, **kwargs) -> FCircuit:
        """Return the circuit preparing the reference (Hartree-Fock) state.

        Parameters
        ----------
        state : list of int, optional
            Occupation of the spin orbitals (up, down, up, down, ...). Default is the reference state of the molecule.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = mol.prepare_reference()  # Hartree-Fock state
        """
        if state is None:
            state = self._reference_state()
        d = {2 * i: i for i in range(self.n_orbitals)}
        d.update({2 * i + 1: i + self.n_orbitals for i in range(self.n_orbitals)})
        n_state = [0] * len(state)
        for i in range(len(state)):
            n_state[d[i]] = state[i]
        w = QubitWaveFunction(
            n_qubits=2 * self.n_orbitals, dense=False, numbering=BitNumbering.LSB
        )
        w[BitString.from_array(n_state)] = 1.0
        # n_state = prepare_product_state(BitString.from_array(n_state))
        U = FCircuit()
        U.initial_state = w
        # prevent trace out in direct wfn simulation
        U.n_qubits = self.n_orbitals * 2
        return U

    def prepare_hardcore_boson_reference(self):
        """Return the reference circuit, same as ``prepare_reference``.

        Fermionic molecules have no hard-core boson qubit reduction: a warning is raised and the electrons are treated as pairs.

        Returns
        -------
        FCircuit
        """
        warnings.warn(
            "No HCB operations (qubit reduction) in Fermionic Backends; treated as electron pairs",
            TequilaWarning,
        )
        return self.prepare_reference()

    def hcb_to_me(self, U=None, condensed=False):
        """Return ``U`` unchanged: fermionic molecules have no separate hard-core boson (HCB)
        encoding.

        Fermionic molecules have no hard-core boson qubit reduction: a warning is raised and the electrons are treated as pairs.

        Parameters
        ----------
        U : FCircuit, optional
            Circuit in the hard-core boson picture. Default is an empty circuit.
        condensed : bool, optional
            Not used.

        Returns
        -------
        FCircuit
        """
        warnings.warn(
            "No HCB operations (qubit reduction) in Fermionic Backends; treated as electron pairs",
            TequilaWarning,
        )
        if U is None:
            return FCircuit()
        return U

    def make_hardcore_boson_upccgd_layer(
        self,
        indices: list = "UpCCGD",
        label: str | None = None,
        assume_real: bool = True,
        *args,
        **kwargs,
    ):
        """Return one layer of pair excitation gates, one for every excitation index.

        Fermionic molecules have no hard-core boson qubit reduction: a warning is raised and the electrons are treated as pairs.

        Parameters
        ----------
        indices : list or str, optional
            Pair excitation indices, or the name of an excitation set that is passed to ``make_upccgsd_indices``. Default is "UpCCGD".
        label : hashable, optional
            Label of the angles, named ``(idx, "D", label)``.
        assume_real : bool, optional
            Passed to ``make_hardcore_boson_excitation_gate``. Default is True.
        *args, **kwargs
            Not used.

        Returns
        -------
        FCircuit
        """
        warnings.warn(
            "No HCB operations (qubit reduction) in Fermionic Backends; treated as electron pairs",
            TequilaWarning,
        )
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices.lower())

        UD = FCircuit()
        for idx in indices:
            UD += self.make_hardcore_boson_excitation_gate(
                indices=idx, angle=(idx, "D", label), assume_real=assume_real
            )

        return UD

    def make_spa_ansatz(
        self,
        edges,
        use_units_of_pi=False,
        label=None,
        ladder=True,
        check_edges=True,
    ) -> FCircuit:
        """Return the Separable Pair Ansatz (SPA) circuit.

        Each electron pair is spread over the orbitals of one edge. Needs an even number of electrons.

        Parameters
        ----------
        edges : list of tuple of int
            Orbitals of each pair, one edge per electron pair and every orbital in at most one edge.
            E.g. ``[(0,), (1, 2, 3), (4, 5)]`` are three pairs built from one, three and two orbitals.
        use_units_of_pi : bool, optional
            Give the circuit angles in units of pi.
        label : hashable, optional
            Label added to the variable names.
        ladder : bool, optional
            If True (default) excitations are local, e.g. 1->2 and 2->3 for the edge ``(1, 2, 3)``;
            if False they are coupled-cluster like, 1->2 and 1->3.
        check_edges : bool, optional
            Check that the edges are valid (number of edges, no shared orbitals). Default is True;
            only switch it off if you are sure about the edges.

        Returns
        -------
        FCircuit

        Raises
        ------
        ValueError
            If ``edges`` is None, has the wrong number of edges or edges overlap.

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        References
        ----------
        J. S. Kottmann and A. Aspuru-Guzik, Phys. Rev. A 105, 032449 (2022), arXiv:2105.03836.

        Examples
        --------
        >>> U = mol.make_spa_ansatz(edges=[(0, 1), (2, 3)])
        """

        if edges is None:
            raise ValueError(
                "SPA ansatz within a standard orbital basis needs edges. Please provide with the keyword edges.\nExample: edges=[(0,1,2),(3,4)] would correspond to two edges created from orbitals (0,1,2) and (3,4), note that orbitals can only be assigned to a single edge"
            )

        # sanity checks
        # current SPA implementation needs even number of electrons
        if self.n_electrons % 2 != 0:
            raise TequilaException(
                f"need even number of electrons for SPA ansatz.\n{self.n_electrons} active electrons"
            )

        if check_edges:
            # making sure that enough edges are assigned
            if len(edges) != self.n_electrons // 2:
                raise ValueError(
                    f"number of edges need to be equal to number of active electrons//2\n{len(edges)} edges given\n{self.n_electrons} active electrons\nfrozen core is {self.parameters.frozen_core}"
                )
            # making sure that orbitals are uniquely assigned to edges
            for edge_qubits in edges:
                for q1 in edge_qubits:
                    for edge2 in edges:
                        if edge2 == edge_qubits:
                            continue
                        elif q1 in edge2:
                            raise ValueError(
                                f"make_spa_ansatz: faulty list of edges, orbitals are overlapping e.g. orbital {q1} is in edge {edge_qubits} and edge {edge2}"
                            )

        # auto assign if the circuit construction is optimized
        # depending on the current qubit encoding (if hcb_to_me is implemnented we can optimize)

        return FCircuit.from_edges(
            edges=edges,
            use_units_of_pi=use_units_of_pi,
            label=label,
            n_orb=self.n_orbitals,
            ladder=ladder,
        )

    def make_ansatz(self, name: str, *args, **kwargs) -> FCircuit:
        """Return the circuit of a named ansatz, calling the matching ``make_*_ansatz`` method.

        Parameters
        ----------
        name : str
            Ansatz name: ``"UpCCD"``, ``"UpCCSD"``, ``"UpCCGSD"``, ``"k-UpCCGSD"``, ``"UCCSD"``, ``"SPA"``.
            Ansatze can be combined with ``+``, e.g. ``"SPA+UpCCD"``; the later parts get no reference state and an additional label.
            An empty name gives an empty circuit.
        *args, **kwargs
            Passed to the corresponding ``make_*_ansatz`` method, e.g. ``edges`` for SPA.
            For combined ansatze such as "SPA+UpCCD", ``label`` labels the variables of each part
            (``hcb_optimization`` is ignored there).

        Returns
        -------
        FCircuit

        Raises
        ------
        ValueError
            If the name is unknown or contains ``HCB`` (not supported for fermionic molecules).

        See Also
        --------
        make_spa_ansatz : Separable Pair Ansatz (SPA).
        make_upccgsd_ansatz : (k-)UpCCGSD-type ansatz.
        make_uccsd_ansatz : Closed-shell UCCSD ansatz.

        Examples
        --------
        >>> U = mol.make_ansatz("UpCCSD")
        >>> U = mol.make_ansatz("SPA", edges=[(0, 1), (2, 3)])
        """
        name = name.lower()
        if name.strip() == "":
            return FCircuit()

        if "+" in name:
            U = FCircuit()
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
            if "hcb" in kwargs or "hcb" in name.lower():
                warnings.warn(
                    "No HCB operations (qubit reduction) in Fermionic Backends; treated as electron pairs",
                    TequilaWarning,
                )
            return self.make_spa_ansatz(*args, **kwargs)
        elif "d" in name or "s" in name:
            return self.make_upccgsd_ansatz(*args, name=name, **kwargs)
        else:
            raise ValueError(f"unknown ansatz with name={name}")

    def make_upccgsd_ansatz(
        self,
        include_reference: bool = True,
        name: str = "UpCCGSD",
        label: str | None = None,
        order: int | None = None,
        spin_adapt_singles: bool = True,
        mix_sd: bool = False,
        *args,
        **kwargs,
    ) -> FCircuit:
        """Return a (k-)UpCCGSD-type ansatz circuit, similar to Lee et al.

        The name selects the excitations: ``S`` includes singles, ``D`` doubles, ``G`` generalized
        (not only occupied to virtual) excitations, ``p`` pair doubles, and a prefix ``k-`` the order.

        Parameters
        ----------
        include_reference : bool, optional
            Start from the reference state. Default is True.
        name : str, optional
            Name of the ansatz, e.g. ``"UpCCD"``, ``"UpCCSD"``, ``"2-UpCCGSD"``. Default is ``"UpCCGSD"``.
        label : hashable, optional
            Label added to the variable names. The variables are named ``(label, k)`` for layer ``k``, plus the indices.
        order : int, optional
            Number of repetitions of the layer, with independent parameters. Default is taken from the name (``k-``) or 1.
        spin_adapt_singles : bool, optional
            Use spin-adapted singles (one variable for up and down). Default is True.
        mix_sd : bool, optional
            Alternate singles and doubles acting on the same orbitals (DS-DS-...) instead of
            all doubles followed by all singles. Useful for strongly correlated systems with a
            large Trotterization error of the UCC ansatz.
        **kwargs
            Passed to ``make_upccgsd_layer``.

        Returns
        -------
        FCircuit

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

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except Exception:  # noqa: BLE001
                order = 1

        indices = self.make_upccgsd_indices(key=name)

        if "HCB" in name:
            warnings.warn(
                "No HCB operations (qubit reduction) in Fermionic Backends; treated as electron pairs",
                TequilaWarning,
            )

        # convenience
        S = "S" in name.upper()
        D = "D" in name.upper()

        # first layer
        U = FCircuit()
        if include_reference:
            U = self.prepare_reference()
        U += self.make_upccgsd_layer(
            *args,
            include_singles=S,
            include_doubles=D,
            indices=indices,
            label=(label, 0),
            mix_sd=mix_sd,
            spin_adapt_singles=spin_adapt_singles,
            **kwargs,
        )
        for k in range(1, order):
            U += self.make_upccgsd_layer(
                include_singles=S,
                include_doubles=D,
                indices=indices,
                label=(label, k),
                spin_adapt_singles=spin_adapt_singles,
                mix_sd=mix_sd,
            )

        return U

    def make_upccgsd_layer(
        self,
        indices,
        include_singles: bool = True,
        include_doubles: bool = True,
        label=None,
        spin_adapt_singles: bool = True,
        angle_transform=None,
        mix_sd: bool = False,
        *args,
        **kwargs,
    ) -> FCircuit:
        """Return one layer of paired excitations for the given orbital-pair indices.

        Parameters
        ----------
        indices : list
            Orbital pairs, each given as ``[(p, q)]`` (spatial orbitals), e.g. from ``make_upccgsd_indices``.
        include_singles, include_doubles : bool, optional
            Which excitation types the layer contains. Default is both.
        label : hashable, optional
            Label of the variables.
        spin_adapt_singles : bool, optional
            Use spin-adapted singles. Default is True.
        angle_transform : callable, optional
            Function applied to the angle names of the singles.
        mix_sd : bool, optional
            Alternate doubles and singles on the same orbitals instead of all doubles first.

        Returns
        -------
        FCircuit
        """
        U = FCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = ((idx,), "D", label)
            if include_doubles:
                U += self.UC(i=idx[0], j=idx[1], angle=angle, **kwargs)
            if include_singles and mix_sd:
                U += self.make_upccgsd_singles(
                    indices=[(idx,)],
                    label=label,
                    spin_adapt_singles=spin_adapt_singles,
                    angle_transform=angle_transform,
                )

        if include_singles and not mix_sd:
            U += self.make_upccgsd_singles(
                indices=indices,
                label=label,
                spin_adapt_singles=spin_adapt_singles,
                angle_transform=angle_transform,
            )
        return U

    def make_upccgsd_singles(
        self,
        indices="UpCCGSD",
        spin_adapt_singles=True,
        label=None,
        angle_transform=None,
        *args,
        **kwargs,
    ) -> FCircuit:
        """Return the singles (orbital rotations) of a UpCCGSD-type layer for the given orbital-pair
        indices.

        Parameters
        ----------
        indices : str or list, optional
            Name of an ansatz (e.g. ``"UpCCGSD"``) whose orbital pairs are used, or the pairs as ``[(p, q)]`` entries.
        spin_adapt_singles : bool, optional
            If True one rotation per pair is used, otherwise separate excitations for up and down spin.
        label : hashable, optional
            Label of the variables.
        angle_transform : callable, optional
            Function applied to the angle names.

        Returns
        -------
        FCircuit
        """
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices)

        U = FCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            if spin_adapt_singles:
                angle = (idx, "S", label)
                if angle_transform is not None:
                    angle = angle_transform(angle)
                U += self.UR(i=idx[0], j=idx[1], angle=angle)
            else:
                angle1 = (idx, "SU", label)
                angle2 = (idx, "SD", label)
                if angle_transform is not None:
                    angle1 = angle_transform(angle1)
                    angle2 = angle_transform(angle2)
                U += self.make_excitation_gate(
                    angle=angle1, indices=[(2 * idx[0], 2 * idx[1])], **kwargs
                )
                U += self.make_excitation_gate(
                    angle=angle2, indices=[(2 * idx[0] + 1, 2 * idx[1] + 1)], **kwargs
                )

        return U

    def make_uccsd_ansatz(
        self,
        trotter_steps: int = 1,
        initial_amplitudes: str | Amplitudes | ClosedShellAmplitudes | None = None,
        include_reference_ansatz=True,
        parametrized=True,
        threshold=1.0e-8,
        add_singles=None,
        screening=True,
        *args,
        **kwargs,
    ) -> FCircuit:
        """Return a closed-shell UCCSD ansatz circuit, optionally initialized with amplitudes.

        Parameters
        ----------
        trotter_steps : int, optional
            Number of Trotter steps. Default is 1.
        initial_amplitudes : str or Amplitudes or ClosedShellAmplitudes, optional
            Amplitudes to start from: ``"mp2"``, ``"ccsd"``, another method name accepted by ``compute_amplitudes``,
            or an amplitudes object. Default is None (all amplitudes zero).
        include_reference_ansatz : bool, optional
            Prepare the Hartree-Fock reference first. Default is True.
        parametrized : bool, optional
            Use variables (named by the amplitude indices) instead of fixed numbers. Default is True.
        threshold : float, optional
            Amplitudes below this value are dropped if ``screening`` is True.
        add_singles : bool, optional
            Include singles. Default is True (with ``"mp2"`` initial amplitudes the singles start at zero).
        screening : bool, optional
            Drop amplitudes smaller than ``threshold``. Default is True.

        Returns
        -------
        FCircuit

        Raises
        ------
        TequilaException
            If the number of electrons is odd.

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        Examples
        --------
        >>> U = mol.make_ansatz("UCCSD", add_singles=False)
        """

        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2" and add_singles is None:
                add_singles = True
        elif initial_amplitudes is not None and add_singles is not None:
            warnings.warn(
                "make_uccsd_anstatz: add_singles has no effect when explicit amplitudes are passed down",
                TequilaWarning,
            )
        elif add_singles is None:
            add_singles = True

        if self.n_electrons % 2 != 0:
            raise TequilaException(
                "make_uccsd_ansatz currently only for closed shell systems"
            )

        nocc = self.n_electrons // 2
        nvirt = self.n_orbitals - nocc

        U = FCircuit()
        if include_reference_ansatz:
            U = self.prepare_reference()

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            else:
                try:
                    amplitudes = self.compute_amplitudes(
                        method=initial_amplitudes.lower()
                    )
                except Exception as exc:
                    raise TequilaException(
                        f"{exc}\nDon't know how to initialize '{initial_amplitudes}' amplitudes"
                    ) from exc
        if amplitudes is None:
            tia = None
            if add_singles:
                tia = numpy.zeros(shape=[nocc, nvirt])
            amplitudes = ClosedShellAmplitudes(
                tIjAb=numpy.zeros(shape=[nocc, nocc, nvirt, nvirt]), tIA=tia
            )
            screening = False

        closed_shell = isinstance(amplitudes, ClosedShellAmplitudes)
        indices = {}

        if not screening:
            threshold = 0.0

        if not isinstance(amplitudes, dict):
            amplitudes = amplitudes.make_parameter_dictionary(
                threshold=threshold, screening=screening
            )
            amplitudes = dict(
                sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True)
            )
        for key, t in amplitudes.items():
            assert len(key) % 2 == 0
            if not numpy.isclose(t, 0.0, atol=threshold) or not screening:
                if closed_shell:
                    if len(key) == 2 and add_singles:
                        # singles
                        angle = 2.0 * t
                        if parametrized:
                            angle = 2.0 * Variable(name=key)
                        idx_a = (2 * key[0], 2 * key[1])
                        idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                        indices[idx_a] = angle
                        indices[idx_b] = angle
                    else:
                        assert len(key) == 4
                        angle = 2.0 * t
                        if parametrized:
                            angle = 2.0 * Variable(name=key)
                        idx_abab = (
                            2 * key[0] + 1,
                            2 * key[1] + 1,
                            2 * key[2],
                            2 * key[3],
                        )
                        indices[idx_abab] = angle
                        if key[0] != key[2] and key[1] != key[3]:
                            idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                            idx_bbbb = (
                                2 * key[0] + 1,
                                2 * key[1] + 1,
                                2 * key[2] + 1,
                                2 * key[3] + 1,
                            )
                            partner = (key[2], key[1], key[0], key[3])
                            anglex = 2.0 * (t - amplitudes[partner])
                            if parametrized:
                                anglex = 2.0 * (Variable(name=key) - Variable(partner))
                            indices[idx_aaaa] = anglex
                            indices[idx_bbbb] = anglex
                else:
                    raise NotImplementedError(
                        "only closed-shell supported, please assemble yourself .... sorry :-)"
                    )

        factor = 1.0 / trotter_steps
        for _ in range(trotter_steps):
            for idx, angle in indices.items():
                # the amplitudes list the virtual orbital first, (a, i): the excitation destroys i and creates a
                converted = [(idx[2 * i + 1], idx[2 * i]) for i in range(len(idx) // 2)]
                U += self.make_excitation_gate(indices=converted, angle=factor * angle)
        if (
            hasattr(initial_amplitudes, "lower")
            and initial_amplitudes.lower() == "mp2"
            and parametrized
            and add_singles
        ):
            # mp2 has no singles, need to initialize them here (if not parametrized initializling as 0.0 makes no sense though)
            U += self.make_upccgsd_layer(
                indices="upccsd", include_singles=True, include_doubles=False
            )
        return U

    @property
    def rdm1(self):
        """numpy.ndarray or None : one-particle RDM from the last ``compute_rdms`` call, None if not
        computed yet.
        """
        if self._rdm1 is not None:
            return self._rdm1
        else:
            print("1-RDM has not been computed. Return None for 1-RDM.")
            return None

    @property
    def rdm2(self):
        """numpy.ndarray or None : two-particle RDM from the last ``compute_rdms`` call (ordering as
        requested there, by default Dirac, i.e. physics ordering), None if not computed yet.
        """
        if self._rdm2 is not None:
            return self._rdm2
        else:
            print("2-RDM has not been computed. Return None for 2-RDM.")
            return None

    def compute_rdms(
        self,
        U: FCircuit | None = None,
        variables: Variables | None = None,
        spin_free: bool = True,
        get_rdm1: bool = True,
        get_rdm2: bool = True,
        ordering="dirac",
        rdm_trafo: QubitHamiltonian | None = None,
        evaluate=True,
        backend: str | None = None,
        use_hcb: bool = False,
    ):
        r"""Compute the one- and two-particle reduced density matrices (RDMs) of the state ``U|0>``.

        Only real orbitals and real RDMs are considered. No qubit encoding is involved: they are the usual
        second-quantized RDMs in the occupation picture. The results are also stored and available
        as the properties ``rdm1`` and ``rdm2``.

        .. math::
            \gamma^p_q = \langle U 0 | a^\dagger_p a_q | U 0 \rangle, \qquad
            \gamma^{pq}_{rs} = \langle U 0 | a^\dagger_p a^\dagger_q a_s a_r | U 0 \rangle

        By default the 2-RDM is returned in the physics (Dirac) ordering,
        ``rdm2[p, q, r, s]`` :math:`= \gamma^{pq}_{rs}`, see ``ordering``.

        Parameters
        ----------
        U : FCircuit
            Circuit preparing the state. Required.
        variables : dict or tq.Variables, optional
            Values of the variables of ``U``, if it is parametrized.
        spin_free : bool, optional
            Sum over spin (indices are spatial orbitals) if True (default), otherwise indices are spin orbitals.
        get_rdm1, get_rdm2 : bool, optional
            Which RDMs to compute. Computing both at once is cheaper.
        ordering : str {"dirac", "phys", "chem", "openfermion"}, optional
            Index ordering of the 2-RDM. Default is "dirac", the physics ordering
            ``rdm2[p, q, r, s] = <psi|a^p a^q a_s a_r|psi>``; "chem" (Mulliken) and "openfermion" are
            also possible.
        rdm_trafo : QubitHamiltonian, optional
            Transformation ``T`` applied to the operators, ``a^dagger_i a_j -> T^dagger a^dagger_i a_j T``. Default is identity.
        evaluate : bool, optional
            If True (default) the expectation values are evaluated directly; if False the RDMs are returned as tq.QTensor.
        backend : str, optional
            Simulation backend. Default is the ``fermionic_backend`` of the molecule.
        use_hcb : bool, optional
            The state is restricted to doubly occupied spatial orbitals (hard-core boson picture) so fewer operators are needed.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            ``rdm1``, ``rdm2`` or both (as tuple), depending on ``get_rdm1`` and ``get_rdm2``.

        Examples
        --------
        >>> rdm1, rdm2 = mol.compute_rdms(U, variables=variables)
        """
        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException("Need to specify a Quantum Circuit.")

        # Set up number of spin-orbitals and molecular orbitals respectively
        n_SOs = 2 * self.n_orbitals
        n_MOs = self.n_orbitals

        def _get_of_op(operator_tuple):
            """Return operator given by a operator tuple as OpenFermion - Fermion operator."""
            op = openfermion.FermionOperator(operator_tuple)
            return op

        def _get_of_hcb_op(op_tuple):
            r"""Build the hard-core boson operators: b^\dagger_ib_j + h.c. in qubit encoding"""
            if len(op_tuple) == 2:
                return openfermion.FermionOperator(op_tuple, 2)
            elif len(op_tuple) == 4:
                if (
                    op_tuple[0][0] == op_tuple[1][0]
                    and op_tuple[2][0] == op_tuple[3][0]
                ):  # iijj uddu+duud
                    return openfermion.FermionOperator(op_tuple, 0.5)
                elif (
                    (op_tuple[0][0] == op_tuple[2][0])
                    and (op_tuple[1][0] == op_tuple[3][0])
                    and (op_tuple[0][0] != op_tuple[1][0])
                    and (op_tuple[2][0] != op_tuple[3][0])
                ):  # ijij uuuu+dddd
                    return openfermion.FermionOperator(op_tuple, 2)
                elif (
                    (op_tuple[0][0] == op_tuple[3][0])
                    and (op_tuple[1][0] == op_tuple[2][0])
                    and (op_tuple[0][0] != op_tuple[1][0])
                    and (op_tuple[2][0] != op_tuple[3][0])
                ):  # ijji abba
                    return openfermion.FermionOperator(op_tuple)
                else:
                    return 0
            else:
                return 0

        def _get_qop_hermitian(of_operator):
            """Return the Hermitian part (O + O^dagger) / 2 of a FermionOperator (numbers are returned as they are)."""
            if isinstance(of_operator, Number):
                return of_operator
            return 0.5 * (of_operator + hermitian_conjugated(of_operator))

        def _build_1bdy_operators_spinful() -> list:
            """Return spinful one-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp
            ops = []
            for p in range(n_SOs):
                for q in range(p + 1):
                    op_tuple = ((p, 1), (q, 0))
                    op = _get_of_op(op_tuple)
                    ops += [op]

            return ops

        def _build_2bdy_operators_spinful() -> list:
            """Return spinful two-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = -pqsr = -qprs = qpsr
            #                and      =  rspq
            ops = []
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            if p * n_SOs + q >= r * n_SOs + s:
                                op_tuple = ((p, 1), (q, 1), (s, 0), (r, 0))
                                op = _get_of_op(op_tuple)
                                ops += [op]

            return ops

        def _build_1bdy_operators_spinfree() -> list:
            """Return spinfree one-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(n_MOs):
                for q in range(p + 1):
                    # Spin aa
                    op_tuple = ((2 * p, 1), (2 * q, 0))
                    op = _get_of_op(op_tuple)
                    # Spin bb
                    op_tuple = ((2 * p + 1, 1), (2 * q + 1, 0))
                    op += _get_of_op(op_tuple)
                    ops += [op]

            return ops

        def _build_2bdy_operators_spinfree() -> list:
            """Return spinfree two-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    # Spin aaaa
                    op_tuple = (
                        ((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0))
                        if (p != q and r != s)
                        else "0.0 []"
                    )
                    op = _get_of_op(op_tuple)
                    # Spin abab
                    op_tuple = (
                        ((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0))
                        if (2 * p != 2 * q + 1 and 2 * r != 2 * s + 1)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    # Spin baba
                    op_tuple = (
                        ((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0))
                        if (2 * p + 1 != 2 * q and 2 * r + 1 != 2 * s)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    # Spin bbbb
                    op_tuple = (
                        ((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0))
                        if (p != q and r != s)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    ops += [op]
            return ops

        def _assemble_rdm1(evals, rdm1=None) -> numpy.ndarray:
            """Return the spin-ful or spin-free 1-RDM built from the expectation values with the symmetry pq = qp."""
            N = n_MOs if spin_free else n_SOs
            if rdm1 is None:
                rdm1 = numpy.zeros([N, N])
            ctr: int = 0
            for p in range(N):
                for q in range(p + 1):
                    rdm1[p, q] = evals[ctr]
                    # Symmetry pq = qp
                    rdm1[q, p] = rdm1[p, q]
                    ctr += 1

            return rdm1

        def _assemble_rdm2_spinful(evals, rdm2=None) -> numpy.ndarray:
            """Return spin-ful two-particle RDM built by symmetry conditions."""
            ctr: int = 0
            if rdm2 is None:
                rdm2 = numpy.zeros([n_SOs, n_SOs, n_SOs, n_SOs])
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            if p * n_SOs + q >= r * n_SOs + s:
                                rdm2[p, q, r, s] = evals[ctr]
                                # Symmetry pqrs = rspq
                                rdm2[r, s, p, q] = rdm2[p, q, r, s]
                                ctr += 1

            # Further permutational symmetries due to anticommutation relations
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            rdm2[p, q, s, r] = -1 * rdm2[p, q, r, s]  # pqrs = -pqsr
                            rdm2[q, p, r, s] = -1 * rdm2[p, q, r, s]  # pqrs = -qprs
                            rdm2[q, p, s, r] = rdm2[p, q, r, s]  # pqrs =  qpsr

            return rdm2

        def _assemble_rdm2_spinfree(evals, rdm2=None) -> numpy.ndarray:
            """Return spin-free two-particle RDM built by symmetry conditions."""
            ctr: int = 0
            if rdm2 is None:
                rdm2 = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    rdm2[p, q, r, s] = evals[ctr]
                    # Symmetry pqrs = rspq
                    rdm2[r, s, p, q] = rdm2[p, q, r, s]
                    ctr += 1

            # Further permutational symmetry: pqrs = qpsr
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p >= q or r >= s:
                    rdm2[q, p, s, r] = rdm2[p, q, r, s]

            return rdm2

        def _build_1bdy_hcb_operators_spinfree() -> list:
            """Return spinfree one-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(n_MOs):
                for q in range(p + 1):
                    if p == q:
                        # Spin aa
                        op_tuple = ((2 * p, 1), (2 * p, 0))
                        op = _get_of_op(op_tuple)
                        # Spin bb
                        op_tuple = ((2 * p + 1, 1), (2 * p + 1, 0))
                        op += _get_of_op(op_tuple)
                        ops += [op]
                    else:
                        ops += [0.0]
            return ops

        def _build_1bdy_hcb_operators_spinful() -> list:
            """Return spinful one-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp
            ops = []
            for p in range(n_SOs):
                for q in range(p + 1):
                    if p == q:
                        op_tuple = ((p, 1), (q, 0))
                        op = _get_of_op(op_tuple)
                        ops += [op]
                    else:
                        ops += [0.0]
            return ops

        def _build_2bdy_hcb_operators_spinful() -> list:
            """Return spinful two-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = -pqsr = -qprs = qpsr
            #                and      =  rspq
            ops = []
            for p in range(n_SOs):
                for q in range(p):
                    for r in range(n_SOs):
                        for s in range(r):
                            if p * n_SOs + q >= r * n_SOs + s and (
                                (p // 2 == q // 2 and s == r)
                                or (p != q and r != s and p == r and s == q)
                                or (p != q and r != s and p == s and q == r)
                            ):
                                op_tuple = (
                                    ((p, 1), (q, 1), (s, 0), (r, 0))
                                    if (
                                        (
                                            ((p == r and q == s) or (p == s and q == r))
                                            and p % 2 == q % 2 == r % 2 == s % 2
                                        )  # Spin aaaa and bbbb
                                        or (
                                            (
                                                p % 2 == s % 2
                                                and r % 2 == q % 2
                                                and p % 2 != q % 2
                                                and r % 2 != s % 2
                                            )
                                            and (
                                                (p // 2 == q // 2 and r // 2 == s // 2)
                                                or (
                                                    q // 2 == s // 2
                                                    and p // 2 == p // 2
                                                )
                                            )
                                        )  # Spin abba and baab
                                    )
                                    else "0.0 []"
                                )
                                op = _get_of_op(op_tuple)
                                ops += [op]
            return ops

        def _build_2bdy_hcb_operators_spinfree() -> list:
            """Return spinfree two-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    # Spin aaaa
                    op_tuple = (
                        ((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0))
                        if (p == r and q == s) or (p == s and q == r)
                        else "0.0 []"
                    )
                    op = _get_of_op(op_tuple)
                    # Spin abba
                    op_tuple = (
                        ((2 * p, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r, 0))
                        if (p == q and r == s) or (q == s and p == r)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    # Spin baab
                    op_tuple = (
                        ((2 * p + 1, 1), (2 * q, 1), (2 * s, 0), (2 * r + 1, 0))
                        if (p == q and r == s) or (q == s and p == r)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    # Spin bbbb
                    op_tuple = (
                        ((2 * p + 1, 1), (2 * q + 1, 1), (2 * s + 1, 0), (2 * r + 1, 0))
                        if (p == r and q == s) or (p == s and q == r)
                        else "0.0 []"
                    )
                    op += _get_of_op(op_tuple)
                    ops += [op]
            return ops

        # Build operator lists
        qops = []
        if spin_free:
            if use_hcb:
                qops += _build_1bdy_hcb_operators_spinfree() if get_rdm1 else []
                qops += _build_2bdy_hcb_operators_spinfree() if get_rdm2 else []
            else:
                qops += _build_1bdy_operators_spinfree() if get_rdm1 else []
                qops += _build_2bdy_operators_spinfree() if get_rdm2 else []
        else:
            if use_hcb:
                qops += _build_1bdy_hcb_operators_spinful() if get_rdm1 else []
                qops += _build_2bdy_hcb_operators_spinful() if get_rdm2 else []
            else:
                qops += _build_1bdy_operators_spinful() if get_rdm1 else []
                qops += _build_2bdy_operators_spinful() if get_rdm2 else []

        # Transform operator lists to QubitHamiltonians
        qops = [_get_qop_hermitian(op) for op in qops]
        # Compute expected values
        rdm1 = None
        rdm2 = None
        if backend is None:
            backend = self.fermionic_backend
        if evaluate:
            if rdm_trafo is None:
                evals = simulate(
                    Braket(operator=qops, ket=U, shape=[len(qops)], molecule=self),
                    backend=backend,
                    variables=variables,
                )
            else:
                qops = [
                    rdm_trafo.dagger() * qops[i] * rdm_trafo for i in range(len(qops))
                ]
                evals = simulate(
                    Braket(operator=qops, ket=U, shape=[len(qops)], molecule=self),
                    backend=backend,
                    variables=variables,
                )
        else:
            if rdm_trafo is None:
                evals = [Braket(operator=x, ket=U, molecule=self) for x in qops]
                N = n_MOs if spin_free else n_SOs
                rdm1 = QTensor(shape=[N, N])
                rdm2 = QTensor(shape=[N, N, N, N])
            else:
                raise TequilaException(
                    "compute_rdms: rdm_trafo was set but evaluate flag is False (not supported)"
                )

        # Assemble density matrices
        # If self._rdm1, self._rdm2 exist, reset them if they are of the other spin-type
        def _reset_rdm(rdm):
            """Drop a stored RDM if its dimension does not fit the requested spin type."""
            if rdm is not None:
                if (spin_free) and rdm.shape[0] != n_MOs:
                    return None
                if not spin_free and rdm.shape[0] != n_SOs:
                    return None
            return rdm

        self._rdm1 = _reset_rdm(self._rdm1)
        self._rdm2 = _reset_rdm(self._rdm2)
        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = (
                n_MOs * (n_MOs + 1) // 2 if (spin_free) else n_SOs * (n_SOs + 1) // 2
            )
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        # Build matrices using the expectation values
        self._rdm1 = _assemble_rdm1(evals_1, rdm1=rdm1) if get_rdm1 else self._rdm1
        if spin_free:
            self._rdm2 = (
                _assemble_rdm2_spinfree(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2
            )
        else:
            self._rdm2 = (
                _assemble_rdm2_spinful(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2
            )

        if get_rdm2:
            rdm2 = NBodyTensor(elems=self.rdm2, ordering="dirac", verify=False)
            rdm2.reorder(to=ordering)
            rdm2 = rdm2.elems
            self._rdm2 = rdm2

        if get_rdm1:
            if get_rdm2:
                return self.rdm1, self.rdm2
            else:
                return self.rdm1
        elif get_rdm2:
            return self.rdm2
        else:
            warnings.warn(
                "compute_rdms called with instruction to not compute?", TequilaWarning
            )

    def compute_energy(self, method, *args, **kwargs):
        """Compute the energy of a classical or quantum method.

        Parameters
        ----------
        method : str
            Classical (needs PySCF): ``"HF"``, ``"MP2"``, ``"CCSD"``, ``"CCSD(T)"``, ``"FCI"``.
            Quantum: any ansatz name of ``make_upccgsd_ansatz`` such as ``"UpCCD"``, ``"UpCCSD"``, ``"UpCCGSD"``, ``"k-UpCCGSD"``;
            the circuit is optimized with ``sunrise.minimize``.
        *args, **kwargs
            For quantum methods, passed to the minimizer.

        Returns
        -------
        float
            The energy.
        """
        if any(x in method.upper() for x in ["U"]):
            # simulate entirely in HCB representation if no singles are involved
            if "S" not in method.upper().split("-")[-1] and "HCB" not in method.upper():
                method = "HCB-" + method
            U = self.make_ansatz(name=method)
            if "hcb" in method.lower():
                H = self.make_hardcore_boson_hamiltonian()
            else:
                H = self.make_hamiltonian()
            E = Braket(operator=H, ket=U, mol=self)
            from sunrise import minimize

            return minimize(*args, objective=E, **kwargs).energy
        else:
            from sunrise.molecules.qubit_base import INSTALLED_QCHEMISTRY_BACKENDS

            if "pyscf" not in INSTALLED_QCHEMISTRY_BACKENDS:
                raise TequilaException(
                    f"PySCF needs to be installed to compute {method}/{self.parameters.basis_set}"
                )
            else:
                from sunrise.molecules.fermionic_base import QuantumChemistryPySCF

                molx = QuantumChemistryPySCF.from_tequila(self)
                return molx.compute_energy(method=method, **kwargs)

    def n_rotation(self, i, phi) -> FCircuit:
        """Return a phase rotation circuit on both spin orbitals of spatial orbital ``i``.

        Parameters
        ----------
        i : int
            Spatial orbital index.
        phi : float
            Rotation angle; the applied phase is -2 * phi on both components.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> U = mol.n_rotation(0, numpy.pi)
        """
        return _n_rotation(i=i, phi=phi)

    def get_givens_circuit(self, unitary, tol=1e-12, ordering=OPTIMIZED_ORDERING):
        """Return the circuit implementing a real orthogonal matrix with Givens and phase rotations.

        Parameters
        ----------
        unitary : numpy.ndarray
            Real unitary (orthogonal) matrix to decompose.
        tol : float, optional
            Matrix elements below this value are considered zero.
        ordering : list of tuple or str {"Optimized"}, optional
            Order of the index pairs of the Givens rotations, or ``"Optimized"`` (default) to determine it automatically.

        Returns
        -------
        FCircuit

        Examples
        --------
        >>> opt = sun.optimize_orbitals(molecule=mol, circuit=U, silent=True)
        >>> UR = mol.get_givens_circuit(opt.mo_coeff)  # rotation to the optimized orbitals
        """
        return _get_givens_circuit(unitary=unitary, tol=tol, ordering=ordering)

    def __str__(self) -> str:
        """Return a summary of the parameters, the reference state and the basis."""
        result = str(type(self)) + "\n"
        result += "Parameters\n"
        for k, v in self.parameters.__dict__.items():
            result += f"{k!s:15} : {v!s:15} \n"

        result += "{key:15} : {value:15} \n".format(
            key="n_qubits", value=str(self.n_orbitals * 2)
        )
        result += "{key:15} : {value:15} \n".format(
            key="reference state", value=str(self._reference_state())
        )

        result += "\nBasis\n"
        result += str(self.integral_manager)
        result += "\nmore information with: self.print_basis_info()\n"

        return result
