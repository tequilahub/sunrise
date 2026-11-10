"""Quantum chemistry base class: molecule data, orbitals, fermion-to-qubit mapping, Hamiltonians and ansatz circuits."""

from __future__ import annotations

import copy
import inspect
import json
import numbers
import typing
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations, product

import h5py
import numpy
import pyscf
import tequila
import tequila.grouping.fermionic_functions as ff
from tequila import BitString, QTensor, TequilaException, TequilaWarning
from tequila.circuit import QCircuit, gates
from tequila.hamiltonian import QubitHamiltonian
from tequila.hamiltonian.paulis import I, Sm, Sp, Z, Zero
from tequila.objective.objective import ExpectationValue, Objective, Variable, Variables
from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float

from sunrise.symmetry_adaptation.point_group import PointGroup
from sunrise.symmetry_adaptation.point_group_representation import (
    PointGroupRepresentation,
)

from .chemistry_tools import (
    ActiveSpaceData,
    Amplitudes,
    ClosedShellAmplitudes,
    FermionicGateImpl,
    IntegralManager,
    NBodyTensor,
    ParametersQC,
    prepare_product_state,
    to_json_data,
)
from .encodings import EncodingBase, known_encodings

try:
    # if you are experiencing import errors you need to update openfermion
    # required is version >= 1.0
    # otherwise replace with from openfermion.hamiltonians import MolecularData
    import openfermion
    from openfermion.chem import MolecularData
except Exception:  # noqa: BLE001
    try:
        from openfermion.hamiltonians import MolecularData
    except Exception as E:  # noqa: BLE001
        raise TequilaException(
            f"{E!s}\nIssue with Tequila Chemistry: Please update openfermion"
        )
import warnings

OPTIMIZED_ORDERING = "Optimized"


class QuantumChemistryBase:
    """Base class of qubit molecules: integrals, orbitals and a fermion-to-qubit encoding.

    This is what ``sunrise.Molecule(...)`` creates. Main methods (Hamiltonians, ansatz
    circuits, orbital transformations, RDMs) are defined here, derived classes only
    interface specific backends (e.g. Psi4, PySCF, MADNESS).

    Parameters
    ----------
    parameters : ParametersQC
        Molecular parameters (geometry, basis set, charge, ...).
    transformation : str or callable, optional
        Fermion-to-qubit encoding, e.g. "JordanWigner" or "BravyiKitaev". Default is
        Jordan-Wigner. See ``known_encodings`` in encodings.py.
    active_orbitals : list of int, optional
        Active orbitals, all others are frozen. The first ``N//2`` orbitals are
        treated as occupied when the active space is created.
    frozen_orbitals : list of int, optional
        Orbitals removed from the active orbitals.
    orbital_type : str, optional
        Type of the orbitals, e.g. "native". Default is "unknown".
    reference_orbitals : list of int, optional
        Orbitals considered occupied when the active space is created, as total
        indices (frozen orbitals included). Default is the first ``N//2``.
    orbitals : list or dict, optional
        Orbital information in OrbitalData format.
    *args, **kwargs
        Passed on to ``initialize_integral_manager``. For the base class
        ``one_body_integrals``, ``two_body_integrals`` and optionally
        ``nuclear_repulsion`` are required. Keys starting with ``transformation__``
        are passed to the encoding.
        Also ``n_electrons`` (overrides the number from ``parameters``), ``point_group`` and
        ``integral_manager`` (an existing integral manager to reuse).
    """

    def __init__(
        self,
        parameters: ParametersQC,
        transformation: str | typing.Callable | None = None,
        active_orbitals: list | None = None,
        frozen_orbitals: list | None = None,
        orbital_type: str | None = None,
        reference_orbitals: list | None = None,
        orbitals: list | None = None,
        *args,
        **kwargs,
    ):
        """See the class docstring."""

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

        self.transformation = self._initialize_transformation(
            *args, transformation=transformation, **kwargs
        )

        self._rdm1 = None
        self._rdm2 = None
        # given with the point_group keyword, otherwise detected when it is first needed
        self.point_group = kwargs.get("point_group")

    @classmethod
    def from_tequila(cls, molecule, transformation=None, *args, **kwargs):
        """Create a qubit molecule from another molecule, reusing its integrals and orbitals.

        Parameters
        ----------
        molecule : QuantumChemistryBase
            Molecule to copy the data from.
        transformation : str or callable, optional
            Default is the transformation of ``molecule``.
        **kwargs
            Passed on to the constructor, e.g. ``active_orbitals`` (default is the
            active orbitals of ``molecule``).

        Returns
        -------
        QuantumChemistryBase
            New molecule (frozen core is switched off in its parameters).
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
        parameters = molecule.parameters
        parameters.frozen_core = False
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

    def save(self, filename: str):
        """Save the molecule to an HDF5 file without pickle.

        Like a PySCF chkfile, only plain data is stored: parameters, integrals in the basis,
        orbitals, active space, transformation, the settings of the molecule nature and the
        backend data (e.g. the Psi4 reference wavefunction, see ``_backend_data``). So the file
        can be opened from any sunrise installation with ``sunrise.load_molecule``.

        Parameters
        ----------
        filename : str
            Name of the HDF5 file.

        See Also
        --------
        sunrise.load_molecule : Load the saved molecule.

        Examples
        --------
        >>> mol.save("molecule.h5")
        >>> mol = sun.load_molecule("molecule.h5")
        """
        from sunrise.molecules import NATURE_BACKENDS

        nature, backend = next(
            (
                (n, b)
                for n in ("qubit", "hybrid", "fermionic")
                for b, cls in NATURE_BACKENDS[n].items()
                if type(self) is cls
            ),
            (None, None),
        )
        if nature is None:
            raise TequilaException(
                f"can not save molecules of type {type(self).__name__}"
            )

        parameters = asdict(self.parameters)
        if parameters["geometry"].split(".")[-1] == "xyz":
            # keep the coordinates instead of the file name, the file is not around where the molecule is loaded
            parameters["geometry"], comment = self.parameters.read_xyz_from_file(
                parameters["geometry"]
            )
            if not parameters["description"]:
                parameters["description"] = comment

        transformation = None
        if isinstance(self.transformation, EncodingBase):
            # the molecule sets the sizes and the select, the ordering is in the name
            given = (
                "self",
                "n_electrons",
                "n_orbitals",
                "up_then_down",
                "select",
                "args",
                "kwargs",
            )
            transformation = {
                "name": ("Reordered" if self.transformation.up_then_down else "")
                + type(self.transformation).__name__,
                "arguments": {
                    k: getattr(self.transformation, k)
                    for k in inspect.signature(
                        type(self.transformation).__init__
                    ).parameters
                    if k not in given and hasattr(self.transformation, k)
                },
            }
        elif self.transformation is not None:
            warnings.warn(
                f"transformation {self.transformation} can not be saved, give it again with "
                "sun.load_molecule(filename, transformation=...), otherwise the default one is used",
                TequilaWarning,
            )

        # keyword arguments of the nature
        if nature == "hybrid":
            settings = {
                "select": "".join(self.select.values()),
                "condense": self.condense,
                "two_qubit": self.two_qubit,
                "integral_tresh": self.integral_tresh,
            }
        elif nature == "fermionic" and self.fermionic_backend is not None:
            settings = {"fermionic_backend": self.fermionic_backend}
        else:
            settings = {}

        manager = self.integral_manager
        data = {
            "nature": nature,
            "backend": backend,
            "parameters": parameters,
            "constant_term": manager.constant_term,
            "basis_name": manager._basis_name,
            "orbital_type": manager._orbital_type,
            "active_orbitals": manager.active_space.active_orbitals,
            "reference_orbitals": manager.active_space.reference_orbitals,
            "orbitals": [asdict(o) for o in manager.orbitals],
            "point_group": None
            if self._point_group is None
            else self._point_group.schoenflies_label,
            "transformation": transformation,
            "settings": settings,
            "irreps": getattr(self, "irreps", None),
            "pyscf_molecule": self.pyscf_molecule.dumps()
            if hasattr(self, "pyscf_molecule")
            else None,
        }
        backend_data = self._backend_data()
        with h5py.File(filename, "w") as f:
            # numpy scalars and arrays (e.g. orbital energies) as python types
            f["molecule"] = json.dumps(data, default=lambda x: x.tolist())
            if backend_data is not None:
                f["backend_data"] = json.dumps(to_json_data(backend_data))
            f["one_body_integrals"] = manager.one_body_integrals
            f["two_body_integrals"] = manager.two_body_integrals.reorder(
                to="chem"
            ).elems
            f["overlap_integrals"] = manager.overlap_integrals
            f["orbital_coefficients"] = manager.orbital_coefficients

    def _backend_data(self) -> dict | None:
        """Return the backend data that ``save`` stores, so ``load_molecule`` can rebuild the molecule in its backend.

        The data is given back as keyword arguments to the backend's ``__init__``. Nested
        dictionaries, lists, tuples and numpy arrays are allowed (see ``to_json_data``).

        Returns
        -------
        dict or None
            None if the backend has no data of its own: the molecule is then loaded with the base
            backend, unless the backend can be rebuilt from the integrals alone (like PySCF).
        """
        return None

    def supports_ucc(self):
        """Return True if the encoding supports UCC-type operations (e.g. ``make_excitation_gate``)."""
        return self.transformation.supports_ucc

    def _initialize_transformation(self, transformation=None, *args, **kwargs):
        """Return the encoding object from its name (or the callable unchanged).

        Keyword arguments starting with ``transformation__`` are passed to the encoding.
        """

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

        if hasattr(transformation, "upper"):
            # format to conventions
            transformation = transformation.replace("_", "").replace("-", "").upper()
            encodings = known_encodings()
            if transformation in encodings:
                transformation = encodings[transformation](**trafo_args)
            else:
                raise TequilaException(
                    f"Unkown Fermion-to-Qubit encoding {transformation}. Try something like: {list(encodings.keys())}"
                )

        return transformation

    @classmethod
    def from_openfermion(
        cls,
        molecule: openfermion.MolecularData,
        transformation: str | typing.Callable | None = None,
        *args,
        **kwargs,
    ):
        """Create a molecule from an openfermion ``MolecularData`` object.

        Parameters
        ----------
        molecule : openfermion.MolecularData
            The openfermion molecule.
        transformation : str or callable, optional
            Fermion-to-qubit encoding.
        **kwargs
            Passed on to the constructor.

        Returns
        -------
        QuantumChemistryBase
            The molecule.
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
        return cls(
            *args,
            parameters=parameters,
            transformation=transformation,
            molecule=molecule,
            **kwargs,
        )

    def make_excitation_generator(
        self,
        indices: typing.Iterable[tuple[int, int]],
        form: str | None = None,
        remove_constant_term: bool = True,
    ) -> QubitHamiltonian:
        r"""Return the generator of a UCC-type excitation as qubit operator.

        The generator is :math:`M(a^\dagger_{a_0} a_{i_0} a^\dagger_{a_1} a_{i_1} \dots - h.c.)`
        (times :math:`i`), where the qubit map :math:`M` is given by the transformation.

        Parameters
        ----------
        indices : iterable of tuple of int
            Pairs ``[(i_0, a_0), (i_1, a_1), ...]``, each pair destroys ``i`` and creates
            ``a``, in spin orbital notation. Can also be given as flat list
            ``[i_0, a_0, i_1, a_1, ...]``.
        form : str {"involution", "projector"}, optional
            "involution" or "projector" to manipulate the generator. Default is the
            standard fermionic excitation operator.
        remove_constant_term : bool, optional
            Remove the constant term, which does not change the generated unitary unless
            it is controlled. Default is True.

        Returns
        -------
        QubitHamiltonian
            Transformed qubit excitation operator.

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

        qop = qop.simplify()

        if len(qop) == 0:
            warnings.warn(
                "Excitation generator is a unit operator.\n"
                "Non-standard transformations might not work with general fermionic operators\n"
                "indices = " + str(indices),
                category=TequilaWarning,
            )
        return qop

    def make_hardcore_boson_excitation_gate(
        self, indices, angle, control=None, assume_real=True, compile_options="optimize"
    ):
        """Return an excitation gate in the hard-core boson (HCB) approximation (electrons forced
        into spin pairs).

        Use only together with ``make_hardcore_boson_hamiltonian``.

        Parameters
        ----------
        indices : list of tuple of int
            Pairs of spatial orbitals, each pair destroys the first and creates the second.
        angle : float, hashable or tq.Objective
            Rotation angle.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real (halves gradient costs). Default is True.
        compile_options : str, optional
            Passed to the excitation gate. Default is "optimize".

        Returns
        -------
        tq.QCircuit
        """

        def make_hcb_exct_tapered(pair):
            # the tapered orbital L is implicit (see make_hardcore_boson_hamiltonian): the pair only
            # moves between k and L if the orbitals other than k and L hold one pair less than the reference
            L = self.n_orbitals - 1
            k = pair[0] if pair[1] == L else pair[1]
            others = [self.transformation.up(i) for i in range(L) if i != k]
            parity = (-1) ** (self.n_electrons // 2 - 1)
            Qk = 0.5 + 0.5 * parity * (Z(others) if others else I())
            uk = self.transformation.up(k)
            G = Sp(uk) if pair[1] == L else Sm(uk)
            G = (1.0j * (G - G.dagger()) * Qk).simplify()
            p0 = (1.0 - Qk).simplify()
            return gates.GeneralizedRotation(
                angle=angle,
                generator=G,
                p0=p0,
                # without other orbitals P0 vanishes (see make_excitation_gate)
                assume_real=assume_real and len(p0) > 0,
                control=control,
            )

        U = QCircuit()
        target = []
        for pair in indices:
            assert len(pair) == 2
            if (
                "TAPEREDBINARY" in self.transformation.name.upper()
                and self.n_orbitals - 1 in pair
            ):
                U += make_hcb_exct_tapered(pair)
            else:
                target += [
                    self.transformation.up(pair[0]),
                    self.transformation.up(pair[1]),
                ]
        if self.transformation.up_then_down:
            consistency = [x < self.n_orbitals for x in target]
        else:
            consistency = [x % 2 == 0 for x in target]
        if not all(consistency):
            raise TequilaException(
                f"make_hardcore_boson_excitation_gate: Inconsistencies in indices={indices} for encoding: {self.transformation}"
            )
        if len(target):
            U += gates.QubitExcitation(
                angle=angle,
                target=target,
                assume_real=assume_real,
                control=control,
                compile_options=compile_options,
            )
        return U

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
    ):
        """Return the orbital rotation circuit between spatial orbitals ``i`` and ``j``.

        Variables are named automatically, see ``angle`` and ``label``. See arXiv:2207.12421 (Quantum 7, 1073 (2023))
        Eq. 6 for ``UR(0, 1)``.

        Parameters
        ----------
        i, j : int
            Spatial (not spin) orbital indices. A positive angle moves electrons from ``i`` to ``j``.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is the variable ``("R", i, j)`` (times pi).
        label : hashable, optional
            Gives the variable the name ``("R", i, j, label)``, useful for repeated gates
            with individual variables. Ignored if ``angle`` is given.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real (halves gradient costs). Default is True.

        Returns
        -------
        tq.QCircuit

        Examples
        --------
        >>> U = mol.UR(0, 1, angle="a") + mol.UR(2, 3, angle="a")
        """
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
    ):
        """Return the orbital correlator circuit between spatial orbitals ``i`` and ``j``.

        It correlates the two orbitals through a spin-paired double excitation. See
        arXiv:2207.12421 (Quantum 7, 1073 (2023)) Eq. 22 for ``UC(1, 2)``.

        Parameters
        ----------
        i, j : int
            Spatial (not spin) orbital indices. A positive angle moves electrons from ``i`` to ``j``.
        angle : float, hashable or tq.Objective, optional
            Rotation angle. Default is the variable ``("C", i, j)`` (times pi).
        label : hashable, optional
            Gives the variable the name ``("C", i, j, label)``, useful for repeated gates
            with individual variables. Ignored if ``angle`` is given.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real (halves gradient costs). Default is True.

        Returns
        -------
        tq.QCircuit

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
            return gates.QubitExcitation(
                *args,
                target=[2 * i, 2 * j, 2 * i + 1, 2 * j + 1],
                angle=angle,
                control=control,
                assume_real=assume_real,
                **kwargs,
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

    def make_orbital_rotation_gate(self, indices: tuple, *args, **kwargs):
        # backward compatibility
        """Alias of ``UR(indices[0], indices[1], ...)`` kept for backward compatibility."""
        return self.UR(indices[0], indices[1], *args, **kwargs)

    def make_excitation_gate(
        self, indices, angle, control=None, assume_real=True, **kwargs
    ):
        r"""Return a fermionic excitation gate ``exp(-i a/2 G)``.

        The gate is :math:`e^{-i\frac{a}{2} G}` with generator
        :math:`G = i(\prod_{k} a_{q_k}^\dagger a_{p_k} - h.c.)` defined by the indices
        ``[(p0, q0), (p1, q1), ...]``.

        Parameters
        ----------
        indices : list of tuple of int
            Pairs ``(p, q)``, each destroys ``p`` and creates ``q`` (spin orbitals).
        angle : float, hashable or tq.Objective
            Rotation angle.
        control : int or list of int, optional
            Control qubits.
        assume_real : bool, optional
            Assume the wavefunction stays real (halves gradient costs). Default is True.
            Ignored if the generator has no zero eigenvalues (e.g. singles in tapered binary
            encodings with one explicit orbital per spin).

        Returns
        -------
        tq.QCircuit

        Raises
        ------
        TequilaException
            If the transformation does not support UCC operations.

        Examples
        --------
        >>> U = mol.make_excitation_gate(indices=[(0, 4), (1, 5)], angle="a")  # pair excitation
        """

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
        if len(p0) == 0:
            # P0 vanishes if the generator has no zero eigenvalues, e.g. singles in tapered binary
            # encodings with one explicit orbital per spin: use the shift rule that does not need it
            assume_real = False
        if self.transformation.up_then_down:
            idx = []
            for pair in indices:
                idx.append(
                    (
                        pair[0] // 2 + (pair[0] % 2) * self.n_orbitals,
                        pair[1] // 2 + (pair[1] % 2) * self.n_orbitals,
                    )
                )
        else:
            idx = indices
        return QCircuit.wrap_gate(
            FermionicGateImpl(
                angle=angle,
                generator=generator,
                p0=p0,
                transformation=type(self.transformation).__name__.lower(),
                indices=idx,
                assume_real=assume_real,
                control=control,
                **kwargs,
            )
        )

    def make_molecule(self, *args, **kwargs) -> MolecularData:
        """Return the molecule in openfermion MolecularData format.

        The data is computed by ``do_make_molecule``, which backends implement.

        Returns
        -------
        openfermion.MolecularData
        """
        molecule = MolecularData(**self.parameters.molecular_data_param)

        do_compute = True
        # try:
        #     import os
        #     if os.path.exists(self.parameters.filename):
        #         molecule.load()
        #         do_compute = False
        # except OSError:
        #     do_compute = True

        if do_compute:
            molecule = self.do_make_molecule(*args, **kwargs)

        # molecule.save()
        return molecule

    def initialize_integral_manager(self, *args, **kwargs):
        """Return the integral manager, called by ``__init__`` with its arguments passed through.

        Override this in derived classes. The base class requires the keyword arguments
        ``one_body_integrals`` (matrix), ``two_body_integrals`` (NBodyTensor or
        numpy.ndarray with four indices, openfermion ordering) and optionally
        ``nuclear_repulsion`` (constant part of the Hamiltonian).
        """

        n_electrons = self.parameters.total_n_electrons
        if "n_electrons" in kwargs:
            n_electrons = kwargs["n_electrons"]

        assert "one_body_integrals" in kwargs
        assert "two_body_integrals" in kwargs
        one_body_integrals = kwargs["one_body_integrals"]
        kwargs.pop("one_body_integrals")
        two_body_integrals = kwargs["two_body_integrals"]
        kwargs.pop("two_body_integrals")

        if not isinstance(two_body_integrals, NBodyTensor):
            # assuming two_body_integrals are given in openfermion ordering
            ordering = None  # will be auto-detected
            if "ordering" in kwargs:
                ordering = kwargs["ordering"]
                kwargs.pop("ordering")  # let's not confuse the IntegralManager
            two_body_integrals = NBodyTensor(two_body_integrals, ordering=ordering)

        two_body_integrals = two_body_integrals.reorder(to="chem")

        constant_part = 0.0
        if "constant_term" in kwargs:
            constant_part += kwargs["constant_term"]
            kwargs.pop("constant_term")
        if "nuclear_repulsion" in kwargs:
            constant_part += kwargs["nuclear_repulsion"]
            kwargs.pop("nuclear_repulsion")

        if "active_space" not in kwargs:
            active_orbitals = [i for i in range(one_body_integrals.shape[0])]
            if "active_orbitals" in kwargs and kwargs["active_orbitals"] is not None:
                active_orbitals = deepcopy(kwargs["active_orbitals"])
            if "frozen_orbitals" in kwargs and kwargs["frozen_orbitals"] is not None:
                for fo in kwargs["frozen_orbitals"]:
                    if fo in active_orbitals:
                        active_orbitals.remove(fo)

            reference_orbitals = [i for i in range(n_electrons // 2)]
            if (
                "reference_orbitals" in kwargs
                and kwargs["reference_orbitals"] is not None
            ):
                reference_orbitals = kwargs["reference_orbitals"]

            active_space = ActiveSpaceData(
                active_orbitals=sorted(active_orbitals),
                reference_orbitals=sorted(reference_orbitals),
            )
            kwargs["active_space"] = active_space

        if "basis_name" not in kwargs:
            kwargs["basis_name"] = self.parameters.basis_set

        manager = IntegralManager(
            *args,
            one_body_integrals=one_body_integrals,
            two_body_integrals=two_body_integrals,
            constant_term=constant_part,
            **kwargs,
        )

        return manager

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
            Transformation matrix, first index is the old orbital (summed over), second
            the new one. By default defined on the active space.
        ignore_active_space : bool, optional
            If True the matrix is not assumed to be given on the active space only.
        name : str, optional
            Name of the new orbitals.

        Returns
        -------
        QuantumChemistryBase

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
        result = QuantumChemistryBase(
            parameters=self.parameters,
            integral_manager=integral_manager,
            transformation=self.transformation,
        )
        return result

    def rotate_molecule(self, U, angles: dict | None = None):
        """Return the molecule rotated by a circuit of orbital rotations.

        Parameters
        ----------
        U : FCircuit
            Circuit with only ``UR`` gates.
        angles : dict, optional
            Maps the circuit variables to values if they are not mapped yet.

        Returns
        -------
        QuantumChemistryBase
            Rotated molecule.
        """
        U = U.to_udud(norb=self.n_orbitals)
        if angles is not None:
            U = U.map_variables(angles)
        core = 0
        while self.integral_manager.orbitals[core].idx is None:
            core += 1
        n_mo = len(self.integral_manager.orbital_coefficients)
        params = []
        indices = []
        for gate in reversed(U.gates):
            assert gate._name == "UR"
            # Extract parameters and indices from gates
            params.append(gate.variables)
            g = []
            for idx in gate.indices[0]:  # we only need the first index of each UR
                g.extend((i + 2 * core) // 2 for i in idx)
            indices.append(g)
        # Create a matrix for each UR gate
        rot_matrices = []
        for k, index in enumerate(indices):
            tmp = numpy.eye(n_mo)
            if isinstance(params[k], Objective):
                params[k] = params[k]()
            tmp[index[0]][index[0]] = numpy.cos(params[k] / 2)
            tmp[index[0]][index[1]] = numpy.sin(params[k] / 2)
            tmp[index[1]][index[0]] = -numpy.sin(params[k] / 2)
            tmp[index[1]][index[1]] = numpy.cos(params[k] / 2)
            rot_matrices.append(tmp)

        # Multiply all matrices
        tmatrix = rot_matrices[0]
        for matrix in rot_matrices[1:]:
            tmatrix = numpy.dot(tmatrix, matrix)
        self.integral_manager.transform_orbitals(tmatrix.T)
        return self

    def orthonormalize_basis_orbitals(self):
        # backward compatibility
        """Alias of ``use_native_orbitals`` kept for backward compatibility."""
        return self.use_native_orbitals()

    def use_native_orbitals(
        self, inplace=False, core: list | None = None, *args, **kwargs
    ):
        """Return the molecule in the native (orthonormalized atomic) orbital basis.

        For standard basis sets these are orthonormalized Gaussian basis functions.

        Parameters
        ----------
        inplace : bool, optional
            Update the current molecule instead of returning a new one. Default is False.
        core : list of int, optional
            Core orbitals (frozen, doubly occupied), given in the currently used orbitals
            of the molecule (see ``print_basis_info``). Default is inherited from the
            active space of the integral manager. If given, the active native orbitals
            are chosen by their smallest overlap with the core orbitals.
        active : list of int, optional
            Keyword argument. Active orbitals in the native orbital basis. Default is all
            orbitals, or the ``N-len(core)`` orbitals with smallest overlap with ``core``
            if ``core`` is given. For H, He, H, ``active=[0, 1, 2]`` selects the three
            atomic 1s orbitals; with ``core=[0]`` and no ``active``, ``active=[0, 2]`` is
            selected, since the He 1s orbital overlaps most with the lowest HF orbital.
        reference_orbitals : list of int, optional
            Keyword argument. Doubly occupied orbitals, ``N//2`` entries (N = total
            number of electrons). Sets the number of electrons in the active space.

        Returns
        -------
        QuantumChemistryBase
            Molecule in the native basis.

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
                )
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
                )
                parameters = copy.deepcopy(self.parameters)
                result = QuantumChemistryBase(
                    parameters=parameters,
                    integral_manager=integral_manager,
                    transformation=self.transformation,
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
            result = QuantumChemistryBase(
                parameters=self.parameters,
                integral_manager=integral_manager,
                transformation=self.transformation,
            )
            return result

    def get_xyz(self) -> str:
        """Return the geometry as a string in xyz format."""
        geom = self.parameters.get_geometry()
        f = ""
        f += f"{len(geom)}\n"
        f += f"{self.parameters.name}\n"
        for at in geom:
            f += f"{at[0]} {at[1][0]} {at[1][1]} {at[1][2]}\n"
        return f

    def get_spa_edges(self, use_active: bool = True, **kwargs):
        """Return the SPA edges determined by the pure Python JANPA CLPO algorithm.

        The CLPO bonds between atoms come from the atom connectivity (Hartree-Fock density, MP2 with
        ``use_mp2``); each bond gives a pair of orbitals, used as SPA edge.

        Parameters
        ----------
        use_active : bool, optional
            Restrict to the active space. Default is True.
        **kwargs
            Passed on to ``generate_CLPO_molecule_edges``.

        Returns
        -------
        list
            Pairs of orbital indices, one per bond (single indices for lone pairs).

        Examples
        --------
        >>> edges = mol.get_spa_edges()
        >>> U = mol.make_spa_ansatz(edges=edges)
        """
        from sunrise.CLPO.orbital_transformation import generate_CLPO_molecule_edges

        _, edges = generate_CLPO_molecule_edges(
            self, use_active=use_active, silent=True, **kwargs
        )
        return edges

    def use_HAO_orbitals(
        self, use_active: bool = True, inplace: bool = False, *args, **kwargs
    ):
        """Return the molecule in Hybrid Atomic Orbitals (HAO) from the pure Python JANPA pipeline.

        JANPA determines the hybridization from the 3D geometry, so ``sp_list`` is ignored.

        Parameters
        ----------
        use_active : bool, optional
            Restrict to the active space. Default is True.
        inplace : bool, optional
            Update the current molecule instead of returning a new one. Default is False.
        **kwargs
            Passed on to ``generate_HAO_molecule``.

        Returns
        -------
        QuantumChemistryBase
            Molecule in the HAO basis.
        """
        from sunrise.CLPO.orbital_transformation import generate_HAO_molecule

        hao_mol = generate_HAO_molecule(
            self, use_active=use_active, silent=True, **kwargs
        )

        if inplace:
            self.integral_manager.orbital_coefficients = (
                hao_mol.integral_manager.orbital_coefficients.copy()
            )
            try:
                # hybrid molecules also take the encoding of the new molecule
                self.transformation = hao_mol.transformation
                self.update_select(hao_mol.select)
            except AttributeError:
                pass
            return self
        else:
            return hao_mol

    def use_CLPO_orbitals(
        self,
        use_active: bool = True,
        inplace: bool = False,
        edges: list | None = None,
        *args,
        **kwargs,
    ):
        """Return the molecule in Chemist's Localized Property-optimized Orbitals (CLPO).

        The orbitals come from the pure Python JANPA pipeline, which uses the 3D
        geometry, so ``core`` is ignored.

        Parameters
        ----------
        use_active : bool, optional
            Restrict to the active space. Default is True.
        inplace : bool, optional
            Update the current molecule instead of returning a new one. Default is False.
        edges : list, optional
            Edges the CLPO algorithm is restricted to. Default is determined by JANPA.

        Returns
        -------
        QuantumChemistryBase
            Molecule in the CLPO basis.
        """
        clpo_mol, _ = self.use_CLPO_orbitals_and_edges(
            *args, use_active=use_active, inplace=inplace, edges=edges, **kwargs
        )
        return clpo_mol

    def use_CLPO_orbitals_and_edges(
        self,
        use_active: bool = True,
        inplace=False,
        edges: list | None = None,
        *args,
        **kwargs,
    ):
        """Return the CLPO molecule together with its SPA edges.

        Same as ``use_CLPO_orbitals``, but the edges found by the same run are returned
        too, so JANPA is executed only once.

        Parameters
        ----------
        use_active : bool, optional
            Keep the active space of the molecule (the CLPOs are rotated into it). Default is True.
        inplace : bool, optional
            Change this molecule instead of returning a new one. Default is False.
        edges : list of tuple of int, optional
            Bonds imposed on the CLPOs, as 0-based atom indices (see ``generate_CLPO_molecule_edges``);
            the other bonds come from the atom connectivity.

        Returns
        -------
        tuple
            Molecule in the CLPO basis and the orbital pairs of each bond, to use as SPA edges.

        Examples
        --------
        >>> mol, edges = mol.use_CLPO_orbitals_and_edges()
        """
        from sunrise.CLPO.orbital_transformation import generate_CLPO_molecule_edges

        clpo_mol, clpo_edges = generate_CLPO_molecule_edges(
            self, edges=edges, use_active=use_active, silent=True, **kwargs
        )

        if inplace:
            self.integral_manager.orbital_coefficients = (
                clpo_mol.integral_manager.orbital_coefficients.copy()
            )
            try:
                # hybrid molecules also take the encoding of the new molecule
                self.transformation = clpo_mol.transformation
                self.update_select(clpo_mol.select)
            except AttributeError:
                pass
            return self, clpo_edges
        else:
            return clpo_mol, clpo_edges

    @property
    def point_group(self):
        """PointGroup : point group of the molecule.

        If it is not given with the ``point_group`` keyword, it is detected from the geometry as the
        largest point group defined in sunrise: the full point group if it is C3v, C4v, D2d, D3h, D3d,
        D4h, D6h, D8h, Td or Oh, the largest Abelian subgroup otherwise. PySCF labels the orbitals in
        ``abelian_point_group``. It can be set with a PointGroup or its Schoenflies label, e.g. to work in
        a subgroup.
        """
        if self._point_group is None:
            # the point group only depends on the nuclei, so the basis is not needed (it can be custom)
            pyscf_mol = pyscf.gto.M(
                atom=str(self.parameters.geometry),
                charge=self.parameters.charge,
                unit=self.parameters.units,
                symmetry=True,
                verbose=0,
            )
            self.point_group = (
                pyscf_mol.topgroup
                if hasattr(PointGroup, pyscf_mol.topgroup)
                else pyscf_mol.groupname
            )
        return self._point_group

    @point_group.setter
    def point_group(self, point_group):
        if isinstance(point_group, str):
            label = pyscf.symm.std_symb(point_group)
            # PySCF keeps the linear groups and SO3 for atoms, their largest Abelian subgroups are D2h and C2v
            label = {"Dooh": "D2h", "Coov": "C2v", "SO3": "D2h"}.get(label, label)
            # the Abelian groups are taken from PySCF, the others are defined in sunrise
            point_group = (
                PointGroup.from_pyscf(label)
                if label in pyscf.symm.param.CHARACTER_TABLE
                else getattr(PointGroup, label)
            )
        self._point_group = point_group

    @property
    def abelian_point_group(self):
        """PointGroup : the point group if it is Abelian, its largest Abelian subgroup otherwise.

        For example D2h for D4h and D6h, C2v for C4v. PySCF can only label the orbitals in Abelian groups.
        """
        label = self.point_group.schoenflies_label
        if label in pyscf.symm.param.CHARACTER_TABLE:
            return self.point_group
        return PointGroup.from_pyscf(
            {
                "C3v": "Cs",
                "C4v": "C2v",
                "D2d": "D2",
                "D3h": "C2v",
                "D3d": "C2h",
                "D4h": "D2h",
                "D6h": "D2h",
                "D8h": "D2h",
                "Td": "D2",
                "Oh": "D2h",
            }[label]
        )

    def mo_representation(self):
        r"""Return the representation of the point group on the active orbitals.

        The symmetry operations are matrices acting on the active orbitals and the states are
        QubitWaveFunctions with ``2 * n_orbitals`` qubits. The geometric operations are projected into
        the basis of the active orbitals with exact AO rotation matrices (s and p orbitals): the operation
        :math:`g` transforms the orbitals as

        .. math::
            g \phi_j = \sum_i D_{ij} \phi_i, \qquad D_{ij} = \langle \phi_i | g \phi_j \rangle.

        :math:`D` is a signed permutation if :math:`g` maps every orbital onto another one (e.g. localized
        orbitals in the standard orientation); in general it mixes orbitals (e.g. degenerate canonical
        orbitals or p orbitals under C3). The operations act directly on the bitstrings of the states, no
        circuits are needed.

        Returns
        -------
        PointGroupRepresentation

        Raises
        ------
        ValueError
            If an atom is not mapped onto an atom, or an operation maps the active orbitals out of the
            active space (choose active orbitals that the operations map onto each other, e.g. all
            orbitals of a degenerate set).
        NotImplementedError
            For basis functions beyond p orbitals.

        Examples
        --------
        >>> rep = mol.use_native_orbitals().mo_representation()
        >>> rep.character_vector(wfn)  # characters of a state of a one-dimensional irrep
        """
        geom_repr = PointGroupRepresentation.get_geometric_representation(
            self.point_group, self.parameters.get_geometry()
        )
        pyscf_mol = pyscf.gto.M(
            atom=str(self.parameters.geometry),
            basis=self.parameters.basis_set,
            charge=self.parameters.charge,
            unit=self.parameters.units,
            verbose=0,
        )
        S_ao = pyscf_mol.intor("int1e_ovlp")
        coords = pyscf_mol.atom_coords()
        center = numpy.mean(coords, axis=0)
        aoslice = pyscf_mol.aoslice_by_atom()
        n_ao = pyscf_mol.nao_nr()

        # the matrices are written in the orbitals the states are written in, signs included
        C_mo = numpy.asarray(self.integral_manager.orbital_coefficients)[
            :, [orbital.idx_total for orbital in self.orbitals]
        ]

        operations = {}
        for label, R_3x3 in geom_repr.operations.items():
            # exact AO transformation matrix: atom A is moved onto atom B, its s orbitals
            # keep their sign and its p orbitals transform like x, y, z
            coords_rot = (coords - center) @ R_3x3.T + center
            U_AO = numpy.zeros((n_ao, n_ao))
            for A, c_rot in enumerate(coords_rot):
                dists = numpy.linalg.norm(coords - c_rot, axis=1)
                B = numpy.argmin(dists)
                if dists[B] > 1e-4:
                    raise ValueError(
                        f"Atom {A} does not map to any atom under {label}."
                    )
                for sh_A, sh_B in zip(range(*aoslice[A][:2]), range(*aoslice[B][:2])):
                    l = pyscf_mol.bas_angular(sh_A)
                    ao_A = slice(pyscf_mol.ao_loc[sh_A], pyscf_mol.ao_loc[sh_A + 1])
                    ao_B = slice(pyscf_mol.ao_loc[sh_B], pyscf_mol.ao_loc[sh_B + 1])
                    if l == 0:
                        U_AO[ao_B, ao_A] = 1.0
                    elif l == 1:
                        U_AO[ao_B, ao_A] = R_3x3
                    else:
                        raise NotImplementedError(
                            f"Angular momentum l={l} not supported."
                        )

            D = C_mo.T @ S_ao @ U_AO @ C_mo

            # D is orthogonal only if the operation maps the active orbitals onto themselves
            if not numpy.allclose(D.T @ D, numpy.eye(self.n_orbitals), atol=1e-4):
                raise ValueError(
                    f"The operation {label} maps the active orbitals out of the active space: choose active "
                    f"orbitals that the operations of {self.point_group.schoenflies_label} map onto each other, "
                    f"e.g. all orbitals of a degenerate set."
                )
            operations[label] = D

        def _apply(D, state):
            r"""Apply the orbital transformation :math:`a_j^\dagger \to \sum_i D_{ij} a_i^\dagger` (both spins) to the bitstrings of ``state``.

            With the up operators written before the down ones, a determinant with up orbitals :math:`I` and
            down orbitals :math:`J` goes to :math:`\sum_{I', J'} \det(D_{I'I}) \det(D_{J'J}) |I', J'\rangle`.
            """
            n_orbitals = self.n_orbitals
            n_qubits = 2 * n_orbitals
            result_terms = {}
            # expansions of the up and down orbital sets, shared by all determinants of the state
            expansions = {}

            for bitstring, coef in state.items():
                bs_str = format(int(bitstring), f"0{n_qubits}b")
                up = tuple(i for i in range(n_orbitals) if bs_str[2 * i] == "1")
                down = tuple(i for i in range(n_orbitals) if bs_str[2 * i + 1] == "1")

                for occupied in (up, down):
                    if occupied not in expansions:
                        targets = list(combinations(range(n_orbitals), len(occupied)))
                        targets = numpy.array(targets, dtype=int).reshape(
                            len(targets), len(occupied)
                        )
                        dets = numpy.linalg.det(
                            D[targets[:, :, None], numpy.array(occupied, dtype=int)]
                        )
                        expansions[occupied] = [
                            (target, det)
                            for target, det in zip(targets, dets)
                            if abs(det) > 1e-12
                        ]

                # fermionic sign of moving the down operators behind the up ones, and back
                sign = (-1) ** sum(j < i for i in up for j in down)
                for new_up, det_up in expansions[up]:
                    for new_down, det_down in expansions[down]:
                        new_sign = (-1) ** sum(j < i for i in new_up for j in new_down)
                        new_bits = ["0"] * n_qubits
                        for i in new_up:
                            new_bits[2 * i] = "1"
                        for j in new_down:
                            new_bits[2 * j + 1] = "1"
                        new_bs_int = int("".join(new_bits), 2)
                        result_terms[new_bs_int] = (
                            result_terms.get(new_bs_int, 0)
                            + sign * new_sign * det_up * det_down * coef
                        )

            result = tequila.QubitWaveFunction(n_qubits=n_qubits)
            for bs_int, c in result_terms.items():
                if abs(c) > 1e-15:
                    result[bs_int] = c
            return result

        return PointGroupRepresentation(
            operations=operations,
            application_function=_apply,
            is_close_function=tequila.QubitWaveFunction.isclose,
        )

    def compute_orbital_irreps(self):
        """Return the irreducible representations of the active orbitals.

        They are determined by PySCF in the Abelian point group of the molecule, so the
        orbitals have to be symmetry adapted (e.g. canonical orbitals).

        Examples
        --------
        >>> irreps = mol.compute_orbital_irreps()
        """
        pyscf_mol = pyscf.gto.M(
            atom=str(self.parameters.geometry),
            basis=self.parameters.basis_set,
            symmetry=True,
            symmetry_subgroup=self.abelian_point_group.schoenflies_label,
            charge=self.parameters.charge,
            unit=self.parameters.units,
            verbose=0,
        )
        orbital_coefficients = numpy.asarray(
            self.integral_manager.orbital_coefficients
        )[:, [orbital.idx_total for orbital in self.orbitals]]

        return pyscf.symm.label_orb_symm(
            pyscf_mol, pyscf_mol.irrep_name, pyscf_mol.symm_orb, orbital_coefficients
        ).tolist()

    def do_make_molecule(self, *args, **kwargs):
        """Return the openfermion MolecularData built from the integrals, called by ``make_molecule``.

        Override in derived classes if needed.
        """

        assert hasattr(self, "integral_manager") and self.integral_manager is not None
        constant_term, one_body_integrals, two_body_integrals = (
            self.integral_manager.get_integrals(ordering="of")
        )
        two_body_integrals = two_body_integrals.reorder(to="of")

        if "n_orbitals" in kwargs:
            n_orbitals = kwargs["n_orbitals"]
        else:
            n_orbitals = one_body_integrals.shape[0]
            for i in [0, 1, 2, 3]:
                assert n_orbitals == two_body_integrals.shape[i]

        molecule = MolecularData(**self.parameters.molecular_data_param)

        molecule.one_body_integrals = one_body_integrals
        molecule.two_body_integrals = two_body_integrals.elems
        molecule.nuclear_repulsion = constant_term
        molecule.n_orbitals = n_orbitals
        if "n_electrons" in kwargs:
            molecule.n_electrons = kwargs["n_electrons"]
        # molecule.save()
        return molecule

    @property
    def orbitals(self):
        """list of OrbitalData : active orbitals of the integral manager."""
        return self.integral_manager.active_orbitals

    @property
    def reference_orbitals(self):
        """list of OrbitalData : active orbitals that are occupied in the reference state."""
        return self.integral_manager.active_reference_orbitals

    @property
    def n_orbitals(self) -> int:
        """int : number of active spatial orbitals."""
        return len(self.integral_manager.active_orbitals)

    @property
    def active_space(self):
        """ActiveSpaceData : active space of the integral manager."""
        return self.integral_manager.active_space

    @property
    def n_electrons(self) -> int:
        """int : number of active electrons."""
        return 2 * len(self.integral_manager.active_reference_orbitals)

    def make_annihilation_op(self, orbital, coefficient=1.0):
        """Return the annihilation operator of a spin orbital as qubit operator.

        Spin orbitals are labeled ``2i`` (up) and ``2i+1`` (down) in every encoding. Supported:
        Jordan-Wigner and (tree) Bravyi-Kitaev, also reordered, and fully fermionic hybrid
        molecules. Not supported: ``BravyiKitaevFast``, the tapered encodings and bosonic orbitals.

        Parameters
        ----------
        orbital : int
            Spin orbital index.
        coefficient : float, optional
            Prefactor. Default is 1.0.

        Returns
        -------
        QubitHamiltonian
        """
        assert orbital < self.n_orbitals * 2
        aop = openfermion.ops.FermionOperator(f"{orbital}", coefficient)
        return self.transformation(aop)

    def make_creation_op(self, orbital, coefficient=1.0):
        """Return the creation operator of a spin orbital as qubit operator.

        Spin orbitals are labeled ``2i`` (up) and ``2i+1`` (down) in every encoding. Supported:
        Jordan-Wigner and (tree) Bravyi-Kitaev, also reordered, and fully fermionic hybrid
        molecules. Not supported: ``BravyiKitaevFast``, the tapered encodings and bosonic orbitals.

        Parameters
        ----------
        orbital : int
            Spin orbital index.
        coefficient : float, optional
            Prefactor. Default is 1.0.

        Returns
        -------
        QubitHamiltonian
        """
        assert orbital < self.n_orbitals * 2
        cop = openfermion.ops.FermionOperator(f"{orbital}^", coefficient)
        return self.transformation(cop)

    def make_number_op(self, orbital):
        """Return the number operator of a spin orbital as qubit operator.

        Spin orbitals are labeled ``2i`` (up) and ``2i+1`` (down) in every encoding. Supported:
        Jordan-Wigner and (tree) Bravyi-Kitaev, also reordered, and fully fermionic hybrid
        molecules. Not supported: ``BravyiKitaevFast``, the tapered encodings and bosonic orbitals.

        Parameters
        ----------
        orbital : int
            Spin orbital index (interleaved, see ``make_creation_op``).

        Returns
        -------
        QubitHamiltonian
        """
        num_op = self.make_creation_op(orbital) * self.make_annihilation_op(orbital)
        return num_op

    def make_sz_op(self):
        """Return the total :math:`S_z` operator of the molecule as qubit operator.

        Built from ``make_creation_op`` and ``make_annihilation_op``, so it works for the same
        encodings (see there).
        """
        sz = QubitHamiltonian()
        for i in range(0, self.n_orbitals * 2, 2):
            one = 0.5 * self.make_creation_op(i) * self.make_annihilation_op(i)
            two = 0.5 * self.make_creation_op(i + 1) * self.make_annihilation_op(i + 1)
            sz += one - two
        return sz

    def make_sp_op(self):
        """Return the total :math:`S_+` operator of the molecule as qubit operator.

        Built from ``make_creation_op`` and ``make_annihilation_op``, so it works for the same
        encodings (see there).
        """
        sp = QubitHamiltonian()
        for i in range(self.n_orbitals):
            sp += self.make_creation_op(i * 2) * self.make_annihilation_op(i * 2 + 1)
        return sp

    def make_sm_op(self):
        """Return the total :math:`S_-` operator of the molecule as qubit operator.

        Built from ``make_creation_op`` and ``make_annihilation_op``, so it works for the same
        encodings (see there).
        """
        sm = QubitHamiltonian()
        for i in range(self.n_orbitals):
            sm += self.make_creation_op(i * 2 + 1) * self.make_annihilation_op(i * 2)
        return sm

    def make_s2_op(self):
        """Return the total :math:`S^2` operator of the molecule as qubit operator.

        Built from ``make_creation_op`` and ``make_annihilation_op``, so it works for the same
        encodings (see there).
        """
        s2_op = self.make_sm_op() * self.make_sp_op() + self.make_sz_op() * (
            self.make_sz_op() + 1
        )
        return s2_op

    def make_hamiltonian(self, *args, **kwargs) -> QubitHamiltonian:
        """Return the molecular Hamiltonian as qubit operator, in the encoding of this molecule.

        The active space is fixed by the molecule, it cannot be changed here
        (``active_indices`` is ignored with a warning).

        Returns
        -------
        QubitHamiltonian

        Examples
        --------
        >>> H = mol.make_hamiltonian()
        """

        # warnings for backward comp
        if "active_indices" in kwargs:
            warnings.warn(
                "active space can't be changed in molecule. Will ignore active_orbitals passed to make_hamiltonian"
            )

        of_molecule = self.make_molecule()
        fop = of_molecule.get_molecular_hamiltonian()
        fop = openfermion.transforms.get_fermion_operator(fop)
        try:
            qop = self.transformation(fop)
        except TypeError:
            qop = self.transformation(
                openfermion.transforms.get_interaction_operator(fop)
            )
        qop.is_hermitian()
        return qop

    def make_hardcore_boson_hamiltonian(self, condensed=False):
        """Return the Hamiltonian in the hard-core boson (HCB) approximation (electrons forced into
        spin pairs).

        It is independent of the fermion-to-qubit mapping.

        Parameters
        ----------
        condensed : bool, optional
            If True the Hamiltonian acts on qubits ``0`` to ``n_orbitals - 1``. Otherwise
            its qubits follow the encoding, e.g. the even qubits ``0`` to ``2*n_orbitals``
            for Jordan-Wigner. Default is False.

        Returns
        -------
        QubitHamiltonian

        Examples
        --------
        >>> H = mol.make_hardcore_boson_hamiltonian()
        """

        def hcb_ham_tapered():
            # the last orbital L is implicit: it holds a pair if the other orbitals hold one pair less
            # than the reference, and the parity of their Z operators tells whether they do
            L = n_orbitals - 1
            parity = (-1) ** (self.n_electrons // 2 - 1)
            nL = 0.5 + 0.5 * parity * (Z(list(range(L))) if L > 0 else I())
            H = c + h[L, L] * nL
            for p in range(L):
                # the pair moves between p and L if the other orbitals hold one pair less than the reference
                others = [i for i in range(L) if i != p]
                Qp = 0.5 + 0.5 * parity * (Z(others) if others else I())
                H += (h[p, L] * Sm(p) + h[L, p] * Sp(p)) * Qp
                H += (g[p, L] + g[L, p]) * Sm(p) * Sp(p) * nL
                for q in range(L):
                    H += h[p, q] * Sm(p) * Sp(q)
                    H += g[p, q] * Sm(p) * Sp(p) * Sm(q) * Sp(q)

            if not self.transformation.up_then_down and not condensed:
                alpha_map = {
                    k.idx: self.transformation.up(k.idx) for k in self.orbitals
                }
                H = H.map_qubits(alpha_map)
            return H

        # integrate with QubitEncoding at some point
        n_orbitals = self.n_orbitals
        c, obt, tbt = self.get_integrals()
        h = numpy.zeros(shape=[n_orbitals] * 2)
        g = numpy.zeros(shape=[n_orbitals] * 2)
        for p in range(n_orbitals):
            h[p, p] += 2 * obt[p, p]
            for q in range(n_orbitals):
                h[p, q] += +tbt.elems[p, p, q, q]
                if p != q:
                    g[p, q] += 2 * tbt.elems[p, q, q, p] - tbt.elems[p, q, p, q]

        if "TAPEREDBINARY" in self.transformation.name.upper():
            return hcb_ham_tapered()

        H = c
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                up = p
                uq = q
                H += h[p, q] * Sm(up) * Sp(uq) + g[p, q] * Sm(up) * Sp(up) * Sm(
                    uq
                ) * Sp(uq)

        if not self.transformation.up_then_down and not condensed:
            alpha_map = {k.idx: self.transformation.up(k.idx) for k in self.orbitals}
            H = H.map_qubits(alpha_map)
        return H

    def make_molecular_hamiltonian(self, occupied_indices=None, active_indices=None):
        """Return the Hamiltonian as openfermion MolecularHamiltonian (used internally)."""
        return self.make_molecule().get_molecular_hamiltonian(
            occupied_indices=occupied_indices, active_indices=active_indices
        )

    def get_integrals(self, *args, **kwargs):
        """Return the constant part and the one- and two-body integrals.

        Parameters
        ----------
        ordering : str {"openfermion", "chem", "phys"}, optional
            Index ordering of the two-body integrals (``two_body_ordering`` is accepted too).
        ignore_active_space : bool, optional
            Return the integrals of the full space.

        Returns
        -------
        tuple
            Constant part (nuclear repulsion plus integrated frozen parts), one-body
            integrals and two-body integrals.

        Examples
        --------
        >>> c, h1, h2 = mol.get_integrals()
        """

        # backward compatibility
        if "two_body_ordering" in kwargs:
            kwargs["ordering"] = kwargs["two_body_ordering"]

        return self.integral_manager.get_integrals(*args, **kwargs)

    def compute_one_body_integrals(self):
        """Return the one-body integrals."""
        _c, h1, _h2 = self.get_integrals()
        return h1

    def compute_two_body_integrals(self, ordering="openfermion"):
        """Return the two-body integrals with the given ordering ("openfermion", "chem" or "phys")."""
        _c, _h1, h2 = self.get_integrals(ordering=ordering)
        return h2

    def compute_constant_part(self):
        """Return the constant part of the Hamiltonian (nuclear repulsion plus frozen parts)."""
        c, _h1, _h2 = self.get_integrals()
        return c

    def compute_ccsd_amplitudes(self) -> ClosedShellAmplitudes:
        """Return the closed-shell CCSD amplitudes (implemented in the backends).

        Examples
        --------
        >>> amplitudes = mol.compute_ccsd_amplitudes()
        >>> U = mol.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes=amplitudes)
        """
        raise TequilaException("BaseClass Method")

    def _reference_state(self):
        """Return the reference occupation vector in Jordan-Wigner notation (up, down, up, down, ...)."""
        assert self.n_electrons % 2 == 0
        state = [0] * (self.n_orbitals * 2)
        for i in self.reference_orbitals:
            state[2 * i.idx] = 1
            state[2 * i.idx + 1] = 1
        return state

    def prepare_reference(self, state=None, *args, **kwargs):
        """Return the circuit preparing the reference (Hartree-Fock) state in the encoding of this
        molecule.

        Parameters
        ----------
        state : list of int, optional
            Occupation vector of the spin orbitals (up, down, up, down, ...). Default is the
            closed-shell Hartree-Fock reference.

        Returns
        -------
        tq.QCircuit

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
        U.n_qubits = self.n_orbitals * 2  # adapt when tapered transformations work
        return U

    def prepare_hardcore_boson_reference(self):
        """Return the circuit preparing the reference state in the hard-core boson (HCB)
        approximation.

        Each qubit represents two spin-paired electrons.

        Returns
        -------
        tq.QCircuit
        """
        U = gates.X(
            target=[self.transformation.up(i.idx) for i in self.reference_orbitals]
        )
        U.n_qubits = self.n_orbitals
        return U

    def hcb_to_me(self, U=None, condensed=False):
        """Map a circuit from the hard-core boson (HCB) encoding to the encoding of this molecule.

        HCB is encoded on the alpha qubits of the molecule.

        Parameters
        ----------
        U : tq.QCircuit, optional
            HCB circuit. Default is an empty circuit.
        condensed : bool, optional
            Assume that ``U`` is condensed, i.e. HCB is on the first ``n_orbitals`` qubits
            and not, as e.g. for Jordan-Wigner, on the even qubits. Default is False.

        Returns
        -------
        tq.QCircuit

        Examples
        --------
        >>> U = mol.hcb_to_me(U=U_hcb)  # map a hard-core boson circuit to this encoding
        """
        if U is None:
            U = QCircuit()
        else:
            ups = [self.transformation.up(i.idx) for i in self.orbitals]
            consistency = [x in ups for x in U.qubits]
            if not all(consistency):
                warnings.warn(
                    f"hcb_to_me: given circuit is not defined on all first {self.n_orbitals} qubits. Is this a HCB circuit?"
                )

        # map to alpha qubits
        if condensed:
            alpha_map = {k: self.transformation.up(k) for k in range(self.n_orbitals)}
            alpha_U = U.map_qubits(qubit_map=alpha_map)
        else:
            alpha_U = U

        UX = self.transformation.hcb_to_me()
        if UX is None:
            raise AttributeError(
                f"transformation={self.transformation} has no hcb_to_me function implemented"
            )
        return alpha_U + UX

    def get_pair_specific_indices(
        self,
        pair_info: str | None = None,
        include_singles: bool = True,
        general_excitations: bool = True,
    ) -> list:
        """Return indices for a pair-specific ansatz.

        Excite from a set of references ``i`` to any pair coming from ``i``, i.e. any
        ``(i, j)``/``(j, i)``. With general excitations also from pairs to appendant
        pairs and the reference.

        Parameters
        ----------
        pair_info : list of str or str
            Pair structure as list or as name of a file (first row is skipped). A single
            number is a reference, two numbers a pair. For ``['0', '1', '11', '11', '00', '10']``
            orbitals 0 and 1 are references, the next two orbitals belong to pair 11, one
            to 00 and one is mixed 10. Required.
        include_singles : bool, optional
            Include single excitations. Default is True.
        general_excitations : bool, optional
            Allow general excitations. Default is True.

        Returns
        -------
        list
            Spin-orbital indices for ``make_upccgsd_ansatz``.
        """

        if pair_info is None:
            raise TequilaException("Need to provide some pair information.")
        # If pair-information given on file, load (layout see above)
        if isinstance(pair_info, str):
            pairs = numpy.loadtxt(pair_info, dtype=str, delimiter=",", skiprows=1)
        elif isinstance(pair_info, list):
            pairs = pair_info
        elif not isinstance(pair_info, list):
            raise TequilaException(
                "Pair information needs to be contained in a list or filename."
            )

        connect = [[]] * len(pairs)
        # determine "connectivity"
        for idx, p in enumerate(pairs):
            if len(p) == 1:
                connect[idx] = [
                    i
                    for i in range(len(pairs))
                    if ((len(pairs[i]) == 2) and (str(idx) in pairs[i]))
                ]
            elif (len(p) == 2) and general_excitations:
                connect[idx] = [
                    i
                    for i in range(len(pairs))
                    if (
                        ((p[0] in pairs[i]) or (p[1] in pairs[i]) or str(i) in p)
                        and (i != idx)
                    )
                ]
            elif len(p) > 2:
                raise TequilaException("Invalid reference of pair id.")

        # create generating indices from connectivity
        indices = []
        for i, to in enumerate(connect):
            for a in to:
                indices.append(((2 * i, 2 * a), (2 * i + 1, 2 * a + 1)))
                if include_singles:
                    indices.append(((2 * i, 2 * a),))
                    indices.append(((2 * i + 1, 2 * a + 1),))

        return indices

    def format_excitation_indices(self, idx):
        """Return the indices of a single excitation in a consistent format.

        For ``idx = [(p0, q0), (p1, q1), ..., (pn, qn)]`` each pair is first sorted so that
        ``pi < qi``, then the pairs are reordered such that ``p0 < p1 < ... < pn``.

        Parameters
        ----------
        idx : list of tuple of int
            Pairs ``[(p0, q0), (p1, q1), ...]`` describing one fermionic excitation.

        Returns
        -------
        tuple of tuple of int
            Each pair sorted as ``p < q``, the pairs sorted by their first entry.
        """

        idx = [tuple(sorted(x)) for x in idx]
        idx = sorted(idx, key=lambda x: x[0])
        return tuple(idx)

    def make_upccgsd_indices(self, key, reference_orbitals=None, *args, **kwargs):
        """Return the spatial orbital pairs (singles/doubles generators) of a UpCCGSD-type ansatz.

        Parameters
        ----------
        key : str
            "ladder" gives ``(n, n+1)``. A name without "g" (e.g. "UpCCD") gives
            excitations from the reference orbitals into the others, a name with "g"
            (e.g. "UpCCGSD") gives all ``(n, m)`` with ``n < m``.
        reference_orbitals : list of int, optional
            Default is the reference orbitals of the molecule.

        Returns
        -------
        list
            Formatted excitation indices, see ``format_excitation_indices``.
        """
        if reference_orbitals is None:
            reference_orbitals = [x.idx for x in self.reference_orbitals]
        indices = []
        # add doubles in hcb encoding
        if hasattr(key, "lower") and key.lower() == "ladder":
            # ladder structure of the pair excitations
            # ensures local connectivity
            indices = [[(n, n + 1)] for n in range(self.n_orbitals - 1)]
        elif hasattr(key, "lower") and "g" not in key.lower():
            indices = [
                [(n, m)]
                for n in reference_orbitals
                for m in range(self.n_orbitals)
                if n < m and m not in reference_orbitals
            ]
        elif hasattr(key, "lower") and "g" in key.lower():
            indices = [
                [(n, m)]
                for n in range(self.n_orbitals)
                for m in range(self.n_orbitals)
                if n < m
            ]
        else:
            raise TequilaException(f"Unknown recipe: {key}")
        indices = [self.format_excitation_indices(idx) for idx in indices]

        return indices

    def make_hardcore_boson_upccgd_layer(
        self,
        indices: list = "UpCCGD",
        label: str | None = None,
        assume_real: bool = True,
        *args,
        **kwargs,
    ):
        """Return one layer of hard-core boson (HCB) pair doubles (UpCCGD).

        Parameters
        ----------
        indices : list or str, optional
            Excitation indices, or the name of a recipe for ``make_upccgsd_indices``. Default is "UpCCGD".
        label : hashable, optional
            Label of the variables.
        assume_real : bool, optional
            Assume the wavefunction stays real. Default is True.

        Returns
        -------
        tq.QCircuit
        """
        if hasattr(indices, "lower"):
            indices = self.make_upccgsd_indices(key=indices.lower())

        UD = QCircuit()
        for idx in indices:
            UD += self.make_hardcore_boson_excitation_gate(
                indices=idx, angle=(idx, "D", label), assume_real=assume_real
            )

        return UD

    def make_spa_ansatz(
        self,
        edges,
        hcb=False,
        use_units_of_pi=False,
        label=None,
        optimize=None,
        ladder=True,
        check_edges=True,
    ):
        """Return the Separable Pair Ansatz (SPA) circuit.

        Parameters
        ----------
        edges : list of tuple of int
            Orbitals of each pair, e.g. ``[(0,), (1, 2, 3), (4, 5)]`` is one pair with a
            single orbital, one with three and one with two.
        hcb : bool, optional
            Stay in the hard-core boson space without transforming to the encoding of the
            molecule, e.g. to use with ``make_hardcore_boson_hamiltonian``. Default is False.
        use_units_of_pi : bool, optional
            Give the circuit angles in units of pi. Default is False.
        label : hashable, optional
            Label of the variables.
        optimize : bool, optional
            Build a shallow circuit from Ry and CNOT gates.
        ladder : bool, optional
            If True the excitations are local, e.g. 1->2 and 2->3 in the pair
            ``(1, 2, 3)``. If False they are coupled-cluster-like, i.e. 1->2 and 1->3.
            Default is True.
        check_edges : bool, optional
            Check that the edges are valid (number of edges, no shared orbitals). Default is True;
            only switch it off if you are sure about the edges.

        Returns
        -------
        tq.QCircuit

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        References
        ----------
        J. S. Kottmann and A. Aspuru-Guzik, Phys. Rev. A 105, 032449 (2022), arXiv:2105.03836.

        Examples
        --------
        >>> U = mol.make_spa_ansatz(edges=[(0, 1)])
        """

        def _make_spa_tapered(
            edges,
            hcb=False,
            use_units_of_pi=False,
            label=None,
            optimize=False,
            ladder=True,
        ):
            U = QCircuit()
            cedges = []
            for edge in edges:
                if self.n_orbitals - 1 in edge:
                    cedge = [i for i in edge if i != self.n_orbitals - 1] + [
                        self.n_orbitals - 1
                    ]
                    cedges.append(cedge)
                else:
                    cedges.append(edge)
            if optimize:
                for edge_orbitals in cedges:
                    edge_qubits = [self.transformation.up(i) for i in edge_orbitals]
                    if edge_qubits[0] != self.transformation.up(self.n_orbitals - 1):
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
                        if (i - 1 == 0) and q1 != self.transformation.up(
                            self.n_orbitals - 1
                        ):
                            U += gates.Ry(angle=angle, target=q1, control=None)
                        elif (i - 1 == 0) and (
                            q1 == self.transformation.up(self.n_orbitals - 1)
                        ):
                            # rotate the explicit orbital instead of L, with the angle convention of the other encodings
                            U += gates.Ry(angle=-angle, target=c, control=None)
                        else:
                            U += gates.Ry(angle=angle, target=q1, control=c)
                        if q1 != self.transformation.up(self.n_orbitals - 1):
                            U += gates.CNOT(q1, c)
                if not hcb:
                    U += self.hcb_to_me()
                return U
            else:
                return self.make_spa_ansatz(
                    cedges,
                    hcb=hcb,
                    use_units_of_pi=use_units_of_pi,
                    label=label,
                    optimize=True,
                    ladder=ladder,
                )

        if edges is None:
            raise TequilaException(
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
            if have_hcb_to_me:
                optimize = True
            else:
                optimize = False

        U = QCircuit()
        if "TAPEREDBINARY" in self.transformation.name.upper():
            return _make_spa_tapered(
                edges=edges,
                hcb=hcb,
                use_units_of_pi=use_units_of_pi,
                label=label,
                optimize=optimize,
                ladder=ladder,
            )
        # construction of the optimized circuit
        if optimize:
            # circuit in HCB representation
            # depends a bit on the ordering of the spin-orbitals in the encoding
            # here we transform it to the qubits representing the up-spins
            # the hcb_to_me sequence will then transfer to the actual encoding later
            for edge_orbitals in edges:
                edge_qubits = [self.transformation.up(i) for i in edge_orbitals]
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
                U += self.hcb_to_me()
        else:
            orbs = [edge[0] for edge in edges]
            if hcb:
                U = gates.X([self.transformation.up(i) for i in orbs])
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
        return U

    def make_ansatz(self, name: str, *args, **kwargs):
        """Return the circuit of a named ansatz, calling the matching ``make_*_ansatz`` method.

        Parameters
        ----------
        name : str
            Name of the ansatz, e.g. "UpCCGSD", "UpCCD", "SPA", "UCCSD". Ansatzes can be
            combined with "+", e.g. "SPA+UpCCD". Further arguments are passed on.
        **kwargs
            Passed to the corresponding ``make_*_ansatz`` method, e.g. ``edges`` for SPA. ``label``
            labels the variables and ``hcb`` (bool) builds the circuit in the hard-core boson space
            (also set by "HCB" in the name).

        Returns
        -------
        tq.QCircuit

        See Also
        --------
        make_spa_ansatz : Separable Pair Ansatz (SPA).
        make_upccgsd_ansatz : (k-)UpCCGSD-type ansatz.
        make_uccsd_ansatz : Closed-shell UCCSD ansatz.

        Examples
        --------
        >>> U = mol.make_ansatz("SPA", edges=edges)
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
        *args,
        **kwargs,
    ):
        """Return a (k-)UpCCGSD-type ansatz circuit, similar to Lee et al.

        The name selects the excitations: "S" adds singles, "D" adds (pair) doubles, "G"
        makes them generalized (all orbital pairs, not only occupied to virtual), "HCB"
        works in the hard-core boson space. A leading number sets the order, e.g. "2-UpCCGD".

        Parameters
        ----------
        include_reference : bool, optional
            Start from the reference state. Default is True.
        name : str, optional
            Name of the ansatz, e.g. "UpCCD", "UpCCSD", "UpCCGSD", "HCB-UpCCGD". Default is "UpCCGSD".
        indices : list, optional
            Excitation indices of the layers (e.g. ``[((0, 3),), ((1, 4),)]``), replacing the ones
            given by ``name``. Default is ``make_upccgsd_indices(name)``.
        label : hashable, optional
            Label of the variables, which are named ``(label, (x, (p, q)))`` with layer ``x``
            instead of ``(x, (p, q))``.
        order : int, optional
            Number of repeated layers with independent parameters. Default is taken from
            the name, else 1.
        assume_real : bool, optional
            Assume a real wavefunction, reduces gradient costs from 4 to 2. Default is True.
        hcb_optimization : bool, optional
            Build the doubles in the hard-core boson space and map back. Default is True
            if the encoding supports it and the reference is included.
        spin_adapt_singles : bool, optional
            Spin-adapt the singles, i.e. share the angle between alpha and beta. Default is True.
        neglect_z : bool, optional
            Neglect the Z strings in the singles (needs Jordan-Wigner). Set automatically if the name
            contains "A" (e.g. "UpCCGASD"). Default is False.
        mix_sd : bool, optional
            Order the excitations as doubles and singles on the same orbitals pairwise
            (DS-DS-...) instead of all doubles then all singles. Useful for strongly correlated
            systems with a large Trotterization error of the UCC ansatz. Not usable with the
            hard-core boson optimization.

        Returns
        -------
        tq.QCircuit

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        References
        ----------
        J. Lee, W. J. Huggins, M. Head-Gordon and K. B. Whaley, J. Chem. Theory Comput. 15, 311 (2019), arXiv:1810.02327.

        Examples
        --------
        >>> U = mol.make_upccgsd_ansatz(name="UpCCD")
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
        have_hcb_trafo = True
        try:
            if self.transformation.hcb_to_me() is None:
                have_hcb_trafo = False
        except Exception:  # noqa: BLE001
            have_hcb_trafo = False

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
            raise TequilaException(
                f"name={name}, Singles can't be realized without mapping back to the standard encoding leave S or HCB out of the name"
            )
        if hcb_optimization and mix_sd:
            raise TequilaException(
                "Mixed SD can not be employed together with HCB Optimization"
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
                mix_sd=mix_sd,
                spin_adapt_singles=spin_adapt_singles,
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

            if "HCB" not in name and (include_reference or D):
                U = self.hcb_to_me(U=U)

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
            )

        return U

    def make_upccgsd_layer(
        self,
        indices,
        include_singles: bool = True,
        include_doubles: bool = True,
        assume_real: bool = True,
        label=None,
        spin_adapt_singles: bool = True,
        angle_transform=None,
        mix_sd: bool = False,
        neglect_z: bool = False,
        *args,
        **kwargs,
    ):
        """Return one layer of singles and/or doubles for the given orbital-pair indices.

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

        Returns
        -------
        tq.QCircuit
        """
        U = QCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = ((idx,), "D", label)
            if include_doubles:
                if (
                    "jordanwigner" in self.transformation.name.lower()
                    and not self.transformation.up_then_down
                ):
                    # we can optimize with qubit excitations for the JW representation
                    target = [
                        self.transformation.up(idx[0]),
                        self.transformation.up(idx[1]),
                        self.transformation.down(idx[0]),
                        self.transformation.down(idx[1]),
                    ]
                    U += gates.QubitExcitation(
                        angle=angle, target=target, assume_real=assume_real, **kwargs
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
            if include_singles and mix_sd:
                U += self.make_upccgsd_singles(
                    indices=[(idx,)],
                    assume_real=assume_real,
                    label=label,
                    spin_adapt_singles=spin_adapt_singles,
                    angle_transform=angle_transform,
                    neglect_z=neglect_z,
                )

        if include_singles and not mix_sd:
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
    ):
        """Return the singles of a UpCCGSD-type layer for the given orbital-pair indices.

        With ``neglect_z`` the Z strings are neglected, which needs Jordan-Wigner.

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
            if spin_adapt_singles:
                angle = (idx, "S", label)
                if angle_transform is not None:
                    angle = angle_transform(angle)
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
                        angle=angle, target=targeta, assume_real=assume_real, **kwargs
                    )
                    U += gates.QubitExcitation(
                        angle=angle, target=targetb, assume_real=assume_real, **kwargs
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
                        angle=angle1, target=targeta, assume_real=assume_real, **kwargs
                    )
                    U += gates.QubitExcitation(
                        angle=angle2, target=targetb, assume_real=assume_real, **kwargs
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
    ) -> QCircuit:
        """Return a closed-shell UCCSD ansatz circuit, optionally initialized with amplitudes.

        Parameters
        ----------
        trotter_steps : int, optional
            Number of Trotter steps. Default is 1.
        initial_amplitudes : str or Amplitudes or ClosedShellAmplitudes, optional
            Initial amplitudes, either given explicitly or computed with "mp2", "ccsd"
            or another method of ``compute_amplitudes`` (e.g. "cc2").
        include_reference_ansatz : bool, optional
            Prepare the Hartree-Fock reference first. Default is True.
        parametrized : bool, optional
            Use variables, otherwise fixed numbers. Default is True.
        threshold : float, optional
            Amplitudes below this are neglected if ``screening``. Default is 1e-8.
        add_singles : bool, optional
            Add singles. Default is True, also for "mp2" amplitudes, which have no singles.
        screening : bool, optional
            Neglect small amplitudes. Default is True.

        Returns
        -------
        tq.QCircuit
            Parametrized circuit.

        Raises
        ------
        TequilaException
            If the molecule is not closed shell.

        See Also
        --------
        make_ansatz : Build any named ansatz, e.g. "SPA" or "UpCCGSD".

        Examples
        --------
        >>> amplitudes = mol.compute_ccsd_amplitudes()
        >>> U = mol.make_uccsd_ansatz(trotter_steps=1, initial_amplitudes=amplitudes, include_reference_ansatz=True)
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

        Uref = QCircuit()
        if include_reference_ansatz:
            Uref = self.prepare_reference()

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
                except Exception as exc:  # noqa: BLE001
                    raise TequilaException(
                        f"{exc}\nDon't know how to initialize '{initial_amplitudes}' amplitudes"
                    )
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
                    raise TequilaException(
                        "only closed-shell supported, please assemble yourself .... sorry :-)"
                    )

        UCCSD = QCircuit()
        factor = 1.0 / trotter_steps
        for step in range(trotter_steps):
            for idx, angle in indices.items():
                # the amplitudes list the virtual orbital first, (a, i): the excitation destroys i and creates a
                converted = [(idx[2 * i + 1], idx[2 * i]) for i in range(len(idx) // 2)]
                UCCSD += self.make_excitation_gate(
                    indices=converted, angle=factor * angle
                )
        if (
            hasattr(initial_amplitudes, "lower")
            and initial_amplitudes.lower() == "mp2"
            and parametrized
            and add_singles
        ):
            # mp2 has no singles, need to initialize them here (if not parametrized initializling as 0.0 makes no sense though)
            UCCSD += self.make_upccgsd_layer(
                indices="upccsd", include_singles=True, include_doubles=False
            )
        return Uref + UCCSD

    def compute_amplitudes(self, method: str, *args, **kwargs):
        """Return closed-shell coupled-cluster amplitudes (implemented in the backends).

        Parameters
        ----------
        method : str
            Method such as "cc2", "ccsd", "cc3" or "ccsd(t)". Which ones work depends on
            the backend. MP2 has its own function ``compute_mp2_amplitudes``.

        Examples
        --------
        >>> amplitudes = mol.compute_amplitudes("mp2")
        """
        raise TequilaException("compute amplitudes: Needs to be overwritten by backend")

    def compute_energy(self, method, *args, **kwargs):
        """Compute the energy of a classical or quantum method.

        Classical methods run with PySCF, quantum methods build an ansatz and minimize
        the energy (see ``make_upccgsd_ansatz``).

        Parameters
        ----------
        method : str
            Classical: "HF", "MP2", "CCSD", "CCSD(T)", "FCI". Quantum: "UpCCD", "UpCCSD",
            "UpCCGSD", "k-UpCCGSD", "UCCSD".
        **kwargs
            For quantum methods, keyword arguments of the minimizer.

        Returns
        -------
        float

        Examples
        --------
        >>> e_fci = mol.compute_energy("fci")
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
            E = ExpectationValue(H=H, U=U)
            from tequila import minimize

            return minimize(*args, objective=E, **kwargs).energy
        else:
            from . import INSTALLED_QCHEMISTRY_BACKENDS

            if "pyscf" not in INSTALLED_QCHEMISTRY_BACKENDS:
                raise TequilaException(
                    f"PySCF needs to be installed to compute {method}/{self.parameters.basis_set}"
                )
            else:
                from .pyscf_interface import QuantumChemistryPySCF

                molx = QuantumChemistryPySCF.from_tequila(self)
                return molx.compute_energy(method=method, **kwargs)

    def compute_fci(self, *args, **kwargs):
        """Raise NotImplementedError, FCI is only implemented in the PySCF backend."""
        raise NotImplementedError(
            "compute_fci only implemented for the 'pyscf' backend"
        )

    def compute_fock_matrix(self):
        """Return the closed-shell Fock matrix in the current orbital basis."""
        _c, h, g = self.get_integrals()
        g = g.reorder(to="phys")

        # fock matrix is:
        # Fkl = hkl + 2.0 <k|J|l> - <k|K|l> = hkl + 2.0* <ki|g|li> - <ki|g|il>
        F = numpy.zeros(shape=h.shape)
        for k in range(F.shape[0]):
            for l in range(F.shape[1]):
                tmp = h[k, l]
                for ii in self.reference_orbitals:
                    i = ii.idx
                    tmp += 2.0 * g.elems[k, i, l, i] - g.elems[k, i, i, l]
                F[k, l] = tmp
        return F

    def compute_mp2_amplitudes(self, return_energy=False) -> ClosedShellAmplitudes:
        r"""Return closed-shell MP2 amplitudes (canonical orbitals only).

        :math:`t(a,i,b,j) = 0.25\, g(a,i,b,j) / (e(i) + e(j) - a(i) - b(j))`

        Parameters
        ----------
        return_energy : bool, optional
            Also return the MP2 correlation energy. Default is False.

        Returns
        -------
        ClosedShellAmplitudes or tuple
            The amplitudes, and the energy if ``return_energy``.

        Examples
        --------
        >>> amplitudes, energy = mol.compute_mp2_amplitudes(return_energy=True)
        """
        _c, _h, g = self.get_integrals()
        fi = self.compute_fock_matrix()
        self.is_canonical(verify=True, fock_matrix=fi)
        fi = numpy.diag(fi)
        self.is_closed_shell(verify=True)
        nocc = len(self.reference_orbitals)
        ei = fi[:nocc]
        ai = fi[nocc:]
        abgij = g.elems[nocc:, nocc:, :nocc, :nocc]
        amplitudes = (
            abgij
            * 1.0
            / (
                ei.reshape(1, 1, -1, 1)
                + ei.reshape(1, 1, 1, -1)
                - ai.reshape(-1, 1, 1, 1)
                - ai.reshape(1, -1, 1, 1)
            )
        )

        result = ClosedShellAmplitudes(
            tIjAb=numpy.einsum("abij -> ijab", amplitudes, optimize="greedy")
        )

        if return_energy:
            E = 2.0 * numpy.einsum("abij,abij->", amplitudes, abgij) - numpy.einsum(
                "abji,abij", amplitudes, abgij, optimize="greedy"
            )
            return result, E
        else:
            return result

    def compute_cis_amplitudes(self):
        """Return the CIS excitation energies and amplitudes (not field tested).

        Returns
        -------
        ResultCIS
            Indexing gives the pair (excitation energy, ClosedShellAmplitudes).

        Examples
        --------
        >>> amplitudes = mol.compute_cis_amplitudes()
        """

        @dataclass
        class ResultCIS:
            omegas: list[numbers.Real]  # excitation energies [omega0, ...]
            amplitudes: list[
                ClosedShellAmplitudes
            ]  # corresponding amplitudes [x_{ai}_0, ...]

            def __getitem__(self, item):
                return (self.omegas[item], self.amplitudes[item])

            def __len__(self):
                return len(self.omegas)

        self.is_closed_shell(verify=True)
        _c, _h, g = self.get_integrals()
        g.reorder(to="openfermion")
        g = g.elems
        fij = self.compute_fock_matrix()
        self.is_canonical(verify=True, fock_matrix=fij)
        fij = numpy.diag(fij)

        nocc = self.n_electrons // 2
        nvirt = self.n_orbitals - nocc

        pairs = []
        for i in range(nocc):
            for a in range(nocc, nocc + nvirt):
                pairs.append((a, i))
        M = numpy.ndarray(shape=[len(pairs), len(pairs)])

        for xx, x in enumerate(pairs):
            eia = fij[x[0]] - fij[x[1]]
            a, i = x
            for yy, y in enumerate(pairs):
                b, j = y
                delta = float(y == x)
                gpart = 2.0 * g[a, i, b, j] - g[a, i, j, b]
                M[xx, yy] = eia * delta + gpart

        omega, xvecs = numpy.linalg.eigh(M)

        # convert amplitudes to ndarray sorted by excitation energy
        nex = len(omega)
        amplitudes = []
        for ex in range(nex):
            t = numpy.ndarray(shape=[nvirt, nocc])
            exvec = xvecs[ex]
            for xx, x in enumerate(pairs):
                a, i = x
                t[a - nocc, i] = exvec[xx]
            amplitudes.append(ClosedShellAmplitudes(tIA=t))

        return ResultCIS(omegas=list(omega), amplitudes=amplitudes)

    def is_closed_shell(self, verify=False):
        """Return True if the molecule is closed shell (raise TequilaException if ``verify`` and it is not)."""
        cs = self.n_electrons % 2 == 0
        if verify and not cs:
            raise TequilaException(
                f"not a closed shell molecule: having {self.n_electrons} electrons"
            )
        return cs

    def is_canonical(self, verify=False, fock_matrix=None):
        """Return True if the orbitals are canonical, i.e. the Fock matrix is diagonal and the reference orbitals come first.

        Raises TequilaException if ``verify`` and they are not.

        Parameters
        ----------
        verify : bool, optional
            Raise an exception instead of returning False. Default is False.
        fock_matrix : numpy.ndarray, optional
            Fock matrix to check. Default is ``compute_fock_matrix()``.
        """
        canonical = True
        if fock_matrix is None:
            fock_matrix = self.compute_fock_matrix()

        is_diagonal = numpy.isclose(
            numpy.linalg.norm(fock_matrix - numpy.diag(numpy.diag(fock_matrix))),
            0.0,
            atol=1.0e-4,
        )

        if not is_diagonal:
            canonical = False

        refo = self.reference_orbitals

        if refo[0].idx != 0:
            canonical = False
        for i in range(len(refo) - 1):
            if refo[i].idx_total + 1 != refo[i + 1].idx_total:
                canonical = False

        if verify and not canonical:
            data = {"reference_orbitals": refo, "fock_matrix": fock_matrix}
            raise TequilaException(
                f"orbitals are not canonical or can not be verified as such -> implemented method only works for standard orbitals (preferably from psi4)\n{data}"
            )
        return canonical

    @property
    def rdm1(self):
        """numpy.ndarray or None : one-particle RDM from the last ``compute_rdms`` call, None (with
        a message) if not computed yet.
        """
        if self._rdm1 is not None:
            return self._rdm1
        else:
            print("1-RDM has not been computed. Return None for 1-RDM.")
            return None

    @property
    def rdm2(self):
        """numpy.ndarray or None : two-particle RDM from the last ``compute_rdms`` call (Dirac, i.e.
        physics ordering by default), None (with a message) if not computed yet.
        """
        if self._rdm2 is not None:
            return self._rdm2
        else:
            print("2-RDM has not been computed. Return None for 2-RDM.")
            return None

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

        The representation depends on the qubit encoding, Jordan-Wigner gives the usual
        second-quantized RDMs in the occupation picture. Only real orbitals and real RDMs
        are supported. The results are stored and can be accessed with ``rdm1`` and ``rdm2``.

        .. math::
            \gamma^p_q = \langle \psi | a^p a_q | \psi \rangle, \quad
            \gamma^{pq}_{rs} = \langle \psi | a^p a^q a_s a_r | \psi \rangle, \quad
            \psi = U|0\rangle

        By default the 2-RDM is returned in the physics (Dirac) ordering,
        ``rdm2[p, q, r, s]`` :math:`= \gamma^{pq}_{rs}`, see ``ordering``.

        Parameters
        ----------
        U : tq.QCircuit
            Circuit preparing the state.
        variables : dict or tq.Variables, optional
            Values of the variables if ``U`` is parametrized.
        spin_free : bool, optional
            Sum over spin (RDMs in spatial orbitals) instead of spin orbitals. Default is True.
        get_rdm1, get_rdm2 : bool, optional
            Which RDMs to compute and return. Computing both at once is recommended if both
            are needed. Default is True for both.
        ordering : str {"dirac", "phys", "chem", "openfermion"}, optional
            Index ordering of the 2-RDM. Default is "dirac", the physics ordering
            ``rdm2[p, q, r, s] = <psi|a^p a^q a_s a_r|psi>``; "chem" (Mulliken) and "openfermion" are
            also possible.
        use_hcb : bool, optional
            Compute the spin-free RDMs from a circuit in the hard-core boson encoding. Default is False.
        rdm_trafo : QubitHamiltonian, optional
            Transformation ``T`` applied to the operators, ``a^dagger_i a_j -> T^dagger a^dagger_i a_j T``.
            Default is the identity. Needs ``evaluate=True``.
        evaluate : bool, optional
            Evaluate the expectation values directly with ``tq.simulate``, avoiding repeated
            wavefunction simulations. If False, the RDMs are returned as tq.QTensor. Default is True.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            ``(rdm1, rdm2)`` if both are requested, else the requested one.

        Raises
        ------
        TequilaException
            If ``U`` is None or the encoding is BravyiKitaevFast.

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
        # Set up number of spin-orbitals and molecular orbitals respectively
        n_SOs = 2 * self.n_orbitals
        n_MOs = self.n_orbitals

        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException("Need to specify a Quantum Circuit.")

        def _get_hcb_op(op_tuple):
            r"""Build the hard-core boson operators :math:`b^\dagger_i b_j + \mathrm{h.c.}` in the qubit encoding."""
            if len(op_tuple) == 2:
                return 2 * Sm(op_tuple[0][0]) * Sp(op_tuple[1][0])
            elif len(op_tuple) == 4:
                if (op_tuple[0][0] == op_tuple[1][0]) and (
                    op_tuple[2][0] == op_tuple[3][0]
                ):  # iijj uddu+duud
                    return Sm(op_tuple[0][0]) * Sp(op_tuple[2][0]) + Sm(
                        op_tuple[2][0]
                    ) * Sp(op_tuple[0][0])
                if (
                    (op_tuple[0][0] == op_tuple[2][0])
                    and (op_tuple[1][0] == op_tuple[3][0])
                    and (op_tuple[0][0] != op_tuple[1][0])
                    and (op_tuple[2][0] != op_tuple[3][0])
                ):  # ijij uuuu+dddd
                    return (
                        4
                        * Sm(op_tuple[0][0])
                        * Sm(op_tuple[1][0])
                        * Sp(op_tuple[2][0])
                        * Sp(op_tuple[3][0])
                    )
                if (
                    (op_tuple[0][0] == op_tuple[3][0])
                    and (op_tuple[1][0] == op_tuple[2][0])
                    and (op_tuple[0][0] != op_tuple[1][0])
                    and (op_tuple[2][0] != op_tuple[3][0])
                ):  # ijji abba
                    return (
                        -2
                        * Sm(op_tuple[0][0])
                        * Sm(op_tuple[1][0])
                        * Sp(op_tuple[2][0])
                        * Sp(op_tuple[3][0])
                    )
            else:
                return Zero()

        def _get_of_op(operator_tuple):
            """Return operator given by a operator tuple as OpenFermion - Fermion operator."""
            op = openfermion.FermionOperator(operator_tuple)
            return op

        def _get_qop_hermitian(of_operator) -> QubitHamiltonian:
            """Return the Hermitian part of a FermionOperator as QubitHamiltonian."""
            qop = self.transformation(of_operator)
            # qop = QubitHamiltonian(self.transformation(of_operator))
            real, _imag = qop.split(hermitian=True)
            if real:
                return real
            elif not real:
                raise TequilaException(
                    f"Qubit Hamiltonian does not have a Hermitian part. Operator ={of_operator}"
                )

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

        def _build_1bdy_operators_hcb() -> list:
            """Return hcb one-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetry pq = qp (not changed by spin-summation)
            ops = []
            for p in range(n_MOs):
                for q in range(p + 1):
                    if p == q:
                        if self.transformation.up_then_down:
                            op_tuple = ((p, 1), (p, 0))
                            op = _get_hcb_op(op_tuple)
                        else:
                            op_tuple = ((2 * p, 1), (2 * p, 0))
                            op = _get_hcb_op(op_tuple)
                        ops += [op]
                    else:
                        ops += [Zero()]
            return ops

        def _build_2bdy_operators_hcb() -> list:
            """Return hcb two-body operators as a symmetry-reduced list of QubitHamiltonians."""
            # Exploit symmetries pqrs = qpsr (due to spin summation, '-pqsr = -qprs' drops out)
            #                and      = rspq
            ops = []
            scale = 2
            if self.transformation.up_then_down:
                scale = 1
            for p, q, r, s in product(range(n_MOs), repeat=4):
                if p * n_MOs + q >= r * n_MOs + s and (p >= q or r >= s):
                    # Spin abba+ baab allow p=q=r=s orb iijj
                    op_tuple = (
                        ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0))
                        if (p == q and s == r)
                        else "0.0 []"
                    )
                    op = _get_hcb_op(op_tuple)
                    # Spin abba+ baab dont allow p=q=r=s orb ijij
                    op_tuple = (
                        ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0))
                        if (p != q and r != s and p == r and s == q)
                        else "0.0 []"
                    )
                    op += _get_hcb_op(op_tuple)
                    # Spin aaaa+ bbbb dont allow p=q=r=s  orb ijji
                    op_tuple = (
                        ((scale * p, 1), (scale * q, 1), (scale * r, 0), (scale * s, 0))
                        if (p != q and r != s and p == s and q == r)
                        else "0.0 []"
                    )
                    op += _get_hcb_op(op_tuple)
                    ops += [op]
            return ops

        # Build operator lists
        qops = []
        if spin_free and not use_hcb:
            qops += _build_1bdy_operators_spinfree() if get_rdm1 else []
            qops += _build_2bdy_operators_spinfree() if get_rdm2 else []
        elif use_hcb:
            qops += _build_1bdy_operators_hcb() if get_rdm1 else []
            qops += _build_2bdy_operators_hcb() if get_rdm2 else []
        else:
            if use_hcb:
                raise TequilaException(
                    f"compute_rdms: spin_free={spin_free} and use_hcb={use_hcb} are not compatible"
                )
            qops += _build_1bdy_operators_spinful() if get_rdm1 else []
            qops += _build_2bdy_operators_spinful() if get_rdm2 else []

        # Transform operator lists to QubitHamiltonians
        if not use_hcb:
            qops = [_get_qop_hermitian(op) for op in qops]

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
            if rdm is not None:
                if (spin_free or use_hcb) and rdm.shape[0] != n_MOs:
                    return None
                if not spin_free and rdm.shape[0] != n_SOs:
                    return None
            return rdm

        self._rdm1 = _reset_rdm(self._rdm1)
        self._rdm2 = _reset_rdm(self._rdm2)
        # Split expectation values in 1- and 2-particle expectation values
        if get_rdm1:
            len_1 = (
                n_MOs * (n_MOs + 1) // 2
                if (spin_free or use_hcb)
                else n_SOs * (n_SOs + 1) // 2
            )
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        # Build matrices using the expectation values
        self._rdm1 = _assemble_rdm1(evals_1, rdm1=rdm1) if get_rdm1 else self._rdm1
        if spin_free or use_hcb:
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

    def rdm_spinsum(self, sum_rdm1: bool = True, sum_rdm2: bool = True) -> tuple:
        """Return spin-free RDMs obtained by spin summation of the spin-full RDMs.

        Parameters
        ----------
        sum_rdm1, sum_rdm2 : bool, optional
            Perform the spin summation on the 1-RDM / 2-RDM. Default is True for both.

        Returns
        -------
        tuple
            ``(rdm1_spinsum, rdm2_spinsum)``, None for those not summed.
        """
        n_MOs = self.n_orbitals
        rdm1_spinsum = None
        rdm2_spinsum = None

        # Spin summation on rdm1
        if sum_rdm1:
            # Check whether spin-rdm2 exists
            if self._rdm1 is None:
                raise TequilaException("The spin-RDM for the 1-RDM does not exist!")
            # Check whether existing rdm1 is in spin-orbital basis
            if self._rdm1.shape[0] != 2 * n_MOs:
                raise TequilaException(
                    "The existing RDM needs to be in spin-orbital basis, it is already spin-free!"
                )
            # Do summation
            rdm1_spinsum = numpy.zeros([n_MOs, n_MOs])
            for p in range(n_MOs):
                for q in range(p + 1):
                    rdm1_spinsum[p, q] += self._rdm1[2 * p, 2 * q]
                    rdm1_spinsum[p, q] += self._rdm1[2 * p + 1, 2 * q + 1]
            for p in range(n_MOs):
                for q in range(p):
                    rdm1_spinsum[q, p] = rdm1_spinsum[p, q]

        # Spin summation on rdm2
        if sum_rdm2:
            # Check whether spin-rdm2 exists
            if self._rdm2 is None:
                raise TequilaException("The spin-RDM for the 2-RDM does not exist!")
            # Check whether existing rdm2 is in spin-orbital basis
            if self._rdm2.shape[0] != 2 * n_MOs:
                raise TequilaException(
                    "The existing RDM needs to be in spin-orbital basis, it is already spin-free!"
                )
            # Do summation
            rdm2_spinsum = numpy.zeros([n_MOs, n_MOs, n_MOs, n_MOs])
            for p, q, r, s in product(range(n_MOs), repeat=4):
                rdm2_spinsum[p, q, r, s] += self._rdm2[2 * p, 2 * q, 2 * r, 2 * s]
                rdm2_spinsum[p, q, r, s] += self._rdm2[
                    2 * p + 1, 2 * q, 2 * r + 1, 2 * s
                ]
                rdm2_spinsum[p, q, r, s] += self._rdm2[
                    2 * p, 2 * q + 1, 2 * r, 2 * s + 1
                ]
                rdm2_spinsum[p, q, r, s] += self._rdm2[
                    2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                ]

        return rdm1_spinsum, rdm2_spinsum

    def perturbative_f12_correction(
        self,
        rdm1: numpy.ndarray | None = None,
        rdm2: numpy.ndarray | None = None,
        gamma: float = 1.4,
        n_ri: int | None = None,
        external_info: dict | None = None,
        **kwargs,
    ) -> float:
        """Return the spin-free [2]_R12 energy correction from the 1- and 2-RDM of a reference method.

        Either the RDMs or the information to compute them (``U`` and ``variables`` as
        keyword arguments, see ``compute_rdms``) must be given.

        Parameters
        ----------
        rdm1, rdm2 : numpy.ndarray, optional
            1- and 2-electron reduced density matrices (spin-free), the 2-RDM in the physics
            (Dirac) ordering as returned by ``compute_rdms``.
        gamma : float, optional
            Exponent of the correlation factor ``f_12 = -1/gamma * exp(-gamma*r_12)``. Default is 1.4.
        n_ri : int, optional
            Dimension of the RI basis, to truncate the available one. Must be between the size
            of the orbital basis (where the correction is zero) and the available RI basis.
            Default is the maximum available.
        external_info : dict, optional
            Where to find the f12 tensor ``<rs|f_12|pq>``:
            ``{"f12_filename": file name, "scheme": ordering scheme of the tensor}``.
        **kwargs
            E.g. ``U`` and ``variables`` if the RDMs are not computed yet.

        Returns
        -------
        float
            The correction to the energy.

        Examples
        --------
        >>> dE = mol.perturbative_f12_correction(cabs_type="active", rdm__psi4_method="detci")
        """
        from .f12_corrections._f12_correction_base import ExplicitCorrelationCorrection

        correction = ExplicitCorrelationCorrection(
            mol=self,
            rdm1=rdm1,
            rdm2=rdm2,
            gamma=gamma,
            n_ri=n_ri,
            external_info=external_info,
            **kwargs,
        )
        return correction.compute()

    def n_rotation(self, i, phi):
        """Return a phase rotation circuit on both spin orbitals of spatial orbital ``i``.

        Parameters
        ----------
        i : int
            Spatial orbital index.
        phi : float
            Rotation angle, the angle applied to the up and down number operators is ``-2 * phi``.

        Returns
        -------
        tq.QCircuit

        Examples
        --------
        >>> U = mol.n_rotation(0, numpy.pi)
        """

        # Generate number operators for the up and down components of the qubit.
        n_up = self.make_number_op(2 * i)
        n_down = self.make_number_op(2 * i + 1)

        # Start a new circuit and apply rotations to each component.
        circuit = gates.GeneralizedRotation(generator=n_up, angle=-2 * phi)
        circuit += gates.GeneralizedRotation(generator=n_down, angle=-2 * phi)
        return circuit

    def get_givens_circuit(
        self,
        unitary,
        tol=1e-12,
        ordering=OPTIMIZED_ORDERING,
        fix: bool = True,
        label=None,
    ) -> QCircuit:
        """Return the circuit implementing a real orthogonal matrix with Givens and Rz rotations.

        The matrix is decomposed with ``get_givens_decomposition``.

        Parameters
        ----------
        unitary : numpy.ndarray
            Real unitary matrix.
        tol : float, optional
            Matrix elements below this are treated as zero. Default is 1e-12.
        ordering : list of tuple or str {"Optimized"}, optional
            Order of the index pairs of the Givens rotations, or "Optimized" to generate it.
        fix : bool, optional
            If True the angles are fixed numbers, otherwise ``tq.Variable(idx) + value`` so the
            basis change can be relaxed further. Default is True.
        label : hashable, optional
            Label of the variables, named ``("R", i, j, label)``.

        Returns
        -------
        tq.QCircuit

        Examples
        --------
        >>> opt = sun.optimize_orbitals(molecule=mol, circuit=U, silent=True)
        >>> UR = mol.get_givens_circuit(opt.mo_coeff)  # rotation to the optimized orbitals
        """
        # Decompose the unitary matrix into Givens and phase (Rz) rotations.
        theta_list, phi_list = get_givens_decomposition(unitary, tol, ordering)

        # Initialize an empty quantum circuit.
        circuit = QCircuit()

        # Add all Rz (phase) rotations to the circuit.
        for phi in phi_list:
            if fix:
                circuit += self.n_rotation(i=phi[1], phi=phi[0])
            else:
                circuit += self.n_rotation(
                    i=phi[1],
                    phi=phi[0]
                    + Variable(
                        f"Ph({phi[1]}" + ("," + str(label)) * (label is not None) + ")"
                    ),
                )

        # Add all Givens rotations to the circuit.
        # (UR(i, j) moves an electron from i to j for a positive angle)
        for theta in reversed(theta_list):
            if fix:
                circuit += self.UR(i=theta[1], j=theta[2], angle=theta[0] * 2)
            else:
                circuit += self.UR(
                    i=theta[1],
                    j=theta[2],
                    angle=(theta[0] * 2)
                    + Variable(
                        f"UR({theta[1]},{theta[2]}"
                        + ("," + str(label)) * (label is not None)
                        + ")"
                    ),
                )

        return circuit

    def print_basis_info(self):
        """Print information on the basis and the orbitals of the molecule.

        Prints the basis name, the orbital type, whether the basis is orthogonal, the number of
        basis functions, the active orbitals and the reference (occupied) orbitals, all as total
        indices. Then, for every current orbital, its data (index, irrep, energy, ...) and its
        coefficients in the basis functions.

        Useful to look up orbital indices, e.g. for ``core`` in ``use_native_orbitals`` or for
        ``active_orbitals``.

        Examples
        --------
        >>> mol.print_basis_info()
        """
        return self.integral_manager.print_basis_info()

    def __str__(self) -> str:
        """Return a summary of the encoding, parameters and orbitals."""
        result = str(type(self)) + "\n"
        result += "Qubit Encoding\n"
        result += str(self.transformation) + "\n\n"
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


def givens_matrix(n, p, q, theta):
    """Return the ``n x n`` Givens rotation matrix by angle ``theta`` in the (p, q) plane.

    Parameters
    ----------
    n : int
        Dimension of the matrix.
    p, q : int
        Indices of the rotation plane.
    theta : float, tq.Variable or tq.Objective
        Rotation angle.

    Returns
    -------
    tq.QTensor
        The matrix, which can be complex.
    """
    matrix = QTensor(
        shape=(n, n), objective_list=numpy.eye(n).reshape(n * n)
    )  # Matrix to hold complex numbers
    if isinstance(theta, (Variable, Objective)):
        cos_theta = theta.apply(numpy.cos)
        sin_theta = theta.apply(numpy.sin)
    else:
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)

    # Directly assign cosine and sine without complex phase adjustment
    matrix[p, p] = cos_theta
    matrix[q, q] = cos_theta
    matrix[p, q] = sin_theta
    matrix[q, p] = -sin_theta

    return matrix


def get_givens_decomposition(
    unitary, tol=1e-12, ordering=OPTIMIZED_ORDERING, return_diagonal=False
):
    """Decompose a real unitary matrix into Givens rotations and Rz rotations (phases).

    Parameters
    ----------
    unitary : numpy.ndarray
        Real unitary matrix.
    tol : float, optional
        Matrix elements with smaller absolute value are treated as zero. Default is 1e-12.
    ordering : list of tuple of int or str or None, optional
        Custom ``(row, column)`` elimination order, or ``"Optimized"`` (default) for a linear-depth order.
    return_diagonal : bool, optional
        Also return the diagonal matrix left after all Givens rotations. Default is False.

    Returns
    -------
    theta_list : list of tuple
        Givens rotations as ``(theta, i, j)``.
    phi_list : list of tuple
        Rz rotations as ``(phi, i)``.
    diagonal : numpy.ndarray
        Diagonal matrix left after the eliminations (only if ``return_diagonal``).

    Examples
    --------
    >>> theta_list, phi_list = get_givens_decomposition(unitary)
    >>> R = reconstruct_matrix_from_givens(unitary.shape[0], theta_list, phi_list)  # equals unitary
    """
    U = unitary  # no need to copy as we don't modify the original
    # U[abs(U) < tol] = 0 # Zeroing out the small elements as per the tolerance level. #comented out, its being considered latter again
    n = U.shape[0]

    # Determine optimized ordering if specified.
    if ordering == OPTIMIZED_ORDERING:
        ordering = ff.depth_eff_order_mf(n)

    theta_list = []
    phi_list = []

    def calcTheta(U, c, r):
        """Calculate and apply the Givens rotation for a specific matrix element."""
        t = arctan2(-U[r, c], U[r - 1, c])
        theta_list.append((t, r, r - 1))
        g = givens_matrix(n, r, r - 1, t)  # is a QTensor
        U = g.dot(U)
        return U

    # Apply and store Givens rotations as per the given or computed ordering.
    if ordering is None:
        for c in range(n):
            for r in range(n - 1, c, -1):
                U = calcTheta(U, c, r)
    else:
        for r, c in ordering:
            U = calcTheta(U, c, r)
    # Calculating the Rz rotations based on the phases of the diagonal elements.
    # For real elements this means a 180 degree shift, i.e. a sign change.
    for i in range(n):
        if isinstance(U[i, i], (Variable, Objective)):
            if len(U[i, i].args):
                phi_list.append((U[i, i].apply(numpy.angle), i))
        else:
            phi_list.append((numpy.angle(U[i, i]), i))
    # Filtering out rotations without significance.
    theta_list_new = []
    for i, theta in enumerate(theta_list):
        if isinstance(theta[0], (Variable, Objective)):
            if len(theta[0].args):
                theta_list_new.append(theta)
        elif abs(theta[0] % (2 * numpy.pi)) > tol:
            theta_list_new.append(theta)
    phi_list_new = []
    for i, phi in enumerate(phi_list):
        if isinstance(phi[0], (Variable, Objective)):
            if len(phi[0].args):
                phi_list_new.append(phi)
        elif abs(phi[0]) > tol:
            phi_list_new.append(phi)
    if return_diagonal:
        # Optionally return the resulting diagonal
        return theta_list_new, phi_list_new, U
    else:
        return theta_list_new, phi_list_new


def reconstruct_matrix_from_givens(
    n, theta_list, phi_list, to_real_if_possible=True, tol=1e-12
):
    """Reconstruct a matrix from Givens rotations and Rz diagonal rotations.

    This is effectively the inverse of ``get_givens_decomposition`` and therefore only works with
    data in the same format as its output: the phases are applied to the identity first, then the
    Givens rotations in reverse order.

    Parameters
    ----------
    n : int
        Dimension of the matrix to reconstruct.
    theta_list : list of tuple
        Givens rotations ``(angle, i, j)``: a rotation by ``angle`` radians acting on rows/columns
        ``i`` and ``j``.
    phi_list : list of tuple
        Rz rotations ``(angle, i)``: the ``i``-th diagonal element is multiplied by
        ``exp(1j * angle)``.
    to_real_if_possible : bool, optional
        If True (default), return a real matrix when its imaginary part is zero.
    tol : float, optional
        Tolerance to detect an angle of pi, which is applied as an exact sign change instead of a
        complex phase. Default is 1e-12.

    Returns
    -------
    numpy.ndarray
        The reconstructed matrix, real or complex depending on ``to_real_if_possible`` and on
        the rotations.

    Examples
    --------
    >>> theta_list, phi_list = get_givens_decomposition(unitary)
    >>> R = reconstruct_matrix_from_givens(unitary.shape[0], theta_list, phi_list)  # equals unitary
    """
    # Start with an identity matrix
    reconstructed = numpy.eye(n, dtype=complex)

    # Apply Rz rotations for diagonal elements
    for phi in phi_list:
        angle, i = phi
        # Directly apply a sign flip if the rotation angle is π
        if numpy.isclose(angle, numpy.pi, atol=tol):
            reconstructed[i, i] *= -1
        else:
            reconstructed[i, i] *= numpy.exp(1j * angle)

    # Apply Givens rotations in reverse order
    for theta in reversed(theta_list):
        angle, i, j = theta
        g = givens_matrix(n, i, j, angle)
        reconstructed = numpy.dot(
            g.conj().T, reconstructed
        )  # Transpose of Givens matrix applied to the left

    # givens_matrix returns a QTensor, so with numeric angles the entries are numbers in an object array
    if all(isinstance(x, numbers.Number) for x in numpy.ravel(reconstructed)):
        reconstructed = numpy.asarray(reconstructed, dtype=complex)

    # Convert matrix to real if its imaginary part is negligible unless disabled via to_real_if_possible
    if to_real_if_possible and numpy.all(reconstructed.imag == 0):
        # Directly apply a sign flip if the rotation angle is π
        # Convert to real by taking the real part
        reconstructed = reconstructed.real

    return reconstructed


def arctan2(x1, x2, *args, **kwargs):
    """Return ``arctan2(x1, x2)``, also for tequila Variables and Objectives."""
    if isinstance(x1, (Variable, Objective)) or isinstance(x2, (Variable, Objective)):
        return Objective().binary_operator(left=1 * x1, right=1 * x2, op=numpy.arctan2)
    elif not isinstance(x1, numbers.Complex) and not isinstance(x2, numbers.Complex):
        return numpy.arctan2(x1, x2)
    else:
        return numpy.arctan2(x1.imag, x2.imag) + numpy.arctan2(x1.real, x2.real)
