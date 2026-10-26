from __future__ import annotations

import copy
import inspect
import json
import numbers
import typing
from dataclasses import asdict, dataclass
from itertools import product

import h5py
import numpy
import pyscf
import tequila
import tequila.grouping.fermionic_functions as ff
from scipy.optimize import linear_sum_assignment
from tequila import BitString, QTensor, TequilaException, TequilaWarning
from tequila.circuit import QCircuit, gates
from tequila.hamiltonian import QubitHamiltonian
from tequila.hamiltonian.paulis import I, Sm, Sp, Z, Zero
from tequila.objective.objective import ExpectationValue, Objective, Variable, Variables
from tequila.simulators.simulator_api import simulate
from tequila.utils import to_float

from sunrise.symmetry_adaptation.point_group import PointGroup, PointGroupRepresentation
from sunrise.utils.transformation_helpers import (
    FermionicSWAP,
    GeometricTransformationHelper,
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
    """
    Base Class for tequila chemistry functionality
    This is what is initialized with sun.Molecule(...)
    We try to define all main methods here and only implemented specializations in the derived classes
    Derived classes interface specific backends (e.g. Psi4, PySCF and Madness). See PACKAGE_interface.py for more
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
        """
        Parameters
        ----------
        parameters: the quantum chemistry parameters handed over as instance of the ParametersQC class (see there for content)
        transformation: the fermion to qubit transformation (default is JordanWigner). See encodings.py for supported encodings or to extend
        active_orbitals: list of active orbitals (others will be frozen, if we have N-electrons then the first N//2 orbitals will be considered occpied when creating the active space)
        frozen_orbitals: convenience (will be removed from list of active orbitals)
        reference_orbitals: give list of orbitals that shall be considered occupied when creating a possible active space (default is the first N//2). The indices are expected to be total indices (including possible frozen orbitals in the counting)
        orbitals: information about the orbitals (should be in OrbitalData format, can be a dictionary)
        args
        kwargs
        """

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
        """
        Save the molecule in an HDF5 file without pickle, so it can be opened from any sunrise installation
        with sun.load_molecule. Like the pyscf chkfile only plain data is stored: the parameters, the integrals
        in the basis, the orbitals, the active space, the transformation, the settings of the molecule nature
        and the data of the backend (e.g. the psi4 reference wavefunction, see _backend_data)

        Parameters
        ----------
        filename
            the HDF5 file
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
        """
        Data of the quantum chemistry backend that mol.save stores, so sun.load_molecule can rebuild the molecule
        in its backend instead of the base one. It is given back as keyword arguments on init
        Nested dictionaries, lists, tuples and numpy arrays are allowed (see to_json_data)

        Returns
        -------
            None if the backend has no data of its own, then the molecule is loaded with the base backend
            (unless the backend is rebuilt from the integrals alone, like pyscf)
        """
        return None

    def supports_ucc(self):
        """
        check if the current molecule supports UCC operations
        (e.g. mol.make_excitation_gate)
        """
        return self.transformation.supports_ucc

    def _initialize_transformation(self, transformation=None, *args, **kwargs):
        """
        Helper Function to initialize the Fermion-to-Qubit Transformation
        Parameters
        ----------
        transformation: name of the transformation (passed down from __init__
        args
        kwargs

        Returns
        -------

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
        """
        Initialize direclty from openfermion MolecularData object

        Parameters
        ----------
        molecule
            The openfermion molecule
        Returns
        -------
            The Tequila molecule
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
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\\dagger_{a_0} a_{i_0} a^\\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(i_0, a_0), (i_1, a_1), ... ], each pair destroys i and creates a - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [i_0, a_0, i_1, a_1 ...]
        form : str : (Default value None):
            Manipulate the generator to involution or projector
            set form='involution' or 'projector'
            the default is no manipulation which gives the standard fermionic excitation operator back
        remove_constant_term: bool: (Default value True):
            by default the constant term in the qubit operator is removed since it has no effect on the unitary it generates
            if the unitary is controlled this might not be true!
        Returns
        -------
        type
            1j*Transformed qubit excitation operator, depends on self.transformation
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
        """
        Make excitation generator in the hardcore-boson approximation (all electrons are forced to spin-pairs)
        use only in combination with make_hardcore_boson_hamiltonian()

        Parameters
        ----------
        indices
        angle
        control
        assume_real
        compile_options

        Returns
        -------

        """
        U = QCircuit()
        target = []
        for pair in indices:
            assert len(pair) == 2
            if (
                "TAPEREDBINARY" in self.transformation.name.upper()
                and self.n_orbitals - 1 in pair
            ):
                # the tapered orbital L is implicit (see make_hardcore_boson_hamiltonian): the pair only
                # moves between k and L if the orbitals other than k and L hold one pair less than the reference
                L = self.n_orbitals - 1
                k = pair[0] + pair[1] - L
                others = [self.transformation.up(i) for i in range(L) if i != k]
                parity = (-1) ** (self.n_electrons // 2 - 1)
                Qk = 0.5 + 0.5 * parity * (Z(others) if others else I())
                G = Sp(self.transformation.up(k)) if pair[1] == L else Sm(self.transformation.up(k))
                G = (1.0j * (G - G.dagger()) * Qk).simplify()
                p0 = (1.0 - Qk).simplify()
                U += gates.GeneralizedRotation(
                    angle=angle,
                    generator=G,
                    p0=p0,
                    # without other orbitals P0 vanishes (see make_excitation_gate)
                    assume_real=assume_real and len(p0) > 0,
                    control=control,
                )
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
        """
        Convenience function for orbital rotation circuit (rotating spatial orbital i and j) with standard naming of variables
        See arXiv:2207.12421 Eq.6 for UR(0,1)
        Parameters:
        ----------
            indices:
                two spatial(!) orbital indices i and j, a positive angle moves electrons from i to j
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
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
        """
        Convenience function for orbital correlator circuit (correlating spatial orbital i and j through a spin-paired double excitation) with standard naming of variables
        See arXiv:2207.12421 Eq.22 for UC(1,2)

        Parameters:
        ----------
            indices:
                two spatial(!) orbital indices i and j, a positive angle moves electrons from i to j
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
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
        return self.UR(indices[0], indices[1], *args, **kwargs)

    def make_excitation_gate(
        self, indices, angle, control=None, assume_real=True, **kwargs
    ):
        """
        Initialize a fermionic excitation gate defined as

        .. math::
            e^{-i\\frac{a}{2} G}
        with generator defines by the indices [(p0,q0),(p1,q1),...]
        .. math::
            G = i(\\prod_{k} a_{q_k}^\\dagger a_{p_k} - h.c.)

        Parameters
        ----------
            indices:
                List of tuples that define the generator, (p, q) destroys p and creates q
            angle:
                Numeric or hashable type or tequila objective
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
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
        """Creates a molecule in openfermion format by running psi4 and extracting the data
        Will check for previous outputfiles before running
        Will not recompute if a file was found

        Parameters
        ----------
        parameters :
            An instance of ParametersQC, which also holds an instance of ParametersPsi4 via parameters.psi4
            The molecule will be saved in parameters.filename, if this file exists before the call the molecule will be imported from the file

        Returns
        -------
        type
            the molecule in openfermion.MolecularData format

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
        """
        Called by self.__init__() with args and kwargs passed through
        Override this in derived class such that it returns an intitialized instance of the integral manager

        In the BaseClass it is required to pass the following with kwargs on init:
        - one_body_integrals as matrix
        - two_body_integrals as NBTensor of numpy.ndarray (four indices, openfermion ordering)
        - nuclear_repulsion (constant part of hamiltonian - optional)

        Method sets:
        - result of self.get_integrals()
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
                active_orbitals = kwargs["active_orbitals"]
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
        """
        Parameters
        ----------
        orbital_coefficients: second index is new orbital indes, first is old orbital index (summed over), indices are assumed to be defined on the active space
        ignore_active_space: if true orbital_coefficients are not assumed to be given in the active space
        name: str, name the new orbitals
        args
        kwargs

        Returns
        -------
        New molecule with transformed orbitals
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
        """
        U: a FCircuit with ONLY UR gates
        angles: dictionary to map the circuit variables in case they are not mapped yet
        Output: rotated molecule
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
        return self.use_native_orbitals()

    def use_native_orbitals(
        self, inplace=False, core: list | None = None, *args, **kwargs
    ):
        """
        Parameters
        ----------
            inplace:
                update current molecule or return a new instance
            core:
                list of core orbital indices (optional) — orbitals will be frozen and treated as doubly occupied. The orbitals correspond to
                the currently used orbitals of the molecule (default is usually canonical HF), see mol.print_basis_info() if unsure. Providing core
                orbitals is optional; the default is inherited from the active space set in self.integral_manager. If core is provided, the
                corresponding active native orbitals will be chosen based on their overlap with the core orbitals.
            active(in kwargs):
                list the active orbital indices (optional, in kwargs) - on the Native Orbital schema. Default: All orbitals, if core (see above) is provided,
                then the default is to automatically select the active orbitals based on their overlap with the provided core orbitals (selectint the N-|core|
                orbitals that have smallest overlap with coree).
                As an example, Assume the input geometry was H, He, H. active=[0,1,2] is selecting the (orthonormalized) atomic 1s (left H), 1s (He), 1s (right H).
                If core=[0] and active is not set, then active=[0,2] will be selected automatically (as the 1s He atomic orbital will have the largest overlap
                with the lowest energy HF orbital).
            reference_orbitals(in kwargs):
                list of orbitals doubly occupied orbitals. Must be of len equal to number total electrons//2. Relevant for desired mol.n_electrons in active spaces.
        Returns
        -------
        New molecule in the native (orthonormalized) basis given
        e.g. for standard basis sets the orbitals are orthonormalized Gaussian Basis Functions
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
        geom = self.parameters.get_geometry()
        f = ""
        f += f"{len(geom)}\n"
        f += f"{self.parameters.name}\n"
        for at in geom:
            f += f"{at[0]} {at[1][0]} {at[1][1]} {at[1][2]}\n"
        return f

    def get_spa_edges(self, use_active: bool = True, **kwargs):
        """
        Generates SPA edges using the pure Python JANPA CLPO algorithm.
        """
        from sunrise.CLPO.orbital_transformation import generate_CLPO_molecule_edges

        _, edges = generate_CLPO_molecule_edges(
            self, use_active=use_active, silent=True, **kwargs
        )
        return edges

    def use_HAO_orbitals(
        self, use_active: bool = True, inplace: bool = False, *args, **kwargs
    ):
        """
        Transforms the molecule to use Hybrid Atomic Orbitals (HAO) computed by the pure Python JANPA pipeline.
        Note: `sp_list` is ignored because JANPA automatically determines the optimal hybridization
        directly from the 3D molecular geometry.
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
        """
        Transforms the molecule to use Chemist's Localized Property-optimized Orbitals (CLPO) computed by the
        pure Python JANPA pipeline.
        Note: `core` is ignored because JANPA determines the orbitals directly from the 3D molecular geometry.

        Parameters
        ----------
        inplace: update current molecule or return a new instance
        edges: list of edges the CLPO algorithm is restricted to. If not given, they are determined by JANPA

        Returns
        -------
        New molecule in the CLPO basis
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
        """
        Same as `use_CLPO_orbitals`, but the SPA edges determined by the same CLPO run are returned along
        with the molecule, so that the JANPA pipeline is only executed once.

        Returns
        -------
        Tuple of the molecule in the CLPO basis and the corresponding list of SPA edges
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
        """
        The point group of the molecule. If it is not given with the point_group keyword, it is detected from the
        geometry as the largest point group defined in sunrise: the full point group if it is D4h or D6h,
        the largest Abelian subgroup otherwise. The quantum chemistry backends compute in abelian_point_group.
        It can be set with a PointGroup or its Schoenflies label, e.g. to work in a subgroup.
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
        """
        The point group if it is Abelian, its largest Abelian subgroup otherwise (e.g. D2h for D4h and D6h).
        PySCF can only label the orbitals in Abelian groups.
        """
        if self.point_group.schoenflies_label in pyscf.symm.param.CHARACTER_TABLE:
            return self.point_group
        pyscf_mol = pyscf.gto.M(
            atom=str(self.parameters.geometry),
            charge=self.parameters.charge,
            unit=self.parameters.units,
            symmetry=True,
            verbose=0,
        )
        return PointGroup.from_pyscf(pyscf_mol.groupname)

    def atomic_permutation_representation(self):
        """
        Builds the representation of the point group where the symmetry operations are
        represented as permutation matrices and the states are vectors where each
        component represents an atom of the molecule.

        Examples
        --------
        For linear H4 and inversion:

            [-1,  0,  0]    [0, 0, 0, 1]
            [ 0, -1,  0] -> [0, 0, 1, 0]
            [ 0,  0, -1]    [0, 1, 0, 0]
                            [1, 0, 0, 0]
        """
        geometric_representation = (
            PointGroupRepresentation.get_geometric_representation(self.point_group)
        )

        # This can be presumed as an NDArray
        operations = geometric_representation.operations

        mol_atom_coordinates = numpy.array(
            [atom[1] for atom in self.parameters.get_geometry()]
        )
        mol_center = numpy.mean(mol_atom_coordinates, axis=0)
        mol_hom_atom_coordinates = (
            GeometricTransformationHelper.homogeneous_coordinates(
                mol_atom_coordinates, is_row_vector=True
            ).T
        )

        atomic_permutation_representation_operations = {}
        for l, op in operations.items():
            # Apply the homogeneous operation to the homogeneous coordinates
            op_effect = (
                GeometricTransformationHelper.homogeneous_operation(
                    op, center=mol_center
                )
                @ mol_hom_atom_coordinates
            )
            op_permutations = GeometricTransformationHelper.to_col_permutation(
                mol_hom_atom_coordinates, op_effect
            )
            atomic_permutation_representation_operations[l] = (
                GeometricTransformationHelper.to_permutation_matrix(op_permutations)
            )

        return PointGroupRepresentation(atomic_permutation_representation_operations)

    def ao_permutation_representation(self):
        """
        Builds the representation of the point group where the symmetry operations are
        represented as permutation matrices and the states are vectors where each
        component represents an atomic orbital of the molecule.

        This is similar to the atomic permutation representation, but with additional
        structure within each atom's block corresponding to the orbitals on that atom.
        For example, for a molecule with two atoms ``x`` and ``y``, where ``x`` has 3 orbitals,
        ``y`` has 3 orbitals, and ``z`` has 2 orbitals, the state vector would have 8 components, e.g. ordered as
        ``[x1, x2, x3, y1, y2, y3, z1, z2]``. A symmetry operation that swaps atoms ``x`` and ``y``
        would then be represented by a permutation matrix that swaps the first 3
        rows/columns with the last 3 rows/columns accordingly. Only atomic orbitals of the same atom can
        be permuted with each other, so the permutation matrix will have a block-diagonal structure where
        each block corresponds to an atom and permutes only the orbitals of that atom.

        Notes
        -----
        PySCF is required as this method relies on it to determine the number of atomic orbitals on each atom.
        """

        def get_number_of_core_electrons(n):
            result = 0
            if n > 2:
                result += 2
            if n > 10:
                result += 10 - 2
            if n > 18:
                result += 18 - 10
            if n > 36:
                result += 36 - 18
            if n > 54:
                result += 54 - 36
            if n > 86:
                result += 86 - 54
            return result

        # This can be presumed as an NDArray
        operations = self.atomic_permutation_representation().operations

        # This finds out the number of active spatial orbitals by constructing a one-atom PySCF molecule
        atom_num_aos = []
        for atom in self.parameters.get_atoms():
            # Build a one-atom pyscf molecule with automatic spin
            pyscf_mol = pyscf.gto.M(
                atom=f"{atom} 0 0 0",
                basis=self.parameters.basis_set,
                spin=None,
                verbose=0,
            )
            mf = pyscf.scf.UHF(pyscf_mol).run()  # NOTE: check if RHF works equally well

            atom_num_aos.append(
                mf.mo_coeff.shape[1]
                - get_number_of_core_electrons(pyscf_mol.atom_charges()[0]) // 2
            )

        # Alternative: statically determine the number of active spatial orbitals for STO-3G based on num_active_spatial_orbitals
        # atom_num_aos = GeometricTransformationHelper.num_active_spatial_orbitals([pyscf.gto.charge(atom) for atom in self.parameters.get_atoms()])

        ao_blocks = []
        for i in range(len(atom_num_aos)):
            ao_blocks.extend([i] * atom_num_aos[i])

        ao_permutation_representation_operations = {}
        for l, op in operations.items():
            atomic_permutation_op = GeometricTransformationHelper.to_col_permutation(
                numpy.eye(len(op)), op
            )
            ao_permutation_representation_operations[l] = (
                GeometricTransformationHelper.build_blockwise_permutation_matrix(
                    ao_blocks, atomic_permutation_op
                )
            )

        return PointGroupRepresentation(ao_permutation_representation_operations)

    def qcircuit_representation(self):
        """
        Builds the representation of the point group where the symmetry operations are compiled
        circuits acting on QubitWaveFunctions with 2 * n_orbitals qubits
        """
        # the circuits permute the spin orbitals in the non-reordered Jordan-Wigner encoding (up0, down0, up1, ...),
        # so they only fit molecules whose transformation is that encoding
        if self.transformation is None or not all(
            self.transformation(openfermion.FermionOperator(f"{k}^"))
            == QubitHamiltonian.from_openfermion(
                openfermion.jordan_wigner(openfermion.FermionOperator(f"{k}^"))
            )
            for k in range(2 * self.n_orbitals)
        ):
            raise TequilaException(
                "The QCircuit representation needs the non-reordered Jordan-Wigner encoding (up0, down0, up1, ...): "
                "use a qubit molecule with transformation='JordanWigner' or a hybrid molecule with only fermionic orbitals"
            )

        # This can be presumed as an NDArray
        operations = self.ao_permutation_representation().operations

        # For each operation, we need to convert the permutation matrix into a sequence of swaps,
        # and then convert those swaps into a QCircuit while maintaining fermionic antisymmetry.
        qcircuit_representation_operations = {}
        for l, op in operations.items():
            swaps = GeometricTransformationHelper.perm_matrix_to_swaps(op)

            if swaps == []:
                U = QCircuit()  # Identity operation, no swaps needed
                U += gates.I(target=list(range(self.n_orbitals * 2)))
                qcircuit_representation_operations[l] = U
                continue

            max_qubit_index = max(max(max(swap) for swap in swaps), self.n_orbitals * 2)

            U = QCircuit()
            U += gates.I(
                target=list(range(max_qubit_index))
            )  # self.n_orbitals*2 might not be enough for the fermionic swaps
            for i, j in swaps:
                assert i < U.n_qubits and j < U.n_qubits, (
                    f"Mismatch between the AO permutation representation and the number of qubits in the circuit: swap indices ({i}, {j}) exceed current circuit qubits {U.n_qubits}. This likely indicates an error in the AO permutation representation construction."
                )
                # alternative method with FSWAP
                # U += FSWAP(i=i, j=j, n_orbitals=self.n_electrons, up_then_down=True).construct_circuit()
                U += FermionicSWAP(self).f_swap(i, j)

            qcircuit_representation_operations[l] = U

        def _is_close(state1, state2):
            if numpy.allclose((state1 + state2).to_array(), 0):
                return False
            return tequila.QubitWaveFunction.isclose(state1, state2)

        def _apply(compiled_op, state):
            return compiled_op(initial_state=state)

        # Compile each operation's circuit ONCE here, not per-state in _apply.
        compiled_operations = {
            l: tequila.compile(U) for l, U in qcircuit_representation_operations.items()
        }

        return PointGroupRepresentation(
            application_function=_apply,
            is_close_function=_is_close,
            operations=compiled_operations,
        )

    def _get_exact_ao_transformation_matrix(self, R_3x3):
        """
        Builds the exact AO transformation matrix U_AO for a given 3x3 rotation/reflection matrix.
        Correctly handles sign flips of p-orbitals under rotation.
        """
        pyscf_mol = pyscf.gto.M(
            atom=str(self.parameters.geometry),
            basis=self.parameters.basis_set,
            charge=self.parameters.charge,
            unit=self.parameters.units,
            verbose=0,
        )
        coords = pyscf_mol.atom_coords()
        center = numpy.mean(coords, axis=0)
        coords_rot = (coords - center) @ R_3x3.T + center
        aoslice = pyscf_mol.aoslice_by_atom()
        n_ao = pyscf_mol.nao_nr()
        U_AO = numpy.zeros((n_ao, n_ao))

        for i, c_rot in enumerate(coords_rot):
            dists = numpy.linalg.norm(coords - c_rot, axis=1)
            j = numpy.argmin(dists)
            if dists[j] > 1e-4:
                raise ValueError(f"Atom {i} does not map to any atom under rotation.")

            sh0_A, sh1_A, _, _ = aoslice[i]
            sh0_B, sh1_B, _, _ = aoslice[j]

            for sh_A, sh_B in zip(range(sh0_A, sh1_A), range(sh0_B, sh1_B)):
                l = pyscf_mol.bas_angular(sh_A)
                ao_s_A, ao_e_A = pyscf_mol.ao_loc[sh_A], pyscf_mol.ao_loc[sh_A + 1]
                ao_s_B, ao_e_B = pyscf_mol.ao_loc[sh_B], pyscf_mol.ao_loc[sh_B + 1]

                if l == 0:
                    U_AO[ao_s_B:ao_e_B, ao_s_A:ao_e_A] = 1.0
                elif l == 1:
                    U_AO[ao_s_B:ao_e_B, ao_s_A:ao_e_A] = R_3x3
                else:
                    raise NotImplementedError(f"Angular momentum l={l} not supported.")
        return U_AO

    def compute_mo_perms_and_phases(self):
        """
        Projects the geometric symmetry operations into the current MO basis using exact AO rotation matrices.

        Returns
        -------
        Tuple of dictionaries mapping each operation label to the permutation of the orbitals and to their phases
        """
        geom_repr = PointGroupRepresentation.get_geometric_representation(
            self.point_group
        )
        pyscf_mol = pyscf.gto.M(
            atom=str(self.parameters.geometry),
            basis=self.parameters.basis_set,
            charge=self.parameters.charge,
            unit=self.parameters.units,
            verbose=0,
        )
        S_ao = pyscf_mol.intor("int1e_ovlp")

        try:
            C_all = numpy.asarray(self.integral_manager.orbital_coefficients)
            n_active = self.n_orbitals

            # Slicing assumption: This grabs the last `n_active` columns.
            # This works perfectly if the injected matrix is ordered [active, virtual]
            # or if the active orbitals happen to be at the end.
            # If you inject a full matrix ordered as [core, active, virtual],
            # you may need to adjust this slice to C_all[:, n_core : n_core + n_active].
            C_mo = C_all[:, -n_active:].copy()

        except Exception:  # noqa: BLE001
            C_mo = pyscf.scf.RHF(pyscf_mol).run().mo_coeff[:, : self.n_orbitals].copy()

        # Canonicalize global signs to prevent arbitrary sign flips
        for i in range(C_mo.shape[1]):
            max_idx = numpy.argmax(numpy.abs(C_mo[:, i]))
            if C_mo[max_idx, i] < 0:
                C_mo[:, i] *= -1.0

        perms, phases = {}, {}
        for label, R_3x3 in geom_repr.operations.items():
            U_AO = self._get_exact_ao_transformation_matrix(R_3x3)
            S_mo = C_mo.T @ S_ao @ U_AO @ C_mo

            # Hungarian algorithm for guaranteed 1-to-1 mapping
            row_ind, col_ind = linear_sum_assignment(-numpy.abs(S_mo))
            perm = [0] * self.n_orbitals
            for r, c in zip(row_ind, col_ind):
                perm[c] = r
            perms[label] = perm

            phases_list = []
            for j in range(self.n_orbitals):
                i = perm[j]
                phase = numpy.sign(S_mo[i, j])
                phases_list.append(float(phase if phase != 0 else 1.0))
            phases[label] = phases_list

        return perms, phases

    def compute_orbital_irreps(self):
        """
        Computes the irreducible representations of the active orbitals, determined by PySCF in the Abelian point group
        of the molecule. The orbitals have to be symmetry adapted, e.g. canonical orbitals.
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
        """
        Called by self.make_molecule with args and kwargs passed through
        Override this in derived class if needed
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
        return self.integral_manager.active_orbitals

    @property
    def reference_orbitals(self):
        return self.integral_manager.active_reference_orbitals

    @property
    def n_orbitals(self) -> int:
        """
        Returns
        -------
        The number of active orbitals in this Molecule
        """
        return len(self.integral_manager.active_orbitals)

    @property
    def active_space(self):
        return self.integral_manager.active_space

    @property
    def n_electrons(self) -> int:
        """
        Returns
        -------
        The number of active electrons in this molecule
        """
        return 2 * len(self.integral_manager.active_reference_orbitals)

    def make_annihilation_op(self, orbital, coefficient=1.0):
        """
        Compute annihilation operator on spin-orbital in qubit representation
        Spin-orbital order is always (up,down,up,down,...)
        """
        assert orbital < self.n_orbitals * 2
        aop = openfermion.ops.FermionOperator(f"{orbital}", coefficient)
        return self.transformation(aop)

    def make_creation_op(self, orbital, coefficient=1.0):
        """
        Compute creation operator on spin-orbital in qubit representation
        Spin-orbital order is always (up,down,up,down,...)
        """
        assert orbital < self.n_orbitals * 2
        cop = openfermion.ops.FermionOperator(f"{orbital}^", coefficient)
        return self.transformation(cop)

    def make_number_op(self, orbital):
        """
        Compute number operator on spin-orbital in qubit representation
        Spin-orbital order is always (up,down,up,down,...)
        """
        num_op = self.make_creation_op(orbital) * self.make_annihilation_op(orbital)
        return num_op

    def make_sz_op(self):
        """
        Compute the spin_z operator of the molecule in qubit representation
        """
        sz = QubitHamiltonian()
        for i in range(0, self.n_orbitals * 2, 2):
            one = 0.5 * self.make_creation_op(i) * self.make_annihilation_op(i)
            two = 0.5 * self.make_creation_op(i + 1) * self.make_annihilation_op(i + 1)
            sz += one - two
        return sz

    def make_sp_op(self):
        """
        Compute the spin+ operator of the molecule in qubit representation
        """
        sp = QubitHamiltonian()
        for i in range(self.n_orbitals):
            sp += self.make_creation_op(i * 2) * self.make_annihilation_op(i * 2 + 1)
        return sp

    def make_sm_op(self):
        """
        Compute the spin- operator of the molecule in qubit representation
        """
        sm = QubitHamiltonian()
        for i in range(self.n_orbitals):
            sm += self.make_creation_op(i * 2 + 1) * self.make_annihilation_op(i * 2)
        return sm

    def make_s2_op(self):
        """
        Compute the spin^2 operator of the molecule in qubit representation
        """
        s2_op = self.make_sm_op() * self.make_sp_op() + self.make_sz_op() * (
            self.make_sz_op() + 1
        )
        return s2_op

    def make_hamiltonian(self, *args, **kwargs) -> QubitHamiltonian:
        """
        Parameters
        ----------
        occupied_indices: will be auto-assigned according to specified active space. Can be overridden by passing specific lists (same as in open fermion)
        active_indices: will be auto-assigned according to specified active space. Can be overridden by passing specific lists (same as in open fermion)

        Returns
        -------
        Qubit Hamiltonian in the Fermion-to-Qubit transformation defined in self.parameters
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
        """
        Returns
        -------
        Hamiltonian in Hardcore-Boson approximation (electrons are forced into spin-pairs)
        Indepdent of Fermion-to-Qubit mapping
        condensed: always give Hamiltonian back from qubit 0 to N where N is the number of orbitals
        if condensed=False then JordanWigner would give back the Hamiltonian defined on even qubits between 0 to 2N
        """

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

        # in tapered binary encodings the last orbital L is implicit: it holds a pair if the other
        # orbitals hold one pair less than the reference, which the parity of their Z operators tells
        tapered = "TAPEREDBINARY" in self.transformation.name.upper()
        L = n_orbitals - 1
        parity = (-1) ** (self.n_electrons // 2 - 1)
        ZL = Z(list(range(L))) if L > 0 else I()
        H = c
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                up = p
                uq = q
                if tapered and L in (p, q):
                    k = p + q - L
                    nL = 0.5 + 0.5 * parity * ZL
                    if p == q:
                        H += h[p, q] * nL
                    else:
                        # the pair moves between k and L if the orbitals other than k and L hold one pair less
                        others = [i for i in range(L) if i != k]
                        Qk = 0.5 + 0.5 * parity * (Z(others) if others else I())
                        hop = Sm(k) if p == k else Sp(k)
                        H += h[p, q] * hop * Qk + g[p, q] * Sm(k) * Sp(k) * nL
                    continue
                H += h[p, q] * Sm(up) * Sp(uq) + g[p, q] * Sm(up) * Sp(up) * Sm(
                    uq
                ) * Sp(uq)

        if not self.transformation.up_then_down and not condensed:
            alpha_map = {k.idx: self.transformation.up(k.idx) for k in self.orbitals}
            H = H.map_qubits(alpha_map)
        return H

    def make_molecular_hamiltonian(self, occupied_indices=None, active_indices=None):
        """
        Returns
        -------
        Create a MolecularHamiltonian as openfermion Class
        (used internally here, not used in tequila)
        """
        return self.make_molecule().get_molecular_hamiltonian(
            occupied_indices=occupied_indices, active_indices=active_indices
        )

    def get_integrals(self, *args, **kwargs):
        """
        Returns

        options for kwargs: "ordering = ["openfermion", "chem", "phys"], ignore_active_space = [True, False]"
        -------
        Tuple with:
        constant part (nuclear_repulsion + possible integrated parts from active-spaces)
        one_body_integrals
        two_body_integrals
        """

        # backward compatibility
        if "two_body_ordering" in kwargs:
            kwargs["ordering"] = kwargs["two_body_ordering"]

        return self.integral_manager.get_integrals(*args, **kwargs)

    def compute_one_body_integrals(self):
        """convenience function"""
        _c, h1, _h2 = self.get_integrals()
        return h1

    def compute_two_body_integrals(self, ordering="openfermion"):
        _c, _h1, h2 = self.get_integrals(ordering=ordering)
        return h2

    def compute_constant_part(self):
        c, _h1, _h2 = self.get_integrals()
        return c

    def compute_ccsd_amplitudes(self) -> ClosedShellAmplitudes:
        raise TequilaException("BaseClass Method")

    def _reference_state(self):
        """
        used internally
        gives back reference state occupation vector (in second quantization or JW notation)
        transformation to current encoding is done in def prepare_reference
        """
        assert self.n_electrons % 2 == 0
        state = [0] * (self.n_orbitals * 2)
        for i in self.reference_orbitals:
            state[2 * i.idx] = 1
            state[2 * i.idx + 1] = 1
        return state

    def prepare_reference(self, state=None, *args, **kwargs):
        """
        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in the chosen transformation
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
        """
        Prepare reference state in the Hardcore-Boson approximation (eqch qubit represents two spin-paired electrons)
        Returns
        -------
        tq.QCircuit that prepares the HCB reference
        """
        U = gates.X(
            target=[self.transformation.up(i.idx) for i in self.reference_orbitals]
        )
        U.n_qubits = self.n_orbitals
        return U

    def hcb_to_me(self, U=None, condensed=False):
        """
        Transform a circuit in the hardcore-boson encoding (HCB)
        to the encoding of this molecule
        HCB is supposed to be encoded on the first n_orbitals qubits
        Parameters
        ----------
        U: HCB circuit (using the alpha qubits)
        condensed: assume that incoming U is condensed (HCB on the first n_orbitals; and not, as for example in JW on the first n even orbitals)
        Returns
        -------

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
        """
         Assuming a pair-specific model, create a pair-specific index list
         to be used in make_upccgsd_ansatz(indices = ... )
         Excite from a set of references (i) to any pair coming from (i),
         i.e. any (i,j)/(j,i). If general excitations are allowed, also
         allow excitations from pairs to appendant pairs and reference.

         Parameters
         ----------
         pair_info
             file or list including information about pair structure
             references single number, pair double
             example: as file: "0,1,11,11,00,10" (hand over file name)
                      in file, skip first row assuming some text with information
                      as list:['0','1`','11','11','00','10']
                      ~> two reference orbitals 0 and 1,
                      then two orbitals from pair 11, one from 00, one mixed 10
         include_singles
             include single excitations
         general_excitations
             allow general excitations
        Returns
         -------
             list of indices with pair-specific ansatz
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
        """
        Consistent formatting of excitation indices
        idx = [(p0,q0),(p1,q1),...,(pn,qn)]
        sorted as: p0<p1<pn and pi<qi
        :param idx: list of index tuples describing a single(!) fermionic excitation
        :return: tuple-list of index tuples
        """

        idx = [tuple(sorted(x)) for x in idx]
        idx = sorted(idx, key=lambda x: x[0])
        return tuple(idx)

    def make_upccgsd_indices(self, key, reference_orbitals=None, *args, **kwargs):
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
    ):
        """
        Separable Pair Ansatz (SPA) for general molecules
        see arxiv:
        edges: a list of tuples that contain the orbital indices for the specific pairs
               one example: edges=[(0,), (1,2,3), (4,5)] are three pairs, one with a single orbital [0], one with three orbitals [1,2,3] and one with two orbitals [4,5]
        hcb: spa ansatz in the hcb (hardcore-boson) space without transforming to current transformation (e.g. JordanWigner), use this for example in combination with the self.make_hardcore_boson_hamiltonian() and see the article above for more info
        use_units_of_pi: circuit angles in units of pi
        label: label the variables in the circuit
        optimize: optimize the circuit construction (see article). Results in shallow circuit from Ry and CNOT gates
        ladder: if true the excitation pattern will be local. E.g. in the pair from orbitals (1,2,3) we will have the excitations 1->2 and 2->3, if set to false we will have standard coupled-cluster style excitations - in this case this would be 1->2 and 1->3
        """

        def _make_spa_tappered(
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
                            # rotating the explicit orbital instead of L: same angle convention as the other encodings
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
            return _make_spa_tappered(
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
        """
        Automatically calls the right subroutines to construct ansatze implemented in sunrise.chemistry
        name: namne of the ansatz, examples are: UpCCGSD, UpCCD, SPA, UCCSD, SPA+UpCCD, SPA+GS
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
        """
        UpGCCSD Ansatz similar as described by Lee et. al.

        Parameters
        ----------
        include_reference
            include the HF reference state as initial state
        indices
            pass custom defined set of indices from which the ansatz will be created
            List of tuples of tuples spin-indices e.g. [((2*p,2*q),(2*p+1,2*q+1)), ...]
        label
            An additional label that is set with the variables
            default is None and no label will be set: variables names will be
            (x, (p,q)) for x in range(order)
            with a label the variables will be named
            (label, (x, (p,q)))
        order
            Order of the ansatz (default is 1)
            determines how often the ordering gets repeated
            parameters of repeating layers are independent
        assume_real
            assume a real wavefunction (that is always the case if the reference state is real)
            reduces potential gradient costs from 4 to 2
        mix_sd
            Changes the ordering from first all doubles and then all singles excitations (DDDDD....SSSS....) to
            a mixed order (DS-DS-DS-DS-...) where one DS pair acts on the same MOs. Useful to consider when systems
            with high electronic correlation and system high error associated with the no Trotterized UCC.
        Returns
        -------
            UpGCCSD ansatz
        """

        name = name.upper()

        if ("A" in name) and neglect_z is None:
            neglect_z = True
        else:
            neglect_z = False

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except Exception:  # noqa: BLE001
                order = 1

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
        """

        Parameters
        ----------
        initial_amplitudes :
            initial amplitudes given as ManyBodyAmplitudes structure or as string
            where 'mp2', 'cc2' or 'ccsd' are possible initializations
        include_reference_ansatz :
            Also do the reference ansatz (prepare closed-shell Hartree-Fock) (Default value = True)
        parametrized :
            Initialize with variables, otherwise with static numbers (Default value = True)
        trotter_steps: int :

        initial_amplitudes: typing.Union[str :

        Amplitudes :

        ClosedShellAmplitudes] :
             (Default value = "cc2")

        Returns
        -------
        type
            Parametrized QCircuit

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
        """
        Compute closed-shell CC amplitudes

        Parameters
        ----------
        method :
            coupled-cluster methods like cc2, ccsd, cc3, ccsd(t)
            Success might depend on backend
            got an extra function for MP2
        *args :

        **kwargs :


        Returns
        -------

        """
        raise TequilaException("compute amplitudes: Needs to be overwritten by backend")

    def compute_energy(self, method, *args, **kwargs):
        """
        Call classical methods over PySCF (needs to be installed) or
        use as a shortcut to calculate quantum energies (see make_upccgsd_ansatz)

        Parameters
        ----------
        method: method name
                classical: HF, MP2, CCSD, CCSD(T), FCI -- with pyscf
                quantum: UpCCD, UpCCSD, UpCCGSD, k-UpCCGSD, UCCSD,
                see make_upccgsd_ansatz of the this class for more information
        args
        kwargs: for quantum methods, keyword arguments for minimizer

        Returns
        -------

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
        raise NotImplementedError(
            "compute_fci only implemented for the 'pyscf' backend"
        )

    def compute_fock_matrix(self):
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

    def compute_mp2_amplitudes(
        self, hf_energy=None, return_energy=False
    ) -> ClosedShellAmplitudes:
        """

        Compute closed-shell mp2 amplitudes (canonical amplitudes only)

        .. math::
            t(a,i,b,j) = 0.25 * g(a,i,b,j)/(e(i) + e(j) -a(i) - b(j) )

        :return:

        Parameters
        ----------

        Returns
        -------

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
        """
        Compute the CIS amplitudes of the molecule
        Warning: Not field tested!
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
        cs = self.n_electrons % 2 == 0
        if verify and not cs:
            raise TequilaException(
                f"not a closed shell molecule: having {self.n_electrons} electrons"
            )
        return cs

    def is_canonical(self, verify=False, fock_matrix=None):
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
        """
        Returns RMD1 if computed with compute_rdms function before
        """
        if self._rdm1 is not None:
            return self._rdm1
        else:
            print("1-RDM has not been computed. Return None for 1-RDM.")
            return None

    @property
    def rdm2(self):
        """
        Returns RMD2 if computed with compute_rdms function before
        This is returned in Dirac (physics) notation by default (can be changed in compute_rdms with keyword)!
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
        """
        Computes the one- and two-particle reduced density matrices (rdm1 and rdm2) given
        a unitary U. This method uses the standard ordering in physics as denoted below.
        Note, that the representation of the density matrices depends on the qubit transformation
        used. The Jordan-Wigner encoding corresponds to 'classical' second quantized density
        matrices in the occupation picture.

        We only consider real orbitals and thus real-valued RDMs.
        The matrices are set as private members _rdm1, _rdm2 and can be accessed via the properties rdm1, rdm2.

        .. math :
            \\text{rdm1: } \\gamma^p_q = \\langle \\psi | a^p a_q | \\psi \\rangle
                                     = \\langle U 0 | a^p a_q | U 0 \\rangle
            \\text{rdm2: } \\gamma^{pq}_{rs} = \\langle \\psi | a^p a^q a_s a_r | \\psi \\rangle
                                             = \\langle U 0 | a^p a^q a_s a_r | U 0 \\rangle

        Parameters
        ----------
        U :
            Quantum Circuit to achieve the desired state \\psi = U |0\\rangle, non-optional
        variables :
            If U is parametrized, then need to hand over a set of fixed variables
        spin_free :
            Set whether matrices should be spin-free (summation over spin) or defined by spin-orbitals
        get_rdm1, get_rdm2 :
            Set whether either one or both rdm1, rdm2 should be computed. If both are needed at some point,
            it is recommended to compute them at once.
        rdm_trafo :
            The rdm operators can be transformed, e.g., a^dagger_i a_j -> U^dagger a^dagger_i a_j U,
            where U represents the transformation. The default is set to None, implying that U equas the identity.
        evaluate :
            if true, the tequila expectation values are evaluated directly via the tq.simulate command.
            the protocol is optimized to avoid repetation of wavefunction simulation
            if false, the rdms are returned as tq.QTensors
        Returns
        -------
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
            """Build the hardcore boson operators: b^\\dagger_ib_j + h.c. in qubit encoding"""
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
            """Returns operator given by a operator tuple as OpenFermion - Fermion operator"""
            op = openfermion.FermionOperator(operator_tuple)
            return op

        def _get_qop_hermitian(of_operator) -> QubitHamiltonian:
            """Returns Hermitian part of Fermion operator as QubitHamiltonian"""
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
            """Returns spinful one-body operators as a symmetry-reduced list of QubitHamiltonians"""
            # Exploit symmetry pq = qp
            ops = []
            for p in range(n_SOs):
                for q in range(p + 1):
                    op_tuple = ((p, 1), (q, 0))
                    op = _get_of_op(op_tuple)
                    ops += [op]

            return ops

        def _build_2bdy_operators_spinful() -> list:
            """Returns spinful two-body operators as a symmetry-reduced list of QubitHamiltonians"""
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
            """Returns spinfree one-body operators as a symmetry-reduced list of QubitHamiltonians"""
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
            """Returns spinfree two-body operators as a symmetry-reduced list of QubitHamiltonians"""
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
            """
            Returns spin-ful or spin-free one-particle RDM built by symmetry conditions
            Same symmetry with or without spin, so we can use the same function
            """
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
            """Returns spin-ful two-particle RDM built by symmetry conditions"""
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
            """Returns spin-free two-particle RDM built by symmetry conditions"""
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
            """Returns hcb one-body operators as a symmetry-reduced list of QubitHamiltonians"""
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
            """Returns hcb two-body operators as a symmetry-reduced list of QubitHamiltonians"""
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
        """
        Given the spin-ful 1- and 2-particle reduced density matrices, compute the spin-free RDMs by spin summation.

        Parameters
        ----------
            sum_rdm1, sum_rdm2 :
               If set to true, perform spin summation on rdm1, rdm2

        Returns
        -------
            rdm1_spinsum, rdm2_spinsum :
                The desired spin-free matrices
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
        """
        Computes the spin-free [2]_R12 correction, needing only the 1- and 2-RDM of a reference method
        Requires either 1-RDM, 2-RDM or information to compute them in kwargs

        Parameters
        ----------
        rdm1 :
            1-electron reduced density matrix
        rdm2 :
            2-electron reduced density matrix
        gamma :
            f12-exponent, for a correlation factor f_12 = -1/gamma * exp[-gamma*r_12]
        n_ri :
            dimensionality of RI-basis; specify only, if want to truncate available RI-basis
            if None, then the maximum available via tensors / basis-set is used
            must not be larger than size of available RI-basis, and not smaller than size of OBS
            for n_ri==dim(OBS), the correction returns zero
        external_info :
            for usage in qc_base, need to provide information where to find one-body tensor f12-tensor <rs|f_12|pq>;
            pass dictionary with {"f12_filename": where to find f12-tensor, "scheme": ordering scheme of tensor}
        kwargs :
            e.g. RDM-information via {"U": QCircuit, "variables": optimal angles}, needs to be passed if rdm1,rdm2 not
            yet computed

        Returns
        -------
            the f12 correction for the energy
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
        """
        Creates a quantum circuit that applies a phase rotation based on phi to both components (up and down) of a given qubit.

        Parameters:
        - i (int): The index of the qubit to which the rotation will be applied.
        - phi (float): The rotation angle. The actual rotation applied will be multiplied with -2 for both components.

        Returns:
        - QCircuit: A quantum circuit object containing the sequence of rotations applied to the up and down components of the specified qubit.
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
        """
        Constructs a quantum circuit from a given real unitary matrix using Givens rotations.

        This method decomposes a unitary matrix into a series of Givens and Rz (phase) rotations,
        then constructs and returns a quantum circuit that implements this sequence of rotations.

        Parameters:
        - unitary (numpy.array): A real unitary matrix representing the transformation to implement.
        - tol (float): A tolerance threshold below which matrix elements are considered zero.
        - ordering (list of tuples or 'Optimized'): Custom ordering of indices for Givens rotations or 'Optimized' to generate them automatically.
        - fix (bool): whether to set the angle as fixed or as inial value as: angle=tq.Variable(idx) + value. Useful to let further relaxation to the basis change
        - label: can be passed instead of angle to have auto-naming with label ("R",i,j,label) useful for repreating gates with individual variables

        Returns:
        - QCircuit: A quantum circuit implementing the series of rotations decomposed from the unitary.
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
        return self.integral_manager.print_basis_info()

    def __str__(self) -> str:
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
    """
    Construct a complex Givens rotation matrix of dimension n by theta between rows/columns p and q.
    """
    """
    Generates a Givens rotation matrix of size n x n to rotate by angle theta in the (p, q) plane. This matrix can be complex

    Parameters:
    - n (int): The size of the Givens rotation matrix.
    - p (int): The first index for the rotation plane.
    - q (int): The second index for the rotation plane.
    - theta (float): The rotation angle.

    Returns:
    - numpy.array: The Givens rotation matrix.
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
    """
    Decomposes a real unitary matrix into Givens rotations (theta) and Rz rotations (phi).

    Parameters:
    - unitary (numpy.array): A real unitary matrix to decompose. It cannot be complex.
    - tol (float): Tolerance for considering matrix elements as zero. Elements with absolute value less than tol are treated as zero.
    - ordering (list of tuples or 'Optimized'): Custom ordering of indices for Givens rotations or 'Optimized' to generate them automatically.
    - return_diagonal (bool): If True, the function also returns the diagonal matrix as part of the output.

    Returns:
    - list: A list of tuples, each representing a Givens rotation. Each tuple contains the rotation angle theta and indices (i,j) of the rotation.
    - list: A list of tuples, each representing an Rz rotation. Each tuple contains the rotation angle phi and the index (i) of the rotation.
    - numpy.array (optional): The diagonal matrix after applying all Givens rotations, returned if return_diagonal is True.
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
    """
    Reconstructs a matrix from given Givens rotations and Rz diagonal rotations.
    This function is effectively an inverse of get_givens_decomposition, and therefore only works with data in the same format as its output.

    Parameters:
    - n (int): The size of the unitary matrix to be reconstructed.
    - theta_list (list of tuples): Each tuple contains (angle, i, j) representing a Givens rotation of `angle` radians, applied to rows/columns `i` and `j`.
    - phi_list (list of tuples): Each tuple contains (angle, i), representing an Rz rotation by `angle` radians applied to the `i`th diagonal element.
    - to_real_if_possible (bool): If True, converts the matrix to real if its imaginary part is effectively zero.
    - tol (float): The tolerance whether to swap a complex rotation for a sign change.

    Returns:
    - numpy.ndarray: The reconstructed complex or real matrix, depending on the `to_real_if_possible` flag and matrix composition.
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
    if isinstance(x1, (Variable, Objective)) or isinstance(x2, (Variable, Objective)):
        return Objective().binary_operator(left=1 * x1, right=1 * x2, op=numpy.arctan2)
    elif not isinstance(x1, numbers.Complex) and not isinstance(x2, numbers.Complex):
        return numpy.arctan2(x1, x2)
    else:
        return numpy.arctan2(x1.imag, x2.imag) + numpy.arctan2(x1.real, x2.real)
