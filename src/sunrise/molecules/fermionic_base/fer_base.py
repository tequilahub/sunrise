import copy
from dataclasses import dataclass
from tequila import TequilaException, BitString, TequilaWarning
from tequila.hamiltonian import QubitHamiltonian
from sunrise.fermionic_operations import FCircuit
from sunrise.fermionic_operations import gates
from tequila.objective.objective import Variable, Variables,Objective
from tequila import QTensor

# from tequila.simulators.simulator_api import simulate
from ...expval.minimize import simulate
from ...expval import Braket
from tequila.quantumchemistry.chemistry_tools import (
    prepare_product_state,
    ClosedShellAmplitudes,
    Amplitudes,
    ParametersQC,
    NBodyTensor,
)
import typing
import numpy
from itertools import product
from sunrise.fermionic_operations.givens_rotations import get_givens_circuit as __get_givens_circuit
from sunrise.fermionic_operations.givens_rotations import n_rotation as __n_rotation
from sunrise.fermionic_operations.givens_rotations import reconstruct_matrix_from_circuit
from sunrise.hybridization.hybridization import Graph

from tequila.quantumchemistry.qc_base import QuantumChemistryBase
try:
    # if you are experiencing import errors you need to update openfermion
    # required is version >= 1.0
    # otherwise replace with from openfermion.hamiltonians import MolecularData
    import openfermion
    from openfermion.chem import MolecularData
    from openfermion.ops import FermionOperator as OFFermionOperator
    from openfermion.utils import hermitian_conjugated
except Exception:
    try:
        from openfermion.hamiltonians import MolecularData
    except Exception as E:
        raise Exception("{}\nIssue with Tequila Chemistry: Please update openfermion".format(str(E)))
import warnings

OPTIMIZED_ORDERING = "Optimized"

class FermionicBase(QuantumChemistryBase):
    """
    Base Class for tequila chemistry functionality
    This is what is initialized with tq.Molecule(...)
    We try to define all main methods here and only implemented specializations in the derived classes
    Derived classes interface specific backends (e.g. Psi4, PySCF and Madness). See PACKAGE_interface.py for more
    """

    def __init__(
        self,
        parameters: ParametersQC,
        active_orbitals: list = None,
        frozen_orbitals: list = None,
        orbital_type: str = None,
        reference_orbitals: list = None,
        orbitals: list = None,
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
        self.transformation = None
        self.fermionic_backend = None
        if 'fermionic_backend' in kwargs:
            self.fermionic_backend = kwargs['fermionic_backend']
            kwargs.pop('fermionic_backend')
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
        overriding_freeze_instruction = orbital_type is not None and orbital_type.lower() == "native"
        # determine frozen core automatically if set
        # only if molecule is computed from scratch and not passed down from above
        overriding_freeze_instruction = overriding_freeze_instruction or n_electrons != parameters.total_n_electrons
        overriding_freeze_instruction = overriding_freeze_instruction or frozen_orbitals is not None
        if not overriding_freeze_instruction and self.parameters.frozen_core:
            n_core_electrons = self.parameters.get_number_of_core_electrons()
            if frozen_orbitals is None:
                frozen_orbitals = [i for i in range(n_core_electrons // 2)]

        # initialize integral manager
        if "integral_manager" in kwargs:
            self.integral_manager = kwargs["integral_manager"]
        else:
            self.integral_manager = self.initialize_integral_manager(
                active_orbitals=active_orbitals,
                reference_orbitals=reference_orbitals,
                orbitals=orbitals,
                frozen_orbitals=frozen_orbitals,
                orbital_type=orbital_type,
                basis_name=self.parameters.basis_set,
                *args,
                **kwargs,
            )

        if orbital_type is not None and orbital_type.lower() == "native":
            self.integral_manager.transform_to_native_orbitals()

        self._rdm1 = None
        self._rdm2 = None

    @classmethod
    def from_tequila(cls, molecule,*args, **kwargs):
        c = molecule.integral_manager.constant_term
        h1 = molecule.integral_manager.one_body_integrals
        h2 = molecule.integral_manager.two_body_integrals
        S = molecule.integral_manager.overlap_integrals
        if "active_orbitals" not in kwargs:
            active_orbitals = [o.idx_total for o in molecule.integral_manager.active_orbitals]
        else:
            active_orbitals = kwargs["active_orbitals"]
            kwargs.pop("active_orbitals")
        parameters = molecule.parameters
        return cls(
            nuclear_repulsion=c,
            one_body_integrals=h1,
            two_body_integrals=h2,
            overlap_integrals=S,
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            active_orbitals=active_orbitals,
            orbital_type=molecule.integral_manager._orbital_type,
            parameters=parameters,
            reference_orbitals=molecule.integral_manager.active_space.reference_orbitals,
            *args,
            **kwargs,
        )

    def supports_ucc(self):
        """
        check if the current molecule supports UCC operations
        (e.g. mol.make_excitation_gate)
        """
        return True

    @classmethod
    def from_openfermion(
        cls,
        molecule: openfermion.MolecularData,
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
        parameters = ParametersQC(
            basis_set=molecule.basis,
            geometry=molecule.geometry,
            units="angstrom",
            description=molecule.description,
            multiplicity=molecule.multiplicity,
            charge=molecule.charge,
        )
        return cls(parameters=parameters, molecule=molecule, *args, **kwargs)

    def make_excitation_generator(
        self, indices: typing.Iterable[typing.Tuple[int, int]], form: str = None,
    ) -> openfermion.FermionOperator:
        """
        Notes
        ----------
        Creates the transformed hermitian generator of UCC type unitaries:
              M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
              where the qubit map M depends is self.transformation

        Parameters
        ----------
        indices : typing.Iterable[typing.Tuple[int, int]] :
            List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
            can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
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
                "Molecule with transformation {} does not support general UCC operations".format(self.transformation)
            )

        # check indices and convert to list of tuples if necessary
        if len(indices) == 0:
            raise TequilaException("make_excitation_operator: no indices given")
        elif not isinstance(indices[0], typing.Iterable):
            if len(indices) % 2 != 0:
                raise TequilaException(
                    "make_excitation_generator: unexpected input format of indices\n"
                    "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                    "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                    "you gave: {}".format(indices)
                )
            converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        else:
            converted = indices

        # convert everything to native python int
        # otherwise openfermion will complain
        converted = [(int(pair[0]), int(pair[1])) for pair in converted]

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
                op += openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            elif form.lower() == "g-":
                op += openfermion.FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, 1.0)
                op += openfermion.FermionOperator(Ni + Ma, 1.0)
            elif form.lower() == "p0":
                # P0: we only construct P0 and don't keep the original generator
                op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
                op += openfermion.FermionOperator(Na + Mi, -1.0)
                op += openfermion.FermionOperator(Ni + Ma, -1.0)
            else:
                raise TequilaException("Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))
        return  op
    
    def make_hardcore_boson_excitation_gate(
        self, indices, angle, control=None, assume_real=True, compile_options="optimize"
    )->FCircuit:
        raise TequilaException('No HCB operations in Fermionic Backends')

    def UR(self, i, j, angle=None, label=None, control=None, *args, **kwargs)->FCircuit:
        """
        Convenience function for orbital rotation circuit (rotating spatial orbital i and j) with standard naming of variables
        See arXiv:2207.12421 Eq.6 for UR(0,1)
        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
            angle:
                Numeric or hashable type or tequila objective. Default is None and results
                in automatic naming as ("R",i,j)
            label:
                can be passed instead of angle to have auto-naming with label ("R",i,j,label)
                useful for repreating gates with individual variables
            control:
                List of possible control qubits 
        """
        i, j = self.format_excitation_indices([(i, j)])[0]
        if angle is None:
            if label is None:
                angle = Variable(name=("R", i, j)) * numpy.pi
            else:
                angle = Variable(name=("R", i, j, label)) * numpy.pi
        if control is not None: 
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        return gates.UR(i,j,angle)

    def UC(self, i, j, angle=None, label=None, control=None, *args, **kwargs) ->FCircuit:
        """
        Convenience function for orbital correlator circuit (correlating spatial orbital i and j through a spin-paired double excitation) with standard naming of variables
        See arXiv:2207.12421 Eq.22 for UC(1,2)

        Parameters:
        ----------
            indices:
                tuple of two spatial(!) orbital indices
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
        i, j = self.format_excitation_indices([(i, j)])[0]
        if angle is None:
            if label is None:
                angle = Variable(name=("C", i, j)) * numpy.pi
            else:
                angle = Variable(name=("C", i, j, label)) * numpy.pi
        if control is not None: 
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        return gates.UC(i,j,angle)

    def make_excitation_gate(self, indices, angle, control=None,  **kwargs) -> FCircuit:
        """
        Initialize a fermionic excitation gate defined as

        .. math::
            e^{-i\\frac{a}{2} G}
        with generator defines by the indices [(p0,q0),(p1,q1),...]
        .. math::
            G = i(\\prod_{k} a_{p_k}^\\dagger a_{q_k} - h.c.)

        Parameters
        ----------
            indices:
                List of tuples that define the generator
            angle:
                Numeric or hashable type or tequila objective
            control:
                List of possible control qubits
            assume_real:
                Assume that the wavefunction will always stay real.
                Will reduce potential gradient costs by a factor of 2
        """
        if control is not None: 
            raise NotImplementedError("Not implemented yet for Fermionic Backends")
        return gates.FermionicExcitation(indices=indices,variables=angle)

    def transform_orbitals(self, orbital_coefficients:typing.Union[FCircuit,numpy.array], ignore_active_space=False, name=None, *args, **kwargs):
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
        if isinstance(orbital_coefficients,FCircuit):
            orbital_coefficients = numpy.array(reconstruct_matrix_from_circuit(orbital_coefficients,self.n_orbitals,tol=1.e-6)).T
        U = numpy.eye(self.integral_manager.orbital_coefficients.shape[0])
        # mo_coeff by default only acts on the active space
        active_indices = [o.idx_total for o in self.integral_manager.active_orbitals]

        if ignore_active_space:
            U = orbital_coefficients
        else:
            for kk, k in enumerate(active_indices):
                for ll, l in enumerate(active_indices):
                    if isinstance(orbital_coefficients[kk][ll],Objective):
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

    def use_native_orbitals(self, inplace=False, core: list = None, *args, **kwargs):
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
        Returns
        -------
        New molecule in the native (orthonormalized) basis given
        e.g. for standard basis sets the orbitals are orthonormalized Gaussian Basis Functions
        """
        c = copy.deepcopy(self.integral_manager.orbital_coefficients)
        s = self.integral_manager.overlap_integrals
        d = self.integral_manager.get_orthonormalized_orbital_coefficients()

        def inner(a, b, s):
            return numpy.sum(numpy.multiply(numpy.outer(a, b), s))

        def orthogonalize(c, d, s):
            """
            :return: orthogonalized orbitals with core HF orbitals and active Orthongonalized Native orbitals.
            """
            ### Computing Core-Active overlap Matrix
            # sbar_{ki} = \langle \phi_k | \varphi_i \rangle = \sum_{m,n} c_{nk}d_{mi}\langle \chi_n | \chi_m \rangle
            # c_{nk} = HF coeffs, d_{mi} = nat orb coef s_{mn} = Atomic Overlap Matrix
            # k \in active orbs, i \in core orbs, m,n \in basis coeffs
            # sbar = np.einsum('nk,mi,nm->ki', c, d, s) #only works if active == to_active
            c = c.T
            d = d.T
            sbar = numpy.zeros(shape=s.shape)
            for k in active:
                for i in core:
                    sbar[i][to_active[k]] = inner(c[i], d[k], s)
            ### Projecting out Core orbitals from the Native ones
            # dbar_{ji} = d_{ji} - \sum_k sbar_{ki}c_{jk}
            # k \in active, i \in core, j in basis coeffs
            dbar = numpy.zeros(shape=s.shape)

            for j in active:
                dbar[to_active[j]] = d[j]
                for i in core:
                    temp = sbar[i][to_active[j]] * c[i]
                    dbar[to_active[j]] -= temp
            ### Projected-out Nat Orbs Normalization
            for i in to_active.values():
                norm = numpy.sqrt(numpy.sum(numpy.multiply(numpy.outer(dbar[i], dbar[i]), s.T)))
                if not numpy.isclose(norm, 0):
                    dbar[i] = dbar[i] / norm
            ### Reintroducing the New Coeffs on the HF coeff matrix
            for j in to_active.values():
                c[j] = dbar[j]
            ### Compute new orbital overlap matrix:
            sprima = numpy.eye(len(c))
            for idx, i in enumerate(to_active.values()):
                for j in [*to_active.values()][idx:]:
                    sprima[i][j] = inner(c[i], c[j], s)
                    sprima[j][i] = sprima[i][j]
            ### Symmetric orthonormalization
            lam_s, l_s = numpy.linalg.eigh(sprima)
            lam_s = lam_s * numpy.eye(len(lam_s))
            lam_sqrt_inv = numpy.sqrt(numpy.linalg.inv(lam_s))
            symm_orthog = numpy.dot(l_s, numpy.dot(lam_sqrt_inv, l_s.T))
            return symm_orthog.dot(c).T

        def get_active(core):
            ov = numpy.zeros(shape=(len(self.integral_manager.orbitals)))
            for i in core:
                for j in range(len(d)):
                    ov[j] += numpy.abs(inner(c.T[i], d.T[j], s))
            act = []
            for i in range(len(self.integral_manager.orbitals) - len(core)):
                idx = numpy.argmin(ov)
                act.append(idx)
                ov[idx] = 1 * len(core)
            act.sort()
            return act

        def get_core(active):
            ov = numpy.zeros(shape=(len(self.integral_manager.orbitals)))
            for i in active:
                for j in range(len(d)):
                    ov[j] += numpy.abs(inner(d.T[i], c.T[j], s))
            co = []
            for i in range(len(self.integral_manager.orbitals) - len(active)):
                idx = numpy.argmin(ov)
                co.append(idx)
                ov[idx] = 1 * len(active)
            co.sort()
            return co

        active = None
        if not self.integral_manager.active_space_is_trivial() and core is None:
            core = [i.idx_total for i in self.integral_manager.orbitals if i.idx is None]
        if "active" in kwargs:
            active = kwargs["active"]
            kwargs.pop("active")
            if core is None:
                core = get_core(active)
        else:
            if active is None:
                if core is None:
                    core = []
                    active = [i for i in range(len(self.integral_manager.orbitals))]
                else:
                    if isinstance(core, int):
                        core = [core]
                    active = get_active(core)
        assert len(active) + len(core) == len(self.integral_manager.orbitals)
        to_active = [i for i in range(len(self.integral_manager.orbitals)) if i not in core]
        to_active = {active[i]: to_active[i] for i in range(len(active))}
        if len(core):
            coeff = orthogonalize(c, d, s)
            if inplace:
                self.integral_manager = self.initialize_integral_manager(
                    one_body_integrals=self.integral_manager.one_body_integrals,
                    two_body_integrals=self.integral_manager.two_body_integrals,
                    constant_term=self.integral_manager.constant_term,
                    active_orbitals=[*to_active.values()],
                    reference_orbitals=[i.idx_total for i in self.integral_manager.reference_orbitals],
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
                    reference_orbitals=[i.idx_total for i in self.integral_manager.reference_orbitals],
                    frozen_orbitals=core,
                    orbital_coefficients=coeff,
                    overlap_integrals=s,
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
                parameters=self.parameters, integral_manager=integral_manager, fermionic_backend= self.fermionic_backend
            )
            return result

    def make_annihilation_op(self, orbital, coefficient=1.0)->OFFermionOperator:
        """
        Compute annihilation operator on spin-orbital
        Spin-orbital order is always (up,down,up,down,...)
        """
        assert orbital <= self.n_orbitals * 2
        return OFFermionOperator(f"{orbital}", coefficient)

    def make_creation_op(self, orbital, coefficient=1.0)->OFFermionOperator:
        """
        Compute creation operator on spin-orbital
        Spin-orbital order is always (up,down,up,down,...)
        """
        assert orbital <= self.n_orbitals * 2
        return OFFermionOperator(f"{orbital}^", coefficient)

    def make_number_op(self, orbital)->OFFermionOperator:
        """
        Compute number operator on spin-orbital
        Spin-orbital order is always (up,down,up,down,...)
        """
        return self.make_creation_op(orbital) * self.make_annihilation_op(orbital)

    def make_sz_op(self)->OFFermionOperator:
        """
        Compute the spin_z operator of the molecule
        """
        sz = OFFermionOperator()
        for i in range(0, self.n_orbitals * 2, 2):
            one = 0.5 * self.make_creation_op(i) * self.make_annihilation_op(i)
            two = 0.5 * self.make_creation_op(i + 1) * self.make_annihilation_op(i + 1)
            sz += one - two
        return sz

    def make_sp_op(self)->OFFermionOperator:
        """
        Compute the spin+ operator of the molecule
        """
        sp = OFFermionOperator
        for i in range(self.n_orbitals):
            sp += self.make_creation_op(i * 2) * self.make_annihilation_op(i * 2 + 1)
        return sp

    def make_sm_op(self)->OFFermionOperator:
        """
        Compute the spin- operator of the molecule
        """
        sm = OFFermionOperator()
        for i in range(self.n_orbitals):
            sm += self.make_creation_op(i * 2 + 1) * self.make_annihilation_op(i * 2)
        return sm

    def make_s2_op(self)->OFFermionOperator:
        """
        Compute the spin^2 operator of the molecule
        """
        s2_op = self.make_sm_op() * self.make_sp_op() + self.make_sz_op() * (self.make_sz_op() + 1)
        return s2_op

    def make_hamiltonian(self, *args, **kwargs) -> OFFermionOperator:
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
        return fop

    def make_hardcore_boson_hamiltonian(self, condensed=False)->OFFermionOperator:
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

        H = OFFermionOperator(term=c)
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                up = p
                uq = q
                H += h[p, q] * self.make_creation_op(up) * self.make_annihilation_op(uq) + g[p, q] * self.make_number_op(up) * self.make_number_op(uq)
        return H

    def prepare_reference(self, state=None, *args, **kwargs)->FCircuit:
        """
        Returns
        -------
        A tequila circuit object which prepares the reference of this molecule in the chosen transformation
        """
        if state is None:
            state = self._reference_state()
        d = {2*i:i for i in range(self.n_orbitals)}
        d.update({2*i+1:i+self.n_orbitals for i in range(self.n_orbitals)})
        n_state = [0]*len(state)
        for i in range(len(state)):
            n_state[d[i]] = state[i]
        n_state = prepare_product_state(BitString.from_array(n_state)) 
        U = FCircuit()
        U.initial_state = n_state
        # prevent trace out in direct wfn simulation
        U.n_qubits = self.n_orbitals * 2
        return U

    def prepare_hardcore_boson_reference(self):
        raise TequilaException('No HCB operations in Fermionic Backends')

    def hcb_to_me(self, U=None, condensed=False):
        raise TequilaException('No HCB operations in Fermionic Backends')

    def make_hardcore_boson_upccgd_layer(
        self, indices: list = "UpCCGD", label: str = None, assume_real: bool = True, *args, **kwargs
    ):
        raise TequilaException('No HCB operations in Fermionic Backends')

    def make_spa_ansatz(self, edges,use_units_of_pi=False, label=None, ladder=True)->FCircuit:
        """
        Separable Pair Ansatz (SPA) for general molecules
        see arxiv:
        edges: a list of tuples that contain the orbital indices for the specific pairs
               one example: edges=[(0,), (1,2,3), (4,5)] are three pairs, one with a single orbital [0], one with three orbitals [1,2,3] and one with two orbitals [4,5]
        use_units_of_pi: circuit angles in units of pi
        label: label the variables in the circuit
        ladder: if true the excitation pattern will be local. E.g. in the pair from orbitals (1,2,3) we will have the excitations 1->2 and 2->3, if set to false we will have standard coupled-cluster style excitations - in this case this would be 1->2 and 1->3
        """

        if edges is None:
            raise TequilaException(
                "SPA ansatz within a standard orbital basis needs edges. Please provide with the keyword edges.\nExample: edges=[(0,1,2),(3,4)] would correspond to two edges created from orbitals (0,1,2) and (3,4), note that orbitals can only be assigned to a single edge"
            )

        # sanity checks
        # current SPA implementation needs even number of electrons
        if self.n_electrons % 2 != 0:
            raise TequilaException(
                "need even number of electrons for SPA ansatz.\n{} active electrons".format(self.n_electrons)
            )
        # making sure that enough edges are assigned
        n_edges = len(edges)
        if len(edges) != self.n_electrons // 2:
            raise TequilaException(
                "number of edges need to be equal to number of active electrons//2\n{} edges given\n{} active electrons\nfrozen core is {}".format(
                    len(edges), self.n_electrons, self.parameters.frozen_core
                )
            )
        # making sure that orbitals are uniquely assigned to edges
        for edge_qubits in edges:
            for q1 in edge_qubits:
                for edge2 in edges:
                    if edge2 == edge_qubits:
                        continue
                    elif q1 in edge2:
                        raise TequilaException(
                            "make_spa_ansatz: faulty list of edges, orbitals are overlapping e.g. orbital {} is in edge {} and edge {}".format(
                                q1, edge_qubits, edge2
                            )
                        )

        # auto assign if the circuit construction is optimized
        # depending on the current qubit encoding (if hcb_to_me is implemnented we can optimize)
    
        return FCircuit.from_edges(edges=edges,use_units_of_pi=use_units_of_pi,label=label,n_orb=self.n_orbitals,ladder=ladder)

    def make_ansatz(self, name: str, *args, **kwargs)->FCircuit:
        """
        Automatically calls the right subroutines to construct ansatze implemented in tequila.chemistry
        name: namne of the ansatz, examples are: UpCCGSD, UpCCD, SPA, UCCSD, SPA+UpCCD, SPA+GS
        """
        name = name.lower()
        if name.strip() == "":
            return FCircuit()

        if "+" in name:
            U = FCircuit()
            subparts = name.split("+")
            U = self.make_ansatz(name=subparts[0], *args, **kwargs)
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
                    name=subpart, *args, label=(label, i), include_reference=False, hcb_optimization=False, **kwargs
                )
            return U

        if name == "uccsd":
            return self.make_uccsd_ansatz(*args, **kwargs)
        elif "spa" in name.lower():
            if "hcb" not in kwargs:
                hcb = False
                if "hcb" in name.lower():
                    raise TequilaException('No HCB operations in Fermionic Backends')
            return self.make_spa_ansatz(*args, **kwargs)
        elif "d" in name or "s" in name:
            return self.make_upccgsd_ansatz(name=name, *args, **kwargs)
        else:
            raise TequilaException("unknown ansatz with name={}".format(name))

    def make_upccgsd_ansatz(
        self,
        include_reference: bool = True,
        name: str = "UpCCGSD",
        label: str = None,
        order: int = None,
        spin_adapt_singles: bool = True,
        mix_sd: bool = False,
        *args,
        **kwargs,
    )->FCircuit:
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

        if order is None:
            try:
                if "-" in name:
                    order = int(name.split("-")[0])
                else:
                    order = 1
            except Exception:
                order = 1

        indices = self.make_upccgsd_indices(key=name)

        if "HCB" in name:
            raise TequilaException('No HCB operations in Fermionic Backends')
        
        # convenience
        S = "S" in name.upper()
        D = "D" in name.upper()

        # first layer
        U = FCircuit()
        if include_reference:
            U = self.prepare_reference()
        U += self.make_upccgsd_layer(
            include_singles=S,
            include_doubles=D,
            indices=indices,
            label=(label, 0),
            mix_sd=mix_sd,
            spin_adapt_singles=spin_adapt_singles,
            *args,
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
    )->FCircuit:
        U = FCircuit()
        for idx in indices:
            assert len(idx) == 1
            idx = idx[0]
            angle = (tuple([idx]), "D", label)
            if include_doubles:
                U += self.UC(i=idx[0],j=idx[1],angle=angle,**kwargs)
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
    )->FCircuit:
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
                U += self.UR(i=idx[0],j=idx[1],angle=angle)
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
        initial_amplitudes: typing.Union[str, Amplitudes, ClosedShellAmplitudes] = None,
        include_reference_ansatz=True,
        parametrized=True,
        threshold=1.0e-8,
        add_singles=None,
        screening=True,
        *args,
        **kwargs,
    ) -> FCircuit:
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
            Parametrized FCircuit

        """

        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2" and add_singles is None:
                add_singles = True
        elif initial_amplitudes is not None and add_singles is not None:
            warnings.warn(
                "make_uccsd_anstatz: add_singles has no effect when explicit amplitudes are passed down", TequilaWarning
            )
        elif add_singles is None:
            add_singles = True

        if self.n_electrons % 2 != 0:
            raise TequilaException("make_uccsd_ansatz currently only for closed shell systems")

        nocc = self.n_electrons // 2
        nvirt = self.n_orbitals - nocc

        U = FCircuit()
        if include_reference_ansatz:
            U.initial_state = self.prepare_reference().initial_state

        amplitudes = initial_amplitudes
        if hasattr(initial_amplitudes, "lower"):
            if initial_amplitudes.lower() == "mp2":
                amplitudes = self.compute_mp2_amplitudes()
            elif initial_amplitudes.lower() == "ccsd":
                amplitudes = self.compute_ccsd_amplitudes()
            else:
                try:
                    amplitudes = self.compute_amplitudes(method=initial_amplitudes.lower())
                except Exception as exc:
                    raise TequilaException(
                        "{}\nDon't know how to initialize '{}' amplitudes".format(exc, initial_amplitudes)
                    )
        if amplitudes is None:
            tia = None
            if add_singles:
                tia = numpy.zeros(shape=[nocc, nvirt])
            amplitudes = ClosedShellAmplitudes(tIjAb=numpy.zeros(shape=[nocc, nocc, nvirt, nvirt]), tIA=tia)
            screening = False

        closed_shell = isinstance(amplitudes, ClosedShellAmplitudes)
        indices = {}

        if not screening:
            threshold = 0.0

        if not isinstance(amplitudes, dict):
            amplitudes = amplitudes.make_parameter_dictionary(threshold=threshold, screening=screening)
            amplitudes = dict(sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True))
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
                        idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                        indices[idx_abab] = angle
                        if key[0] != key[2] and key[1] != key[3]:
                            idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                            idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                            partner = tuple([key[2], key[1], key[0], key[3]])
                            anglex = 2.0 * (t - amplitudes[partner])
                            if parametrized:
                                anglex = 2.0 * (Variable(name=key) - Variable(partner))
                            indices[idx_aaaa] = anglex
                            indices[idx_bbbb] = anglex
                else:
                    raise Exception("only closed-shell supported, please assemble yourself .... sorry :-)")

        factor = 1.0 / trotter_steps
        for step in range(trotter_steps):
            for idx, angle in indices.items():
                converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
                U += self.make_excitation_gate(indices=converted, angle=factor * angle)
        if (
            hasattr(initial_amplitudes, "lower")
            and initial_amplitudes.lower() == "mp2"
            and parametrized
            and add_singles
        ):
            # mp2 has no singles, need to initialize them here (if not parametrized initializling as 0.0 makes no sense though)
            U += self.make_upccgsd_layer(indices="upccsd", include_singles=True, include_doubles=False)
        return U

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
        U: FCircuit = None,
        variables: Variables = None,
        spin_free: bool = True,
        get_rdm1: bool = True,
        get_rdm2: bool = True,
        ordering="dirac",
        rdm_trafo: QubitHamiltonian = None,
        evaluate=True,
        backend:str=None,
        use_hcb:bool=False
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

        # Set up number of spin-orbitals and molecular orbitals respectively
        n_SOs = 2 * self.n_orbitals
        n_MOs = self.n_orbitals

        # Check whether unitary circuit is not 0
        if U is None:
            raise TequilaException("Need to specify a Quantum Circuit.")
        def _get_hcb_op(op_tuple):
            raise TequilaException('No HCB operations in Fermionic Backends')
        
        def _get_of_op(operator_tuple):
            """Returns operator given by a operator tuple as OpenFermion - Fermion operator"""
            op = openfermion.FermionOperator(operator_tuple)
            return op

        def _get_qop_hermitian(of_operator):
            """Returns Hermitian"""
            return 0.5*(of_operator + hermitian_conjugated(of_operator))

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
                    op_tuple = ((2 * p, 1), (2 * q, 1), (2 * s, 0), (2 * r, 0)) if (p != q and r != s) else "0.0 []"
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

        # Build operator lists
        qops = []
        if spin_free:
            qops += _build_1bdy_operators_spinfree() if get_rdm1 else []
            qops += _build_2bdy_operators_spinfree() if get_rdm2 else []
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
                evals = simulate(Braket(H=qops, U=U, shape=[len(qops)],backend=backend,molecule=self), variables=variables)
            else:
                qops = [rdm_trafo.dagger() * qops[i] * rdm_trafo for i in range(len(qops))]
                evals = simulate(Braket(H=qops, U=U, shape=[len(qops)],backend=backend,molecule=self), variables=variables)
        else:
            if rdm_trafo is None:
                evals = [Braket(H=x, U=U,backend=backend,molecule=self) for x in qops]
                N = n_MOs if spin_free else n_SOs
                rdm1 = QTensor(shape=[N, N])
                rdm2 = QTensor(shape=[N, N, N, N])
            else:
                raise TequilaException("compute_rdms: rdm_trafo was set but evaluate flag is False (not supported)")

        # Assemble density matrices
        # If self._rdm1, self._rdm2 exist, reset them if they are of the other spin-type
        def _reset_rdm(rdm):
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
            len_1 = n_MOs * (n_MOs + 1) // 2 if (spin_free) else n_SOs * (n_SOs + 1) // 2
        else:
            len_1 = 0
        evals_1, evals_2 = evals[:len_1], evals[len_1:]
        # Build matrices using the expectation values
        self._rdm1 = _assemble_rdm1(evals_1, rdm1=rdm1) if get_rdm1 else self._rdm1
        if spin_free:
            self._rdm2 = _assemble_rdm2_spinfree(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2
        else:
            self._rdm2 = _assemble_rdm2_spinful(evals_2, rdm2=rdm2) if get_rdm2 else self._rdm2

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
            warnings.warn("compute_rdms called with instruction to not compute?", TequilaWarning)

    def n_rotation(self, i, phi)->FCircuit:
        """
        Creates a quantum circuit that applies a phase rotation based on phi to both components (up and down) of a given qubit.

        Parameters:
        - i (int): The index of the qubit to which the rotation will be applied.
        - phi (float): The rotation angle. The actual rotation applied will be multiplied with -2 for both components.

        Returns:
        - FCircuit: A quantum circuit object containing the sequence of rotations applied to the up and down components of the specified qubit.
        """
        return __n_rotation(i=i,phi=phi)

    def get_givens_circuit(self, unitary, tol=1e-12, ordering=OPTIMIZED_ORDERING):
        """
        Constructs a quantum circuit from a given real unitary matrix using Givens rotations.

        This method decomposes a unitary matrix into a series of Givens and Phase rotations,
        then constructs and returns a quantum circuit that implements this sequence of rotations.

        Parameters:
        - unitary (numpy.array): A real unitary matrix representing the transformation to implement.
        - tol (float): A tolerance threshold below which matrix elements are considered zero.
        - ordering (list of tuples or 'Optimized'): Custom ordering of indices for Givens rotations or 'Optimized' to generate them automatically.

        Returns:
        - FCircuit: A quantum circuit implementing the series of rotations decomposed from the unitary.
        """
        return __get_givens_circuit(unitary=unitary,tol=tol,ordering=ordering)

    def __str__(self) -> str:
        result = str(type(self)) + "\n"
        result += "Parameters\n"
        for k, v in self.parameters.__dict__.items():
            result += "{key:15} : {value:15} \n".format(key=str(k), value=str(v))

        result += "{key:15} : {value:15} \n".format(key="n_qubits", value=str(self.n_orbitals * 2))
        result += "{key:15} : {value:15} \n".format(key="reference state", value=str(self._reference_state()))

        result += "\nBasis\n"
        result += str(self.integral_manager)
        result += "\nmore information with: self.print_basis_info()\n"

        return result
    
    def get_xyz(self)->str:
        geom = self.parameters.get_geometry()
        f = ''
        f += f'{len(geom)}\n'
        f += f'{self.parameters.name}\n'
        for at in geom:
            f += f'{at[0]} {at[1][0]} {at[1][1]} {at[1][2]}\n'
        return f

    def graph(self):
        return Graph.parse_xyz(self.get_xyz())
    
    def get_spa_edges(self,collapse:bool=True,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        return self.graph().get_spa_edges(collapse=collapse,strip_orbitals=strip_orbitals)
    
    def get_spa_guess(self,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        return self.graph().get_orbital_coefficient_matrix(strip_orbitals=strip_orbitals)

    def get_spa_edges_and_guess(self,collapse:bool=True,strip_orbitals:bool=None):
        if strip_orbitals  is None:
            strip_orbitals = not self.integral_manager.active_space_is_trivial()
        g = self.graph()
        return g.get_spa_edges(collapse=collapse,strip_orbitals=strip_orbitals),g.get_orbital_coefficient_matrix(strip_orbitals=strip_orbitals)
