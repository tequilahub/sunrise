"""Fermionic backend ``"fqe"``: simulation with OpenFermion-FQE on number- and spin-conserving
wavefunctions.
"""

from __future__ import annotations

from ctypes import ArgumentError

import fqe
import tequila as tq
from fqe.hamiltonians.diagonal_coulomb import DiagonalCoulomb
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from numpy import ndarray
from tequila import QubitWaveFunction, TequilaException

from sunrise.fermionic_operations.circuit import FCircuit
from sunrise.utils.fermionic_utils import *

from ..fermionic_braket import FermBraketImpl


class FQEBraKet:
    """Expectation value or transition element evaluated with the ``"fqe"`` backend.

    Circuits are converted to products of fermionic excitation generators that
    are applied by time evolution to an FQE wavefunction (Hartree-Fock unless the
    circuit defines an initial state).

    Parameters
    ----------
    braket : FermBraketImpl
        Braket to evaluate. Its molecule is required.
    *args, **kwargs
        Ignored.

    Raises
    ------
    TequilaException
        If the operator is not "H"/"hamiltonian", "I"/"identity", a
        FermionOperator or a float.

    References
    ----------
    N. C. Rubin et al., Quantum 5, 568 (2021).
    """

    def __init__(self, braket: FermBraketImpl, *args, **kwargs):
        self.ket_instructions = None
        self.ket_angles = None  # all variables including function objects
        self.ket_extract_variables_names = None  # only unique variables
        self.ket_original_obj = None
        self.ket_generator = None
        self.bra_instructions = None
        self.bra_angles = None
        self.bra_extract_variables_names = None
        self.bra_original_obj = None
        self.bra_generator = None
        self.fermion_operator = None

        ket = braket.ket
        bra = braket.bra
        mol = braket.molecule
        operator = braket.operator
        self.n_ele = mol.n_electrons
        self.n_orbitals = mol.n_orbitals

        if isinstance(operator, str):
            if operator.lower() == "h" or operator.lower() == "hamiltonian":
                c, h, g = mol.get_integrals()
                h_of = make_fermionic_hamiltonian(
                    one_body_integrals=h, two_body_integrals=g.elems, constant=c
                )
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(
                    h_of, norb=self.n_orbitals
                )
            elif operator.lower() == "hcb":
                # the hardcore-boson Hamiltonian has a 4-body term, which fqe stores as a dense
                # (2*norb)**8 tensor: the restricted 2-body integrals are equivalent on seniority-zero states
                c, h, g = mol.get_integrals(ordering="chem")
                h, g.elems = hcb_restricted_integrals(h, g.elems)
                h_of = make_fermionic_hamiltonian(
                    one_body_integrals=h,
                    two_body_integrals=g.reorder("of").elems,
                    constant=c,
                )
                self.h_fqe = fqe.get_hamiltonian_from_openfermion(
                    h_of, norb=self.n_orbitals
                )
            elif operator.lower() == "i" or operator.lower() == "identity":
                self.h_fqe = None
            else:
                raise TequilaException(f"Not implemented operator {operator}")
        elif isinstance(
            operator, openfermion.ops.operators.fermion_operator.FermionOperator
        ):
            self.h_fqe = fqe.get_hamiltonian_from_openfermion(
                operator, norb=mol.n_orbitals
            )
            self.fermion_operator = operator
        elif isinstance(operator, float):
            operator = openfermion.ops.FermionOperator(term=None, coefficient=operator)
            self.h_fqe = fqe.get_hamiltonian_from_openfermion(
                operator, norb=mol.n_orbitals
            )
        else:
            raise TequilaException(f"Not recognized format {type(operator)}")

        bin_dict = generate_of_binary_dict(self.n_orbitals, self.n_ele // 2)

        # initalize ket properties
        self.ket = ket.to_udud(norb=self.n_orbitals)
        self.ket_wfn = fqe.Wavefunction(
            param=[[self.n_ele, 0, self.n_orbitals]]
        )  # probably only works for H
        if ket.initial_state is None:
            self.ket_wfn.set_wfn(strategy="hartree-fock")
        else:
            set_init_state(
                wfn=self.ket_wfn,
                n_ele=self.n_ele,
                n_orb=self.n_orbitals,
                init_state=ket.initial_state,
                bin_dict=bin_dict,
            )

        self.bra = bra.to_udud(norb=self.n_orbitals)
        self.bra_wfn = fqe.Wavefunction(param=[[self.n_ele, 0, self.n_orbitals]])
        if bra is not None and bra.initial_state is None:
            self.bra_wfn.set_wfn(strategy="hartree-fock")
        else:
            set_init_state(
                wfn=self.bra_wfn,
                n_ele=self.n_ele,
                n_orb=self.n_orbitals,
                init_state=bra.initial_state,
                bin_dict=bin_dict,
            )

        self.ket_time_evolved = None
        self.bra_time_evolved = None

    def __call__(self, variables: dict | list | None = None, *args, **kwargs) -> float:
        """Evaluate the braket for the given variable values.

        Parameters
        ----------
        variables : dict or list or Variables, optional
            Values of the circuit variables used in the time evolution.

        Returns
        -------
        float
            ``<bra|H|ket>``, or ``<bra|ket>`` for the identity operator.

        Raises
        ------
        TequilaException
            If a variable value is missing.
        """
        if variables is None:
            variables = {}
        if self.is_diagonal:
            parameters = self.ket_extract_variables_names
        else:
            parameters = (
                self.ket_extract_variables_names + self.bra_extract_variables_names
            )
        if isinstance(variables, list):
            assert len(variables) == len(self.extract_variables())
            variables = {
                parameters[i]: variables[i] for i in range(len(internal_variables))
            }
        check_variables = {k: k in variables for k in self.extract_variables()}
        if not all(list(check_variables.values())):
            raise TequilaException(
                "Objective did not receive all variables:\n"
                "You gave\n"
                f" {variables}\n"
                " but the objective depends on\n"
                f" {self.extract_variables()}\n"
                " missing values for\n"
                f" {[k for k, v in check_variables.items() if not v]}"
            )
        internal_variables = deepcopy(variables)
        internal_variables = tq.format_variable_dictionary(internal_variables)

        angle_internal_ket = [x(internal_variables) for x in self.ket_angles]
        angle_internal_bra = []
        if not self.is_diagonal:
            angle_internal_bra = [x(internal_variables) for x in self.bra_angles]

        list_gen_vals_ket = []
        for gens in self.ket_generator.values():
            if len(gens) == 1:
                list_gen_vals_ket.append(gens)
            else:
                for URs in gens:
                    list_gen_vals_ket.append([URs])
        zip_ket = zip(angle_internal_ket, list_gen_vals_ket)

        ket_t = deepcopy(self.ket_wfn)
        for arguments in zip_ket:
            for generators in arguments[1]:
                ket_t = ket_t.time_evolve(-0.5 * arguments[0], generators)

        if self.is_diagonal:
            bra_t = None
        else:
            bra_t = deepcopy(self.bra_wfn)
            list_gen_vals_bra = []
            for gens in self.bra_generator.values():
                if len(gens) == 1:
                    list_gen_vals_bra.append(gens)
                else:
                    for URs in gens:
                        list_gen_vals_bra.append([URs])
            zip_bra = zip(angle_internal_bra, list_gen_vals_bra)

            for arguments in zip_bra:
                for generators in arguments[1]:
                    bra_t = bra_t.time_evolve(-0.5 * arguments[0], generators)

        self.ket_time_evolved = ket_t
        self.bra_time_evolved = bra_t
        if self.h_fqe is not None:
            try:
                result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)
            except ArgumentError:
                # fqe crashes evaluating some DiagonalCoulomb operators (e.g. RDM elements n_p n_q), see
                # https://github.com/quantumlib/OpenFermion-FQE/issues/142; as a SparseHamiltonian it works
                if (
                    not isinstance(self.h_fqe, DiagonalCoulomb)
                    or self.fermion_operator is None
                ):
                    raise
                self.h_fqe = SparseHamiltonian(self.fermion_operator)
                result = fqe.expectationValue(wfn=ket_t, ops=self.h_fqe, brawfn=bra_t)
        else:
            result = fqe.dot(bra_t, ket_t)
        return result.real

    def print_ket(self):
        """Print the initial ket wavefunction."""
        self.ket_wfn.print_wfn()

    def print_bra(self):
        """Print the bra."""
        self.bra.print_wfn()

    def print_ket_time_evolved(self):
        """Print the ket after the last evaluation."""
        self.ket_time_evolved.print_wfn()

    def print_bra_time_evolved(self):
        """Print the bra after the last evaluation."""
        self.bra_time_evolved.print_wfn()

    def print_ket_generator(self):
        """Print the excitation generators of the ket."""
        print(self.ket_generator)

    def print_bra_generator(self):
        """Print the excitation generators of the bra."""
        if self.bra_instructions is not None:
            raise ValueError("No bra circuit provided")
        else:
            print(self.bra_generator)

    def extract_ket_variables(self):
        """Return the unique variables of the ket."""
        return self.ket_extract_variables_names

    def extract_bra_variables(self):
        """Return the unique variables of the bra."""
        return self.bra_extract_variables_names

    def extract_variables(self):
        """Return the variables of ket and bra."""
        ket_v = self.extract_ket_variables()

        if self.bra_instructions is not None:
            ket_v += self.extract_bra_variables()
        return ket_v

    @property
    def ket(self) -> FCircuit:
        """FCircuit : circuit preparing the ket state. Setting it builds the excitation generators."""
        return self.ket_original_obj

    @ket.setter
    def ket(self, ket: FCircuit):
        assert isinstance(ket, FCircuit)
        self.ket_instructions = ket.extract_indices()
        self.ket_angles = ket.variables  # all variables including function objects
        self.ket_extract_variables_names = (
            ket.extract_variables()
        )  # only unique variables
        self.ket_original_obj = ket
        self.ket_generator = create_fermionic_generators(
            self.ket_instructions, self.ket_angles
        )

    @property
    def bra(self) -> FCircuit:
        """FCircuit : circuit preparing the bra state. Setting it builds the excitation generators."""
        return self.bra_original_obj

    @bra.setter
    def bra(self, bra: FCircuit):
        if bra is None:
            return
        assert isinstance(bra, FCircuit)
        self.bra_instructions = bra.extract_indices()
        self.bra_angles = bra.variables  # all variables including function objects
        self.bra_extract_variables_names = (
            bra.extract_variables()
        )  # only unique variables
        self.bra_original_obj = bra
        self.bra_generator = create_fermionic_generators(
            self.bra_instructions, self.bra_angles
        )

    @property
    def constant_dict_ket(self):
        """dict : constants of the ket circuit."""
        return self._constant_dict_ket

    @constant_dict_ket.setter
    def constant_dict_ket(self, constant_dict_ket):
        self._constant_dict_ket = constant_dict_ket

    @property
    def U(self):
        """FCircuit : the ket circuit. Needed by tequila Objectives."""
        return self.ket_original_obj

    def count_measurements(self) -> int:
        """Return a cost measure of the braket: the dimension of the Hamiltonian, or 0 for the
        identity.
        """
        mes = 0
        if self.h_fqe is not None:
            mes = self.h_fqe.dim()
        return mes

    def __str__(self):
        res = ""
        if self.is_diagonal:
            res += f"FQE Expectation Value with indices: {self.ket_instructions} with variables {self.ket_angles}"
        else:
            res += f"FQE Braket with Bra= {self.bra_instructions} with variables {self.bra_angles}\n"
            res += f"           with Ket= {self.ket_instructions} with variables {self.ket_angles}"
        return res

    def __repr__(self):
        return self.__str__()

    @property
    def is_diagonal(self):
        """bool : True if no separate bra is set, i.e. the braket is an expectation value."""
        return self.bra is None


def set_init_state(
    wfn: fqe.Wavefunction,
    n_ele,
    n_orb,
    init_state: list[tuple[str, int] | QubitWaveFunction | ndarray] | QubitWaveFunction,
    bin_dict: dict,
) -> None:
    """Set the initial state of an FQE wavefunction in place and normalize it.

    Parameters
    ----------
    wfn : fqe.Wavefunction
        Wavefunction to modify.
    n_ele : int
        Number of electrons.
    n_orb : int
        Number of spatial orbitals.
    init_state : tuple of (str, float) or QubitWaveFunction or numpy.ndarray
        Initial state, as (bitstring, coefficient) pairs, a wavefunction or a
        coefficient array.
    bin_dict : dict
        Maps bitstrings to indices, see ``generate_of_binary_dict``.

    Raises
    ------
    TequilaException
        If the state has the wrong length or number of ones, or has an unknown type.
    """
    coeff = wfn.get_coeff((n_ele, 0))

    if isinstance(init_state, tuple):
        for state in init_state:
            if len(state[0]) != n_ele:
                raise TequilaException("initial state is to long")
            n_ones = 0
            for binary in state[0]:
                if binary == "1":
                    n_ones += 1
            if n_ones != n_ele // 2:
                raise TequilaException(
                    "initial state has to many ones for the reordrerd JW"
                )

        for state in init_state:
            i = bin_dict[state[0]]
            coeff[i][i] += state[1]
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, QubitWaveFunction):
        indices, values = init_state_from_wavefunction(
            wvf=init_state, n_orb=n_orb, bin_dict=bin_dict
        )
        for i, index in enumerate(indices):
            coeff[index][index] = values[i]

        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): coeff})
        wfn.normalize()

    elif isinstance(init_state, np.ndarray):
        wfn.set_wfn(strategy="from_data", raw_data={(n_ele, 0): init_state[0]})
        wfn.normalize()

    else:
        raise TequilaException(f"unkown intitial state type {type(init_state[0])}")


def init_state_from_wavefunction(wvf: QubitWaveFunction, n_orb: int, bin_dict: dict):
    """Return indices (via ``bin_dict``) and amplitudes of the non-negligible components of ``wvf``.

    Parameters
    ----------
    wvf : QubitWaveFunction
        Wavefunction to convert.
    n_orb : int
        Number of spatial orbitals; shorter bitstrings are padded.
    bin_dict : dict
        Map from bitstrings to FQE indices.
    """
    indices = []
    values = []
    for idx, i in wvf.items():
        if abs(i) > 1e-6:
            vec = idx.binary
            if len(vec) < n_orb:
                vec = "0" * (n_orb - len(vec)) + vec
            if len(vec) > n_orb:
                vec = vec[:n_orb]
            vec = vec[::-1]
            # vec = vec[len(vec)//2:]
            indices.append(bin_dict[vec])
            values.append(abs(i))  # todo not sure about this

    return indices, values
