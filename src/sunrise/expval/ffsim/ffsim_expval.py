from __future__ import annotations

from copy import deepcopy
from typing import Callable

import ffsim
from ffsim import FermionOperator
from numpy import (
    array,
    complex128,
    ndarray,
    prod,
)
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
from tequila.utils.bitstrings import BitNumbering

from sunrise.expval import FermBraketImpl
from sunrise.molecules.qubit_base import qc_base

from ..fermionic_operations.circuit import FCircuit


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
            spin = (molecule.parameters.multiplicity - 1) / 2
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
            self.init_state = ffsim.hartree_fock_state(
                self.molecule.norb, self.molecule.nelec
            )

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
        if isinstance(variables, Variables):
            variables = variables.store
        if isinstance(variables, dict):
            v: dict = deepcopy(self.variables)
            if v is None:
                v = variables
            else:
                v.update(variables)
            tvars: list = deepcopy(self.BK.total_variables)
            variables: list = [map_variables(x, v) for x in tvars]
        # FFSim doesnt suport custom operations
        # if variables is None:
        #     if self.is_diagonal:
        #         return real(
        #             vdot(self.init_state_ket, self.operator @ self.init_state_ket)
        #         )
        #     else:
        #         return real(
        #             vdot(self.init_state_bra, self.operator @ self.init_state_ket)
        #         )

        # return real(vdot())

    def extract_variables(self) -> list[Variable]:
        """
        Extract all variables on which the objective depends
        :return: List of all Variables
        """
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
            res += f"{self._name} with indices: {self.ket.extract_indices()} with variables {self.params_ket}"
        else:
            res += f"{self._name} with Bra= {self.bra.extract_indices()} with variables {self.params_bra}\n"
            res += f"{len(self._name) * ' '} with Ket= {self.ket.extract_indices()} with variables {self.params_ket}"
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
            mes += prod(self.molecule.one_body_integrals)
        if self.BK.int2e is not None:
            mes += prod(self.molecule.two_body_integrals)
        if mes:
            return mes
        else:
            return len(
                ffsim.hartree_fock_state(self.molecule.norb, self.molecule.nelec)
            )

    def __civect_to_qwvf(self, civect: ndarray) -> QubitWaveFunction:
        wvf = QubitWaveFunction(n_qubits=2 * self.molecule.norb).to_array()
        for i, coeff in enumerate(self._cistrings):
            wvf[i] = coeff
        return QubitWaveFunction.from_array(array=wvf, numbering=BitNumbering.LSB)

    def __qwvf_to_civect(self, wvf: QubitWaveFunction) -> ndarray:
        wvf.n_qubits = 2 * self.molecule.norb
        wvf = wvf.to_array(out_numbering=BitNumbering.LSB)
        return array([wvf[i] for i in self._cistrings])

    def build_operator(
        self, operator: str | FermionOperator | QubitHamiltonian = None
    ) -> None | Callable | ffsim.FermionOperator:
        """
        Build the expectation value operator.
        Even if it is accepted a QubitHamiltonian, we disencorage its use here for fermionic states limitations.
        It will be applied to the ci_vector and kept these results which doesn't leave the ci_vector.
        """

        def from_string(operator: str) -> ffsim.FermionOperator | LinearOperator:
            if operator.upper() == "I":
                return LinearOperator(
                    shape=(len(self._cistrings), len(self._cistrings)),
                    matvec=lambda x: x,
                    rmatvec=lambda x: x,
                    dtype=complex128,
                )
            elif operator.upper() == "H":
                op = ffsim.fermion_operator(self.molecule.hamiltonian)
                op.simplify()
                return op
            else:
                raise TequilaException(
                    f"No operator str {operator} supported on FFSim BraKet"
                )

        if isinstance(operator, str):
            operator = from_string(operator)
        elif isinstance(operator, FermionOperator):
            pass
        elif isinstance(operator, QubitHamiltonian):

            def f(v):
                wvf = self.__civect_to_qwvf(v)
                wvf.apply_qubitoperator(operator)
                return self.__qwvf_to_civect(wvf)

            if operator.is_hermitian():
                operator = LinearOperator(
                    shape=(len(self._cistrings), len(self._cistrings)),
                    matvec=f,
                    rmatvec=f,
                    n_qubits=2 * self.molecule.norb,
                )
            else:
                operator = LinearOperator(
                    shape=(len(self._cistrings), len(self._cistrings)),
                    matvec=f,
                    n_qubits=2 * self.molecule.norb,
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
