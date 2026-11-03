"""Fermionic backend ``"tcc"``: simulation with TenCirChem in CI space.

``TCCBraket`` wraps a TCC ``EXPVAL`` object. The remaining functions convert
excitation indices and initial states between sunrise/tequila and TCC
conventions (TCC angles are ``-theta/2`` of the tequila values).
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from numbers import Number
from typing import Callable

import numpy as np
import tencirchem as tcc
import tensorcircuit as tc
from numpy import allclose, ceil, eye, pi, prod, zeros
from openfermion import FermionOperator
from openfermion.transforms.opconversions.term_reordering import reorder
from openfermion.utils.indexing import up_then_down
from tencirchem.static.ci_utils import get_ci_strings
from tencirchem.static.hamiltonian import get_h_fcifunc_from_integral
from tequila import (
    Objective,
    QubitHamiltonian,
    QubitWaveFunction,
    TequilaException,
    Variable,
    assign_variable,
    simulate,
)
from tequila.objective.objective import FixedVariable, Variables
from tequila.utils.bitstrings import BitNumbering, BitString

from sunrise.expval.tcc.braket import EXPVAL
from sunrise.expval.tcc.operator_builder import (
    build_sparse_ci_operator,
    extract_restricted_integrals,
)

from ...fermionic_operations.circuit import FCircuit
from ..fermionic_braket import FermBraketImpl


def _is_pyscf_molecule(molecule) -> bool:
    # imported here and not at the top: sunrise.molecules imports sunrise.expval while it loads
    """Return True if the molecule is backed by PySCF."""
    from sunrise.molecules.fermionic_base.pyscf_interface import (
        QuantumChemistryPySCF as FermionicPySCF,
    )
    from sunrise.molecules.hybrid_base.pyscf_interface import PyscfHybridBase
    from sunrise.molecules.qubit_base.pyscf_interface import (
        QuantumChemistryPySCF as QubitPySCF,
    )

    return isinstance(molecule, (QubitPySCF, FermionicPySCF, PyscfHybridBase))


class TCCBraket:
    """Expectation value or transition element evaluated with the ``"tcc"`` backend.

    Parameters
    ----------
    braket : FermBraketImpl
        Braket to evaluate. Its molecule is required.
    backend_kwargs : dict, optional
        Passed as keyword argument. Accepted keys are ``engine`` (one of
        "statevector", "tensornetwork", "civector", "civector-large", "pyscf";
        default "pyscf"), ``backend`` (tensorcircuit backend, default "numpy") and
        ``dtype``.
    name : str, optional
        Label used when printing. Passed as keyword argument.
    *args, **kwargs
        Ignored apart from the keywords above.

    Attributes
    ----------
    BK : EXPVAL
        Underlying TCC object holding integrals, excitation operators and
        parameters.

    References
    ----------
    W. Li et al., J. Chem. Theory Comput. 19, 3966 (2023).
    """

    def __init__(self, braket: FermBraketImpl, *args, **kwargs):
        self.operator = None
        if "backend_kwargs" in kwargs:
            backend_kwargs = kwargs["backend_kwargs"]
            kwargs.pop("backend_kwargs")
        else:
            backend_kwargs = {}
        if "engine" in backend_kwargs:
            engine = backend_kwargs["engine"]
            backend_kwargs.pop("engine")
        else:
            engine = "pyscf"
        if "backend" in backend_kwargs:
            tcc.set_backend(backend_kwargs["backend"])
            backend_kwargs.pop("backend")
        else:
            tcc.set_backend("numpy")
        if "dtype" in backend_kwargs:
            tcc.set_dtype(backend_kwargs["dtype"])
            backend_kwargs.pop("dtype")

        bra = braket.bra
        ket = braket.ket
        molecule = braket.molecule
        operator = braket.operator
        if bra is not None:
            bra = bra.to_upthendown(molecule)
        if ket is not None:
            ket = ket.to_upthendown(molecule)
        run_hf = (bra is None or bra.initial_state is None) and (
            ket is None or ket.initial_state is None
        )
        mo_coeff = molecule.integral_manager.orbital_coefficients
        if not allclose(mo_coeff, eye(len(mo_coeff))) and not _is_pyscf_molecule(
            molecule
        ):
            # the pyscf path below applies mo_coeff to pyscf's AO basis, which only matches pyscf-backed molecules
            # (psi4 e.g. orders the p orbitals differently), so other backends go through their MO integrals instead
            from sunrise.molecules import MoleculeFromSunrise

            molecule = MoleculeFromSunrise(molecule)
            mo_coeff = molecule.integral_manager.orbital_coefficients
        aslst = [i.idx_total for i in molecule.integral_manager.active_orbitals]
        active_space = (molecule.n_electrons, molecule.n_orbitals)
        if allclose(
            mo_coeff, eye(len(mo_coeff))
        ):  # idea when initialized the molecule from integrals, the mo_coeff are setted to identity and the provided integral are saved on the place of the Atomic Integrals
            e_core, int1e, int2e = molecule.get_integrals()
            int2e = int2e.reorder("c").elems
            self.BK: EXPVAL = EXPVAL.from_integral(
                int1e=int1e,
                int2e=int2e,
                n_elec=molecule.n_electrons,
                e_core=e_core,
                mo_coeff=mo_coeff,
                init_method="zeros",
                engine=engine,
                run_hf=run_hf,
                run_mp2=False,
                run_ccsd=False,
                run_fci=False,
                **backend_kwargs,
            )
        else:
            # imported here and not at the top: sunrise.molecules imports sunrise.expval while it loads
            from sunrise.molecules import MoleculeToPyscf

            molecule = MoleculeToPyscf(molecule)
            self.BK: EXPVAL = EXPVAL(
                mol=molecule,
                run_hf=run_hf,
                run_mp2=False,
                run_ccsd=False,
                run_fci=False,
                init_method="zeros",
                aslst=aslst,
                active_space=active_space,
                engine=engine,
                mo_coeff=mo_coeff,
                **backend_kwargs,
            )
        if ket is not None:
            self.ket = ket
        if bra is not None:
            self.bra = bra
        self.opt_res = defaultdict(None)
        if "name" in kwargs:
            self._name = kwargs["name"]
        else:
            self._name = (
                "Expectation Value" if braket.is_diagonal else "Transition Value"
            )
        if isinstance(operator, str) and operator == "I":
            self._name = "Transition Element"
        if operator is not None:
            self.operator = self.build_operator(operator)

    def __call__(self, variables: list | dict | None = None, *args, **kwargs) -> float:
        """Evaluate the braket, see ``simulate``."""
        if variables is None:
            variables = {}
        return self.simulate(variables=variables)

    def simulate(self, variables: list | dict | None = None) -> float:
        """Evaluate the braket for the given variable values.

        Parameters
        ----------
        variables : dict or Variables, optional
            Values of the circuit variables.

        Returns
        -------
        float
            Expectation value (or transition element).

        Raises
        ------
        TequilaException
            If a variable value is missing.
        """
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
        if variables is None:
            return self.BK.expval(hamiltonian=self.operator)
        return self.BK.expval(
            angles=[-0.5 * i for i in variables], hamiltonian=self.operator
        )

    def extract_variables(self) -> list:
        """Return the unique variables the braket depends on."""
        v = []
        for d in self.params:
            if isinstance(d, FixedVariable):
                continue
            elif hasattr(d, "extract_variables"):
                v.extend(d.extract_variables())
            else:
                v.extend(d)
        unique = []
        for i in v:
            if i not in unique:
                unique.append(i)
        return unique

    @property
    def energy(self) -> float:
        """float : energy of the last optimization result."""
        if isinstance(self.opt_res, dict):
            return self.opt_res["e"]
        else:
            return self.opt_res.e

    @property
    def bra(self):
        """list of list of tuple of int : excitation operators of the bra, in tequila format. Can be
        set from an FCircuit or from indices ``[[(0, 2), (1, 3)], ...]`` in up-then-down order.
        """
        return indices_tcc_to_tq(self.BK.ex_ops_bra)

    @bra.setter
    def bra(self, bra: list[list[tuple[int]]] | FCircuit | None):
        if isinstance(bra, FCircuit):
            if bra.initial_state is not None:
                ini_bra = init_state_from_wavefunction(bra.initial_state)
                for i in ini_bra:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in ini_bra:
                    init[i[0]] = i[1]
                self.init_state_bra = init
            self.variables_bra = bra.variables
            bra = bra.to_upthendown(len(self.BK.aslst))
            self.BK.ex_ops_bra, _, _ = indices_tq_to_tcc(bra.extract_indices())
        else:
            bra, params_bra, _ = indices_tq_to_tcc(bra)
            if self.variables_bra is None:
                self.variables_bra = params_bra
            self.BK.ex_ops_bra = bra

    @property
    def ket(self):
        """list of list of tuple of int : excitation operators of the ket, in tequila format. Can be
        set from an FCircuit or from indices ``[[(0, 2), (1, 3)], ...]`` in up-then-down order.
        """
        return indices_tcc_to_tq(self.BK.ex_ops_ket)

    @ket.setter
    def ket(self, ket: list[list[tuple[int]]] | FCircuit | None):
        if isinstance(ket, FCircuit):
            if ket.initial_state is not None:
                ini_ket = init_state_from_wavefunction(ket.initial_state)
                for i in ini_ket:
                    i[0] = self.BK.get_addr(i[0])
                init = [0] * self.BK.civector_size
                for i in ini_ket:
                    init[i[0]] = i[1]
                self.init_state_ket = init
            self.variables_ket = ket.variables
            ket = ket.to_upthendown(len(self.BK.aslst))
            self.BK.ex_ops_ket, _, _ = indices_tq_to_tcc(ket.extract_indices())
        else:
            ket, params_ket, _ = indices_tq_to_tcc(ket)
            if self.variables_ket is None:
                self.variables_ket = params_ket
            self.BK.ex_ops_ket = ket

    @property
    def variables_bra(self) -> dict:
        """dict : tequila variables of the bra circuit."""
        bar: dict = deepcopy(self.BK.var_to_param_bra)
        if bar is not None:
            for i, v in bar.items():  # Here to tequila
                if isinstance(v, (Number, FixedVariable)):
                    bar[i] = assign_variable(-2 * v)
        return bar

    @property
    def params_bra(self):
        """list : TCC parameters of the bra (variable names, or values for fixed angles)."""
        if self.variables_bra is None:
            return []
        d = {v: k for k, v in self.variables_bra.items()}
        return [
            assign_variable(-2 * v)
            if isinstance(v, (Number, FixedVariable))
            else d[assign_variable(v)]
            for v in self.BK.params_bra
        ]

    @variables_bra.setter
    def variables_bra(self, variables_bra):
        for idx, i in enumerate(variables_bra):
            if isinstance(assign_variable(i), (Number, FixedVariable)):
                variables_bra[idx] = assign_variable(-0.5 * (i % (2 * pi)))
        self.BK.params_bra = variables_bra

    @property
    def variables_ket(self):
        """dict : tequila variables of the ket circuit."""
        bar = deepcopy(self.BK.var_to_param_ket)
        if bar is not None:
            for i in bar:  # to tequila
                if isinstance(bar[i], (Number, FixedVariable)):
                    bar[i] = assign_variable(-2 * (bar[i]))
        return bar

    @property
    def params_ket(self):
        """list : TCC parameters of the ket (variable names, or values for fixed angles)."""
        d = {v: k for k, v in self.variables_ket.items()}
        return [
            assign_variable(-2 * v)
            if isinstance(v, (Number, FixedVariable))
            else d[assign_variable(v)]
            for v in self.BK.params_ket
        ]

    @variables_ket.setter
    def variables_ket(self, variables_ket):
        for idx, i in enumerate(variables_ket):
            if isinstance(assign_variable(i), (Number, FixedVariable)):
                variables_ket[idx] = assign_variable(-0.5 * (i % (2 * pi)))
        self.BK.params_ket = variables_ket

    @property
    def variables(self) -> dict:
        """dict : tequila variables of the circuits."""
        bar: dict = deepcopy(self.BK.var_to_param)
        if bar is not None:
            for i, v in bar.items():  # Here to tequila
                if isinstance(v, (Number, FixedVariable)):
                    bar[i] = assign_variable(-2 * v)
        return bar

    @property
    def params(self):
        """list : TCC parameters of bra and ket."""
        k = self.params_ket
        if self.BK.params_bra is None:
            return k
        return self.params_bra + k

    @variables.setter
    def variables(self, variables):
        for idx, i in enumerate(variables):
            if isinstance(assign_variable(i), (Number, FixedVariable)):
                variables[idx] = assign_variable(-0.5 * (i % (2 * pi)))
        self.BK.params = variables

    @property
    def init_state_bra(self):
        """Tensor : initial state of the bra before the excitations, usually Hartree-Fock.

        See ``BK.get_init_state_dataframe`` for a readable form.
        """
        return self.BK.init_state_bra

    @property
    def init_state_ket(self):
        """Tensor : initial state of the ket before the excitations, usually Hartree-Fock.

        See ``BK.get_init_state_dataframe`` for a readable form.
        """
        return self.BK.init_state_ket

    @property
    def init_state(self):
        """tuple : initial states of bra and ket. Setting it sets both."""
        return self.BK.init_state_bra, self.BK.init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self.BK.init_state_bra = init_state_bra

    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        self.BK.init_state_ket = init_state_ket

    @init_state.setter
    def init_state(self, init_state):
        self.BK.init_state_bra = init_state
        self.BK.init_state_ket = init_state

    @property
    def init_guess(self):
        """list of float : initial variable values for minimization in tequila units, default all 0."""
        return [-2 * i for i in self.BK.init_guess]

    @init_guess.setter
    def init_guess(self, init_guess):
        self.BK.init_guess = [-i / 2 for i in init_guess]

    @property
    def param_to_ex_ops_bra(self):
        """dict : maps each bra variable to the excitation operators using it."""
        if self.params_bra is None:
            return {}
        d = defaultdict(list)
        for i, j in enumerate(self.params_bra):
            if hasattr(j, "extract_variables"):
                for v in j.extract_variables():
                    d[v].append(self.bra[i])
            else:
                d[j].append(self.bra[i])
        return d

    @property
    def param_to_ex_ops_ket(self):
        """dict : maps each ket variable to the excitation operators using it."""
        d = defaultdict(list)
        for i, j in enumerate(self.params_ket):
            if hasattr(j, "extract_variables"):
                for v in j.extract_variables():
                    d[v].append(self.ket[i])
            else:
                d[j].append(self.ket[i])
        return d

    @property
    def param_to_ex_ops(self):
        """dict : maps each variable to the excitation operators using it."""
        db = self.param_to_ex_ops_bra if self.params_bra is not None else {}
        dk = self.param_to_ex_ops_ket
        db.update(dk)
        return db

    def __str__(self):
        """Return a readable description of bra, ket and their variables."""
        res = ""
        if self.is_diagonal:
            res += f"{self._name} with indices: {self.ket} with variables {self.params_ket}"
        else:
            res += (
                f"{self._name} with Bra= {self.bra} with variables {self.params_bra}\n"
            )
            res += f"{len(self._name) * ' '} with Ket= {self.ket} with variables {self.params_ket}"
        return res

    def __repr__(self):
        return self.__str__()

    @property
    def is_diagonal(self):
        """bool : True if bra and ket are the same, i.e. the braket is an expectation value."""
        return self.BK.is_diagonal()

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
        if self.BK.int1e is not None:
            mes += prod(self.BK.int1e.shape)
        if self.BK.int2e is not None:
            mes += prod(self.BK.int2e.shape)
        if mes:
            return mes
        else:
            return len(
                self.BK.civector(params=[0.0 for _ in range(self.BK.n_variables_ket)])
            )

    def build_operator(
        self, operator: str | FermionOperator | QubitHamiltonian | Number | None = None
    ) -> Callable | None:
        """Build the operator applied to the ket.

        A FermionOperator is applied directly in CI space: a spin-restricted one- or two-body
        operator (e.g. most physical Hamiltonians and observables) is contracted with PySCF's FCI
        machinery (``get_h_fcifunc_from_integral``); any other operator (higher-body terms,
        non-Hermitian excitation operators, ...) falls back to a Jordan-Wigner sparse matrix built
        once and restricted to the CI-string basis. Neither path ever builds a dense
        ``2**n_qubits`` qubit wavefunction.

        A QubitHamiltonian is accepted but discouraged, since TCC works on fermionic states: it is
        applied to the CI vector through a dense qubit wavefunction, keeping only the results that
        stay in the CI space, with real coefficients.

        Parameters
        ----------
        operator : str or FermionOperator or QubitHamiltonian or number, optional
            "I" (identity), "H" (molecular Hamiltonian, the default of the backend),
            a FermionOperator, a QubitHamiltonian or a constant.

        Returns
        -------
        callable or number or None
            Function mapping a CI vector to the operator applied to it, the constant itself for a
            number, or None for the Hamiltonian stored in ``BK``.

        Raises
        ------
        TequilaException
            If the operator is not supported.
        """

        def from_string(operator: str):
            """Set up ``BK`` for the string operator "I" or "H"."""
            if operator.upper() == "I":
                # identity as N/n_elec: exact on the fixed particle-number CI space
                nmo = len(self.BK.aslst)
                n_elec = int(np.sum(self.BK.n_elec))
                self.BK.hamiltonian = None
                self.BK.int1e = eye(nmo) / n_elec
                self.BK.int2e = zeros((nmo, nmo, nmo, nmo))
                self.BK.e_core = 0.0
                self.BK.hamiltonian_lib = {}
            elif operator.upper() == "H":
                pass
            else:
                raise TequilaException(
                    f"No operator str {operator} supported on TCC BraKet"
                )

        if operator is None:
            return None
        if isinstance(operator, str):
            if operator.upper() == "I":
                return lambda ket: ket
            from_string(operator)
            return None
        elif isinstance(operator, FermionOperator):
            n_orb = len(self.BK.aslst)
            operator = reorder(
                operator=operator, order_function=up_then_down, num_modes=2 * n_orb
            )
            self.BK.e_core = 0
            restricted = extract_restricted_integrals(operator, n_orb)
            if restricted is not None:
                int1e, int2e, constant = restricted
                fci_func = get_h_fcifunc_from_integral(int1e, int2e, self.BK.n_elec)
                if constant == 0.0:
                    return fci_func
                return lambda ket: fci_func(ket) + constant * ket
            ci_vec = get_ci_strings(
                n_elec_s=self.BK.n_elec, n_qubits=2 * n_orb, mode="fermion"
            )
            H_ci = build_sparse_ci_operator(
                operator, n_qubits=2 * n_orb, ci_strings=ci_vec
            )

            def sparse_callable(ket):
                ket_np = tc.backend.numpy(ket).astype(np.float64)
                hket = np.asarray(H_ci @ ket_np, dtype=np.float64)
                return tc.backend.convert_to_tensor(hket).astype(tc.rdtypestr)

            return sparse_callable
        elif isinstance(operator, QubitHamiltonian):
            self.BK.e_core = 0
        elif isinstance(operator, Number):
            from_string("I")
            return operator
        else:
            raise TequilaException(f"No operator {type(operator).__name__} supported")
        ci_vec = get_ci_strings(
            n_elec_s=self.BK.n_elec, n_qubits=2 * len(self.BK.aslst), mode="fermion"
        )
        self.BK.e_core = 0

        def callable_operator(
            ci_vect: list[int],
            operator: QubitHamiltonian,
            n_qubits: int,
            ket: list[float],
        ) -> list[float]:
            """Apply a QubitHamiltonian to a CI vector through a qubit wavefunction."""
            hket = zeros(len(ci_vect))
            wfv = QubitWaveFunction(
                n_qubits=n_qubits, numbering=BitNumbering.LSB, dense=True
            )
            for idx, val in enumerate(ket):
                wfv._state[ci_vect[idx]] = val
            hket = wfv.apply_qubitoperator(operator)
            hket = hket.to_array(out_numbering=BitNumbering.LSB)
            return [hket[i].real for i in ci_vect]

        return lambda ket: callable_operator(
            ci_vect=ci_vec, operator=operator, n_qubits=2 * len(self.BK.aslst), ket=ket
        )


def init_state_from_wavefunction(wvf: QubitWaveFunction):
    """Return the initial state as a list of ``[bitstring, amplitude]`` pairs, in TCC format."""
    if not isinstance(wvf._state, dict):
        return init_state_from_array(wvf=wvf)
    init_state = []
    for i in wvf._state:
        vec = bin(i)[2:].zfill(wvf.n_qubits)
        init_state.append(
            [vec, wvf._state[i].real]
        )  # tcc automatically does this, but with an anoying message everytime
    return init_state


def init_state_from_array(wvf: QubitWaveFunction, tol=1e-6):
    """Return a dense wavefunction in up-then-down order as ``[bitstring, amplitude]`` pairs, dropping amplitudes below ``tol``."""
    if isinstance(wvf._state, dict):
        return init_state_from_wavefunction(wvf)
    init_state = []
    nq = wvf.n_qubits
    nq = int(2 * (ceil(nq / 2)))  # if HCB may be odd amount of qubits
    for i, idx in enumerate(wvf._state):
        vec = BitString.from_int(i)
        vec.nbits = nq
        vec = vec.binary
        if abs(idx) > tol:
            init_state.append(
                [vec, idx.real]
            )  # tcc automatically does this, but with an anoying message everytime
    return init_state


def indices_tq_to_tcc(indices: list[list[tuple[int]]] | None = None):
    """Convert tequila excitation indices into TCC excitation operators.

    Parameters
    ----------
    indices : list of list of tuple of int, optional
        Excitations in up-then-down order, e.g. ``[[(0, 1), (n_mo, n_mo + 1)], ...]``.

    Returns
    -------
    tuple
        ``(ex_ops, params, param_ids)`` with ex_ops as ``[(..., n_mo, 0, 1, n_mo + 1, ...), ...]``,
        or three Nones if ``indices`` is None.
    """
    if indices is None:
        return None, None, None
    if not len(indices):
        return [], [], []
    assert isinstance(indices, (list, tuple))
    if isinstance(indices[0], Number):
        indices = [indices]
    ex_ops = []
    params = []
    param_ids = []
    for exct in indices:
        exc = []
        params.append(str(exct))
        param_ids.append(len(param_ids))
        for idx in exct:
            # (p, q) destroys p and creates q: tcc creates the first half of an ex_op, but applies it with the
            # opposite sign of the angle (angles = -0.5*variables), so p goes in the first half
            exc.append(idx[1])
            exc.insert(0, idx[0])
        ex_ops.append(tuple(exc))
    return ex_ops, params, param_ids


def indices_tcc_to_tq(
    indices: list[list[tuple[int]]] | None = None,
) -> list[list[tuple[int]]]:
    """Convert TCC excitation operators into tequila excitation indices.

    Inverse of ``indices_tq_to_tcc``.

    Parameters
    ----------
    indices : list of tuple of int, optional
        TCC format, e.g. ``[(..., n_mo, 0, 1, n_mo + 1, ...), ...]``.

    Returns
    -------
    list of list of tuple of int or None
        Excitations in up-then-down order, e.g. ``[[(0, 1), (n_mo, n_mo + 1)], ...]``.
    """
    if indices is None:
        return None
    assert isinstance(indices, (list, tuple))
    if not len(indices):
        return []
    if isinstance(indices[0], Number):
        indices = [indices]
    ex_ops = []
    for exct in indices:
        exc = []
        for i in range(len(exct) // 2):
            exc.append((exct[i], exct[-i - 1]))
        ex_ops.append(exc[::-1])
    return ex_ops


def map_variables(x: list[Variable, Objective], dvariables: dict):
    """Return the value of ``x``, evaluating it with ``dvariables`` if it is a Variable or Objective."""
    if isinstance(x, Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x, Objective):
        x = simulate(x, dvariables)
    return x
