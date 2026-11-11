"""Small application that wraps a tequila VQE object and passes it to the PySCF CASSCF solver.
This way we can conveniently optimize orbitals.
This basically does what is described here in the context of orbital-optimized unitary coupled-cluster (oo-ucc): https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.033421
Any circuit can be used instead of UCC, and the VQE itself can be replaced by a custom ``vqe_solver``.

The Interface with the PySCF module follows the original PySCF article  https://arxiv.org/abs/2002.12531 (see Fig.3)

Currently this is a beta version (not extensively used in real life), so be careful when using it and please report issues on github :-)

References
----------
T. Mizukami et al., Phys. Rev. Research 2, 033421 (2020), doi:10.1103/PhysRevResearch.2.033421.
Q. Sun et al., J. Chem. Phys. 153, 024109 (2020), arXiv:2002.12531.
"""

from __future__ import annotations

import copy
import typing
import warnings
from dataclasses import dataclass, field

import numpy
from tequila import ExpectationValue, QCircuit, TequilaException, TequilaWarning

from sunrise.expval import INSTALLED_FERMIONIC_BACKENDS, Braket, pick_backend
from sunrise.expval.minimize import minimize
from sunrise.fermionic_operations.circuit import FCircuit
from sunrise.molecules import MoleculeFromSunrise
from sunrise.molecules.fermionic_base.fer_base import FermionicBase
from sunrise.molecules.hybrid_base.hybridbase import HybridBase

from ..molecules.qubit_base.qc_base import (
    NBodyTensor,
    QuantumChemistryBase,
)


@dataclass
class OptimizeOrbitalsResult:
    """Result of ``optimize_orbitals``.

    Instances are also used as the CASSCF callback, which stores the iteration
    count and the last PySCF local data.

    Attributes
    ----------
    old_molecule : QuantumChemistryBase
        Molecule before the optimization.
    molecule : QuantumChemistryBase
        Molecule with the optimized orbitals.
    mcscf_object : object, optional
        PySCF MCSCF object, only set if ``return_mcscf=True``.
    mcscf_local_data : dict, optional
        Local variables of the last CASSCF iteration.
    mo_coeff : numpy.ndarray
        Optimized molecular orbital coefficients.
    energy : float
        Optimized total energy.
    U : FCircuit or QCircuit
        Circuit used in the last VQE run.
    variables : dict
        Optimized circuit variables of the last VQE run.
    iterations : int
        Number of CASSCF iterations.
    """

    old_molecule: QuantumChemistryBase = None  # the old sunrise molecule
    molecule: QuantumChemistryBase = (
        None  # the new sunrise molecule with transformed orbitals
    )
    mcscf_object: object | None = None  # the pyscf mcscf object
    mcscf_local_data: dict | None = None
    mo_coeff = None  # the optimized mo coefficients
    energy: float | None = None  # the optimized energy
    U: FCircuit | QCircuit | None = None  # the circuit used in the last VQE run
    variables: dict | None = (
        None  # the optimized circuit variables from the last VQE run
    )
    iterations: int = 0

    def __call__(self, local_data, *args, **kwargs):
        # use as callback
        """Store the PySCF local data and count the iteration (CASSCF callback)."""
        if "u" in local_data:
            self.rotation_matrix = copy.deepcopy(local_data["u"])
        self.mcscf_local_data = local_data
        self.iterations += 1


def optimize_orbitals(
    molecule,
    circuit=None,
    vqe_solver=None,
    pyscf_arguments=None,
    silent=False,
    vqe_solver_arguments=None,
    initial_guess=None,
    return_mcscf=False,
    use_hcb=False,
    molecule_factory=None,
    molecule_arguments=None,
    restrict_to_active_space=True,
    read_chkfile: str | None = None,
    save_chkfile: str | None = None,
    *args,
    **kwargs,
) -> OptimizeOrbitalsResult:
    """Optimize the orbitals of a molecule with a VQE and PySCF's CASSCF.

    Parameters
    ----------
    molecule : QuantumChemistryBase or FermionicBase or HybridBase
        Molecule whose orbitals are optimized.
    circuit : FCircuit or QCircuit, optional
        Ansatz of the VQE. Can be None if ``vqe_solver`` builds its own circuit.
    vqe_solver : callable, optional
        Custom solver, called as
        ``vqe_solver(H=H, circuit=circuit, molecule=molecule, **vqe_solver_arguments)``
        and returning a tequila optimization result. If it has a
        ``compute_rdms(U, variables, molecule, use_hcb)`` method, the RDMs are computed with it,
        otherwise with ``molecule.compute_rdms`` (see ``SPASolver`` in
        ``spafastprototype/spa.py``). By default the energy of ``circuit`` is minimized.
    pyscf_arguments : dict, optional
        Attributes of the PySCF MCSCF object. Default is
        ``{"max_cycle_macro": 10, "max_cycle_micro": 3}``.
    silent : bool
        If True, suppress printout. Default is False.
    vqe_solver_arguments : dict, optional
        Arguments for the VQE solver. The default solver accepts
        ``"optimizer_arguments"`` (kwargs of ``minimize``) and ``"restrict_to_hcb"``
        (use the hard-core boson Hamiltonian).
    initial_guess : numpy.ndarray or str, optional
        Initial orbital rotation matrix for PySCF's MCSCF. Default is the identity.
        "random" (width 1) and "near_zero" (width 1e-3) add normal noise to the identity;
        "random_loc=X_scale=Y" sets mean X and width Y of the noise.
    return_mcscf : bool
        If True, store the PySCF MCSCF object in the result.
    use_hcb : bool
        If True, the circuit is in hard-core boson encoding.
    molecule_factory : callable, optional
        Constructor for the molecules built during the optimization.
    molecule_arguments : dict, optional
        Arguments for ``molecule_factory``. Only change if you know what you are doing.
    restrict_to_active_space : bool
        Default is True.
    read_chkfile : str, optional
        PySCF checkpoint file from which the initial orbitals are read.
    save_chkfile : str, optional
        PySCF checkpoint file in which intermediate orbitals are saved.
    *args, **kwargs
        Passed on to the optimizer. For fermionic molecules ``fermionic_backend``
        and ``backend`` select the simulators.
        ``fbackend`` is accepted as alias of ``fermionic_backend``.

    Returns
    -------
    OptimizeOrbitalsResult
        Optimized molecule, orbital coefficients, energy, circuit and variables.

    See Also
    --------
    sunrise.minimize : Optimize only the circuit variables.
    sunrise.spafastprototype.run_spa : SPA with orbital optimization, using ``SPASolver`` as ``vqe_solver``.
    sunrise.graph_selection.run_spa_optimization : Choose the SPA edges from the geometry, then optimize.

    Examples
    --------
    >>> U = mol.make_ansatz(name="SPA", edges=edges)
    >>> opt = sun.optimize_orbitals(molecule=mol, circuit=U, silent=True)
    >>> mol = opt.molecule
    """

    class solver:
        def __init__(
            self,
            fermionic_backend: str | None = None,
            backend: str | None = None,
            circuit: FCircuit | None = None,
        ):
            self.fermionic_backend = fermionic_backend
            self.backend = backend
            self.U = circuit

        def __call__(self, H, circuit, molecule, **vqe_solver_arguments):
            return minimize(
                Braket(molecule=molecule, ket=circuit, operator="H"),
                fermionic_backend=self.fermionic_backend,
                backend=self.backend,
                **vqe_solver_arguments,
            )

    if isinstance(molecule, HybridBase):
        use_hcb = False
        if molecule_arguments is None:
            molecule_arguments = {
                "select": molecule.select,
                "condense": molecule.condense,
                "two_qubit": molecule.two_qubit,
                "integral_tresh": molecule.integral_tresh,
                "parameters": molecule.parameters,
                "transformation": molecule.transformation,
                "backend": "pyscf",
            }
        else:
            mol_args = {
                "select": molecule.select,
                "condense": molecule.condense,
                "two_qubit": molecule.two_qubit,
                "integral_tresh": molecule.integral_tresh,
                "parameters": molecule.parameters,
                "transformation": molecule.transformation,
                "backend": "pyscf",
            }
            mol_args.update(molecule_arguments)
            molecule_arguments = mol_args
            if isinstance(circuit, FCircuit):
                circuit = circuit.to_qcircuit(molecule=molecule)
        if molecule_factory is None:
            molecule_factory = HybridBase
    elif isinstance(molecule, FermionicBase):
        if isinstance(circuit, QCircuit):
            molecule = MoleculeFromSunrise(
                mol=molecule, transformation="reordered-jordan-wigner"
            )
            warnings.warn(
                "Qcircuit detected but Fermionic Molecule provided. Converting the molecule to qubit_base with transformation = reordered-jordan-wigner"
            )
        else:
            if "fermionic_backend" in kwargs:
                fbackend = kwargs["fermionic_backend"]
                kwargs.pop("fermionic_backend")
            elif "fbackend" in kwargs:
                fbackend = kwargs["fbackend"]
                kwargs.pop("fbackend")
            elif (
                "backend" in kwargs
                and kwargs["backend"] in INSTALLED_FERMIONIC_BACKENDS
            ):
                # backward compatible: `backend` used to select the fermionic backend
                fbackend = kwargs["backend"]
                kwargs.pop("backend")
            else:
                fbackend = pick_backend()
            if "backend" in kwargs:
                backend = kwargs["backend"]
                kwargs.pop("backend")
            else:
                backend = None
            if molecule_factory is None:
                molecule_factory = FermionicBase
            molecule.fermionic_backend = fbackend
            if molecule_arguments is None:
                molecule_arguments = {
                    "fermionic_backend": fbackend,
                    "parameters": molecule.parameters,
                }
            else:
                molecule_arguments["fermionic_backend"] = fbackend
                if "parameters" not in molecule_arguments:
                    molecule_arguments["parameters"] = molecule.parameters
            if use_hcb:
                if vqe_solver_arguments is None:
                    vqe_solver_arguments = {}
                vqe_solver_arguments["restrict_to_hcb"] = True
            if vqe_solver is None:
                vqe_solver = solver(
                    backend=backend, circuit=circuit, fermionic_backend=fbackend
                )
    else:  # Plain Qubit molecule
        if isinstance(circuit, FCircuit):
            circuit = circuit.to_qcircuit(molecule=molecule)
    result = optimizer(
        *args,
        molecule=molecule,
        use_hcb=use_hcb,
        circuit=circuit,
        pyscf_arguments=pyscf_arguments,
        silent=silent,
        initial_guess=initial_guess,
        return_mcscf=return_mcscf,
        molecule_factory=molecule_factory,
        molecule_arguments=molecule_arguments,
        restrict_to_active_space=restrict_to_active_space,
        vqe_solver_arguments=vqe_solver_arguments,
        save_chkfile=save_chkfile,
        read_chkfile=read_chkfile,
        vqe_solver=vqe_solver,
        **kwargs,
    )
    if isinstance(molecule, HybridBase):
        result.molecule = HybridBase(
            **molecule_arguments, integral_manager=result.molecule.integral_manager
        )
    elif isinstance(molecule, FermionicBase):
        result.molecule = FermionicBase(
            parameters=result.molecule.parameters,
            integral_manager=result.molecule.integral_manager,
        )
    return result


def optimizer(
    molecule,
    circuit=None,
    vqe_solver=None,
    pyscf_arguments=None,
    silent=False,
    vqe_solver_arguments=None,
    initial_guess=None,
    return_mcscf=False,
    use_hcb=False,
    molecule_factory=None,
    molecule_arguments=None,
    restrict_to_active_space=True,
    read_chkfile: str | None = None,
    save_chkfile: str | None = None,
    *args,
    **kwargs,
):
    """Run the PySCF CASSCF optimization with a VQE solver, see ``optimize_orbitals`` for the arguments."""
    try:
        from pyscf import lib, mcscf

        from ..molecules.qubit_base.pyscf_interface import QuantumChemistryPySCF
    except Exception as exception:  # noqa: BLE001
        raise TequilaException(
            f"{exception}\noptimize_orbitals: Need pyscf to run (pip install pyscf)"
        )

    if pyscf_arguments is None:
        pyscf_arguments = {"max_cycle_macro": 10, "max_cycle_micro": 3}
    no = molecule.n_orbitals

    if not isinstance(molecule, QuantumChemistryPySCF):
        pyscf_molecule = QuantumChemistryPySCF.from_tequila(
            molecule=molecule, transformation=molecule.transformation
        )
    else:
        pyscf_molecule = molecule

    mf = pyscf_molecule._get_hf()
    result = OptimizeOrbitalsResult()
    mc = mcscf.CASSCF(mf, pyscf_molecule.n_orbitals, pyscf_molecule.n_electrons)
    mc.callback = result
    if save_chkfile is not None:
        mc.chkfile = save_chkfile
    c = pyscf_molecule.compute_constant_part()

    if circuit is None and vqe_solver is None:
        raise TequilaException(
            "optimize_orbitals: Either provide a circuit or a callable vqe_solver"
        )

    if use_hcb:
        if vqe_solver_arguments is None:
            vqe_solver_arguments = {}
        vqe_solver_arguments["restrict_to_hcb"] = True
        # consistency check
        n_qubits = len(circuit.qubits)
        n_orbitals = molecule.n_orbitals
        if n_qubits > n_orbitals:
            warnings.warn(
                f"Potential inconsistency in orbital optimization: use_hcb is switched on but we have\n n_qubits={n_qubits} in the circuit\n n_orbital={n_orbitals} in the molecule\n",
                TequilaWarning,
            )

    if molecule_arguments is None:
        molecule_arguments = {
            "parameters": pyscf_molecule.parameters,
            "transformation": molecule.transformation,
        }

    wrapper = PySCFVQEWrapper(
        *args,
        molecule_arguments=molecule_arguments,
        n_electrons=pyscf_molecule.n_electrons,
        const_part=c,
        circuit=circuit,
        vqe_solver_arguments=vqe_solver_arguments,
        silent=silent,
        vqe_solver=vqe_solver,
        molecule_factory=molecule_factory,
        **kwargs,
    )
    mc.fcisolver = wrapper
    mc.internal_rotation = True
    if pyscf_arguments is not None:
        for k, v in pyscf_arguments.items():
            if hasattr(mc, str(k)):
                setattr(mc, str(k), v)
            else:
                print(f"unknown arguments: {k}")
    if not silent:
        print("Optimizing Orbitals with PySCF and VQE Solver:")
        print("{:25} : {}".format("pyscf_arguments", pyscf_arguments))
        print(wrapper)
    if initial_guess is not None:
        if hasattr(initial_guess, "lower"):
            if (
                "random" in initial_guess.lower()
                or "near_zero" in initial_guess.lower()
            ):
                scale = 1.0e-3
                if "random" in initial_guess.lower():
                    scale = 1.0
                loc = 0.0
                if "scale" in initial_guess.lower():
                    scale = float(
                        initial_guess.split("scale")[1].split("_")[0].split("=")[1]
                    )
                if "loc" in initial_guess.lower():
                    loc = float(
                        initial_guess.split("loc")[1].split("_")[0].split("=")[1]
                    )
                initial_guess = numpy.eye(no) + numpy.random.normal(
                    scale=scale, loc=loc, size=no * no
                ).reshape(no, no)
            else:
                raise TequilaException(f"Unknown initial_guess={initial_guess.lower()}")

        assert len(initial_guess.shape) == 2
        assert initial_guess.shape[0] == no
        assert initial_guess.shape[1] == no
        initial_guess = mcscf.project_init_guess(mc, initial_guess)
        mc.kernel(mo_coeff=initial_guess)
    elif read_chkfile is not None:
        chk = lib.chkfile.load(read_chkfile, "mcscf")
        mo_coeff = chk["mo_coeff"]
        ci0 = chk.get("ci", None)
        mc.kernel(mo_coeff=mo_coeff, ci0=ci0)
    else:
        mc.kernel()
    # make new molecule
    mo_coeff = mc.mo_coeff
    transformed_molecule = pyscf_molecule.transform_orbitals(
        orbital_coefficients=mo_coeff, name="optimized"
    )
    result.molecule = transformed_molecule
    result.old_molecule = molecule
    result.mo_coeff = mo_coeff
    result.energy = mc.e_tot
    result.U = wrapper.circuit
    if len(wrapper.history) > 0:
        result.variables = wrapper.history[-1].variables

    if return_mcscf:
        result.mcscf_object = mc

    return result


@dataclass
class PySCFVQEWrapper:
    """Wrapper making a VQE solver usable as the FCI solver of PySCF's CASSCF.

    ``kernel`` builds a molecule from the active-space integrals, runs the VQE and
    stores the energy and reduced density matrices requested by PySCF.

    Attributes
    ----------
    n_electrons : int
        Number of electrons in the active space.
    molecule_arguments : dict
        Arguments used to construct the molecules.
    history : list
        Optimization results of all VQE runs.
    const_part : float
        Constant energy contribution added as nuclear repulsion.
    silent : bool
        If True, suppress printout.
    vqe_solver : callable, optional
        Custom solver, see ``optimize_orbitals``.
    circuit : FCircuit or QCircuit, optional
        VQE ansatz.
    vqe_solver_arguments : dict
        Arguments for the VQE solver.
    molecule_factory : callable, optional
        Constructor for the molecules built at each iteration.
    """

    # needs initialization
    n_electrons: int = None
    molecule_arguments: dict = None

    # internal data
    rdm1: numpy.ndarray = None
    rdm2: numpy.ndarray = None
    one_body_integrals: numpy.ndarray = None
    two_body_integrals: numpy.ndarray = None
    history: list = field(default_factory=list)

    # optional
    const_part: float = 0.0
    silent: bool = False
    vqe_solver: typing.Callable = None
    circuit: FCircuit | QCircuit | None = None
    vqe_solver_arguments: dict = field(default_factory=dict)
    molecule_factory: typing.Callable = None

    def reorder(self, M, ordering, to):
        # convenience since we need to reorder
        # all the time
        """Return the tensor ``M`` converted from index ordering ``ordering`` to ``to``."""
        M = NBodyTensor(elems=M, ordering=ordering)
        M.reorder(to=to)
        return M.elems

    def kernel(self, h1, h2, *args, **kwargs):
        """Run the VQE for integrals ``h1`` and ``h2`` and return the energy and None (PySCF FCI-solver interface)."""
        if self.history is None:
            self.history = []
        h2of = self.reorder(h2, "mulliken", "openfermion")
        restrict_to_hcb = (
            self.vqe_solver_arguments is not None
            and "restrict_to_hcb" in self.vqe_solver_arguments
            and self.vqe_solver_arguments["restrict_to_hcb"]
        )

        if self.molecule_factory is None:
            molecule = QuantumChemistryBase(
                one_body_integrals=h1,
                two_body_integrals=h2of,
                nuclear_repulsion=self.const_part,
                n_electrons=self.n_electrons,
                **self.molecule_arguments,
            )
        else:
            molecule = self.molecule_factory(
                one_body_integrals=h1,
                two_body_integrals=h2of,
                nuclear_repulsion=self.const_part,
                n_electrons=self.n_electrons,
                **self.molecule_arguments,
            )
        if restrict_to_hcb:
            H = molecule.make_hardcore_boson_hamiltonian()
        else:
            H = molecule.make_hamiltonian()

        rdm1 = None
        rdm2 = None
        if self.vqe_solver is not None:
            vqe_solver_arguments = {}
            if self.vqe_solver_arguments is not None:
                vqe_solver_arguments = self.vqe_solver_arguments
            result = self.vqe_solver(
                H=H, circuit=self.circuit, molecule=molecule, **vqe_solver_arguments
            )
            if hasattr(self.vqe_solver, "compute_rdms"):
                rdm1, rdm2 = self.vqe_solver.compute_rdms(
                    U=self.circuit,
                    variables=result.variables,
                    molecule=molecule,
                    use_hcb=restrict_to_hcb,
                )
                rdm2 = self.reorder(rdm2, "dirac", "mulliken")
        elif self.circuit is None:
            raise TequilaException(
                "Orbital Optimizer: Either provide a callable vqe_solver or a circuit"
            )
        else:
            U = self.circuit
            E = ExpectationValue(H=H, U=U)
            optimizer_arguments = {}
            if (
                self.vqe_solver_arguments is not None
                and "optimizer_arguments" in self.vqe_solver_arguments
            ):
                optimizer_arguments = self.vqe_solver_arguments["optimizer_arguments"]
            if self.silent is not None and "silent" not in optimizer_arguments:
                optimizer_arguments["silent"] = True

            result = minimize(E, **optimizer_arguments)
        if hasattr(result, "circuit"):
            # potential adaptive ansatz
            U = result.circuit
            self.circuit = U
        else:
            # static ansatz
            U = self.circuit

        if rdm1 is None or rdm2 is None:
            rdm1, rdm2 = molecule.compute_rdms(
                U=U,
                variables=result.variables,
                spin_free=True,
                get_rdm1=True,
                get_rdm2=True,
                use_hcb=restrict_to_hcb,
            )
            rdm2 = self.reorder(rdm2, "dirac", "mulliken")
        if not self.silent:
            print("{:20} : {}".format("energy", result.energy))
            if len(self.history) > 0:
                print(
                    "{:20} : {}".format(
                        "deltaE", result.energy - self.history[-1].energy
                    )
                )
                print(
                    "{:20} : {}".format(
                        "||delta RDM1||", numpy.linalg.norm(self.rdm1 - rdm1)
                    )
                )
        self.history.append(result)
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.one_body_integrals = h1
        self.two_body_integrals = h2
        return result.energy, None

    def make_rdm12(self, *args, **kwargs):
        """Return the stored one- and two-body reduced density matrices (PySCF FCI-solver interface)."""
        return self.rdm1, self.rdm2

    def __str__(self):
        """Return the wrapper attributes, one per line."""
        result = f"{type(self).__name__}\n"
        for k, v in self.__dict__.items():
            if k == "circuit" and v is not None:
                result += "{:30} : {}\n".format(
                    k, f"{len(v.gates)} gates, {len(v.extract_variables())} parameters"
                )
            else:
                result += f"{k:30} : {v}\n"
        return result
